"""Rigorously verify a MiniMax-M3 (or generic MoE) compressed-tensors checkpoint.

Motivated by the Qwen3 MoE quantization incidents (see
``Model-Optimizer/modelopt-test/BUGS_AND_FIXES.md`` bug #1: wrong expert layout ->
~56k vs ~13k quantizers, and ``pipeline/README.md``: MoE gate pruned from saved
``ignore`` -> vLLM mis-load -> garbage output). The single most transferable check
is: **count what actually got quantized and confirm it matches the architecture**,
and confirm the keep-bf16 modules were *not* quantized and *are* listed in the saved
``ignore``.

This runs on checkpoint metadata only (``config.json`` +
``model.safetensors.index.json``), so it is fast and needs no GPU / full model load.
Pass ``--check-tensors`` to additionally open shards and validate a sample of packed
weights / scales for finiteness and group-scale shape.

Usage:
    python -m pipeline.verify_quant_checkpoint --ckpt artifacts/.../checkpoint
    python -m pipeline.verify_quant_checkpoint --ckpt <dir> --check-tensors
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections import defaultdict
from pathlib import Path

# Compressed-tensors pack-quantized param suffixes (see
# compressed_tensors/compressors/pack_quantized/base.py::compression_param_names).
_PACKED = "weight_packed"
_SCALE = "weight_scale"
_SHAPE = "weight_shape"

# Modules that MUST stay bf16 for MiniMax-M3 (keep-bf16 recipe). Each entry is a
# regex over the module path (without the trailing param name).
_MUST_NOT_QUANTIZE = {
    "vision_tower": re.compile(r"vision_tower"),
    "multi_modal_projector": re.compile(r"multi_modal_projector"),
    "patch_merge": re.compile(r"patch_merge"),
    "moe_router_gate": re.compile(r"\.mlp\.gate$"),
    "shared_experts": re.compile(r"\.mlp\.shared_experts\."),
    "msa_indexer": re.compile(r"\.self_attn\.indexer\."),
    "dense_layers_0_2": re.compile(r"\.layers\.[0-2]\."),
    "lm_head": re.compile(r"lm_head"),
}

# Ignore patterns we expect to be persisted into the saved config (order-independent).
#
# THIS LIST IS MODEL-SPECIFIC. It encodes the MiniMax-M3 keep-bf16 recipe, and on
# any other architecture the missing entries are false failures, not defects: a
# GLM-5.2 checkpoint has no vision_tower, multi_modal_projector, patch_merge,
# indexer or block_sparse_moe to ignore, so this list fails five times on a
# perfectly healthy checkpoint. Pass `expect_ignore=` (or --expect-ignore /
# --expect-ignore-preset) for non-M3 models.
#
# The name is kept because pipeline/tests/test_m3_routed_diagnostics_runner.py
# imports it.
_EXPECTED_IGNORE_SUBSTR = [
    "lm_head",
    "vision_tower",
    "multi_modal_projector",
    "patch_merge",
    "mlp[.]gate$",
    "shared_experts",
    "block_sparse_moe",
    "indexer",
    "layers[.][0-2]",
]

# GLM-5.2 keep-bf16 recipe, from pipeline/configs/glm52_distributed_w4afp8_*.yaml.
# Attention is ignored wholesale here (M3 quantized sparse attention), the dense
# prefix is layers 0-2 as on M3, and layer 78 (the MTP/final layer) is excluded.
# Deliberately does NOT include the layer-restriction regex that a partial-layer
# smoke adds: that is a sampling choice, not a keep-bf16 requirement, and asserting
# it would make this preset reject a full run.
_GLM52_EXPECTED_IGNORE_SUBSTR = [
    "lm_head",
    "mlp[.]gate$",
    "shared_experts",
    "self_attn",
    "layers[.][0-2]",
]

_IGNORE_PRESETS = {
    "m3": _EXPECTED_IGNORE_SUBSTR,
    "glm52": _GLM52_EXPECTED_IGNORE_SUBSTR,
}

# Naming-agnostic classification. Post-linearize the routed experts become a
# ``ModuleList`` that replaces the original fused-experts submodule *in place*, so the
# saved key is ``<layer-prefix>.<container>.<expert_idx>.<proj>`` where ``<container>``
# depends on the model (``mlp.experts``, ``block_sparse_moe.experts``, ``feed_forward``
# ...). Rather than hard-code the container we:
#   * find the layer index via the ``.layers.N.`` segment,
#   * treat any ``.self_attn.`` module as attention,
#   * treat any *other* quantized module whose path has a numeric segment followed by a
#     projection leaf as a routed expert (expert index = that numeric segment).
_LAYER = re.compile(r"\.layers\.(\d+)\.")
_ATTN_PROJ = re.compile(r"\.layers\.(\d+)\.self_attn\.([A-Za-z0-9_]+)$")
# ``...<expert_idx>.<proj_leaf>`` at the end of the path (container-agnostic).
_EXPERT_PROJ = re.compile(r"\.(\d+)\.([A-Za-z0-9_]+)$")


def _module_prefix(key: str) -> str | None:
    """Return the module path for a compressed/plain weight key, else None."""
    for suffix in (
        f".{_PACKED}",
        f".{_SCALE}",
        f".{_SHAPE}",
        ".weight_zero_point",
        ".weight",
    ):
        if key.endswith(suffix):
            return key[: -len(suffix)]
    return None


def _load_weight_keys(ckpt: Path) -> list[str]:
    idx = ckpt / "model.safetensors.index.json"
    if idx.exists():
        with idx.open(encoding="utf-8") as fh:
            return list(json.load(fh)["weight_map"].keys())
    # single-shard fallback
    single = ckpt / "model.safetensors"
    if single.exists():
        from safetensors import safe_open

        with safe_open(str(single), framework="pt") as f:
            return list(f.keys())
    raise FileNotFoundError(f"no safetensors index or single shard under {ckpt}")


# Maps an expected-ignore token to a matcher against MODULE names. The tokens are
# the historical substrings, so the presets keep working; what changed is that they
# are now resolved against the checkpoint's modules instead of against the text of
# the ignore patterns.
_IGNORE_TOKEN_MATCHERS: dict[str, re.Pattern] = {
    "lm_head": re.compile(r"lm_head"),
    "vision_tower": re.compile(r"vision_tower"),
    "multi_modal_projector": re.compile(r"multi_modal_projector"),
    "patch_merge": re.compile(r"patch_merge"),
    "mlp[.]gate$": re.compile(r"\.mlp\.gate$"),
    "shared_experts": re.compile(r"\.mlp\.shared_experts\."),
    "block_sparse_moe": re.compile(r"\.block_sparse_moe\."),
    "indexer": re.compile(r"\.self_attn\.indexer\."),
    "self_attn": re.compile(r"\.self_attn\."),
    "layers[.][0-2]": re.compile(r"\.layers\.[0-2]\."),
}


def _check_ignore_coverage(
    ckpt: Path,
    ignore: list[str],
    expected_ignore: list[str],
    errors: list[str],
    warnings: list[str],
) -> None:
    """Every module that must stay BF16 has to be covered by some ignore entry.

    A component with NO unquantized modules is vacuous rather than a failure: when
    the recipe deliberately FP8-quantizes all 75 shared experts, there is nothing
    left for an ignore entry to protect, and demanding one would require the very
    pattern that shadows them.
    """
    from pipeline.serve_ignore import checkpoint_modules, match_name

    try:
        modules, quantized = checkpoint_modules(ckpt)
    except (FileNotFoundError, KeyError, ValueError) as err:
        warnings.append(f"ignore-coverage check skipped (tensor names unreadable): {err}")
        return

    unquantized = modules - quantized
    for token in expected_ignore:
        matcher = _IGNORE_TOKEN_MATCHERS.get(token)
        if matcher is None:
            warnings.append(f"no module matcher for expected-ignore token {token!r}")
            continue
        targets = sorted(m for m in unquantized if matcher.search(m))
        if not targets:
            print(f"    {token}: no unquantized modules (vacuous)")
            continue
        uncovered = [m for m in targets if not any(match_name(m, p) for p in ignore)]
        if uncovered:
            _fail(
                f"{token}: {len(uncovered)} of {len(targets)} unquantized modules are "
                f"NOT covered by any ignore entry, so a loader will treat them as "
                f"quantized, e.g. {uncovered[:3]}",
                errors,
            )
        else:
            print(f"    {token}: all {len(targets)} unquantized modules covered")


def _fail(msg: str, errors: list[str]) -> None:
    errors.append(msg)
    print(f"  [FAIL] {msg}")


def _ok(msg: str) -> None:
    print(f"  [ok]   {msg}")


def verify(
    ckpt: Path,
    check_tensors: bool,
    dequant_base: Path | None = None,
    expect_ignore: list[str] | None = None,
    allow_fp8_components: set[str] | None = None,
) -> int:
    errors: list[str] = []
    warnings: list[str] = []
    # None keeps the historical M3 behaviour for existing callers. An explicitly
    # EMPTY list is rejected rather than treated as "nothing to check", so the
    # keep-bf16 assertion cannot be silently disabled into a vacuous pass.
    if expect_ignore is not None and not expect_ignore:
        raise ValueError(
            "expect_ignore=[] would make the keep-bf16 check vacuous; pass None "
            "for the M3 default or a non-empty list of expected substrings"
        )
    expected_ignore = (
        _EXPECTED_IGNORE_SUBSTR if expect_ignore is None else list(expect_ignore)
    )

    # ---- 1. config.json quantization_config ---------------------------------
    print("== config.json quantization_config ==")
    cfg = json.loads((ckpt / "config.json").read_text(encoding="utf-8"))
    qc = cfg.get("quantization_config")
    if not qc:
        print("  [FAIL] no quantization_config in config.json")
        return 1
    fmt = qc.get("format")
    ignore = list(qc.get("ignore", []))
    print(f"  format={fmt}")
    groups = qc.get("config_groups", {})
    for gname, g in groups.items():
        w = g.get("weights", {})
        a = g.get("input_activations")
        print(
            f"  {gname}: weights num_bits={w.get('num_bits')} "
            f"type={w.get('type')} strategy={w.get('strategy')} "
            f"group_size={w.get('group_size')} symmetric={w.get('symmetric')}; "
            f"acts={None if not a else a.get('num_bits')}"
            f"{'' if not a else '/' + str(a.get('type')) + '/dyn=' + str(a.get('dynamic'))}"
        )
    # r8 mixed recipes: an 8-bit float weight group (FP8_DYNAMIC on
    # attention / shared experts / dense MLPs) alongside the int4 expert
    # group. Its presence switches the keep-bf16 expectations below and adds
    # the fp8 coverage/format checks.
    fp8_groups = {
        gname: g
        for gname, g in groups.items()
        if g.get("weights", {}).get("num_bits") == 8
        and g.get("weights", {}).get("type") == "float"
    }
    mixed_fp8 = bool(fp8_groups)
    if mixed_fp8:
        print(f"  mixed int4+FP8 checkpoint (fp8 groups: {sorted(fp8_groups)})")
        for gname, g in fp8_groups.items():
            gfmt = g.get("format")
            if gfmt != "float-quantized":
                _fail(
                    f"fp8 group {gname} has format={gfmt!r} (expected "
                    "'float-quantized'): a global pack-quantized override "
                    "runs fp8 weights through the int4 packer — see "
                    "BUGS_AND_FIXES.md, r8 smoke v2 (2026-07-23)",
                    errors,
                )
    print(f"  ignore ({len(ignore)}):")
    for p in ignore:
        print(f"    - {p}")
    # COVERAGE, NOT PATTERN TEXT. This used to assert that each expected substring
    # appeared in some ignore pattern, and that broke the moment
    # _persist_ignore_to_config started resolving broad patterns against the saved
    # tensors (2026-08-28). Two ways it failed on a healthy checkpoint:
    #
    #   - 're:.*mlp[.]shared_experts[.].*' is correctly ABSENT when every shared
    #     expert is FP8-quantized, because keeping it would shadow them and make a
    #     loader serve them unquantized. The full run quantizes all 75, so the
    #     text check would fail every production save.
    #   - the concrete entries replacing 're:.*layers[.][0-2][.].*' name real
    #     modules and so do not contain the literal string 'layers[.][0-2]'.
    #
    # What actually matters is what the original check was a proxy for: every
    # module that must stay BF16 is covered by SOME ignore entry, so no loader
    # treats it as quantized. That is decidable against the artifact, it composes
    # with pattern resolution, and it is strictly stronger -- a pattern can be
    # present and still miss modules.
    _check_ignore_coverage(ckpt, ignore, expected_ignore, errors, warnings)
    if not errors:
        _ok("all expected keep-bf16 ignore patterns present in saved config")

    # ---- 2. bucket weight keys ---------------------------------------------
    print("\n== weight key inventory ==")
    keys = _load_weight_keys(ckpt)
    quantized: set[str] = set()
    plain: set[str] = set()
    scale_keys: set[str] = set()
    for k in keys:
        pref = _module_prefix(k)
        if pref is None:
            continue
        if k.endswith(f".{_PACKED}"):
            quantized.add(pref)
        elif k.endswith(f".{_SCALE}"):
            scale_keys.add(pref)
        elif k.endswith(".weight"):
            plain.add(pref)
    # fp8 float-quantized modules keep a plain fp8 ``.weight`` plus a
    # per-channel ``.weight_scale`` (no ``weight_packed``).
    fp8_quantized = plain & scale_keys
    plain -= fp8_quantized
    print(f"  total tensors: {len(keys)}")
    print(f"  quantized (weight_packed) modules: {len(quantized)}")
    print(f"  fp8 float-quantized modules:       {len(fp8_quantized)}")
    print(f"  plain (.weight) modules:           {len(plain)}")
    if fp8_quantized and not mixed_fp8:
        _fail(
            f"{len(fp8_quantized)} modules look float-quantized but the config "
            f"declares no fp8 weight group, e.g. {sorted(fp8_quantized)[:3]}",
            errors,
        )

    # Self-diagnosing: show the actual leaf (proj) names of quantized modules so a
    # naming mismatch is obvious rather than silently reported as "nothing quantized".
    leaf_counts: dict[str, int] = defaultdict(int)
    for m in quantized:
        leaf_counts[m.rsplit(".", 1)[-1]] += 1
    print("  quantized leaf names:")
    for leaf, c in sorted(leaf_counts.items(), key=lambda kv: -kv[1]):
        print(f"    {c:>6}  *.{leaf}")

    # Show a few full module paths so the real (container) layout is visible -- this is
    # exactly the naming vLLM must load the experts under.
    print("  sample quantized module paths:")
    for m in sorted(quantized)[:4]:
        print(f"    {m}")
    non_attn = sorted(m for m in quantized if ".self_attn." not in m)
    for m in non_attn[:4]:
        print(f"    {m}")

    # every quantized module must also have a scale
    missing_scale = quantized - scale_keys
    if missing_scale:
        _fail(f"{len(missing_scale)} quantized modules missing weight_scale, e.g. "
              f"{sorted(missing_scale)[:3]}", errors)
    else:
        _ok("every quantized module has a matching weight_scale")

    # ---- 3. keep-bf16 modules must NOT be quantized -------------------------
    # In mixed int4+FP8 checkpoints the fp8 group legitimately covers the
    # shared experts and dense MLPs (and attention, which has no entry here);
    # int4 must still never touch them, and every other keep-bf16 module must
    # not be quantized in ANY format.
    print("\n== keep-bf16 modules must not be quantized ==")
    # Components a MIXED int4+FP8 recipe is allowed to FP8-quantize. Extendable
    # per-run because the DSA indexer moved from "must be BF16" to a decision:
    # zai-org's own FP8 release and PhalaCloud's W4AFP8 both quantize
    # indexer.wq_b and indexer.wk, our BF16 stance turned out to rest on no
    # measurement, and it is worth only ~0.7% of decode weight traffic either way.
    # A run that deliberately matches upstream must not be failed by a gate
    # encoding the older stance as a constant.
    fp8_ok_in_mixed = {"shared_experts", "dense_layers_0_2"} | set(allow_fp8_components or ())
    for name, pat in _MUST_NOT_QUANTIZE.items():
        packed_leak = sorted(m for m in quantized if pat.search(m))
        fp8_hits = sorted(m for m in fp8_quantized if pat.search(m))
        allow_fp8 = mixed_fp8 and name in fp8_ok_in_mixed
        leaked = packed_leak + ([] if allow_fp8 else fp8_hits)
        if leaked:
            _fail(f"{name}: {len(leaked)} modules were quantized but should be "
                  f"{'bf16 or fp8' if allow_fp8 else 'bf16'}, "
                  f"e.g. {leaked[:3]}", errors)
        elif allow_fp8 and fp8_hits:
            _ok(f"{name}: no int4 leak; {len(fp8_hits)} fp8-quantized (mixed recipe)")
        else:
            _ok(f"{name}: none quantized")

    # ---- 4. routed experts + sparse attention must BE quantized ------------
    # Layer range, expert count and projection names are all DISCOVERED from the
    # checkpoint below, so this section is model-agnostic; the old heading said
    # "sparse layers 3-59", which is M3's range and merely misleading on GLM-5.2.
    print("\n== expected-quantized coverage (layer range discovered below) ==")
    # experts_by_layer[layer][expert_idx] = {proj names present}
    experts_by_layer: dict[int, dict[int, set[str]]] = defaultdict(lambda: defaultdict(set))
    attn_by_layer: dict[int, set[str]] = defaultdict(set)
    expert_container: set[str] = set()
    for m in quantized:
        lm = _LAYER.search(m)
        if lm is None:
            continue
        layer = int(lm.group(1))
        am = _ATTN_PROJ.search(m)
        if am:
            attn_by_layer[layer].add(am.group(2))
            continue
        # non-attention quantized module in a decoder layer => routed expert.
        em = _EXPERT_PROJ.search(m)
        if em:
            idx, leaf = em.group(1), em.group(2)
            experts_by_layer[layer][int(idx)].add(leaf)
            # container = path segment between ".layers.N." and ".<idx>.<leaf>"
            tail = m[lm.end():]  # e.g. "mlp.experts.0.gate_proj"
            expert_container.add(tail[: -len(f".{idx}.{leaf}")])

    sparse_layers = sorted(experts_by_layer)
    if not sparse_layers:
        _fail("no routed-expert projections detected -- inspect the 'sample quantized "
              "module paths' above; experts may be stored fused (single "
              "experts.gate_up_proj) rather than per-expert Linears",
              errors)
        return _summary(errors, warnings)

    # discover the projection-name set and expert count from the data itself
    proj_names: set[str] = set()
    n_experts = 0
    for lyr in sparse_layers:
        for e, projs in experts_by_layer[lyr].items():
            proj_names |= projs
            n_experts = max(n_experts, e + 1)
    attn_names = set().union(*attn_by_layer.values()) if attn_by_layer else set()
    print(f"  expert container path segment(s): {sorted(expert_container)}")
    print(f"  detected sparse layers with experts: {sparse_layers[0]}..{sparse_layers[-1]} "
          f"({len(sparse_layers)} layers)")
    print(f"  detected experts per layer: {n_experts}")
    print(f"  detected expert projections: {sorted(proj_names)}")
    print(f"  detected sparse-attn projections: {sorted(attn_names)}")

    for lyr in sparse_layers:
        layer_experts = experts_by_layer[lyr]
        # every expert index present with the full projection set
        for e in range(n_experts):
            got = layer_experts.get(e, set())
            missing = proj_names - got
            if missing:
                _fail(f"layer {lyr} expert {e}: missing quantized projections {sorted(missing)}", errors)
        attn_missing = attn_names - attn_by_layer.get(lyr, set())
        if attn_missing:
            _fail(f"layer {lyr}: attention projections not quantized: {sorted(attn_missing)}", errors)
    if not any(e.startswith("layer ") for e in errors):
        _ok(f"all {len(sparse_layers)} sparse layers fully quantized "
            f"({n_experts} experts x {len(proj_names)} proj + {len(attn_names)} attn each)")

    expected_q = len(sparse_layers) * (n_experts * len(proj_names) + len(attn_names))
    print(f"  expected quantized Linears = {len(sparse_layers)} x "
          f"({n_experts}*{len(proj_names)} + {len(attn_names)}) = {expected_q}")
    print(f"  actual quantized Linears   = {len(quantized)}")
    if len(quantized) != expected_q:
        warnings.append(
            f"quantized count {len(quantized)} != expected {expected_q} "
            f"(diff {len(quantized) - expected_q}); inspect extras/missing")
        print(f"  [warn] count mismatch (diff {len(quantized) - expected_q})")
    else:
        _ok("quantized Linear count matches expectation exactly")

    # ---- 4b. fp8 (mixed recipe) coverage ------------------------------------
    if mixed_fp8:
        print("\n== fp8 (mixed recipe) coverage ==")
        fp8_attn = sorted(m for m in fp8_quantized if ".self_attn." in m)
        fp8_shared = sorted(m for m in fp8_quantized if "shared_experts" in m)
        fp8_dense = sorted(
            m for m in fp8_quantized
            if re.search(r"\.layers\.[0-2]\.mlp\.", m)
        )
        fp8_by_layer: dict[int, int] = defaultdict(int)
        for m in fp8_quantized:
            lm = _LAYER.search(m)
            if lm:
                fp8_by_layer[int(lm.group(1))] += 1
        print(f"  fp8 modules: attn={len(fp8_attn)} shared={len(fp8_shared)} "
              f"dense={len(fp8_dense)}")
        print(f"  fp8 layers covered: {sorted(fp8_by_layer)}")
        for label, mods in (("attention", fp8_attn),
                            ("shared_experts", fp8_shared),
                            ("dense mlp (layers 0-2)", fp8_dense)):
            if mods:
                _ok(f"fp8 covers {label} ({len(mods)} modules)")
            else:
                _fail(f"fp8 group present but no fp8-quantized {label} "
                      "modules found (target regex mismatch?)", errors)

    # ---- 5. width / group-size divisibility --------------------------------
    print("\n== geometry (256 serve constraint + group_size) ==")
    tc = cfg.get("text_config", cfg)
    inter = tc.get("intermediate_size")
    gsize = None
    for g in groups.values():
        gsize = g.get("weights", {}).get("group_size", gsize)
    print(f"  routed-expert intermediate_size = {inter}; weight group_size = {gsize}")
    if inter is not None:
        if inter % 256 != 0:
            _fail(f"expert intermediate_size {inter} not a multiple of 256 "
                  f"(vLLM CUTLASS W4A8 MoE kernel constraint at full/EP width)", errors)
        else:
            _ok(f"intermediate_size {inter} = {inter // 256} x 256 (EP width OK)")
        if gsize and inter % gsize != 0:
            _fail(f"intermediate_size {inter} not divisible by group_size {gsize}", errors)

    # ---- 6. optional tensor-level checks -----------------------------------
    if check_tensors:
        print("\n== sampled tensor checks (finiteness + scale shape) ==")
        _check_tensors(ckpt, keys, gsize, errors, warnings)

    # ---- 7. optional dequant-vs-base value check ----------------------------
    if dequant_base is not None:
        print("\n== sampled dequant-vs-base checks ==")
        _check_dequant(ckpt, Path(dequant_base), keys, errors, warnings)
        print("\n== sampled untouched-tensor checks ==")
        _check_untouched(ckpt, Path(dequant_base), keys, errors, warnings)

    return _summary(errors, warnings)


# Candidate checkpoints use transformers projection names; some base exports
# (and all vLLM-portable re-exports) use w1/w2/w3 for routed experts.
_ROUTED_ALIASES = {".gate_proj.": ".w1.", ".down_proj.": ".w2.", ".up_proj.": ".w3."}

# Thresholds calibrated on known checkpoints (2026-07-19): consistent AWQ W4
# g128 (r9) fits base with residual 0.09-0.12 and per-column smoothing scales
# 0.7-1.4; garbage-scale checkpoints (r15/r3) are NaN / O(1). GPTQ (no
# smoothing) fits with scale ~1.0.
# RECALIBRATED 2026-08-29 to 0.40, with external evidence. The prior 0.25 bound,
# and the "up_proj 0.194 / gate_proj 0.089 / down_proj 0.064" numbers that used to
# be quoted here, are both retracted:
#
#   * Those figures could not be reproduced. Re-measuring three checkpoints (two
#     32x512 runs and one 256x2048) with the current separable fit gives
#     gate_proj 0.117, up_proj 0.116, down_proj 0.133 -- identical to three
#     decimals across all three, and independent of calibration size, which is
#     correct because the residual is int4 grid resolution and not a calibration
#     statistic. There is no up_proj asymmetry. A sweep of all 256 experts of
#     layer 3 gives median 0.115, max 0.128, ZERO above 0.25.
#   * The band has an analytic anchor now. For symmetric int4 group-g the
#     relative residual is (max|w|/rms|w|) / (7*sqrt(12)); the measured
#     within-group ratio 2.91 predicts 0.120 against 0.116 measured, and
#     down_proj's ratio 3.38 predicts 0.139 against 0.133. So ~0.12 is the FLOOR
#     for this scheme, not a warning sign.
#   * 0.25 rejected a known-good production checkpoint. PhalaCloud/GLM-5.2-W4AFP8
#     -- served by SGLang, reported at no measurable quality loss -- measures
#     0.26 against the BF16 source, because they apply AWQ weight CLIPPING
#     (their scales run 1.28x below the least-squares optimum, ~9/7) which we do
#     not implement. Clipping deliberately trades weight-space fidelity for
#     activation-weighted error, so a higher residual there is method, not damage.
#
# 0.40 keeps the gate useful against what it was built for -- garbage scales are
# NaN or O(1), and a lost/misdirected write is far above 0.40 -- while no longer
# cutting into either our own healthy distribution or a clipping-based method's.
# Weight-space residual is NOT a cross-method quality metric; do not use it to
# compare our checkpoints against anyone else's.
_DEQUANT_MAX_RESID = 0.40
_DEQUANT_SCALE_RANGE = (0.2, 5.0)

# M3 layers sampled deterministically on top of the even spread: dead
# (8/10-13) or near-dead (9/14) norm channels — the AWQ scale-degeneracy
# class from full r4 — plus the deepest layer (59), where grid-search error
# is largest and r4 kept ~64 partially-corrupt modules that a 20-module
# even spread misses with ~94% probability.
_DEQUANT_RISK_LAYERS = (8, 9, 10, 11, 12, 13, 14, 59)


def _check_dequant(ckpt, base, keys, errors, warnings):
    """Dequantize sampled packed modules and require them to match the base
    weights up to a fitted per-input-column scale (absorbs AWQ smoothing; ~1.0
    for GPTQ) within W4 quantization error. Catches garbage scales, corrupted
    packed weights, and any transform the saved tensors cannot explain —
    value-level coverage the finiteness check alone does not give."""
    import torch
    from safetensors import safe_open

    try:
        from compressed_tensors.compressors.pack_quantized.base import (
            unpack_from_int32,
        )
    except ImportError as err:
        warnings.append(f"dequant check skipped (no unpacker): {err}")
        return

    # Single-shard checkpoints have no index. save_pretrained omits it for anything
    # small enough, which is every subset probe and small smoke -- and this raised
    # FileNotFoundError on exactly those, AFTER a successful quantization and save
    # (2026-08-28, the one-layer indexer smoke). The same assumption was silently
    # skipping the smooth-fold gate in m3_checkpoint_scale_audit; this one at least
    # failed loudly.
    from pipeline.serve_ignore import weight_map_of

    weight_map = weight_map_of(ckpt)
    base_map = weight_map_of(base)

    sample_scales = [k for k in keys if k.endswith(f".{_SCALE}")]
    sample = sample_scales[:: max(1, len(sample_scales) // 20)][:20]
    for layer in _DEQUANT_RISK_LAYERS:
        tag = f"layers.{layer}."
        in_layer = [k for k in sample_scales if tag in k]
        for pick in in_layer[:1] + in_layer[-1:]:
            if pick not in sample:
                sample.append(pick)
    opened: dict[tuple, object] = {}

    def _get(root, wmap, k):
        shard = (root, wmap[k])
        if shard not in opened:
            opened[shard] = safe_open(str(root / wmap[k]), framework="pt")
        return opened[shard].get_tensor(k)

    def _base_key(pref):
        cand = f"{pref}.weight"
        if cand in base_map:
            return cand
        for tf_name, alias in _ROUTED_ALIASES.items():
            if tf_name.strip(".") == pref.rsplit(".", 1)[-1]:
                aliased = pref.rsplit(".", 1)[0] + alias + "weight"
                if aliased in base_map:
                    return aliased
        return None

    checked = 0
    for sk in sample:
        pref = sk[: -len(f".{_SCALE}")]
        bk = _base_key(pref)
        pk, shpk = f"{pref}.{_PACKED}", f"{pref}.{_SHAPE}"
        if bk is None or pk not in weight_map or shpk not in weight_map:
            continue
        scale = _get(ckpt, weight_map, sk).float()
        shape = torch.Size(_get(ckpt, weight_map, shpk).tolist())
        q = unpack_from_int32(_get(ckpt, weight_map, pk), 4, shape).float()
        gsize_mod = shape[1] // scale.shape[1]
        w = q * scale.repeat_interleave(gsize_mod, dim=1)
        base_w = _get(base, base_map, bk).float()
        if base_w.shape != w.shape:
            warnings.append(f"{pref}: base shape {tuple(base_w.shape)} != "
                            f"dequant {tuple(w.shape)}; skipped")
            continue
        checked += 1
        if not torch.isfinite(w).all():
            _fail(f"non-finite dequantized weight in {pref}", errors)
            continue
        # SEPARABLE fit (per-output-row x per-input-column), not per-column only.
        # A per-column fit absorbs the norm fold, which applies to input columns,
        # but the AWQ `up_proj -> down_proj` mapping divides up_proj along its
        # OUTPUT ROWS (down_proj consumes those channels), and no per-column scale
        # can absorb that, so the separable form is the right one to keep.
        #
        # RETRACTED (2026-08-29): this comment used to claim per-column 0.267 /
        # separable 0.194 for up_proj against 0.089 gate_proj and 0.064 down_proj,
        # and explained the gap as "AWQ shifting error on purpose". Both the
        # numbers and the explanation are withdrawn. Re-measuring three
        # checkpoints gives gate 0.117 / up 0.116 / down 0.133 -- no asymmetry to
        # explain -- and the old down_proj figure of 0.064 sat BELOW the analytic
        # int4 floor, which a separable fit cannot do: it has rows+cols free
        # parameters against rows*cols elements, nowhere near enough to absorb
        # rounding error. See the _DEQUANT_MAX_RESID block for the analytic anchor
        # and the external calibration against PhalaCloud's checkpoint.
        row = torch.ones(base_w.shape[0], dtype=base_w.dtype)
        col = torch.ones(base_w.shape[1], dtype=base_w.dtype)
        for _ in range(8):  # alternating least squares; converges in a few passes
            pred = base_w * row.unsqueeze(1)
            col = (w * pred).sum(dim=0) / (pred * pred).sum(dim=0).clamp_min(1e-12)
            pred = base_w * col
            row = (w * pred).sum(dim=1) / (pred * pred).sum(dim=1).clamp_min(1e-12)
        col_scale = col
        resid = ((w - base_w * row.unsqueeze(1) * col).norm() / base_w.norm()).item()
        # the fitted scale is only meaningful for columns carrying real mass,
        # and isolated outliers are legitimate quantization behavior (GPTQ can
        # zero a weak or even single significant column) — only a systematic
        # fraction outside the plausible range indicates a lost/garbage
        # transform, which by nature hits most columns
        col_norm = base_w.norm(dim=0)
        significant = col_norm > 0.1 * col_norm.median()
        sig_scales = col_scale[significant]
        out_of_range = (
            (sig_scales < _DEQUANT_SCALE_RANGE[0])
            | (sig_scales > _DEQUANT_SCALE_RANGE[1])
        )
        out_frac = out_of_range.float().mean().item() if significant.any() else 0.0
        if resid > _DEQUANT_MAX_RESID:
            _fail(f"dequant mismatch in {pref}: resid={resid:.3f} "
                  f"(max {_DEQUANT_MAX_RESID})", errors)
        elif out_frac > 0.01:
            _fail(f"implausible column scales in {pref}: {out_frac:.1%} of "
                  f"significant columns outside {_DEQUANT_SCALE_RANGE}", errors)
    if checked and not any("dequant" in e or "column scales" in e for e in errors):
        _ok(f"sampled {checked} modules: dequantized weights match base "
            f"(resid <= {_DEQUANT_MAX_RESID}, scales sane)")
    elif not checked:
        warnings.append("dequant check matched no modules (name mismatch?)")


# Families the quantization recipe must not touch: bitwise-identical to base.
# Calibrated on r9 (2026-07-19): attention, embeddings, lm_head, and dense-MLP
# tensors round-trip byte-identically. Norms are excluded from the bitwise
# family — the offset-norm calibration context rewrites EVERY norm as
# w -> (1+w) -> (1+w)-1, a bf16 round-trip that is not exact — and get an
# allclose bound instead (smoothed norms' real fold is checked separately by
# the smooth-fold gate; here we only catch garbage/uninitialized values).
_IDENTITY_PATTERNS = (".self_attn.", ".embed_tokens.", "lm_head.")
_NORM_PATTERNS = ("norm.weight",)  # layernorm, q_norm/k_norm, final norm ...
# Smoothed norms legitimately move by the fold ((1+w)/s - 1, |delta| < ~1 for
# observed s in [0.7, 1.4]); only values far outside any plausible fold are
# corruption. The smooth-fold gate checks the fold itself.
_NORM_MAX_DELTA = 5.0


def _check_untouched(ckpt, base, keys, errors, warnings):
    """Sampled comparison of tensors the recipe should leave alone. A lost or
    misdirected write (the disk-offload bug class) that lands outside the
    quantized experts would corrupt exactly these tensors, and neither the
    dequant check nor the fold gate reads them."""
    import torch
    from safetensors import safe_open

    # Single-shard checkpoints have no index. This is the THIRD site with the
    # same assumption: m3_checkpoint_scale_audit._index silently skipped the fold
    # gate, _check_dequant raised FileNotFoundError, and fixing that one merely
    # advanced the verifier far enough to reach this one (2026-08-29, the
    # 256-sample one-layer smoke: quantization and save both succeeded, then rank
    # 0 died here). Route every index read through the one helper that handles
    # both layouts so there is no fourth site.
    from pipeline.serve_ignore import weight_map_of

    weight_map = weight_map_of(ckpt)
    base_map = weight_map_of(base)
    opened: dict[tuple, object] = {}

    def _get(root, wmap, k):
        shard = (root, wmap[k])
        if shard not in opened:
            opened[shard] = safe_open(str(root / wmap[k]), framework="pt")
        return opened[shard].get_tensor(k)

    key_set = set(keys)

    def _sample(patterns, limit, exclude=()):
        hits = [k for k in keys
                if any(p in k for p in patterns)
                and not any(x in k for x in exclude)
                and k.endswith(".weight") and k in base_map
                # quantized-in-any-format modules (e.g. r8's fp8 attention)
                # are not "untouched" — they carry a weight_scale sibling
                and k[: -len(".weight")] + f".{_SCALE}" not in key_set]
        return hits[:: max(1, len(hits) // limit)][:limit]

    ident = _sample(_IDENTITY_PATTERNS, 12, exclude=_NORM_PATTERNS)
    norms = _sample(_NORM_PATTERNS, 12)
    ok_ident = ok_norm = 0
    for k in ident:
        a, b = _get(base, base_map, k), _get(ckpt, weight_map, k)
        if a.shape != b.shape or not torch.equal(a, b):
            _fail(f"untouched tensor differs from base: {k}", errors)
        else:
            ok_ident += 1
    for k in norms:
        a = _get(base, base_map, k).float()
        b = _get(ckpt, weight_map, k).float()
        if a.shape != b.shape or not torch.isfinite(b).all():
            _fail(f"norm tensor corrupt (shape/non-finite): {k}", errors)
            continue
        delta = (b - a).abs().max().item()
        if delta > _NORM_MAX_DELTA:
            _fail(f"norm deviates beyond any plausible fold: {k} "
                  f"(max abs delta {delta:.2f})", errors)
        else:
            ok_norm += 1
    if not ident and not norms:
        warnings.append("untouched check matched no tensors (name mismatch?)")
    elif not any("untouched tensor differs" in e or "norm" in e for e in errors):
        _ok(f"sampled {ok_ident} identity + {ok_norm}/{len(norms)} norm "
            f"tensors match base")


def _check_tensors(ckpt, keys, gsize, errors, warnings):
    import torch
    from safetensors import safe_open

    idx_path = ckpt / "model.safetensors.index.json"
    weight_map = json.loads(idx_path.read_text())["weight_map"] if idx_path.exists() else {
        k: "model.safetensors" for k in keys
    }
    # sample a handful of quantized modules across layers
    sample_scales = [k for k in keys if k.endswith(f".{_SCALE}")]
    sample = sample_scales[:: max(1, len(sample_scales) // 20)][:20]
    opened: dict[str, object] = {}

    def _get(k):
        shard = weight_map[k]
        if shard not in opened:
            opened[shard] = safe_open(str(ckpt / shard), framework="pt")
        return opened[shard].get_tensor(k)

    for sk in sample:
        pref = sk[: -len(f".{_SCALE}")]
        scale = _get(sk)
        if not torch.isfinite(scale).all():
            _fail(f"non-finite weight_scale in {pref}", errors)
        pk = f"{pref}.{_PACKED}"
        wk = f"{pref}.weight"
        if pk in weight_map or pk in keys:
            packed = _get(pk)
            if not torch.isfinite(packed.float()).all():
                _fail(f"non-finite weight_packed in {pref}", errors)
        elif wk in weight_map or wk in keys:
            # float-quantized (fp8) module: plain weight must actually BE fp8
            w = _get(wk)
            if w.dtype != torch.float8_e4m3fn:
                _fail(f"{pref}: has weight_scale but weight dtype {w.dtype} "
                      "(expected float8_e4m3fn for float-quantized modules)",
                      errors)
            if (scale <= 0).any():
                _fail(f"non-positive weight_scale in fp8 module {pref}", errors)
        # group-scale shape: scale should have out_features rows and
        # in_features/group_size columns for GROUP strategy
        shp = f"{pref}.{_SHAPE}"
        if (shp in weight_map or shp in keys) and gsize:
            orig = _get(shp).tolist()
            if len(orig) == 2 and scale.ndim == 2:
                exp_cols = orig[1] // gsize
                if scale.shape[1] != exp_cols:
                    warnings.append(
                        f"{pref}: weight_scale cols {scale.shape[1]} != "
                        f"in_features/group_size {exp_cols}")
    if not any("finite" in e for e in errors):
        _ok(f"sampled {len(sample)} modules: scales/packed finite, group shapes consistent")


def _summary(errors, warnings) -> int:
    print("\n== summary ==")
    for w in warnings:
        print(f"  [warn] {w}")
    if errors:
        print(f"  RESULT: FAIL ({len(errors)} error(s))")
        return 1
    print(f"  RESULT: PASS{f' ({len(warnings)} warning(s))' if warnings else ''}")
    return 0


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--ckpt", required=True, type=Path, help="checkpoint dir")
    ap.add_argument("--check-tensors", action="store_true",
                    help="also open shards and validate sampled scales/weights")
    ap.add_argument("--dequant-base", type=Path, default=None,
                    help="base checkpoint dir: dequantize sampled modules and "
                         "require value-level agreement (fitted per-column "
                         "smoothing scale, W4-error residual)")
    ap.add_argument("--expect-ignore-preset", choices=sorted(_IGNORE_PRESETS),
                    default="m3",
                    help="which keep-bf16 recipe the checkpoint's ignore list is "
                         "expected to satisfy (default: m3). The M3 preset fails "
                         "5x on a healthy GLM-5.2 checkpoint, which has no "
                         "vision_tower/projector/patch_merge/indexer to ignore.")
    ap.add_argument("--expect-ignore", action="append", default=None,
                    metavar="SUBSTR",
                    help="expected ignore substring; repeatable. Overrides "
                         "--expect-ignore-preset entirely.")
    args = ap.parse_args(argv)
    if not (args.ckpt / "config.json").exists():
        print(f"error: {args.ckpt}/config.json not found")
        return 2
    expect_ignore = args.expect_ignore or _IGNORE_PRESETS[args.expect_ignore_preset]
    print(f"== keep-bf16 expectations: "
          f"{'explicit --expect-ignore' if args.expect_ignore else args.expect_ignore_preset} "
          f"({len(expect_ignore)} patterns) ==")
    return verify(args.ckpt, args.check_tensors, args.dequant_base, expect_ignore)


if __name__ == "__main__":
    sys.exit(main())
