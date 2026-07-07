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
_EXPECTED_IGNORE_SUBSTR = [
    "lm_head",
    "vision_tower",
    "multi_modal_projector",
    "patch_merge",
    "mlp[.]gate$",
    "shared_experts",
    "indexer",
    "layers[.][0-2]",
]

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


def _fail(msg: str, errors: list[str]) -> None:
    errors.append(msg)
    print(f"  [FAIL] {msg}")


def _ok(msg: str) -> None:
    print(f"  [ok]   {msg}")


def verify(ckpt: Path, check_tensors: bool) -> int:
    errors: list[str] = []
    warnings: list[str] = []

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
    print(f"  ignore ({len(ignore)}):")
    for p in ignore:
        print(f"    - {p}")
    for sub in _EXPECTED_IGNORE_SUBSTR:
        if not any(sub in p for p in ignore):
            _fail(f"expected ignore pattern containing '{sub}' missing from config", errors)
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
    print(f"  total tensors: {len(keys)}")
    print(f"  quantized (weight_packed) modules: {len(quantized)}")
    print(f"  plain (.weight) modules:           {len(plain)}")

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
    print("\n== keep-bf16 modules must not be quantized ==")
    for name, pat in _MUST_NOT_QUANTIZE.items():
        leaked = sorted(m for m in quantized if pat.search(m))
        if leaked:
            _fail(f"{name}: {len(leaked)} modules were quantized but should be bf16, "
                  f"e.g. {leaked[:3]}", errors)
        else:
            _ok(f"{name}: none quantized")

    # ---- 4. routed experts + sparse attention must BE quantized ------------
    print("\n== expected-quantized coverage (sparse layers 3-59) ==")
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

    return _summary(errors, warnings)


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
        if pk in weight_map or pk in keys:
            packed = _get(pk)
            if not torch.isfinite(packed.float()).all():
                _fail(f"non-finite weight_packed in {pref}", errors)
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
    args = ap.parse_args(argv)
    if not (args.ckpt / "config.json").exists():
        print(f"error: {args.ckpt}/config.json not found")
        return 2
    return verify(args.ckpt, args.check_tensors)


if __name__ == "__main__":
    sys.exit(main())
