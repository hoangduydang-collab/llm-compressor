"""Convert a compressed-tensors W4AFP8 checkpoint into SGLang's ``w4afp8`` layout.

WHY A CONVERSION AND NOT A RE-QUANTIZATION. SGLang cannot serve our
compressed-tensors artifact: its compressed-tensors MoE path refuses the scheme
outright ("The W4A8Int8 Fused MoE scheme is implemented only for NPU for now",
measured on GPU), while its ``quant_method: w4afp8`` path serves the same
numbers in a different encoding -- confirmed in production by
PhalaCloud/GLM-5.2-W4AFP8 on H200 with SGLang >= v0.5.13.post1. Every format
fact below was read off SGLang's loader or measured on that checkpoint; none is
assumed.

WHAT CHANGES, TENSOR BY TENSOR

  routed experts (int4, group 128), per expert and projection:
      weight_packed  I32 [out, in/8]   ->  weight           I8   [out, in/2]
      weight_scale   BF16 [out, in/128] -> weight_scale_inv BF16 [out, in/128]
      weight_shape   I64 [2]           ->  dropped
    Only the nibble packing changes: int32-with-8-nibbles becomes
    int8-with-2-nibbles, adjacent pairs, EVEN column in the low nibble. The
    scale is bit-identical and merely renamed. w13_weight_scale_inv is
    registered fp32 by the loader, but ``copy_`` casts on load and PhalaCloud
    ships bf16, so bf16 is both reference-matching and lossless here.

  non-expert FP8 (attention, shared experts, dense MLP 0-2):
      weight       F8_E4M3 [out, in] ->  weight           F8_E4M3 [out, in]
      weight_scale BF16 [out, 1]     ->  weight_scale_inv F32 [out/128, in/128]
    REBUILT FROM THE BF16 SOURCE, not transcoded from the fp8 on disk.
    W4AFp8Config HARDCODES weight_block_size=[128,128], so
    Fp8LinearMethod.block_quant is always True on this path and a per-channel
    weight_scale has no fallback in either branch -- it simply fails to load.
    Rebuilding from source costs one rounding (residual 0.0265) where
    transcoding the existing per-channel fp8 would cost two (~0.051), and it
    comes with a built-in cross-check: the rebuilt weight must agree with the
    on-disk per-channel dequant to within those two roundings.

  router, norms, indexer, embeddings, lm_head: copied unchanged.

THE FOLD. AWQ's compensation fold multiplies the balance layers of the
post_attention_layernorm mapping by a per-input-channel scale s and divides the
norm by it. Balance layers there are the router, shared_experts.gate_proj/up_proj
and the int4 expert gate/up -- the mapping is grid-searched because the experts
are integer-quantized, and float-schemed modules still receive the apply-time
fold (see _is_grid_search_targeted in the AWQ modifier). So rebuilding those two
shared-expert projections from the raw source would drop the fold and break the
identity by exactly s, silently. s is recovered exactly as
``norm_base / norm_ckpt`` because the checkpoint's norm is unquantized BF16;
that carries only BF16 rounding (~0.1%), which vanishes against fp8's 2.65%.

Mappings whose balance layers are ALL float -- attention, dense MLP 0-2,
shared_experts.down_proj -- are never grid-searched, so their s is exactly 1.
This module VERIFIES that per layer rather than trusting the derivation: a
wrong assumption here is silent, and code reading is weaker evidence than the
checkpoint itself.

ONE NUMERICS CHANGE THAT IS NOT A FORMAT CHANGE. Our scheme specifies dynamic
per-token fp8 activations for the experts. SGLang's w4afp8 MoE path is static
only ("dynamic" passes config validation but is never implemented), and
process_weights_after_loading collapses the per-expert input scales to a single
max. Emitting 1.0 matches PhalaCloud and matches the loader's own pre-load
default, so it is the validated behaviour -- but the served model does NOT
reproduce our checkpoint's activation scheme, and only an eval can price that.
Recorded in the output manifest so it cannot be forgotten.

Usage:
    python -m pipeline.to_sglang_w4afp8 \
        --ckpt <compressed-tensors checkpoint> \
        --base <BF16 source snapshot> \
        --out  <new directory> \
        [--layers 3,4] [--dry-run] [--shard-bytes 10000000000]

Writes a NEW directory. It never mutates the input, because the
compressed-tensors artifact is the one that passed the gates and conversion is
not reversible.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections import defaultdict
from pathlib import Path

from pipeline.sglang_w4afp8_kernels import (
    DEFAULT_BLOCK,
    dequantize_block_fp8,
    pack_nibbles_int8,
    quantize_block_fp8,
    unpack_nibbles_int8,
)

# A fold factor is exactly 1.0 for unsmoothed mappings. Allow only BF16 rounding
# on the recovered ratio: 2^-8 relative spacing means a true 1.0 can read as
# 1.0039 at worst. 1% is comfortably above that and far below any real fold
# (healthy AWQ folds have norm-implied scale means of 0.7-0.9).
_UNFOLDED_TOL = 0.01

# Cross-check bound for rebuilt-vs-ondisk agreement. Two independent e4m3
# roundings give sqrt(2) * 0.0265 = 0.037; 0.06 leaves headroom for the fold
# reconstruction while still catching a dropped or inverted fold, which would
# land at |1 - s| -- typically 0.1 to 0.3.
_REBUILD_CROSSCHECK_MAX = 0.06

_LAYER_RE = re.compile(r"\.layers\.(\d+)\.")
_EXPERT_RE = re.compile(r"\.mlp\.experts\.(\d+)\.(gate_proj|up_proj|down_proj)$")


def _layer_of(name: str) -> int | None:
    match = _LAYER_RE.search(name)
    return int(match.group(1)) if match else None


def _module_of(name: str) -> str:
    """Strip the tensor suffix, leaving the module path."""
    for suffix in (
        ".weight_packed",
        ".weight_scale_inv",
        ".weight_scale",
        ".weight_shape",
        ".input_scale",
        ".weight",
    ):
        if name.endswith(suffix):
            return name[: -len(suffix)]
    return name


class Plan:
    """Which conversion each module needs, derived from the recipe, not guessed."""

    def __init__(self, fp8_targets: list[str]):
        from pipeline.serve_ignore import match_name

        self._fp8_targets = fp8_targets
        self._match = match_name

    def is_fp8_rest(self, module: str) -> bool:
        return any(self._match(module, target) for target in self._fp8_targets)

    def is_expert(self, module: str) -> bool:
        return bool(_EXPERT_RE.search(module))


def recover_fold_scale(base_norm, ckpt_norm):
    """Per-input-channel fold factor s, from the norm the fold was divided into.

    Ordinary RMSNorm (offset 0), which is GLM's form as resolved by
    quantize.resolve_norm_gain_offset: ``norm_ckpt = norm_base / s``.
    """
    import torch

    base = base_norm.float()
    ckpt = ckpt_norm.float()
    # A base gain of exactly 0 carries no information about s and would produce
    # 0/0. Those channels are dead for the fold too (AWQ's dead-channel path),
    # so pin them to 1 rather than propagating a nan into every weight column.
    safe = ckpt.abs() > 0
    scale = torch.ones_like(base)
    scale[safe] = base[safe] / ckpt[safe]
    return scale


def build_config(src_config: dict, group_size: int = 128) -> dict:
    """The output config.json: compressed-tensors' quantization_config replaced.

    PhalaCloud/GLM-5.2-W4AFP8 carries only ``{"quant_method": "w4afp8"}``, and
    W4AFp8Config.from_config hardcodes group_size 128, weight_block_size
    [128,128], linear_activation_scheme "dynamic" and moe_activation_scheme
    "static" regardless of what the file says. The extra keys are therefore
    documentation for humans, not inputs to the loader; ignored_layers is the one
    field the loader genuinely reads.
    """
    config = dict(src_config)
    ignored = []
    quant = src_config.get("quantization_config", {}) or {}
    for entry in quant.get("ignore", []) or []:
        # Concrete module names only. The loader's is_layer_skipped does prefix
        # matching, not regex, so an unresolved `re:` pattern would silently
        # match nothing.
        if not str(entry).startswith("re:"):
            ignored.append(entry)
    config["quantization_config"] = {
        "quant_method": "w4afp8",
        "group_size": group_size,
        "ignored_layers": ignored,
    }
    return config


def default_unpacker(packed, shape):
    """Unpack int32-packed 4-bit values, delegating to compressed-tensors.

    Deliberately NOT reimplemented. The storage convention (two's-complement
    nibbles vs an offset-by-2^(bits-1) representation) is an internal detail of
    whichever compressed-tensors version WROTE the checkpoint, and a second
    implementation here would be a guess that silently disagrees. The one that
    wrote the bytes is the one that should read them; it is also the unpacker the
    dequant gate used to validate this checkpoint at residual 0.102.

    Tests inject their own to exercise the plumbing without the dependency.
    """
    from compressed_tensors.compressors.pack_quantized.base import unpack_from_int32

    return unpack_from_int32(packed, 4, shape)


def convert(
    ckpt: Path,
    base: Path,
    out: Path,
    layers: list[int] | None = None,
    shard_bytes: int = 10_000_000_000,
    emit_input_scale: bool = True,
    dry_run: bool = False,
    unpacker=default_unpacker,
) -> int:
    import torch
    from safetensors import safe_open
    from safetensors.torch import save_file

    from pipeline.serve_ignore import weight_map_of

    ckpt_map = weight_map_of(ckpt)
    base_map = weight_map_of(base)
    src_config = json.loads((ckpt / "config.json").read_text(encoding="utf-8"))

    recipe_path = ckpt.parent / "recipe.json"
    if recipe_path.is_file():
        fp8_targets = json.loads(recipe_path.read_text(encoding="utf-8")).get(
            "fp8_dynamic_targets", []
        )
    else:
        # Fall back to the saved quantization_config's float group targets.
        fp8_targets = []
        for group in (src_config.get("quantization_config", {})
                      .get("config_groups", {}) or {}).values():
            weights = group.get("weights", {}) or {}
            if weights.get("type") == "float" and weights.get("num_bits") == 8:
                fp8_targets.extend(group.get("targets", []) or [])
    if not fp8_targets:
        print("error: could not determine fp8_dynamic_targets", flush=True)
        return 2
    print(f"[convert] fp8-rest targets: {len(fp8_targets)}", flush=True)

    plan = Plan(fp8_targets)
    opened: dict[tuple, object] = {}

    def _get(root: Path, wmap: dict, key: str):
        shard = (root, wmap[key])
        if shard not in opened:
            opened[shard] = safe_open(str(root / wmap[key]), framework="pt")
        return opened[shard].get_tensor(key)

    # ---- group the checkpoint's tensors by module ---------------------------
    modules: dict[str, list[str]] = defaultdict(list)
    for key in ckpt_map:
        modules[_module_of(key)].append(key)

    selected = sorted(
        modules,
        key=lambda m: (_layer_of(m) if _layer_of(m) is not None else -1, m),
    )
    if layers is not None:
        selected = [
            m for m in selected
            if _layer_of(m) is None or _layer_of(m) in layers
        ]

    n_expert = sum(1 for m in selected if plan.is_expert(m))
    n_fp8 = sum(1 for m in selected if plan.is_fp8_rest(m))
    n_copy = len(selected) - n_expert - n_fp8
    print(f"[convert] modules: expert={n_expert} fp8-rest={n_fp8} "
          f"passthrough={n_copy}", flush=True)
    if dry_run:
        return 0

    out.mkdir(parents=True, exist_ok=True)

    # ---- fold factors, recovered and verified per layer --------------------
    fold_cache: dict[tuple[int, str], object] = {}
    unfolded_violations: list[str] = []

    def fold_for(layer: int, norm: str):
        key = (layer, norm)
        if key not in fold_cache:
            name = f"model.layers.{layer}.{norm}.weight"
            if name not in ckpt_map or name not in base_map:
                fold_cache[key] = None
            else:
                fold_cache[key] = recover_fold_scale(
                    _get(base, base_map, name), _get(ckpt, ckpt_map, name)
                )
        return fold_cache[key]

    def check_input_layernorm_unfolded(layer: int) -> None:
        """Verify the attention-input mapping was never smoothed.

        This is a LAYER property, not a per-module one, and it only covers the
        modules whose smooth layer is ``input_layernorm`` -- the attention
        entry projections. q_b_proj, kv_b_proj, o_proj and
        shared_experts.down_proj sit in mappings whose smooth layer is another
        linear, so a fold on them would live along the preceding layer's output
        rows and no norm ratio can see it. Those are covered instead by the
        rebuilt-vs-ondisk cross-check, which compares against what the run
        actually saved and so catches a fold from any source.
        """
        if layer in checked_layers:
            return
        checked_layers.add(layer)
        scale = fold_for(layer, "input_layernorm")
        if scale is None:
            return
        deviation = (scale - 1.0).abs().max().item()
        if deviation > _UNFOLDED_TOL:
            unfolded_violations.append(
                f"layer {layer} input_layernorm: max|s-1| = {deviation:.4f}"
            )

    checked_layers: set[int] = set()
    shard: dict[str, object] = {}
    shard_size = 0
    shard_index = 0
    total_bytes = 0
    weight_map: dict[str, str] = {}
    stats = {"expert": 0, "fp8": 0, "copy": 0}
    crosscheck: list[float] = []

    def flush() -> None:
        nonlocal shard, shard_size, shard_index
        if not shard:
            return
        shard_index += 1
        name = f"model-{shard_index:05d}-of-SHARDS.safetensors"
        save_file(shard, str(out / name), metadata={"format": "pt"})
        for key in shard:
            weight_map[key] = name
        print(f"[convert] wrote {name}: {len(shard)} tensors, "
              f"{shard_size / 1e9:.1f} GB", flush=True)
        shard = {}
        shard_size = 0

    def emit(key: str, tensor) -> None:
        nonlocal shard_size, total_bytes
        shard[key] = tensor
        nbytes = tensor.numel() * tensor.element_size()
        shard_size += nbytes
        total_bytes += nbytes
        if shard_size >= shard_bytes:
            flush()

    for module in selected:
        keys = modules[module]
        layer = _layer_of(module)

        if plan.is_expert(module):
            packed = _get(ckpt, ckpt_map, f"{module}.weight_packed")
            shape = torch.Size(_get(ckpt, ckpt_map, f"{module}.weight_shape").tolist())
            values = unpacker(packed, shape)
            emit(f"{module}.weight", pack_nibbles_int8(values))
            emit(
                f"{module}.weight_scale_inv",
                _get(ckpt, ckpt_map, f"{module}.weight_scale"),
            )
            if emit_input_scale:
                emit(
                    f"{module}.input_scale",
                    torch.ones(1, dtype=torch.bfloat16),
                )
            stats["expert"] += 1
            continue

        if plan.is_fp8_rest(module):
            base_key = f"{module}.weight"
            if base_key not in base_map:
                print(f"[convert] WARNING: no base weight for {module}; copying "
                      f"the per-channel fp8 unchanged (will NOT load)", flush=True)
                for key in keys:
                    emit(key, _get(ckpt, ckpt_map, key))
                stats["copy"] += 1
                continue

            weight = _get(base, base_map, base_key).float()
            # gate_proj / up_proj of the shared experts are balance layers of the
            # post_attention_layernorm mapping and carry the fold; everything
            # else on this path must be unfolded, which we check rather than
            # assume.
            folded = (
                "shared_experts" in module
                and module.rsplit(".", 1)[-1] in ("gate_proj", "up_proj")
            )
            if layer is not None:
                if folded:
                    scale = fold_for(layer, "post_attention_layernorm")
                    if scale is not None:
                        weight = weight * scale.unsqueeze(0)
                else:
                    check_input_layernorm_unfolded(layer)

            qweight, scale_inv = quantize_block_fp8(weight, DEFAULT_BLOCK)
            emit(f"{module}.weight", qweight)
            emit(f"{module}.weight_scale_inv", scale_inv)

            # Cross-check against what the run itself saved. Independent of the
            # rebuild: disagreement beyond two roundings means the fold was
            # dropped, inverted, or applied on the wrong axis.
            old_scale_key = f"{module}.weight_scale"
            if old_scale_key in ckpt_map:
                on_disk = (
                    _get(ckpt, ckpt_map, f"{module}.weight").float()
                    * _get(ckpt, ckpt_map, old_scale_key).float()
                )
                rebuilt = dequantize_block_fp8(qweight, scale_inv, DEFAULT_BLOCK)
                denom = on_disk.norm()
                if denom > 0:
                    crosscheck.append(
                        ((rebuilt - on_disk).norm() / denom).item()
                    )
            stats["fp8"] += 1
            continue

        for key in keys:
            emit(key, _get(ckpt, ckpt_map, key))
        stats["copy"] += 1

    flush()

    # ---- rename shards to the final count and write the index --------------
    total = shard_index
    renames = {}
    for old in sorted({v for v in weight_map.values()}):
        new = old.replace("of-SHARDS", f"of-{total:05d}")
        (out / old).rename(out / new)
        renames[old] = new
    weight_map = {k: renames[v] for k, v in weight_map.items()}

    (out / "model.safetensors.index.json").write_text(
        json.dumps(
            {"metadata": {"total_size": total_bytes}, "weight_map": weight_map},
            indent=2,
        ),
        encoding="utf-8",
    )
    (out / "config.json").write_text(
        json.dumps(build_config(src_config), indent=2), encoding="utf-8"
    )
    for extra in (
        "generation_config.json",
        "tokenizer.json",
        "tokenizer_config.json",
        "chat_template.jinja",
        "special_tokens_map.json",
    ):
        src = ckpt / extra
        if src.is_file():
            (out / extra).write_bytes(src.read_bytes())

    # ---- report -----------------------------------------------------------
    print(f"\n[convert] modules converted: {stats}", flush=True)
    if crosscheck:
        from statistics import median

        ordered = sorted(crosscheck)
        print(f"[convert] rebuilt-vs-ondisk: n={len(ordered)} "
              f"min={ordered[0]:.4f} median={median(ordered):.4f} "
              f"max={ordered[-1]:.4f} (bound {_REBUILD_CROSSCHECK_MAX})",
              flush=True)
        if ordered[-1] > _REBUILD_CROSSCHECK_MAX:
            print("[convert] FAIL: rebuilt non-expert weights disagree with the "
                  "saved checkpoint by more than two e4m3 roundings; the fold "
                  "reconstruction is wrong", flush=True)
            return 1
    if unfolded_violations:
        print("[convert] FAIL: mappings expected to be unsmoothed carry a fold:",
              flush=True)
        for line in unfolded_violations[:20]:
            print(f"    {line}", flush=True)
        return 1

    (out / "conversion_manifest.json").write_text(
        json.dumps(
            {
                "source_checkpoint": str(ckpt),
                "base_model": str(base),
                "modules": stats,
                "block": list(DEFAULT_BLOCK),
                "expert_activation_scheme_change": (
                    "source specifies dynamic per-token fp8 expert activations; "
                    "SGLang's w4afp8 MoE path is static only and collapses the "
                    "per-expert input scales to one max, so the served model "
                    "uses a static scale of 1.0. Matches "
                    "PhalaCloud/GLM-5.2-W4AFP8 and the loader's pre-load "
                    "default, but is NOT our checkpoint's activation scheme -- "
                    "price it with an eval."
                ),
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"[convert] OK -> {out}", flush=True)
    # Said out loud rather than left to a docstring: the source checkpoint has no
    # layer 78, because the recipe ignores it and the BF16 release does not build
    # an MTP layer. Serving without it silently disables speculative decoding
    # instead of failing, so this is exactly the kind of omission that survives
    # every structural check.
    if 78 not in {_layer_of(m) for m in selected if _layer_of(m) is not None}:
        print("[convert] NOTE: no layer 78 (MTP). Graft it from the vendor FP8 "
              "release as a separate step, or the served model has no draft "
              "head and speculative decoding is silently off.", flush=True)
    return 0


def conformance_check(ckpt: Path, samples: int = 6) -> int:
    """Prove the int8 repack is lossless on REAL packed tensors.

    The unit tests pack and unpack with our own kernels, which cannot catch a
    disagreement with the convention compressed-tensors actually wrote. This
    takes genuine ``weight_packed`` tensors, unpacks with upstream, repacks to
    int8 nibbles, unpacks again with our kernel and requires bit equality --
    the only check that closes that gap. Run it in the environment where the
    conversion will run, before converting 394 GB.
    """
    import torch
    from safetensors import safe_open

    from pipeline.serve_ignore import weight_map_of

    weight_map = weight_map_of(ckpt)
    packed_keys = [k for k in weight_map if k.endswith(".weight_packed")]
    if not packed_keys:
        print("error: no weight_packed tensors found", flush=True)
        return 2
    picks = packed_keys[:: max(1, len(packed_keys) // samples)][:samples]
    opened: dict[str, object] = {}

    def _get(key):
        shard = weight_map[key]
        if shard not in opened:
            opened[shard] = safe_open(str(ckpt / shard), framework="pt")
        return opened[shard].get_tensor(key)

    failures = 0
    for key in picks:
        module = key[: -len(".weight_packed")]
        shape = torch.Size(_get(f"{module}.weight_shape").tolist())
        values = default_unpacker(_get(key), shape)
        repacked = pack_nibbles_int8(values)
        recovered = unpack_nibbles_int8(repacked)
        ok = torch.equal(recovered.to(values.dtype), values)
        lo, hi = int(values.min()), int(values.max())
        grid = max(abs(lo), abs(hi))
        print(f"  {'ok  ' if ok else 'FAIL'} {module}: shape={tuple(shape)} "
              f"range=[{lo},{hi}] grid={grid} "
              f"int8={tuple(repacked.shape)}", flush=True)
        if not ok:
            failures += 1
        # Symmetric int4 must use the full [-7,7] grid; a max below 7 means the
        # scales were computed against a different range than we assume.
        if grid != 7:
            print(f"       WARNING: grid maximum is {grid}, expected 7", flush=True)

    print(f"[conformance] {len(picks) - failures}/{len(picks)} modules "
          f"round-trip bit-exactly", flush=True)
    return 1 if failures else 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ckpt", required=True, type=Path)
    parser.add_argument("--base", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--layers", default=None,
                        help="comma-separated layer subset, for testing")
    parser.add_argument("--shard-bytes", type=int, default=10_000_000_000)
    parser.add_argument("--no-input-scale", action="store_true",
                        help="omit expert input_scale (loader defaults to 1.0)")
    parser.add_argument("--conformance-only", action="store_true",
                        help="check the local nibble packer against "
                             "compressed-tensors' unpacker and exit; run this "
                             "wherever the real conversion will run")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)

    if args.conformance_only:
        return conformance_check(args.ckpt)

    layers = None
    if args.layers:
        layers = [int(p) for p in args.layers.split(",") if p.strip()]

    if args.out.exists() and any(args.out.iterdir()):
        print(f"error: {args.out} exists and is not empty; refusing to overwrite")
        return 2

    return convert(
        args.ckpt, args.base, args.out,
        layers=layers,
        shard_bytes=args.shard_bytes,
        emit_input_scale=not args.no_input_scale,
        dry_run=args.dry_run,
    )


if __name__ == "__main__":
    sys.exit(main())
