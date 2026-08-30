"""Quantize GLM's MTP layer (78) from BF16 and append it to a w4afp8 checkpoint.

WHY THE LAYER IS MISSING IN THE FIRST PLACE. ``AutoModelForCausalLM`` does not
instantiate GLM's next-token-prediction layer, so the quantization run never
loaded it and never saved it. The recipe's ``re:.*layers[.]78[.].*`` ignore was
belt-and-braces on top of that. The layer is NOT missing from the source.

WHY BF16 AND NOT THE VENDOR FP8 RELEASE. GLM-5.3-BF16 -- the same repo and
revision the main model was quantized from -- carries all 791 layer-78 tensors in
BF16 (shards 270-274 of 282, already staged). The vendor FP8 release
(``zai-org/GLM-5.3``) is worse on every axis: its layer 78 holds 1,536 expert
tensors in BLOCK FP8, which SGLang's w4afp8 MoE loader cannot consume because it
wants int8-packed int4; it needs a ~16 GB three-shard download; and its snapshot
is a DIFFERENT REVISION from the BF16 one, so grafting it risks a draft head
subtly mismatched to this target model -- which costs acceptance rate silently
rather than failing.

TREATMENT MIRRORS OUR OWN RECIPE, NOT THE VENDOR'S. Per layer-78 module:

  256 x 3 routed experts   -> int4 group-128, RTN, then int8 nibble pack
  q_a/q_b/kv_a/kv_b/o_proj -> block fp8 (5 modules)
  shared_experts g/u/d     -> block fp8 (3 modules)
  norms, router (+bias), eh_proj, indexer, shared_head.norm -> copied BF16

Two deliberate divergences from the vendor, both to match OUR main model rather
than theirs:
  * the DSA indexer stays BF16. The vendor FP8-quantizes indexer.wk / wq_b; our
    recipe does not, and layer 78 must look like the other 78 layers of the
    artifact it is joining, not like a different checkpoint.
  * eh_proj stays BF16, which happens to agree with the vendor (their eh_proj
    carries no weight_scale_inv either).

RTN, NOT AWQ, FOR THE EXPERTS -- and why that is acceptable HERE specifically.
There are no calibration statistics for layer 78 and borrowing a neighbouring
layer's smoothing scales would be wrong. So the experts get round-to-nearest.
Speculative decoding is lossless BY CONSTRUCTION: the draft head only proposes
tokens and the target model verifies them, so a weaker draft costs ACCEPTANCE
RATE, never output quality. That makes RTN a reasonable trade for this layer
alone; it would not be for the main model.

NO FOLD. Layer 78 was never calibrated, so no AWQ compensation fold was ever
applied to it and s is exactly 1 everywhere. Nothing to recover, nothing to
verify -- unlike the main conversion, where the fold is the correctness crux.

Usage:
    python -m pipeline.graft_mtp_w4afp8 \
        --base <GLM-5.3-BF16 snapshot> \
        --out  <existing w4afp8 checkpoint to extend> \
        [--layer 78] [--dry-run]

Appends new shards and REWRITES the index and config of ``--out`` in place;
existing shards are never touched. Idempotent: refuses if the target already
carries the layer, so a re-run cannot double-append.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from pipeline.to_sglang_w4afp8 import ENGINE_FP8_SUFFIXES
from pipeline.sglang_w4afp8_kernels import (
    DEFAULT_BLOCK,
    dequantize_block_fp8,
    pack_nibbles_int8,
    quantize_block_fp8,
)

# Measured on our own AWQ output, not assumed: max|w_group|/scale_group is
# 7.958-8.040 and values span [-8, 7], so compressed-tensors uses max/8 and the
# full two's-complement range. The graft MUST use the same convention or layer
# 78's scales are systematically 14% off relative to every other layer.
INT4_LEVELS = 8.0
INT4_MIN, INT4_MAX = -8, 7
GROUP = 128

# Block-fp8 targets, mirroring the main recipe's fp8_dynamic_targets restricted
# to what layer 78 actually has. Named explicitly rather than regex-matched: the
# list is short, closed, and a silent miss here means a tensor ships BF16 into a
# slot the loader expects fp8.
#
# ENGINE_FP8_SUFFIXES is IMPORTED, not restated. This file used to carry its own
# list that omitted the DSA indexer, so the graft shipped a BF16
# layers.78.self_attn.indexer.{wk,wq_b} while the converter handled layers 0-77
# correctly -- and ignored_layers could not save it, because
# W4AFp8Config.from_config never passes that field. Layer 78 is not special:
# dsa_indexer.py builds wk/wq_b with a quant_config in every layer, and
# zai-org/GLM-5.3 plus both PhalaCloud w4afp8 releases all ship them with
# weight_scale_inv. weights_proj and k_norm stay BF16, built without one.
_FP8_SUFFIXES = ENGINE_FP8_SUFFIXES + (
    "self_attn.q_a_proj",
    "self_attn.q_b_proj",
    "self_attn.kv_a_proj_with_mqa",
    "self_attn.kv_b_proj",
    "self_attn.o_proj",
    "mlp.shared_experts.gate_proj",
    "mlp.shared_experts.up_proj",
    "mlp.shared_experts.down_proj",
)


def quantize_int4_group_rtn(weight):
    """Round-to-nearest int4, group ``GROUP`` along the input dimension.

    Returns ``(values, scale)`` with ``values`` int8 in [-8, 7] of the input
    shape and ``scale`` bf16 [out, in // GROUP], such that
    ``values * scale.repeat_interleave(GROUP, 1) ~= weight``.

    Matches the convention measured on the AWQ output: scale = max|w|/8 over each
    group, values clamped to [-8, 7]. The clamp is load-bearing rather than
    defensive -- a group whose extreme is POSITIVE rounds to +8, which is not
    representable, so it must saturate to +7. That is real (small) clipping,
    measured at 0.56-0.62% of elements on the main model, and it is inherent to
    the max/8 choice rather than a defect here.
    """
    import torch

    if weight.ndim != 2:
        raise ValueError(f"expected a 2-D weight, got {tuple(weight.shape)}")
    out_features, in_features = weight.shape
    if in_features % GROUP:
        raise ValueError(
            f"input dim {in_features} is not a multiple of the group size {GROUP}"
        )

    work = weight.float().reshape(out_features, in_features // GROUP, GROUP)
    amax = work.abs().amax(dim=2, keepdim=True)
    # An all-zero group reproduces exactly at any positive scale; 1.0 keeps the
    # stored scale finite rather than denormal.
    scale = torch.where(amax > 0, amax / INT4_LEVELS, torch.ones_like(amax))
    values = torch.round(work / scale).clamp(INT4_MIN, INT4_MAX)
    return (
        values.reshape(out_features, in_features).to(torch.int8),
        scale.reshape(out_features, in_features // GROUP).bfloat16(),
    )


def dequantize_int4_group(values, scale):
    """Inverse of :func:`quantize_int4_group_rtn`, for the residual check."""
    return values.float() * scale.float().repeat_interleave(GROUP, dim=1)


def classify(name: str, layer: int) -> str:
    """``expert``, ``fp8``, or ``copy`` for one layer-78 tensor name."""
    stem = name[: -len(".weight")] if name.endswith(".weight") else name
    if f".layers.{layer}.mlp.experts." in name:
        return "expert"
    for suffix in _FP8_SUFFIXES:
        if stem.endswith(suffix):
            return "fp8"
    return "copy"


def graft(
    base: Path,
    out: Path,
    layer: int = 78,
    shard_bytes: int = 10_000_000_000,
    emit_input_scale: bool = True,
    dry_run: bool = False,
) -> int:
    import torch
    from safetensors import safe_open
    from safetensors.torch import save_file

    from pipeline.serve_ignore import weight_map_of

    index_path = out / "model.safetensors.index.json"
    if not index_path.is_file():
        print(f"error: {out} has no model.safetensors.index.json", flush=True)
        return 2

    out_index = json.loads(index_path.read_text(encoding="utf-8"))
    out_map = dict(out_index["weight_map"])
    if any(f".layers.{layer}." in k for k in out_map):
        print(f"error: {out} already carries layer {layer}; refusing to "
              f"double-append. Remove those tensors first if you mean to redo "
              f"the graft.", flush=True)
        return 2

    base_map = weight_map_of(base)
    names = sorted(k for k in base_map if f".layers.{layer}." in k)
    if not names:
        print(f"error: no layer-{layer} tensors in {base}", flush=True)
        return 2

    buckets: dict[str, list[str]] = {"expert": [], "fp8": [], "copy": []}
    for name in names:
        buckets[classify(name, layer)].append(name)
    print(f"[graft] layer {layer} in source: {len(names)} tensors "
          f"(expert={len(buckets['expert'])} fp8={len(buckets['fp8'])} "
          f"copy={len(buckets['copy'])})", flush=True)
    for name in buckets["copy"]:
        print(f"[graft]   BF16 copy: {name}", flush=True)
    if dry_run:
        return 0

    opened: dict[str, object] = {}

    def _get(key: str):
        shard = base_map[key]
        if shard not in opened:
            opened[shard] = safe_open(str(base / shard), framework="pt")
        return opened[shard].get_tensor(key)

    # New shards continue the existing numbering and keep the existing files'
    # names untouched, so the graft cannot invalidate what is already there.
    existing = sorted({v for v in out_map.values()})
    start = len(existing)
    total_shards = start  # grows as we flush
    added_bytes = 0
    shard: dict[str, object] = {}
    shard_size = 0
    new_names: list[str] = []
    int4_resid: list[float] = []
    fp8_resid: list[float] = []

    def flush() -> None:
        nonlocal shard, shard_size, total_shards
        if not shard:
            return
        total_shards += 1
        name = f"model-mtp{total_shards - start:05d}.safetensors"
        save_file(shard, str(out / name), metadata={"format": "pt"})
        for key in shard:
            out_map[key] = name
        new_names.append(name)
        print(f"[graft] wrote {name}: {len(shard)} tensors, "
              f"{shard_size / 1e9:.2f} GB", flush=True)
        shard = {}
        shard_size = 0

    def emit(key: str, tensor) -> None:
        nonlocal shard_size, added_bytes
        shard[key] = tensor
        nbytes = tensor.numel() * tensor.element_size()
        shard_size += nbytes
        added_bytes += nbytes
        if shard_size >= shard_bytes:
            flush()

    for name in buckets["expert"]:
        module = name[: -len(".weight")]
        weight = _get(name)
        values, scale = quantize_int4_group_rtn(weight)
        emit(f"{module}.weight", pack_nibbles_int8(values))
        emit(f"{module}.weight_scale_inv", scale)
        if emit_input_scale:
            emit(f"{module}.input_scale", torch.ones(1, dtype=torch.bfloat16))
        if len(int4_resid) < 16:
            back = dequantize_int4_group(values, scale)
            ref = weight.float()
            int4_resid.append(((back - ref).norm() / ref.norm()).item())

    for name in buckets["fp8"]:
        module = name[: -len(".weight")]
        weight = _get(name).float()
        qweight, scale_inv = quantize_block_fp8(weight, DEFAULT_BLOCK)
        emit(f"{module}.weight", qweight)
        emit(f"{module}.weight_scale_inv", scale_inv)
        back = dequantize_block_fp8(qweight, scale_inv, DEFAULT_BLOCK)
        fp8_resid.append(((back - weight).norm() / weight.norm()).item())

    # Linears this graft leaves BF16. The loader consults ignored_layers to
    # decide whether a module gets UnquantizedLinearMethod or Fp8LinearMethod,
    # and layer 78 is not covered by anything the AWQ run wrote: its only entry
    # was the pattern `re:.*layers[.]78[.].*`, which matched nothing at the time
    # because the layer did not exist yet. Left alone, eh_proj and the DSA
    # indexer would be handed Fp8LinearMethod and asked for a weight_scale_inv
    # that was never written.
    #
    # The pattern also cannot simply be expanded now: it says "ignore ALL of
    # layer 78", whereas this graft deliberately quantizes layer 78's experts
    # and attention. Expanding it would make the loader read grafted int4
    # nibbles as BF16 -- no error, just noise. So the entries are derived from
    # what was actually copied, not from the recipe.
    newly_ignored: list[str] = []
    for name in buckets["copy"]:
        tensor = _get(name)
        emit(name, tensor)
        # 2-D is what separates a Linear from a norm without instantiating the
        # model; norms and the router bias need no entry.
        if name.endswith(".weight") and tensor.dim() == 2:
            newly_ignored.append(name[: -len(".weight")])

    flush()

    # ---- residual report, fail-closed on either bound -----------------------
    from statistics import median

    ok = True
    if int4_resid:
        ordered = sorted(int4_resid)
        print(f"[graft] int4 RTN residual: n={len(ordered)} "
              f"min={ordered[0]:.4f} median={median(ordered):.4f} "
              f"max={ordered[-1]:.4f}", flush=True)
        # RTN has no smoothing, so expect somewhat above the AWQ layers' raw
        # 0.122-0.128. 0.30 is loose enough not to fail on that and tight enough
        # to catch a wrong scale convention, which would land far higher.
        if ordered[-1] > 0.30:
            print(f"[graft] FAIL: int4 residual {ordered[-1]:.4f} > 0.30; the "
                  f"scale convention is probably wrong", flush=True)
            ok = False
    if fp8_resid:
        ordered = sorted(fp8_resid)
        print(f"[graft] block-fp8 residual: n={len(ordered)} "
              f"min={ordered[0]:.4f} median={median(ordered):.4f} "
              f"max={ordered[-1]:.4f}", flush=True)
        if ordered[-1] > 0.05:
            print(f"[graft] FAIL: fp8 residual {ordered[-1]:.4f} > 0.05; "
                  f"expected ~0.0265", flush=True)
            ok = False
    if not ok:
        for name in new_names:
            (out / name).unlink(missing_ok=True)
        print("[graft] removed the shards written by this failed graft; the "
              "target checkpoint is unchanged", flush=True)
        return 1

    # ---- rewrite index and config ------------------------------------------
    out_index["weight_map"] = out_map
    out_index.setdefault("metadata", {})
    out_index["metadata"]["total_size"] = (
        int(out_index["metadata"].get("total_size", 0)) + added_bytes
    )
    index_path.write_text(json.dumps(out_index, indent=2), encoding="utf-8")

    config_path = out / "config.json"
    config = json.loads(config_path.read_text(encoding="utf-8"))
    config["num_nextn_predict_layers"] = 1
    quant = config.setdefault("quantization_config", {})
    ignored = list(quant.get("ignored_layers", []) or [])
    added_ignores = [m for m in newly_ignored if m not in set(ignored)]
    quant["ignored_layers"] = ignored + added_ignores
    config_path.write_text(json.dumps(config, indent=2), encoding="utf-8")

    print(f"[graft] OK: added {len(names)} source tensors as "
          f"{len(new_names)} shard(s), {added_bytes / 1e9:.2f} GB; "
          f"num_nextn_predict_layers=1; "
          f"ignored_layers += {len(added_ignores)} "
          f"({', '.join(m.rsplit('.', 1)[-1] for m in added_ignores[:6])})",
          flush=True)
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", required=True, type=Path,
                        help="BF16 source snapshot (the one the model was "
                             "quantized from)")
    parser.add_argument("--out", required=True, type=Path,
                        help="existing w4afp8 checkpoint to extend in place")
    parser.add_argument("--layer", type=int, default=78)
    parser.add_argument("--shard-bytes", type=int, default=10_000_000_000)
    parser.add_argument("--no-input-scale", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)
    return graft(
        args.base, args.out,
        layer=args.layer,
        shard_bytes=args.shard_bytes,
        emit_input_scale=not args.no_input_scale,
        dry_run=args.dry_run,
    )


if __name__ == "__main__":
    sys.exit(main())
