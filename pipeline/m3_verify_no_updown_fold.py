"""Fail-closed gate: verify a MiniMax-M3 W4 checkpoint carries NO up->down
AWQ smoothing fold.

Context (BUGS_AND_FIXES.md "AWQ up->down smoothing fold is not
function-preserving on MiniMax-M3"): the r5 recipe folded ``up_rows /= s_r`` /
``down_cols *= s_r`` per expert, which rescales the effective swiglu beta and
up-clamp per channel (M3's expert activation is ``(clamp(up)+1)*glu``). r6
removes that mapping; this gate proves the produced checkpoint really has no
residual fold, by dequantizing sampled expert ``down_proj`` weights and
comparing them column-wise against the BF16 base.

Detector logic: ``down_proj`` is touched by NO other transform (the MoE-input
mapping scales the *hidden* axis, which is down's OUTPUT rows -- not folded
into down at all), so with the up->down mapping gone,
``dequant(down) ~= base_down`` up to int4 rounding noise per column. A
residual fold shows up as per-column relative errors of order ``|s_r - 1|``
(r5: median 0.66 at layer 30) versus rounding noise of a few percent.

Self-validating: if the unpack/dequant convention were wrong, the cosine
similarity check would fail for EVERY column and the gate errors out rather
than passing (fail-closed).

Usage:
    python -m pipeline.m3_verify_no_updown_fold \
        --base /mnt/nfs/hoangduy/hf_assets/MiniMaxAI/MiniMax-M3 \
        --checkpoint <run>/awq/MiniMax-M3-awq-W4AFP8/<ts>/checkpoint \
        --output <run>/no_updown_fold_gate.json
Exit code 0 = gate PASSED; non-zero = fold detected or evidence unusable.
"""

from __future__ import annotations

import argparse
import json
import os
import sys

import torch
from safetensors import safe_open

from compressed_tensors.compressors.pack_quantized.helpers import unpack_from_int32

# Sampled sparse layers (span the depth; 30 is where r5's fold was largest)
DEFAULT_LAYERS = "3,8,20,30,45,59"
DEFAULT_EXPERTS = "0,31,64,101,127"
# Per-column relative L2 of dequant-vs-base for group-128 symmetric int4 RTN
# sits at ~0.03-0.08 on M3 experts. A residual r5-scale fold produces column
# errors of order |s_r-1| (median 0.12-0.66 depending on layer).
MEDIAN_RELERR_MAX = 0.15
P99_RELERR_MAX = 0.30
# Sanity floor proving the unpack/dequant convention is right at all.
MIN_MEDIAN_COSINE = 0.95


def _weight_map(root: str) -> dict:
    with open(os.path.join(root, "model.safetensors.index.json")) as fh:
        return json.load(fh)["weight_map"]


def _load(root: str, wmap: dict, key: str) -> torch.Tensor:
    with safe_open(os.path.join(root, wmap[key]), framework="pt") as fh:
        return fh.get_tensor(key)


def _dequant_packed_w4(ckpt: str, cmap: dict, prefix: str) -> torch.Tensor:
    packed = _load(ckpt, cmap, f"{prefix}.weight_packed")
    scale = _load(ckpt, cmap, f"{prefix}.weight_scale").float()
    shape = torch.Size(_load(ckpt, cmap, f"{prefix}.weight_shape").tolist())
    q = unpack_from_int32(packed, num_bits=4, shape=shape).float()
    group = shape[1] // scale.shape[1]
    return (q.reshape(shape[0], -1, group) * scale.unsqueeze(-1)).reshape(shape)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--base", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--layers", default=DEFAULT_LAYERS)
    parser.add_argument("--experts", default=DEFAULT_EXPERTS)
    parser.add_argument("--output", required=True)
    args = parser.parse_args(argv)

    bmap, cmap = _weight_map(args.base), _weight_map(args.checkpoint)
    layers = [int(x) for x in args.layers.split(",")]
    experts = [int(x) for x in args.experts.split(",")]

    report: dict = {
        "base": args.base,
        "checkpoint": args.checkpoint,
        "thresholds": {
            "median_relerr_max": MEDIAN_RELERR_MAX,
            "p99_relerr_max": P99_RELERR_MAX,
            "min_median_cosine": MIN_MEDIAN_COSINE,
        },
        "layers": {},
    }
    passed = True
    for layer in layers:
        rel_all, cos_all = [], []
        for expert in experts:
            base_key = (
                f"language_model.model.layers.{layer}"
                f".block_sparse_moe.experts.{expert}.w2.weight"
            )
            ckpt_prefix = (
                f"language_model.model.layers.{layer}"
                f".block_sparse_moe.experts.{expert}.down_proj"
            )
            base = _load(args.base, bmap, base_key).float()  # [hidden, inter]
            deq = _dequant_packed_w4(args.checkpoint, cmap, ckpt_prefix)
            if deq.shape != base.shape:
                print(f"ERROR: shape mismatch {deq.shape} vs {base.shape}", file=sys.stderr)
                return 3
            diff = deq - base
            # per-COLUMN (down input channel = the folded axis) relative L2
            rel = diff.norm(dim=0) / base.norm(dim=0).clamp(min=1e-12)
            cos = torch.nn.functional.cosine_similarity(deq.T, base.T, dim=1)
            rel_all.append(rel)
            cos_all.append(cos)
        rel = torch.cat(rel_all)
        cos = torch.cat(cos_all)
        stats = {
            "relerr_median": rel.median().item(),
            "relerr_p99": rel.quantile(0.99).item(),
            "relerr_max": rel.max().item(),
            "cosine_median": cos.median().item(),
            "columns": rel.numel(),
        }
        convention_ok = stats["cosine_median"] >= MIN_MEDIAN_COSINE
        no_fold = (
            stats["relerr_median"] <= MEDIAN_RELERR_MAX
            and stats["relerr_p99"] <= P99_RELERR_MAX
        )
        stats["convention_ok"] = convention_ok
        stats["no_fold"] = no_fold
        report["layers"][str(layer)] = stats
        passed = passed and convention_ok and no_fold
        print(
            f"L{layer}: relerr med={stats['relerr_median']:.4f} "
            f"p99={stats['relerr_p99']:.4f} cos_med={stats['cosine_median']:.4f} "
            f"convention_ok={convention_ok} no_fold={no_fold}",
            flush=True,
        )

    report["passed"] = passed
    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    with open(args.output, "w") as fh:
        json.dump(report, fh, indent=1)
    print(f"gate {'PASSED' if passed else 'FAILED'} -> {args.output}", flush=True)
    return 0 if passed else 4


if __name__ == "__main__":
    raise SystemExit(main())
