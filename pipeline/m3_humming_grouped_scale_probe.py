"""Batch-scale probe: does Humming's grouped_contiguous GEMM diverge at
specific tuning buckets (larger M), racing CTAs, or oversized buffers?

Motivation. After the exact-total bounds patch
(pipeline/slurm/patch_humming_grouped_expert_bounds.py), the grouped arm of
the 3-arm perf window 20260725T122256Z completed rc=0 with a valid
attestation and a passing comprehension gate -- and yet its generations
terminate early at scale: 1,594 OSL-mismatch warnings vs 9-11 on the
CUTLASS/indexed arms, 169/640 requests short at reasoning conc-64 (pile-up at
the 2000-token min-token clamp), agentic-warm average OSL 48 vs ~99.5.
Same checkpoint, same prompts, same sampling; only the MoE GEMM differs.

The M-geometry of the broken vs clean regimes points at batch size:

  clean : reasoning conc-1/4  -> prefill 1000 tokens alone -> ~500 valid
          rows per EP rank; decode M in [1, 16] rows.
  broken: agentic-warm conc-1 -> 10.6k-token prefill in 2048-token chunks;
          reasoning conc-16/64 -> merged prefills; both ~1024-4096 valid
          rows per rank, a different tuning bucket (larger block_m, and
          block_k 128 instead of 256).

The first bounds probe (pipeline/m3_humming_grouped_bounds_probe.py) ran a
single M (512 tokens -> 261 valid rows): one bucket, one comparison
(full-buffer vs exact-slice). It cannot see

  (a) a bug that corrupts BOTH runs identically (e.g. a stream-K slice
      miscount specific to one compiled config), or
  (b) a nondeterministic race (concurrent stream-K reducers, epilogue
      barrier misuse), which needs repeated runs, or
  (c) a bucket other than the one it ran.

This probe closes all three holes. For each M in a sweep spanning the decode
and prefill-chunk buckets, and for both layer geometries (w13: N=6144 K=6144,
w2: N=6144 K=3072), it runs the grouped kernel

  1. on the full (M*topk, K) buffer, `--repeats` times -> bitwise
     run-to-run comparison (race detector);
  2. on the exact (total_valid, K) slice -> full-vs-exact comparison
     (bounds-bug detector);
  3. against an fp32 torch reference computed from the dequantized
     operands -> absolute-correctness check that catches both-wrong-equally
     defects. The uint4 dequant convention is validated against the
     original bf16 weight before use (reconstruction error must be within
     quantisation rounding), so the reference cannot silently disagree
     with the kernel about representation.

Tuning config is selected exactly the way production does
(fused_humming_moe.estimate_local_valid_shape_m == ceil(M*topk*local/global)),
and the chosen config is recorded per M so a bad bucket is directly
identifiable from the report.

Pass/fail is structural, not a single number: kernel-vs-reference noise from
fp8 accumulation order should stay below ~2% relative per row; a scheduling
or race bug shows up as O(1) relative error on whole tiles/rows and, for
races, as run-to-run bitwise differences.

Usage (1 GPU is enough)::

    python -m pipeline.m3_humming_grouped_scale_probe --out <dir>
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import torch

# MiniMax-M3 routed experts under TP8 + EP: 16 local experts of 128 global.
LOCAL_EXPERTS = 16
GLOBAL_EXPERTS = 128
TOPK = 4
GROUP_SIZE = 128

# w13 consumes hidden (K=6144) and produces gate+up (N=2*3072=6144);
# w2 consumes the activated intermediate (K=3072) and produces hidden
# (N=6144).
GEOMETRIES = {
    "w13": {"shape_n": 6144, "shape_k": 6144},
    "w2": {"shape_n": 6144, "shape_k": 3072},
}

# Spans the decode buckets (small M) through the chunked-prefill buckets
# (2048-token chunks -> ~1024 valid rows; merged prefills -> more).
DEFAULT_TOKEN_SWEEP = [64, 128, 256, 512, 1024, 2048, 4096]

REL_ERR_ROW_THRESHOLD = 0.05  # fp8 accum-order noise stays well below this


def build_layer_config(shape_n: int, shape_k: int) -> dict[str, Any]:
    return {
        "shape_n": shape_n,
        "shape_k": shape_k,
        "num_experts": LOCAL_EXPERTS,
        "b_dtype": "uint4",
        "a_dtype": "float8e4m3",
        "c_dtype": "bfloat16",
        "input_scale_group_size": 0,
        "weight_scale_group_size": GROUP_SIZE,
        "weight_scale_group_size_n": 0,
        "use_int_weight_scale": False,
        "use_fused_e8m0_scale": False,
        "has_zero_point": False,
        "is_fp_zero_point": False,
        "has_bias": False,
    }


def realistic_offsets(
    num_tokens: int, generator: torch.Generator
) -> tuple[torch.Tensor, int]:
    """Exact cumulative per-expert offsets, as vLLM's moe_permute produces."""
    topk_ids = torch.randint(
        0, GLOBAL_EXPERTS, (num_tokens, TOPK), generator=generator, device="cpu"
    )
    local_hits = topk_ids[topk_ids < LOCAL_EXPERTS]
    counts = torch.bincount(local_hits, minlength=LOCAL_EXPERTS)
    offsets = torch.zeros(LOCAL_EXPERTS + 1, dtype=torch.int64)
    offsets[1:] = torch.cumsum(counts, dim=0)
    return offsets, int(offsets[-1].item())


def production_valid_estimate(num_tokens: int) -> int:
    """Mirror HummingExpertsBase.estimate_local_valid_shape_m."""
    return math.ceil(num_tokens * TOPK * LOCAL_EXPERTS / GLOBAL_EXPERTS)


def dequantize_weight_reference(
    quanted: torch.Tensor,
    scale: torch.Tensor,
    original: torch.Tensor,
) -> torch.Tensor:
    """Reconstruct the fp values the kernel dequantizes to, and validate the
    uint4 convention against the original bf16 weight.

    Symmetric uint4 normally stores q in [0, 15] with an implicit zero point
    of 8, but rather than bet a GPU round-trip on that, try the plausible
    conventions and keep the one whose reconstruction error stays within
    quantisation rounding (0.51 * scale). A wrong convention is off by
    O(scale * 8) and cannot pass the bound by accident.
    """
    e, n, k = original.shape
    q = quanted.to(torch.float32).view(e, n, k // GROUP_SIZE, GROUP_SIZE)
    s = scale.to(torch.float32).view(e, n, k // GROUP_SIZE, 1)
    bound = (s * 0.51).expand_as(q).reshape(e, n, k)
    orig = original.to(torch.float32)

    best: tuple[float, float, torch.Tensor] | None = None
    for zero_point in (8.0, 0.0, 7.5):
        w_deq = ((q - zero_point) * s).view(e, n, k)
        frac_bad = ((w_deq - orig).abs() > bound).float().mean().item()
        if best is None or frac_bad < best[1]:
            best = (zero_point, frac_bad, w_deq)
        if frac_bad < 1e-4:
            break
    assert best is not None and best[1] < 1e-4, (
        f"no uint4 dequant convention matched (best zp={best[0]}, "
        f"{best[1]:.2%} weights beyond rounding bound) -- the fp32 "
        "reference would be meaningless"
    )
    print(f"uint4 dequant convention: zero_point={best[0]} frac_bad={best[1]:.2e}")
    return best[2]


def reference_output(
    quant: torch.Tensor,
    scale: torch.Tensor,
    w_deq: torch.Tensor,
    offsets: torch.Tensor,
) -> torch.Tensor:
    """C_ref[r] = a_scale[r] * (a_q[r] @ w_deq[expert(r)].T), fp32."""
    total_valid = int(offsets[-1].item())
    out = torch.empty(
        total_valid, w_deq.size(1), dtype=torch.float32, device=quant.device
    )
    a_q = quant[:total_valid].to(torch.float32)
    a_s = scale[:total_valid].to(torch.float32).view(-1, 1)
    for e in range(LOCAL_EXPERTS):
        lo, hi = int(offsets[e]), int(offsets[e + 1])
        if hi <= lo:
            continue
        out[lo:hi] = (a_q[lo:hi] @ w_deq[e].T) * a_s[lo:hi]
    return out


def row_metrics(
    kernel_out: torch.Tensor,
    ref: torch.Tensor,
    offsets: torch.Tensor,
) -> dict[str, Any]:
    diff = (kernel_out.float() - ref).abs()
    ref_row_mag = ref.abs().amax(dim=1).clamp_min(1e-6)
    row_rel = diff.amax(dim=1) / ref_row_mag
    bad_rows = row_rel > REL_ERR_ROW_THRESHOLD

    per_expert = []
    for e in range(LOCAL_EXPERTS):
        lo, hi = int(offsets[e]), int(offsets[e + 1])
        if hi <= lo:
            continue
        per_expert.append(
            {
                "expert": e,
                "rows": hi - lo,
                "max_row_rel_err": float(row_rel[lo:hi].max().item()),
                "frac_rows_bad": float(bad_rows[lo:hi].float().mean().item()),
            }
        )
    return {
        "max_row_rel_err": float(row_rel.max().item()),
        "median_row_rel_err": float(row_rel.median().item()),
        "frac_rows_bad": float(bad_rows.float().mean().item()),
        "num_rows_bad": int(bad_rows.sum().item()),
        "finite": bool(torch.isfinite(kernel_out.float()).all().item()),
        "per_expert_bad": [p for p in per_expert if p["frac_rows_bad"] > 0],
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", required=True)
    parser.add_argument(
        "--tokens",
        type=int,
        nargs="+",
        default=DEFAULT_TOKEN_SWEEP,
        help="batch sizes (tokens) to sweep; rows = tokens * topk",
    )
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--garbage-scale", type=float, default=100.0)
    parser.add_argument(
        "--geometries", nargs="+", default=["w2", "w13"], choices=list(GEOMETRIES)
    )
    args = parser.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    from humming import dtypes, ops
    from humming.layer import HummingLayerMeta
    from humming.tune import get_heuristics_config
    from humming.utils.weight import (
        prepare_humming_weight,
        prepare_humming_weight_scale,
        quantize_weight,
    )

    torch.cuda.set_device(0)
    generator = torch.Generator(device="cpu").manual_seed(0)

    report: dict[str, Any] = {
        "topk": TOPK,
        "local_experts": LOCAL_EXPERTS,
        "global_experts": GLOBAL_EXPERTS,
        "repeats": args.repeats,
        "rel_err_row_threshold": REL_ERR_ROW_THRESHOLD,
        "cases": [],
    }
    any_fail = False

    for geom_name in args.geometries:
        geom = GEOMETRIES[geom_name]
        shape_n, shape_k = geom["shape_n"], geom["shape_k"]
        layer_config = build_layer_config(shape_n, shape_k)
        meta = HummingLayerMeta(**layer_config)

        weight_bf16 = (
            torch.randn(
                LOCAL_EXPERTS,
                shape_n,
                shape_k,
                generator=generator,
                dtype=torch.float32,
            )
            * 0.02
        ).to(torch.bfloat16)

        packed_weight, weight_scale_raw, _zp, _gs = quantize_weight(
            weight=weight_bf16,
            dtype=dtypes.uint4,
            scale_dtype=dtypes.bfloat16,
            group_size=GROUP_SIZE,
            pack=True,
        )
        unpacked_weight, unpacked_scale, _zp2, _gs2 = quantize_weight(
            weight=weight_bf16,
            dtype=dtypes.uint4,
            scale_dtype=dtypes.bfloat16,
            group_size=GROUP_SIZE,
            pack=False,
        )
        w_deq = dequantize_weight_reference(
            unpacked_weight.cuda(), unpacked_scale.cuda(), weight_bf16.cuda()
        )

        weight = prepare_humming_weight(
            weight=packed_weight,
            b_dtype=dtypes.uint4,
            a_dtype=dtypes.float8e4m3,
            zero_point=None,
            use_wgmma=True,
            packed=True,
        )
        weight_scale = prepare_humming_weight_scale(
            weight_scale_raw,
            to_apply_on_c=meta.should_apply_bs_on_c,
            is_blockwise=False,
        )

        buckets = get_heuristics_config(
            meta=meta,
            gemm_type="grouped_contiguous",
            use_f16_accum=False,
            use_batch_invariant=False,
        )

        for num_tokens in args.tokens:
            offsets, total_valid = realistic_offsets(num_tokens, generator)
            buffer_rows = num_tokens * TOPK

            estimate = production_valid_estimate(num_tokens)
            tuning = None
            for lo, hi, cfg in buckets:
                if lo < estimate <= hi:
                    tuning = dict(cfg)
                    break
            assert tuning is not None, f"no bucket for estimate={estimate}"
            tuning.pop("num_sms", None)

            a_full = torch.empty(
                buffer_rows, shape_k, dtype=torch.bfloat16, device="cuda"
            )
            a_full.normal_(0.0, 1.0, generator=None)
            a_full[total_valid:] *= args.garbage_scale

            quant_full, scale_full = ops.quant_input(
                inputs=a_full, dtype=str(meta.a_dtype), group_size=None
            )
            quant_exact, scale_exact = ops.quant_input(
                inputs=a_full[:total_valid].contiguous(),
                dtype=str(meta.a_dtype),
                group_size=None,
            )
            assert torch.equal(scale_full[:total_valid], scale_exact)

            compute_config = {
                "use_f16_accum": False,
                "use_batch_invariant": False,
                "gemm_type": "grouped_contiguous",
            }
            locks = torch.zeros(1024, dtype=torch.int32, device="cuda")
            offsets_cuda = offsets.cuda()

            def run(inputs: torch.Tensor, input_scale: torch.Tensor) -> torch.Tensor:
                out = ops.humming_gemm(
                    layer_config=json.dumps(layer_config),
                    compute_config=json.dumps(compute_config),
                    tuning_config=json.dumps(tuning),
                    inputs=inputs,
                    weight=weight,
                    input_scale=input_scale,
                    weight_scale=weight_scale,
                    expert_layout=offsets_cuda,
                    locks=locks,
                    valid_shape_m=estimate,
                )
                torch.cuda.synchronize()
                return out

            runs = [
                run(quant_full, scale_full)[:total_valid].clone()
                for _ in range(args.repeats)
            ]
            deterministic = all(torch.equal(runs[0], r) for r in runs[1:])
            out_exact = run(quant_exact, scale_exact)

            full_vs_exact_identical = torch.equal(runs[0], out_exact)
            ref = reference_output(quant_full, scale_full, w_deq, offsets_cuda)
            metrics = row_metrics(runs[0], ref, offsets)

            case = {
                "geometry": geom_name,
                "num_tokens": num_tokens,
                "buffer_rows": buffer_rows,
                "total_valid_rows": total_valid,
                "valid_estimate_used_for_tuning": estimate,
                "tuning_config": tuning,
                "deterministic_across_repeats": deterministic,
                "full_vs_exact_identical": full_vs_exact_identical,
                "vs_reference": metrics,
            }
            report["cases"].append(case)

            failed = (
                not deterministic
                or not full_vs_exact_identical
                or metrics["frac_rows_bad"] > 0
                or not metrics["finite"]
            )
            any_fail = any_fail or failed
            status = "FAIL" if failed else "ok"
            print(
                f"[{status}] {geom_name} tokens={num_tokens:>5} "
                f"valid={total_valid:>5} bm={tuning['block_shape'][0]:>3} "
                f"bk={tuning['block_shape'][2]:>3} "
                f"det={deterministic} full==exact={full_vs_exact_identical} "
                f"bad_rows={metrics['num_rows_bad']} "
                f"max_rel={metrics['max_row_rel_err']:.4f}"
            )

    report["any_fail"] = any_fail
    (out_dir / "scale-probe.json").write_text(json.dumps(report, indent=2))
    print(f"\nany_fail={any_fail}  report: {out_dir / 'scale-probe.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
