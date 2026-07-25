"""Root-cause probe: does Humming's grouped_contiguous GEMM mishandle an
oversized permuted input buffer?

Arm 3 (grouped_contiguous) passed the correctness qualification -- ten "2+2"
smokes, positive attestation, no IMA/NaN -- and then, in the perf window,
misread the prompt "What is the weather in Paris?" as being about "Skik" at
temperature 0, while CUTLASS and Humming-indexed both answered the identical
request correctly. Fluent output, wrong content, no crash: a silent numerical
fault, not a config problem.

Reading the scheduler gives a specific suspect. For GROUPED_CONTIGUOUS,
humming/include/humming/scheduler.cuh derives the LAST expert's row count from
the scalar kernel argument ``shape_m``::

    smem.expert_tokens[kNumExperts - 1] = shape_m - smem.expert_offset[kNumExperts - 1];

and every other expert's from differences of ``expert_offset``. So the kernel
requires ``shape_m == expert_offset[kNumExperts]`` -- the exact number of valid
permuted rows.

But ``shape_m`` is ``a.size(0)`` (csrc/launcher/launcher.cpp:83), and vLLM hands
the grouped path the whole permuted workspace: HummingGroupedExperts.main_apply
passes ``buffers["quanted_gate_up_input"]``, sized ``(M * topk, K)``
(fused_humming_moe.py get_buffer_metas). Under expert parallelism only a
fraction of those rows are ever filled -- with 16 local experts of 128 and
topk=4, about M/2 of M*4. So the last local expert is told it owns roughly
``4M - offset[15]`` rows instead of ``M/2 - offset[15]``: thousands of unfilled
rows that ``may_quant_input`` has already quantized as garbage.

Whether that is merely wasted work or actually corrupts the valid rows cannot be
settled by reading the CUDA source, so this probe measures it. One variable:

  A) inputs = the full (M*topk, K) buffer            <- what vLLM passes today
  B) inputs = the same buffer sliced to (total_valid, K)

Same weights, same expert_layout, same tuning config, same data in the valid
rows. If ``A[:total_valid]`` differs from ``B``, the oversized buffer corrupts
real tokens and this is arm 3's root cause.

Usage (1 GPU is enough)::

    python -m pipeline.m3_humming_grouped_bounds_probe --out <dir>
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import torch

# w2 geometry of MiniMax-M3's routed experts under TP8 + EP: 16 local experts of
# 128, hidden 6144, moe intermediate 3072. Small M keeps the probe quick while
# preserving every shape the kernel specialises on.
LOCAL_EXPERTS = 16
GLOBAL_EXPERTS = 128
SHAPE_N = 6144
SHAPE_K = 3072
TOPK = 4
GROUP_SIZE = 128


def build_layer_config() -> dict[str, Any]:
    return {
        "shape_n": SHAPE_N,
        "shape_k": SHAPE_K,
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
    """Exact cumulative per-expert offsets, as vLLM's moe_permute produces them.

    Sampling actual routing rather than assuming a uniform split matters: the
    whole question is what happens when the true total differs from the buffer
    size, and a hand-rounded split would beg it.
    """
    topk_ids = torch.randint(
        0, GLOBAL_EXPERTS, (num_tokens, TOPK), generator=generator, device="cpu"
    )
    # Expert-parallel shard: this rank owns the first LOCAL_EXPERTS global ids.
    local_hits = topk_ids[topk_ids < LOCAL_EXPERTS]
    counts = torch.bincount(local_hits, minlength=LOCAL_EXPERTS)
    offsets = torch.zeros(LOCAL_EXPERTS + 1, dtype=torch.int64)
    offsets[1:] = torch.cumsum(counts, dim=0)
    return offsets, int(offsets[-1].item())


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", required=True)
    parser.add_argument("--num-tokens", type=int, default=512)
    parser.add_argument(
        "--garbage-scale",
        type=float,
        default=100.0,
        help="magnitude of the unfilled rows; large values make any leak obvious",
    )
    args = parser.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    from humming import dtypes, ops
    from humming.layer import HummingLayerMeta
    from humming.tune import get_heuristics_config
    from humming.utils.weight import quantize_weight

    torch.cuda.set_device(0)
    generator = torch.Generator(device="cpu").manual_seed(0)

    layer_config = build_layer_config()
    meta = HummingLayerMeta(**layer_config)

    offsets, total_valid = realistic_offsets(args.num_tokens, generator)
    buffer_rows = args.num_tokens * TOPK

    # --- weights: quantized once, shared by both calls --------------------
    weight_bf16 = (
        torch.randn(
            LOCAL_EXPERTS, SHAPE_N, SHAPE_K, generator=generator, dtype=torch.float32
        )
        * 0.02
    ).to(torch.bfloat16)
    packed_weight, weight_scale, _zp, _gs = quantize_weight(
        weight=weight_bf16,
        dtype=dtypes.uint4,
        scale_dtype=dtypes.bfloat16,
        group_size=GROUP_SIZE,
        pack=True,
    )
    from humming.utils.weight import (
        prepare_humming_weight,
        prepare_humming_weight_scale,
    )

    # Mirror transform_humming_layer exactly: weight and scale each get their own
    # layout transform, and use_wgmma must match the kernel's MmaType (WGMMA on
    # SM90) or the repacked layout is not the one the kernel indexes.
    weight = prepare_humming_weight(
        weight=packed_weight,
        b_dtype=dtypes.uint4,
        a_dtype=dtypes.float8e4m3,
        zero_point=None,
        use_wgmma=True,
        packed=True,
    )
    weight_scale = prepare_humming_weight_scale(
        weight_scale,
        to_apply_on_c=meta.should_apply_bs_on_c,
        is_blockwise=False,
    )

    # --- activations: valid rows real, the rest deliberately hostile ------
    a_full = torch.empty(buffer_rows, SHAPE_K, dtype=torch.bfloat16, device="cuda")
    a_full.normal_(0.0, 1.0, generator=None)
    a_full[total_valid:] *= args.garbage_scale
    valid_reference = a_full[:total_valid].clone()

    quant_full, scale_full = ops.quant_input(
        inputs=a_full, dtype=str(meta.a_dtype), group_size=None
    )
    quant_exact, scale_exact = ops.quant_input(
        inputs=a_full[:total_valid].contiguous(),
        dtype=str(meta.a_dtype),
        group_size=None,
    )

    # Per-token scales must agree on the shared rows, or the comparison below
    # would be confounded by quantization rather than by scheduling.
    assert torch.equal(scale_full[:total_valid], scale_exact), "per-row scales diverged"

    buckets = get_heuristics_config(
        meta=meta,
        gemm_type="grouped_contiguous",
        use_f16_accum=False,
        use_batch_invariant=False,
    )
    tuning = None
    for lo, hi, cfg in buckets:
        if lo < total_valid <= hi:
            tuning = dict(cfg)
            break
    assert tuning is not None, f"no bucket for shape_m={total_valid}"
    tuning.pop("num_sms", None)

    compute_config = {
        "use_f16_accum": False,
        "use_batch_invariant": False,
        "gemm_type": "grouped_contiguous",
    }
    locks = torch.zeros(1024, dtype=torch.int32, device="cuda")
    offsets_cuda = offsets.cuda()

    def run(inputs: torch.Tensor, input_scale: torch.Tensor) -> torch.Tensor:
        return ops.humming_gemm(
            layer_config=json.dumps(layer_config),
            compute_config=json.dumps(compute_config),
            tuning_config=json.dumps(tuning),
            inputs=inputs,
            weight=weight,
            input_scale=input_scale,
            weight_scale=weight_scale,
            expert_layout=offsets_cuda,
            locks=locks,
            valid_shape_m=total_valid,
        )

    out_full = run(quant_full, scale_full)
    torch.cuda.synchronize()
    out_exact = run(quant_exact, scale_exact)
    torch.cuda.synchronize()

    a = out_full[:total_valid].float()
    b = out_exact.float()
    diff = (a - b).abs()
    mismatch = (diff > 1e-3).float().mean().item()

    # Localise: which experts' row ranges disagree?
    per_expert = []
    for e in range(LOCAL_EXPERTS):
        lo, hi = int(offsets[e]), int(offsets[e + 1])
        if hi <= lo:
            per_expert.append({"expert": e, "rows": 0, "max_abs_diff": 0.0})
            continue
        d = diff[lo:hi]
        per_expert.append(
            {
                "expert": e,
                "rows": hi - lo,
                "max_abs_diff": float(d.max().item()),
                "frac_rows_wrong": float((d.amax(dim=1) > 1e-3).float().mean().item()),
            }
        )

    report = {
        "num_tokens": args.num_tokens,
        "topk": TOPK,
        "buffer_rows": buffer_rows,
        "total_valid_rows": total_valid,
        "oversize_factor": round(buffer_rows / max(total_valid, 1), 2),
        "last_expert_true_rows": int(offsets[-1] - offsets[-2]),
        "last_expert_rows_unpatched_kernel_infers": buffer_rows - int(offsets[-2]),
        "identical": bool(torch.equal(a, b)),
        "frac_elements_mismatched": mismatch,
        "max_abs_diff": float(diff.max().item()),
        "out_full_finite": bool(torch.isfinite(out_full[:total_valid]).all().item()),
        "out_exact_finite": bool(torch.isfinite(out_exact).all().item()),
        "per_expert": per_expert,
        "valid_reference_norm": float(valid_reference.float().norm().item()),
    }
    (out / "bounds-probe.json").write_text(json.dumps(report, indent=2))

    print(f"valid rows      : {total_valid}")
    print(f"buffer rows     : {buffer_rows}  ({report['oversize_factor']}x oversized)")
    unpatched = report["last_expert_rows_unpatched_kernel_infers"]
    print(
        f"last expert rows: true={report['last_expert_true_rows']} "
        f"unpatched-would-infer={unpatched}"
    )
    print(f"identical       : {report['identical']}")
    print(
        f"mismatched elems: {mismatch:.4%}   max_abs_diff={report['max_abs_diff']:.4g}"
    )
    full_finite = report["out_full_finite"]
    exact_finite = report["out_exact_finite"]
    print(f"finite          : full={full_finite} exact={exact_finite}")
    print("\nper-expert disagreement (rows, max|diff|, frac rows wrong):")
    for row in per_expert:
        if row["rows"]:
            wrong = row.get("frac_rows_wrong", 0)
            print(
                f"  e{row['expert']:>2}  rows={row['rows']:>5}  "
                f"max={row['max_abs_diff']:.4g}  wrong={wrong:.2%}"
            )
    print(f"\nreport: {out / 'bounds-probe.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
