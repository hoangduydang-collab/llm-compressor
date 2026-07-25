"""Focused repro for the w13/512 grouped_contiguous corruption found by
pipeline/m3_humming_grouped_scale_probe.py (window 20260725T155431Z).

That sweep showed expert 3 (the case's largest expert, 23 rows) getting
13/23 rows of garbage at magnitude ~1e33 -- i.e. torch.empty allocator
garbage, suggesting rows the kernel NEVER WROTE -- nondeterministically,
in a tile that belongs to the data-parallel (not stream-K) phase.

This probe pins the mechanism:
  1. sentinel-filled explicit `outputs` tensor -> never-written rows are
     exactly the sentinel, unambiguously separating "store clipped/skipped"
     from "wrong value written";
  2. per-(row, n_block) 2D bad mask -> identifies WHICH tiles failed,
     hence which CTA/phase issued (or skipped) the store;
  3. tuning-config bisection: baseline / use_stream_k=False / use_tma=False
     -> whichever toggle goes clean localizes the broken code path
     (stream-K scheduling vs TMA-C tensor-map clamping vs generic).

Routing geometry is hardcoded to the failing case (offsets reproduced from
the sweep's seed walk); weights/activations are fresh random -- the failure
is expected to be data-independent scheduling/store misbehaviour.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import torch

TOPK = 4
LOCAL_EXPERTS = 16
GROUP_SIZE = 128

# w13 geometry (gate+up): N = 2 * 3072, K = hidden = 6144.
SHAPE_N = 6144
SHAPE_K = 6144
N_BLOCK = 128  # BN of the failing tuning config -> 48 n-blocks

# Exact per-expert token counts of the failing sweep case (w13, tokens=512,
# BM=32: every expert fits one m-block; expert 3 is the unique largest).
FAILING_COUNTS = [19, 20, 19, 23, 10, 15, 10, 15, 19, 15, 13, 15, 12, 13, 15, 13]

SENTINEL = 30000.0  # bf16-exact, far above any real |C| for this data
REL_ERR_THRESHOLD = 0.05


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


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", required=True)
    parser.add_argument("--repeats", type=int, default=8)
    parser.add_argument("--buffer-tokens", type=int, default=512)
    parser.add_argument("--garbage-scale", type=float, default=100.0)
    args = parser.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    from humming import dtypes, ops
    from humming.layer import HummingLayerMeta
    from humming.tune import get_heuristics_config
    from humming.utils.weight import (
        dequantize_weight,
        prepare_humming_weight,
        prepare_humming_weight_scale,
        quantize_weight,
    )

    torch.cuda.set_device(0)
    torch.manual_seed(1234)

    layer_config = build_layer_config()
    meta = HummingLayerMeta(**layer_config)

    counts = torch.tensor(FAILING_COUNTS, dtype=torch.int64)
    offsets = torch.zeros(LOCAL_EXPERTS + 1, dtype=torch.int64)
    offsets[1:] = torch.cumsum(counts, dim=0)
    total_valid = int(offsets[-1].item())
    buffer_rows = args.buffer_tokens * TOPK
    estimate = 256  # production estimate the failing sweep case used

    weight_bf16 = (
        torch.randn(LOCAL_EXPERTS, SHAPE_N, SHAPE_K, dtype=torch.float32) * 0.02
    ).to(torch.bfloat16)
    packed_weight, weight_scale_raw, _zp, _gs = quantize_weight(
        weight=weight_bf16,
        dtype=dtypes.uint4,
        scale_dtype=dtypes.bfloat16,
        group_size=GROUP_SIZE,
        pack=True,
    )
    w_deq = dequantize_weight(
        weight=packed_weight,
        weight_scale=weight_scale_raw,
        zero_point=None,
        global_scale=None,
        dtype=dtypes.uint4,
        packed=True,
    ).float()

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

    a_full = torch.empty(buffer_rows, SHAPE_K, dtype=torch.bfloat16, device="cuda")
    a_full.normal_(0.0, 1.0)
    a_full[total_valid:] *= args.garbage_scale
    quant_full, scale_full = ops.quant_input(
        inputs=a_full, dtype=str(meta.a_dtype), group_size=None
    )

    # fp32 reference on the valid slice.
    a_q32 = quant_full[:total_valid].to(torch.float32)
    a_s32 = scale_full[:total_valid].to(torch.float32).view(-1, 1)
    ref = torch.empty(total_valid, SHAPE_N, dtype=torch.float32, device="cuda")
    for e in range(LOCAL_EXPERTS):
        lo, hi = int(offsets[e]), int(offsets[e + 1])
        if hi > lo:
            ref[lo:hi] = (a_q32[lo:hi] @ w_deq[e].T) * a_s32[lo:hi]

    buckets = get_heuristics_config(
        meta=meta,
        gemm_type="grouped_contiguous",
        use_f16_accum=False,
        use_batch_invariant=False,
    )
    base_tuning = None
    for lo, hi, cfg in buckets:
        if lo < estimate <= hi:
            base_tuning = dict(cfg)
            break
    assert base_tuning is not None
    base_tuning.pop("num_sms", None)
    assert base_tuning["block_shape"][0] == 32, base_tuning

    variants = [
        ("baseline", dict(base_tuning)),
        ("no_stream_k", {**base_tuning, "use_stream_k": False}),
        ("no_tma", {**base_tuning, "use_tma": False}),
    ]

    offsets_cuda = offsets.cuda()
    locks = torch.zeros(1024, dtype=torch.int32, device="cuda")
    compute_config = {
        "use_f16_accum": False,
        "use_batch_invariant": False,
        "gemm_type": "grouped_contiguous",
    }

    report: dict[str, Any] = {
        "counts": FAILING_COUNTS,
        "offsets": offsets.tolist(),
        "buffer_rows": buffer_rows,
        "total_valid": total_valid,
        "estimate": estimate,
        "sentinel": SENTINEL,
        "repeats": args.repeats,
        "variants": [],
    }

    for name, tuning in variants:
        entry: dict[str, Any] = {"name": name, "tuning": tuning, "repeats": []}
        try:
            for rep in range(args.repeats):
                out = torch.full(
                    (buffer_rows, SHAPE_N),
                    SENTINEL,
                    dtype=torch.bfloat16,
                    device="cuda",
                )
                ops.humming_gemm(
                    layer_config=json.dumps(layer_config),
                    compute_config=json.dumps(compute_config),
                    tuning_config=json.dumps(tuning),
                    inputs=quant_full,
                    weight=weight,
                    outputs=out,
                    input_scale=scale_full,
                    weight_scale=weight_scale,
                    expert_layout=offsets_cuda,
                    locks=locks,
                    valid_shape_m=estimate,
                )
                torch.cuda.synchronize()

                valid = out[:total_valid].float()
                never_written = valid == SENTINEL
                diff = (valid - ref).abs()
                ref_row_mag = ref.abs().amax(dim=1).clamp_min(1e-6).view(-1, 1)
                bad2d = (diff / ref_row_mag) > REL_ERR_THRESHOLD
                wrong_written = bad2d & ~never_written

                def blocks_of(mask: torch.Tensor) -> list[dict[str, Any]]:
                    """Summarize a 2D bad mask as (expert, row, n_block) hits."""
                    rows = mask.any(dim=1).nonzero().flatten().tolist()
                    out_list = []
                    for r in rows:
                        nb = (
                            mask[r]
                            .view(SHAPE_N // N_BLOCK, N_BLOCK)
                            .any(dim=1)
                            .nonzero()
                            .flatten()
                            .tolist()
                        )
                        e = int(
                            torch.searchsorted(offsets, r, right=True).item() - 1
                        )
                        out_list.append(
                            {
                                "row": r,
                                "expert": e,
                                "row_in_expert": r - int(offsets[e]),
                                "n_blocks": nb,
                            }
                        )
                    return out_list

                rep_entry = {
                    "rep": rep,
                    "never_written_rows": blocks_of(never_written),
                    "wrong_written_rows": blocks_of(wrong_written),
                    "num_never_written_elems": int(never_written.sum().item()),
                    "num_wrong_written_elems": int(wrong_written.sum().item()),
                    "max_rel_err_excl_sentinel": float(
                        (diff / ref_row_mag)[~never_written].max().item()
                    )
                    if (~never_written).any()
                    else None,
                }
                entry["repeats"].append(rep_entry)
                nw = len(rep_entry["never_written_rows"])
                ww = len(rep_entry["wrong_written_rows"])
                print(
                    f"[{name}] rep={rep} never_written_rows={nw} "
                    f"wrong_written_rows={ww} "
                    f"max_rel={rep_entry['max_rel_err_excl_sentinel']}"
                )
                for r in rep_entry["never_written_rows"][:8]:
                    print(
                        f"    NEVER-WRITTEN row={r['row']} expert={r['expert']} "
                        f"row_in_expert={r['row_in_expert']} n_blocks={r['n_blocks']}"
                    )
                for r in rep_entry["wrong_written_rows"][:8]:
                    print(
                        f"    WRONG-VALUE row={r['row']} expert={r['expert']} "
                        f"row_in_expert={r['row_in_expert']} n_blocks={r['n_blocks']}"
                    )
        except Exception as exc:  # noqa: BLE001 -- record and continue bisection
            entry["error"] = repr(exc)
            print(f"[{name}] ERROR: {exc!r}")
        report["variants"].append(entry)

    (out_dir / "dp-tile-probe.json").write_text(json.dumps(report, indent=2))
    print(f"\nreport: {out_dir / 'dp-tile-probe.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
