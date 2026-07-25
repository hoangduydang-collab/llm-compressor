"""Fingerprint the corrupted tiles of the grouped_contiguous w13/512 case.

pipeline/m3_humming_grouped_dp_tile_probe.py established: ~25% of runs one
whole (expert, n_block) tile is WRITTEN with wrong values (sentinel probe:
rows are written, not clipped); it reproduces with use_stream_k=False but
NOT with use_tma=False -> a TMA-path race.

This script reruns the failing config many times and, for every corrupted
tile, compares its content against candidate reconstructions to identify
which data the kernel actually computed/stored there:

  store-misroute-m : ref tile of ANOTHER expert e', same n_block
                     (C store landed at wrong row_offset)
  store-misroute-n : ref tile of the same expert, different n_block
                     (C store landed at wrong col_offset)
  b-swap           : a[rows(e)] @ w_deq[e'] -- B tiles of a different expert
                     (weight loader desync)
  a-swap-own-scale : a[rows(e')] @ w_deq[e] scaled by rows(e') scales
  a-swap-cur-scale : a[rows(e')] @ w_deq[e] scaled by rows(e) scales
                     (A loader row desync, two scale hypotheses)
  no-a-scale       : a[rows(e)] @ w_deq[e] without the input scale
  stale-plus-ref   : ref + ref (double accumulation, e.g. reduce-add onto
                     an already-stored tile)

Anything not matching (residual > 5%) is reported with its raw stats.
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
SHAPE_N = 6144
SHAPE_K = 6144
N_BLOCK = 128
BM = 32

FAILING_COUNTS = [19, 20, 19, 23, 10, 15, 10, 15, 19, 15, 13, 15, 12, 13, 15, 13]
REL_ERR_THRESHOLD = 0.05
MATCH_THRESHOLD = 0.02  # bf16 store rounding is ~0.4% worst-case


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


def rel_residual(t: torch.Tensor, cand: torch.Tensor) -> float:
    denom = cand.abs().amax().clamp_min(1e-6)
    return float(((t - cand).abs().amax() / denom).item())


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", required=True)
    parser.add_argument("--repeats", type=int, default=48)
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
    estimate = 256

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

    a_q32 = quant_full.to(torch.float32)  # full buffer incl. garbage rows
    a_s32 = scale_full.to(torch.float32).view(-1, 1)
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
    tuning = None
    for lo, hi, cfg in buckets:
        if lo < estimate <= hi:
            tuning = dict(cfg)
            break
    assert tuning is not None and tuning["block_shape"][0] == BM
    tuning.pop("num_sms", None)

    offsets_cuda = offsets.cuda()
    locks = torch.zeros(1024, dtype=torch.int32, device="cuda")
    compute_config = {
        "use_f16_accum": False,
        "use_batch_invariant": False,
        "gemm_type": "grouped_contiguous",
    }

    report: dict[str, Any] = {
        "counts": FAILING_COUNTS,
        "tuning": tuning,
        "repeats": args.repeats,
        "events": [],
    }

    def fingerprint(e_bad: int, j: int, out_valid: torch.Tensor) -> dict[str, Any]:
        lo, hi = int(offsets[e_bad]), int(offsets[e_bad + 1])
        rows = hi - lo
        cols = slice(j * N_BLOCK, (j + 1) * N_BLOCK)
        t = out_valid[lo:hi, cols].float()

        cands: list[tuple[str, torch.Tensor]] = []
        for e2 in range(LOCAL_EXPERTS):
            lo2, hi2 = int(offsets[e2]), int(offsets[e2 + 1])
            r2 = min(rows, hi2 - lo2)
            if e2 != e_bad and r2 > 0:
                cands.append(
                    (f"store-misroute-m:e{e2}", ref[lo2 : lo2 + r2, cols])
                )
                aq2 = a_q32[lo2 : lo2 + r2]
                cands.append(
                    (
                        f"a-swap-own-scale:e{e2}",
                        (aq2 @ w_deq[e_bad][cols].T) * a_s32[lo2 : lo2 + r2],
                    )
                )
                cands.append(
                    (
                        f"a-swap-cur-scale:e{e2}",
                        (aq2 @ w_deq[e_bad][cols].T) * a_s32[lo : lo + r2],
                    )
                )
            if e2 != e_bad:
                cands.append(
                    (
                        f"b-swap:e{e2}",
                        (a_q32[lo:hi] @ w_deq[e2][cols].T) * a_s32[lo:hi],
                    )
                )
        for j2 in range(SHAPE_N // N_BLOCK):
            if j2 != j:
                cols2 = slice(j2 * N_BLOCK, (j2 + 1) * N_BLOCK)
                cands.append((f"store-misroute-n:n{j2}", ref[lo:hi, cols2]))
        cands.append(("no-a-scale", a_q32[lo:hi] @ w_deq[e_bad][cols].T))
        cands.append(("stale-plus-ref", 2.0 * ref[lo:hi, cols]))
        # garbage-region A rows at BM-aligned offsets in the tail
        for gofs in range(total_valid, buffer_rows - rows + 1, BM):
            aq2 = a_q32[gofs : gofs + rows]
            cands.append(
                (
                    f"a-swap-garbage:r{gofs}",
                    (aq2 @ w_deq[e_bad][cols].T) * a_s32[gofs : gofs + rows],
                )
            )

        scored = []
        for name, cand in cands:
            r = min(cand.size(0), rows)
            scored.append((rel_residual(t[:r], cand[:r].float()), name))
        scored.sort()
        return {
            "expert": e_bad,
            "n_block": j,
            "rows": rows,
            "tile_absmax": float(t.abs().amax().item()),
            "ref_absmax": float(ref[lo:hi, cols].abs().amax().item()),
            "best_matches": [
                {"residual": s, "candidate": n} for s, n in scored[:5]
            ],
        }

    n_bad_reps = 0
    for rep in range(args.repeats):
        out = torch.full(
            (buffer_rows, SHAPE_N), 30000.0, dtype=torch.bfloat16, device="cuda"
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

        out_valid = out[:total_valid].float()
        diff = (out_valid - ref).abs()
        ref_row_mag = ref.abs().amax(dim=1).clamp_min(1e-6).view(-1, 1)
        bad2d = (diff / ref_row_mag) > REL_ERR_THRESHOLD
        if not bad2d.any():
            continue

        n_bad_reps += 1
        # collect bad tiles (expert x n_block granularity)
        bad_tiles = set()
        bad_rows = bad2d.any(dim=1).nonzero().flatten().tolist()
        for r in bad_rows:
            e = int(torch.searchsorted(offsets, r, right=True).item() - 1)
            for j in (
                bad2d[r]
                .view(SHAPE_N // N_BLOCK, N_BLOCK)
                .any(dim=1)
                .nonzero()
                .flatten()
                .tolist()
            ):
                bad_tiles.add((e, j))

        for e, j in sorted(bad_tiles):
            fp = fingerprint(e, j, out_valid)
            fp["rep"] = rep
            report["events"].append(fp)
            best = fp["best_matches"][0]
            print(
                f"rep={rep} BAD TILE expert={e} n_block={j} rows={fp['rows']} "
                f"tile_absmax={fp['tile_absmax']:.3g} ref_absmax={fp['ref_absmax']:.3g}"
            )
            for m in fp["best_matches"]:
                print(f"    residual={m['residual']:.4f}  {m['candidate']}")
            verdict = (
                best["candidate"]
                if best["residual"] < MATCH_THRESHOLD
                else "NO-MATCH"
            )
            print(f"    => {verdict}")

    print(f"\nbad_reps={n_bad_reps}/{args.repeats}")
    report["bad_reps"] = n_bad_reps
    (out_dir / "tile-forensics.json").write_text(json.dumps(report, indent=2))
    print(f"report: {out_dir / 'tile-forensics.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
