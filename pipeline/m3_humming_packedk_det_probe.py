"""Bisect the humming 0.1.11 packed-K nondeterminism at w13/4096.

The 0.1.11 packed-K qualification (m3-humming-0111-packedk-qual/20260726T031338Z)
was clean except one bucket: w13 (N=6144, K=6144), tokens=4096, heuristic
config block=[128,256,128] warp=[128,32,128] stream-K -- bitwise
nondeterministic across repeats (and consequently full-vs-exact mismatched),
while every value stayed within fp8 accumulation noise of the reference
(max_row_rel_err 0.0085, zero rows beyond threshold). The same config on
w2 (K=3072) was deterministic, and 0.1.10's heuristic config for the same
bucket (block=[184,128,128], also stream-K) was deterministic on both
geometries after the TMA commit-group fix.

Candidate causes, separated by this probe's variant matrix:

  packedk-baseline      packed-K, heuristic config  -> reproduce det=False
  packedk-no-stream-k   packed-K, use_stream_k=False
        det=True  => the variation enters via stream-K partial-sum
                     accumulation; combined with ulp-scale diffs that is
                     fp reduce-order variation, not a data race.
        det=False => race in the packed-K mainloop/epilogue itself.
  packedk-bn128         packed-K, block=[128,128,128]
        isolates the BN=256 epilogue (4x 64-col store chunks / bigger
        reduce smem) from the packed-K B-layout.
  classic-same-tuning   packed-K OFF, same [128,256,128] config
        det=False => the config shape (BN=256/warp-N=32 + stream-K at
                     K_BLOCKS=48), not the packed-K layout, is the trigger
                     -- an upstream latent config bug 0.1.10 heuristics
                     never selected.
        det=True  => packed-K-specific.
  classic-0110-tuning   packed-K OFF, 0.1.10's [184,128,128] config
        sanity anchor, expected det=True (matches the 0.1.10 verify run).

Each variant also fingerprints the first differing repeat pair: how many
elements differ, the max |diff| and max relative diff, and how many
(expert, n_block) tiles contain diffs -- ulp-scale scattered diffs point at
fp ordering; whole-tile garbage points at a store/sync race.

Usage (1 GPU)::

    python -m pipeline.m3_humming_packedk_det_probe --out <dir>
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import torch

from pipeline.m3_humming_grouped_scale_probe import (
    GEOMETRIES,
    GROUP_SIZE,
    LOCAL_EXPERTS,
    TOPK,
    build_layer_config,
    dequantize_weight_reference,
    production_valid_estimate,
    realistic_offsets,
    reference_output,
)

REL_ERR_ROW_THRESHOLD = 0.05


def diff_fingerprint(
    a: torch.Tensor, b: torch.Tensor, offsets: torch.Tensor, n_block: int
) -> dict[str, Any]:
    d = (a.float() - b.float()).abs()
    mask = d > 0
    n_diff = int(mask.sum().item())
    if n_diff == 0:
        return {"n_diff_elements": 0}
    denom = a.float().abs().clamp_min(1e-6)
    rows = mask.any(dim=1).nonzero().flatten()
    tiles = set()
    for r in rows.tolist():
        e = int(torch.searchsorted(offsets, r, right=True).item() - 1)
        cols = mask[r].nonzero().flatten()
        for j in torch.unique(cols // n_block).tolist():
            tiles.add((e, int(j)))
    return {
        "n_diff_elements": n_diff,
        "frac_diff_elements": n_diff / d.numel(),
        "max_abs_diff": float(d.amax().item()),
        "max_rel_diff": float((d / denom).amax().item()),
        "n_rows_with_diffs": int(rows.numel()),
        "n_tiles_with_diffs": len(tiles),
        "tiles": sorted(tiles)[:40],
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", required=True)
    parser.add_argument("--geometry", default="w13", choices=list(GEOMETRIES))
    parser.add_argument("--tokens", type=int, default=4096)
    parser.add_argument("--repeats", type=int, default=16)
    parser.add_argument("--garbage-scale", type=float, default=100.0)
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

    geo = GEOMETRIES[args.geometry]
    shape_n, shape_k = geo["shape_n"], geo["shape_k"]
    layer_config = build_layer_config(shape_n, shape_k)
    meta = HummingLayerMeta(**layer_config)
    assert getattr(meta, "use_packed_k_layout", False), (
        "this probe targets the 0.1.11 packed-K path; meta did not enable it "
        "-- wrong humming on PYTHONPATH?"
    )

    n_block = 128  # tile-attribution granularity only

    offsets, total_valid = realistic_offsets(args.tokens, generator)
    buffer_rows = args.tokens * TOPK
    estimate = production_valid_estimate(args.tokens)

    weight_bf16 = (
        torch.randn(
            LOCAL_EXPERTS, shape_n, shape_k, generator=generator, dtype=torch.float32
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
    w_deq = dequantize_weight_reference(packed_weight, weight_scale_raw, weight_bf16)

    weights = {}
    for packed_k in (True, False):
        weights[packed_k] = prepare_humming_weight(
            weight=packed_weight,
            b_dtype=dtypes.uint4,
            a_dtype=dtypes.float8e4m3,
            zero_point=None,
            use_wgmma=True,
            packed=True,
            use_packed_k_layout=packed_k,
        )
    weight_scale = prepare_humming_weight_scale(
        weight_scale_raw,
        to_apply_on_c=meta.should_apply_bs_on_c,
        is_blockwise=False,
    )

    a_full = torch.empty(buffer_rows, shape_k, dtype=torch.bfloat16, device="cuda")
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

    ref = reference_output(quant_full, scale_full, w_deq, offsets.cuda())

    buckets = get_heuristics_config(
        meta=meta,
        gemm_type="grouped_contiguous",
        use_f16_accum=False,
        use_batch_invariant=False,
    )
    heuristic = None
    for lo, hi, cfg in buckets:
        if lo < estimate <= hi:
            heuristic = dict(cfg)
            break
    assert heuristic is not None
    heuristic.pop("num_sms", None)

    tuning_0110 = {
        "block_shape": [184, 128, 128],
        "warp_shape": [184, 16, 128],
        "use_stream_k": True,
        "use_f16_accum": False,
        "num_stages": 4,
        "use_warp_spec": True,
        "use_tma": True,
        "use_mbarrier": True,
    }

    variants: list[tuple[str, bool, dict[str, Any]]] = [
        ("packedk-baseline", True, dict(heuristic)),
        ("packedk-no-stream-k", True, {**heuristic, "use_stream_k": False}),
        (
            "packedk-bn128",
            True,
            {
                **heuristic,
                "block_shape": [128, 128, 128],
                "warp_shape": [128, 16, 128],
            },
        ),
        ("classic-same-tuning", False, dict(heuristic)),
        ("classic-0110-tuning", False, tuning_0110),
    ]

    compute_config = {
        "use_f16_accum": False,
        "use_batch_invariant": False,
        "gemm_type": "grouped_contiguous",
    }
    offsets_cuda = offsets.cuda()
    locks = torch.zeros(1024, dtype=torch.int32, device="cuda")

    report: dict[str, Any] = {
        "geometry": args.geometry,
        "tokens": args.tokens,
        "total_valid": total_valid,
        "estimate": estimate,
        "repeats": args.repeats,
        "heuristic_tuning": heuristic,
        "variants": [],
    }

    for name, packed_k, tuning in variants:
        lc = dict(layer_config)
        lc["use_packed_k_layout"] = packed_k
        entry: dict[str, Any] = {
            "name": name,
            "use_packed_k_layout": packed_k,
            "tuning": tuning,
        }

        def run(inputs: torch.Tensor, input_scale: torch.Tensor) -> torch.Tensor:
            out = ops.humming_gemm(
                layer_config=json.dumps(lc),
                compute_config=json.dumps(compute_config),
                tuning_config=json.dumps(tuning),
                inputs=inputs,
                weight=weights[packed_k],
                input_scale=input_scale,
                weight_scale=weight_scale,
                expert_layout=offsets_cuda,
                locks=locks,
                valid_shape_m=estimate,
            )
            torch.cuda.synchronize()
            return out

        try:
            runs = [
                run(quant_full, scale_full)[:total_valid].clone()
                for _ in range(args.repeats)
            ]
            out_exact = run(quant_exact, scale_exact)[:total_valid].clone()
        except Exception as exc:  # config invalid for this kernel family
            entry["error"] = f"{type(exc).__name__}: {exc}"
            report["variants"].append(entry)
            print(f"[err] {name}: {entry['error']}")
            continue

        deterministic = all(torch.equal(runs[0], r) for r in runs[1:])
        entry["deterministic_across_repeats"] = deterministic
        entry["full_vs_exact_identical"] = torch.equal(runs[0], out_exact)
        rel = (runs[0].float() - ref).abs() / ref.abs().amax(dim=1).clamp_min(
            1e-6
        ).view(-1, 1)
        entry["max_row_rel_err"] = float(rel.amax().item())
        entry["rows_beyond_threshold"] = int(
            (rel > REL_ERR_ROW_THRESHOLD).any(dim=1).sum().item()
        )

        if not deterministic:
            other = next(r for r in runs[1:] if not torch.equal(runs[0], r))
            entry["repeat_diff"] = diff_fingerprint(
                runs[0], other, offsets, n_block
            )
        if not entry["full_vs_exact_identical"]:
            entry["full_exact_diff"] = diff_fingerprint(
                runs[0], out_exact, offsets, n_block
            )

        report["variants"].append(entry)
        print(
            f"[{'ok' if deterministic else 'NONDET'}] {name} "
            f"packed_k={packed_k} det={deterministic} "
            f"full==exact={entry['full_vs_exact_identical']} "
            f"max_rel={entry['max_row_rel_err']:.4f} "
            f"bad_rows={entry['rows_beyond_threshold']}"
        )

    path = out_dir / "packedk-det-probe.json"
    path.write_text(json.dumps(report, indent=1))
    print(f"report: {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
