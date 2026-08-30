"""Compare two ``sglang_load_probe`` artifacts and judge their agreement.

This is the numerical equivalence gate for a format conversion. Structural
verification proves the bytes are encoded as intended; it cannot prove the
ENGINE reads them as intended. Two probes over identical prompts -- one on the
reference (BF16), one on the converted artifact -- close that gap without
needing a reference implementation of the model's forward pass.

HOW TO READ THE NUMBERS. The arms differ by real quantization error, so exact
agreement is neither expected nor desirable as a bound:

  top-1 agreement   The fraction of positions where both arms rank the same
                    token first. This is the headline. A correct W4AFP8
                    conversion of a well-behaved model should be high but NOT
                    1.0 -- int4 experts genuinely change some rankings, most
                    often where the top two candidates were nearly tied.
  logprob delta     Mean and max |delta| on the shared top-k. Sensitive to
                    scale errors: a dropped fold or a reciprocal-vs-multiplier
                    mistake shows up as a large, systematic shift rather than
                    the small scatter quantization produces.
  token id match    Whether the greedy continuations are identical. Brittle by
                    nature -- one divergence early cascades -- so it is
                    reported, not gated.

A LOW SCORE DOES NOT LOCALISE THE FAULT. It says the engine's view of the two
artifacts differs; it does not say whether the cause is the conversion, the
engine's kernel for this scheme, or genuine quantization error. Structural
verification (``verify_sglang_w4afp8``) is what distinguishes those, which is
why it should pass BEFORE this is interpreted.

Usage:
    python -m pipeline.compare_logit_probes --ref bf16.json --test w4afp8.json \\
        [--min-top1 0.8] [--max-mean-delta 1.0]
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path


def _load(path: Path) -> dict:
    record = json.loads(path.read_text(encoding="utf-8"))
    if record.get("status") != "ok":
        raise ValueError(
            f"{path} has status {record.get('status')!r}, so the probe did not "
            f"complete; comparing it would report agreement on missing data"
        )
    return record


def compare(ref: dict, test: dict) -> dict:
    if ref.get("prompts") != test.get("prompts"):
        raise ValueError(
            "the two probes used different prompts, so no comparison between "
            "them is meaningful"
        )

    ref_out = ref.get("outputs") or []
    test_out = test.get("outputs") or []
    if len(ref_out) != len(test_out):
        raise ValueError(
            f"probe shape differs: {len(ref_out)} vs {len(test_out)} results"
        )

    top1_hits = top1_total = 0
    deltas: list[float] = []
    exact_sequences = 0
    per_prompt: list[dict] = []

    for index, (a, b) in enumerate(zip(ref_out, test_out)):
        a_lp = a.get("output_top_logprobs") or []
        b_lp = b.get("output_top_logprobs") or []
        hits = total = 0
        local: list[float] = []
        for pa, pb in zip(a_lp, b_lp):
            if not pa or not pb:
                continue
            total += 1
            if pa[0][1] == pb[0][1]:
                hits += 1
            # Compare only tokens BOTH arms ranked in their top-k. A token
            # present in one and absent from the other has no comparable
            # logprob, and inventing one (-inf, or the k-th value) would
            # manufacture either a huge or a tiny delta from nothing.
            shared = dict(
                (tid, lp) for lp, tid in ((e[0], e[1]) for e in pa)
            )
            for lp, tid in ((e[0], e[1]) for e in pb):
                if tid in shared:
                    local.append(abs(shared[tid] - lp))
        top1_hits += hits
        top1_total += total
        deltas.extend(local)
        same_ids = (a.get("output_ids") is not None
                    and a.get("output_ids") == b.get("output_ids"))
        exact_sequences += bool(same_ids)
        per_prompt.append({
            "prompt": (ref.get("prompts") or [None] * (index + 1))[index],
            "positions": total,
            "top1_agreement": (hits / total) if total else None,
            "mean_delta": (sum(local) / len(local)) if local else None,
            "max_delta": max(local) if local else None,
            "identical_output_ids": same_ids,
            "ref_text": a.get("text"),
            "test_text": b.get("text"),
        })

    return {
        "positions": top1_total,
        "top1_agreement": (top1_hits / top1_total) if top1_total else None,
        "mean_logprob_delta": (sum(deltas) / len(deltas)) if deltas else None,
        "max_logprob_delta": max(deltas) if deltas else None,
        "compared_logprob_pairs": len(deltas),
        "identical_sequences": exact_sequences,
        "sequences": len(ref_out),
        "per_prompt": per_prompt,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ref", required=True, type=Path,
                        help="reference probe, normally the BF16 arm")
    parser.add_argument("--test", required=True, type=Path)
    parser.add_argument("--min-top1", type=float, default=0.8,
                        help="minimum top-1 agreement to pass")
    parser.add_argument("--max-mean-delta", type=float, default=1.0,
                        help="maximum mean |logprob delta| to pass")
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args(argv)

    try:
        ref, test = _load(args.ref), _load(args.test)
        result = compare(ref, test)
    except ValueError as err:
        print(f"[compare] FAIL: {err}", flush=True)
        return 2

    print(f"[compare] ref : {ref['model']} (quant={ref['quantization']})")
    print(f"[compare] test: {test['model']} (quant={test['quantization']})")
    print(f"[compare] positions compared      : {result['positions']}")
    print(f"[compare] top-1 agreement         : "
          f"{_fmt(result['top1_agreement'])}")
    print(f"[compare] mean |logprob delta|    : "
          f"{_fmt(result['mean_logprob_delta'])} "
          f"over {result['compared_logprob_pairs']} shared pairs")
    print(f"[compare] max  |logprob delta|    : "
          f"{_fmt(result['max_logprob_delta'])}")
    print(f"[compare] identical continuations : "
          f"{result['identical_sequences']}/{result['sequences']}")
    for item in result["per_prompt"]:
        print(f"    {item['prompt']!r}: top1={_fmt(item['top1_agreement'])} "
              f"mean={_fmt(item['mean_delta'])} "
              f"ids_match={item['identical_output_ids']}")
        print(f"      ref : {item['ref_text']!r}")
        print(f"      test: {item['test_text']!r}")

    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(result, indent=2), encoding="utf-8")

    failures = []
    if result["positions"] == 0:
        failures.append("no comparable positions; the probes carry no logprobs")
    top1 = result["top1_agreement"]
    if top1 is not None and top1 < args.min_top1:
        failures.append(f"top-1 agreement {top1:.3f} < {args.min_top1}")
    mean = result["mean_logprob_delta"]
    if mean is not None and (math.isnan(mean) or mean > args.max_mean_delta):
        failures.append(f"mean logprob delta {mean:.4f} > {args.max_mean_delta}")

    if failures:
        for line in failures:
            print(f"[compare] FAIL: {line}", flush=True)
        return 1
    print("[compare] RESULT: PASS", flush=True)
    return 0


def _fmt(value) -> str:
    return "n/a" if value is None else f"{value:.4f}"


if __name__ == "__main__":
    sys.exit(main())
