#!/usr/bin/env python3
"""Stage nvidia/SPEED-Bench prompts for EAGLE3 phase D (M3_SPECDEC_EAGLE3_PLAN.md).

SPEED-Bench is NVIDIA's benchmark built specifically to measure speculative
decoding across semantic domains AND input sequence lengths -- exactly the
question phase D asks. We adopt it rather than hand-building a trace.

Two release facts force a staging step instead of `--public-dataset speed_bench_*`:

1. **~45% of the public parquet is masked.** Licensed source text is replaced by
   "FULL BENCHMARK DATA SHOULD BE FETCHED FROM THE SOURCE USING SPECDEC_BENCH".
   aiperf's SpeedBenchLoader does NOT filter those rows, so a straight
   --public-dataset run would mix ~10-token placeholders into a "1k ISL" cell and
   corrupt both the length axis and the acceptance number (placeholder text is
   trivially draftable). We drop them and assert none survive.
2. **The `mixed` entropy tier is 100% masked** in every throughput split, so it
   cannot be used from the public release at all. Phase D reports `low_entropy`
   and `high_entropy` only, and says so.

Output: one JSONL per (bucket, tier) in aiperf single_turn form ({"text": ...}),
plus a manifest carrying sha256 + measured M3 token stats per file so the arm
launcher can gate on it.

    python pipeline/stage_speedbench.py --out artifacts/aiperf-datasets/speedbench
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import statistics as st

MASK = "FULL BENCHMARK DATA SHOULD BE FETCHED"
DATASET = "nvidia/SPEED-Bench"
# `mixed` is deliberately absent: 512/512 masked in every throughput split.
TIERS = ["low_entropy", "high_entropy"]
BUCKETS = ["throughput_1k", "throughput_8k", "throughput_32k"]
TOKENIZER = "/mnt/nfs/hoangduy/hf_assets/MiniMaxAI/MiniMax-M3"


def sha256(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="artifacts/aiperf-datasets/speedbench")
    ap.add_argument("--max-per-cell", type=int, default=256,
                    help="cap entries written per (bucket,tier); cells need <=100")
    ap.add_argument("--tokenizer", default=TOKENIZER)
    args = ap.parse_args()

    from datasets import load_dataset
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(args.tokenizer, trust_remote_code=True)
    os.makedirs(args.out, exist_ok=True)
    manifest = {"dataset": DATASET, "dropped_tier": "mixed",
                "mask_sentinel": MASK, "files": {}}

    for bucket in BUCKETS:
        ds = load_dataset(DATASET, name=bucket, split="test")
        by_tier: dict[str, list[str]] = {t: [] for t in TIERS}
        masked = {t: 0 for t in TIERS}
        for row in ds:
            tier = row.get("category")
            if tier not in by_tier:
                continue
            turns = row.get("turns") or []
            text = turns[0] if turns else ""
            if not text or MASK in text:
                masked[tier] += 1
                continue
            by_tier[tier].append(text)

        for tier, texts in by_tier.items():
            keep = texts[: args.max_per_cell]
            name = f"{bucket.replace('throughput_', '')}-{tier.replace('_entropy', '')}.jsonl"
            path = os.path.join(args.out, name)
            with open(path, "w") as fh:
                for text in keep:
                    fh.write(json.dumps({"text": text}) + "\n")
            counts = [len(tok(t, add_special_tokens=False)["input_ids"]) for t in keep]
            manifest["files"][name] = {
                "bucket": bucket, "tier": tier,
                "entries": len(keep), "clean_available": len(texts),
                "masked_dropped": masked[tier],
                "tokens_mean": round(st.mean(counts), 1),
                "tokens_median": st.median(counts),
                "tokens_min": min(counts), "tokens_max": max(counts),
                "sha256": sha256(path),
            }
            print(f"{name:22s} n={len(keep):4d} (clean {len(texts)}, dropped {masked[tier]}) "
                  f"tok mean={st.mean(counts):.0f} med={st.median(counts)} "
                  f"min={min(counts)} max={max(counts)}")

    mpath = os.path.join(args.out, "manifest.json")
    with open(mpath, "w") as fh:
        json.dump(manifest, fh, indent=2)
    print(f"\nmanifest: {mpath}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
