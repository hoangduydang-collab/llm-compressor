"""Build an immutable, calibration-disjoint teacher-forced probe corpus."""

from __future__ import annotations

import random
from collections.abc import Iterable
from typing import Any


# Bounded to 49,152 tokens/model so the probe stays secondary to benchmark eval.
DEFAULT_BUCKETS = {
    "short": (4, 2_048),
    "8k": (1, 8_192),
    "32k": (1, 32_768),
}


def build_probe_corpus(
    texts: Iterable[str],
    tokenizer,
    *,
    seed: int,
    buckets: dict[str, tuple[int, int]] | None = None,
) -> list[dict[str, Any]]:
    bucket_spec = buckets or DEFAULT_BUCKETS
    tokens: list[int] = []
    for text in texts:
        if not isinstance(text, str) or not text.strip():
            continue
        tokens.extend(tokenizer.encode(text, add_special_tokens=False))

    windows = [
        (bucket, length)
        for bucket, (count, length) in bucket_spec.items()
        for _ in range(count)
    ]
    required = sum(length for _, length in windows)
    if len(tokens) < required:
        raise ValueError(
            f"distributional probe requires {required} tokens, got {len(tokens)}"
        )

    random.Random(seed).shuffle(windows)
    counters = {bucket: 0 for bucket in bucket_spec}
    rows: list[dict[str, Any]] = []
    cursor = 0
    for bucket, length in windows:
        index = counters[bucket]
        counters[bucket] += 1
        end = cursor + length
        rows.append(
            {
                "schema_version": 1,
                "prompt_id": f"{bucket}-{index}",
                "length_bucket": bucket,
                "start_token": cursor,
                "end_token": end,
                "prompt_token_ids": [int(token) for token in tokens[cursor:end]],
            }
        )
        cursor = end
    return sorted(rows, key=lambda row: row["prompt_id"])
