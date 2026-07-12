"""Pure normalization and comparison for teacher-forced prompt logprobs."""

from __future__ import annotations

import math
from collections import defaultdict
from collections.abc import Iterable
from typing import Any


def _logprob_value(value: object) -> tuple[float, int | None]:
    if isinstance(value, (int, float)):
        return float(value), None
    if isinstance(value, dict):
        logprob = value.get("logprob")
        rank = value.get("rank")
    else:
        logprob = getattr(value, "logprob", None)
        rank = getattr(value, "rank", None)
    if not isinstance(logprob, (int, float)):
        raise ValueError(f"invalid prompt logprob entry: {value!r}")
    return float(logprob), int(rank) if isinstance(rank, int) else None


def normalize_prompt_logprobs(output, prompt_meta: dict) -> list[dict[str, Any]]:
    token_ids = list(output.prompt_token_ids)
    positions = list(output.prompt_logprobs)
    if len(token_ids) != len(positions):
        raise ValueError("prompt token/logprob lengths differ")

    rows: list[dict[str, Any]] = []
    for position, entries in enumerate(positions):
        if position == 0 or entries is None:
            continue
        normalized: list[dict[str, Any]] = []
        observed_logprob = None
        for token_id, value in entries.items():
            logprob, rank = _logprob_value(value)
            token_id = int(token_id)
            if token_id == int(token_ids[position]):
                observed_logprob = logprob
            normalized.append(
                {"token_id": token_id, "logprob": logprob, "rank": rank}
            )
        normalized.sort(
            key=lambda item: (
                item["rank"] if item["rank"] is not None else 10**9,
                -item["logprob"],
                item["token_id"],
            )
        )
        for rank, item in enumerate(normalized, start=1):
            if item["rank"] is None:
                item["rank"] = rank
        if observed_logprob is None:
            raise ValueError(
                f"observed token {token_ids[position]} absent at position {position}"
            )
        rows.append(
            {
                "schema_version": 1,
                "corpus_sha256": prompt_meta["corpus_sha256"],
                "prompt_id": prompt_meta["prompt_id"],
                "length_bucket": prompt_meta["length_bucket"],
                "prompt_token_count": len(token_ids),
                "position": position,
                "observed_token_id": int(token_ids[position]),
                "observed_logprob": observed_logprob,
                "top_logprobs": normalized,
            }
        )
    return rows


def _record_key(record: dict) -> tuple[str, int]:
    return str(record["prompt_id"]), int(record["position"])


def _index_records(records: Iterable[dict]) -> dict[tuple[str, int], dict]:
    indexed: dict[tuple[str, int], dict] = {}
    for record in records:
        key = _record_key(record)
        if key in indexed:
            raise ValueError(f"duplicate distributional record: {key}")
        indexed[key] = record
    return indexed


def _quantile(values: list[float], fraction: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    position = (len(ordered) - 1) * fraction
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    weight = position - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def _top_ids(record: dict, limit: int) -> list[int]:
    ordered = sorted(
        record.get("top_logprobs") or [],
        key=lambda item: (int(item.get("rank", 10**9)), -float(item["logprob"])),
    )
    return [int(item["token_id"]) for item in ordered[:limit]]


def _jaccard(left: list[int], right: list[int]) -> float:
    union = set(left) | set(right)
    return len(set(left) & set(right)) / len(union) if union else 1.0


def _compare_pairs(pairs: list[tuple[dict, dict]]) -> dict[str, Any]:
    for reference, candidate in pairs:
        for record in (reference, candidate):
            values = [float(record["observed_logprob"])] + [
                float(item["logprob"])
                for item in record.get("top_logprobs") or []
            ]
            if not all(math.isfinite(value) for value in values):
                raise ValueError("distributional record contains non-finite logprobs")
    ref_lps = [float(reference["observed_logprob"]) for reference, _ in pairs]
    cand_lps = [float(candidate["observed_logprob"]) for _, candidate in pairs]
    drifts = [candidate - reference for reference, candidate in zip(ref_lps, cand_lps, strict=True)]
    reference_nll = -sum(ref_lps) / len(ref_lps)
    candidate_nll = -sum(cand_lps) / len(cand_lps)
    top1 = [
        _top_ids(reference, 1) == _top_ids(candidate, 1)
        for reference, candidate in pairs
    ]
    top5 = [
        _jaccard(_top_ids(reference, 5), _top_ids(candidate, 5))
        for reference, candidate in pairs
    ]
    top20 = [
        _jaccard(_top_ids(reference, 20), _top_ids(candidate, 20))
        for reference, candidate in pairs
    ]
    retained = [
        bool(_top_ids(reference, 1))
        and _top_ids(reference, 1)[0] in set(_top_ids(candidate, 20))
        for reference, candidate in pairs
    ]
    return {
        "paired_tokens": len(pairs),
        "reference_nll": reference_nll,
        "candidate_nll": candidate_nll,
        "reference_perplexity": math.exp(reference_nll),
        "candidate_perplexity": math.exp(candidate_nll),
        "perplexity_ratio": math.exp(candidate_nll - reference_nll),
        "bits_per_token_increase": (candidate_nll - reference_nll) / math.log(2),
        "mean_observed_logprob_drift": sum(drifts) / len(drifts),
        "median_observed_logprob_drift": _quantile(drifts, 0.5),
        "p95_abs_observed_logprob_drift": _quantile([abs(value) for value in drifts], 0.95),
        "p99_abs_observed_logprob_drift": _quantile([abs(value) for value in drifts], 0.99),
        "top1_agreement": sum(top1) / len(top1),
        "top5_jaccard": sum(top5) / len(top5),
        "top20_jaccard": sum(top20) / len(top20),
        "bf16_top1_retained_top20": sum(retained) / len(retained),
    }


def _position_quartile(record: dict) -> str:
    count = max(1, int(record.get("prompt_token_count", 1)))
    fraction = int(record["position"]) / count
    return f"q{min(4, int(fraction * 4) + 1)}"


def compare_distributional_records(
    reference_records: Iterable[dict],
    candidate_records: Iterable[dict],
) -> dict[str, Any]:
    reference = _index_records(reference_records)
    candidate = _index_records(candidate_records)
    if set(reference) != set(candidate):
        raise ValueError(
            "distributional coverage mismatch: "
            f"reference={len(reference)} candidate={len(candidate)}"
        )
    if not reference:
        raise ValueError("distributional comparison requires paired records")

    pairs: list[tuple[dict, dict]] = []
    for key in sorted(reference):
        ref = reference[key]
        cand = candidate[key]
        if ref.get("corpus_sha256") != cand.get("corpus_sha256"):
            raise ValueError(f"corpus provenance mismatch at {key}")
        if int(ref["observed_token_id"]) != int(cand["observed_token_id"]):
            raise ValueError(f"observed token identity mismatch at {key}")
        pairs.append((ref, cand))

    result = _compare_pairs(pairs)
    by_bucket: dict[str, list[tuple[dict, dict]]] = defaultdict(list)
    by_quartile: dict[str, list[tuple[dict, dict]]] = defaultdict(list)
    for pair in pairs:
        by_bucket[str(pair[0]["length_bucket"])].append(pair)
        by_quartile[_position_quartile(pair[0])].append(pair)
    result["by_length_bucket"] = {
        name: _compare_pairs(values) for name, values in sorted(by_bucket.items())
    }
    result["by_position_quartile"] = {
        name: _compare_pairs(values) for name, values in sorted(by_quartile.items())
    }
    result["corpus_sha256"] = next(iter(reference.values()))["corpus_sha256"]
    return result
