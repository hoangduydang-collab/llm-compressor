"""Generation-health diagnostics for quantized model evaluation."""

from __future__ import annotations

import json
import math
from collections.abc import Sequence
from pathlib import Path
from typing import Any


def unwrap_singleton(value: Any) -> Any:
    """Unwrap backend singleton containers without flattening structured data."""
    while isinstance(value, (list, tuple)) and len(value) == 1:
        value = value[0]
    return value


def _periodic_suffix(token_ids: Sequence[int]) -> tuple[bool, int | None, int]:
    values = list(token_ids)
    for period in range(1, min(16, len(values)) + 1):
        pattern = values[-period:]
        repeats = 1
        cursor = len(values) - period
        while cursor >= period and values[cursor - period : cursor] == pattern:
            repeats += 1
            cursor -= period
        repeated_tokens = repeats * period
        if repeats >= 4 and repeated_tokens >= 16:
            return True, period, repeated_tokens
    return False, None, 0


def _repeated_ngram_fraction(token_ids: Sequence[int], size: int) -> float | None:
    if len(token_ids) < size:
        return None
    ngrams = [
        tuple(token_ids[index : index + size])
        for index in range(len(token_ids) - size + 1)
    ]
    return 1.0 - len(set(ngrams)) / len(ngrams)


def analyze_generation(
    text: str | None,
    *,
    token_ids: Sequence[int] | None,
    max_gen_toks: int | None,
    extracted_answer: object,
) -> dict[str, Any]:
    missing = text is None
    empty = missing or not text.strip()
    tokens = list(token_ids) if token_ids is not None else None
    periodic_loop, loop_period, repeated_tokens = (
        _periodic_suffix(tokens) if tokens is not None else (False, None, 0)
    )
    token_count = len(tokens) if tokens is not None else None
    cap_hit = (
        token_count is not None
        and max_gen_toks is not None
        and token_count >= max_gen_toks
    )
    return {
        "applicable": True,
        "missing": missing,
        "empty": empty,
        "answer_extraction_failed": bool(not empty and extracted_answer is None),
        "length_cap_hit": cap_hit,
        "periodic_loop": periodic_loop,
        "loop_period": loop_period,
        "loop_repeated_tokens": repeated_tokens,
        "repeated_3gram_fraction": (
            _repeated_ngram_fraction(tokens, 3) if tokens is not None else None
        ),
        "repeated_4gram_fraction": (
            _repeated_ngram_fraction(tokens, 4) if tokens is not None else None
        ),
        "token_count": token_count,
        "token_count_source": "backend" if tokens is not None else "unavailable",
        "max_gen_toks": max_gen_toks,
    }


def enrich_generation_rows(
    rows: Sequence[dict],
    *,
    encode,
    max_gen_toks: int | None,
) -> list[dict]:
    """Re-encode textual responses when the backend omitted output token IDs."""
    enriched: list[dict] = []
    for original in rows:
        row = dict(original)
        health = dict(row.get("health") or {})
        response = unwrap_singleton(row.get("response"))
        if isinstance(response, str):
            row["response"] = response
        if isinstance(response, str) and health.get("token_count") is None:
            token_ids = [int(token_id) for token_id in encode(response)]
            extracted_answer = None if health.get("answer_extraction_failed") else True
            health = analyze_generation(
                response,
                token_ids=token_ids,
                max_gen_toks=max_gen_toks,
                extracted_answer=extracted_answer,
            )
            health["token_count_source"] = "tokenizer_reencode"
            row["response_token_ids"] = token_ids
            row["health"] = health
        enriched.append(row)
    return enriched


def enrich_samples_file(
    path: str | Path,
    *,
    encode,
    max_gen_toks: int | None,
) -> dict[str, Any]:
    path = Path(path)
    rows = [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").split("\n")
        if line.strip()
    ]
    enriched = enrich_generation_rows(
        rows,
        encode=encode,
        max_gen_toks=max_gen_toks,
    )
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        for row in enriched:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    temporary.replace(path)

    summary = summarize_generation_health(enriched)
    summary_path = path.parent.parent / "generation_health" / f"{path.stem}.json"
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_temporary = summary_path.with_suffix(summary_path.suffix + ".tmp")
    summary_temporary.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    summary_temporary.replace(summary_path)
    return summary


def _quantile(sorted_values: list[int], fraction: float) -> float | None:
    if not sorted_values:
        return None
    position = (len(sorted_values) - 1) * fraction
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return float(sorted_values[lower])
    weight = position - lower
    return sorted_values[lower] * (1.0 - weight) + sorted_values[upper] * weight


def _length_summary(values: list[int]) -> dict[str, float | int | None]:
    ordered = sorted(values)
    return {
        "count": len(ordered),
        "min": ordered[0] if ordered else None,
        "median": _quantile(ordered, 0.5),
        "p95": _quantile(ordered, 0.95),
        "max": ordered[-1] if ordered else None,
    }


def summarize_generation_health(rows: Sequence[dict]) -> dict[str, Any]:
    applicable_rows = [
        row for row in rows if (row.get("health") or {}).get("applicable", True)
    ]
    count = len(applicable_rows)
    health_rows = [row.get("health") or {} for row in applicable_rows]

    def total(field: str) -> int:
        return sum(bool(health.get(field)) for health in health_rows)

    token_counts = [
        int(health["token_count"])
        for health in health_rows
        if health.get("token_count") is not None
    ]
    nonfinite = sum(
        isinstance(row.get("metric_value"), (int, float))
        and not math.isfinite(float(row["metric_value"]))
        for row in applicable_rows
    )
    result: dict[str, Any] = {
        "samples": count,
        "not_applicable_count": len(rows) - count,
        "missing_count": total("missing"),
        "empty_count": total("empty"),
        "answer_extraction_failure_count": total("answer_extraction_failed"),
        "length_cap_hit_count": total("length_cap_hit"),
        "periodic_loop_count": total("periodic_loop"),
        "nonfinite_metric_count": nonfinite,
        "reasoning_failure_count": sum(
            bool(
                health.get("answer_extraction_failed")
                or health.get("length_cap_hit")
                or health.get("empty")
                or health.get("periodic_loop")
            )
            for health in health_rows
        ),
        "output_tokens": _length_summary(token_counts),
    }
    for field in (
        "missing",
        "empty",
        "answer_extraction_failure",
        "length_cap_hit",
        "periodic_loop",
        "nonfinite_metric",
        "reasoning_failure",
    ):
        result[f"{field}_rate"] = result[f"{field}_count"] / count if count else None
    return result
