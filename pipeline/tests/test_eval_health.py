"""Tests for generation-health and degeneration diagnostics."""

from __future__ import annotations

import math

import pytest

from pipeline.evalsuite.health import (
    analyze_generation,
    enrich_generation_rows,
    enrich_samples_file,
    summarize_generation_health,
)


@pytest.mark.parametrize(
    "token_ids,period",
    [
        ([1, 2] * 16, 2),
        ([4, 5, 6] * 12, 3),
        ([9] * 20, 1),
        (list(range(8)) * 8, 8),
    ],
)
def test_detects_periodic_token_loop_at_length_cap(token_ids, period):
    result = analyze_generation(
        "looping output",
        token_ids=token_ids,
        max_gen_toks=len(token_ids),
        extracted_answer=None,
    )

    assert result["periodic_loop"] is True
    assert result["loop_period"] == period
    assert result["length_cap_hit"] is True
    assert result["token_count"] == len(token_ids)


def test_short_healthy_answer_is_not_a_loop():
    result = analyze_generation(
        "Paris.",
        token_ids=[10, 11],
        max_gen_toks=64,
        extracted_answer="Paris",
    )

    assert result["missing"] is False
    assert result["empty"] is False
    assert result["periodic_loop"] is False
    assert result["length_cap_hit"] is False
    assert result["answer_extraction_failed"] is False


def test_missing_and_empty_responses_are_distinct():
    missing = analyze_generation(
        None, token_ids=None, max_gen_toks=64, extracted_answer=None
    )
    empty = analyze_generation(
        "   ", token_ids=[], max_gen_toks=64, extracted_answer=None
    )
    assert missing["missing"] is True
    assert missing["empty"] is True
    assert empty["missing"] is False
    assert empty["empty"] is True


def test_summary_counts_health_and_nonfinite_metrics():
    rows = [
        {
            "metric_value": 1.0,
            "health": analyze_generation(
                "ok", token_ids=[1, 2], max_gen_toks=8, extracted_answer="ok"
            ),
        },
        {
            "metric_value": math.nan,
            "health": analyze_generation(
                "bad", token_ids=[3, 4] * 8, max_gen_toks=16, extracted_answer=None
            ),
        },
    ]

    result = summarize_generation_health(rows)

    assert result["samples"] == 2
    assert result["periodic_loop_count"] == 1
    assert result["length_cap_hit_count"] == 1
    assert result["answer_extraction_failure_count"] == 1
    assert result["reasoning_failure_count"] == 1
    assert result["reasoning_failure_rate"] == 0.5
    assert result["nonfinite_metric_count"] == 1
    assert result["output_tokens"]["max"] == 16


def test_reasoning_failure_is_row_union_not_sum_of_failure_flags():
    rows = [
        {
            "health": {
                "applicable": True,
                "empty": True,
                "answer_extraction_failed": True,
                "length_cap_hit": True,
                "periodic_loop": True,
            }
        },
        {"health": {"applicable": True}},
    ]

    result = summarize_generation_health(rows)

    assert result["reasoning_failure_count"] == 1
    assert result["reasoning_failure_rate"] == 0.5


def test_enrich_generation_rows_reencodes_missing_token_ids():
    rows = [
        {
            "response": "loop",
            "health": analyze_generation(
                "loop", token_ids=None, max_gen_toks=None, extracted_answer=None
            ),
        }
    ]

    enriched = enrich_generation_rows(
        rows,
        encode=lambda _: [1, 2] * 8,
        max_gen_toks=16,
    )

    assert enriched[0]["response_token_ids"] == [1, 2] * 8
    assert enriched[0]["health"]["token_count_source"] == "tokenizer_reencode"
    assert enriched[0]["health"]["periodic_loop"] is True
    assert enriched[0]["health"]["length_cap_hit"] is True


def test_enrich_samples_file_rewrites_rows_and_health_summary(tmp_path):
    import json

    samples = tmp_path / "samples" / "gsm8k.jsonl"
    samples.parent.mkdir()
    samples.write_text(
        json.dumps(
            {
                "sample_uid": "x",
                "response": "loop",
                "metric_value": 0.0,
                "health": {"applicable": True, "token_count": None},
            }
        )
        + "\n",
        encoding="utf-8",
    )

    summary = enrich_samples_file(
        samples,
        encode=lambda _: [1, 2] * 8,
        max_gen_toks=16,
    )

    row = json.loads(samples.read_text(encoding="utf-8"))
    assert row["health"]["periodic_loop"] is True
    assert summary["periodic_loop_count"] == 1
    summary_path = tmp_path / "generation_health" / "gsm8k.json"
    assert json.loads(summary_path.read_text()) == summary


def test_enrich_generation_rows_unwraps_nested_text_but_not_likelihood():
    rows = [
        {"response": ["loop"], "health": {"applicable": False}},
        {"response": [[-1.2, False]], "health": {"applicable": False}},
    ]

    enriched = enrich_generation_rows(
        rows,
        encode=lambda _: [1, 2] * 8,
        max_gen_toks=16,
    )

    assert enriched[0]["response"] == "loop"
    assert enriched[0]["health"]["applicable"] is True
    assert enriched[0]["health"]["periodic_loop"] is True
    assert enriched[1] == rows[1]
