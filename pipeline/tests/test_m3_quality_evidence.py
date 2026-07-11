"""CPU-only tests for MiniMax-M3 smoke quality and evidence decisions."""

from __future__ import annotations

from pipeline.m3_quality_evidence import (
    M3_QUALITY_CASES,
    assess_output,
    assess_quality_outputs,
    classify_pair,
)


def test_assess_output_accepts_factual_answer():
    result = assess_output(" Paris.", ("paris",))

    assert result["passed"] is True
    assert result["expected_match"] is True
    assert result["repetitive"] is False


def test_assess_output_rejects_concatenated_repetition():
    result = assess_output("arring" * 20, ("paris",))

    assert result["passed"] is False
    assert result["repetitive"] is True
    assert "character_chunk" in result["repetition_reasons"]


def test_assess_output_rejects_token_repetition():
    result = assess_output("seringk " * 20, ("paris",))

    assert result["passed"] is False
    assert result["repetitive"] is True
    assert "dominant_token" in result["repetition_reasons"]


def test_quality_suite_preserves_raw_outputs():
    outputs = [" Paris.", "4"]

    result = assess_quality_outputs(outputs)

    assert result["quality_ok"] is True
    assert [case["text"] for case in result["quality_cases"]] == outputs
    assert len(result["quality_cases"]) == len(M3_QUALITY_CASES)


def test_quality_suite_requires_one_output_per_case():
    result = assess_quality_outputs(["Paris"])

    assert result["quality_ok"] is False
    assert result["complete"] is False
    assert "output_count" in result["errors"][0]


def test_reference_failure_stops_pair():
    result = classify_pair({"quality_ok": False}, {}, {})

    assert result["verdict"] == "invalid_reference"


def test_candidate_quality_pass_wins_after_valid_reference():
    result = classify_pair(
        {"quality_ok": True},
        {"quality_ok": True},
        {"required_complete": True},
    )

    assert result["verdict"] == "candidate_quality_pass"


def test_missing_required_evidence_is_inconclusive():
    result = classify_pair(
        {"quality_ok": True},
        {"quality_ok": False},
        {"required_complete": False, "missing": ["candidate.parameter_fingerprints"]},
    )

    assert result["verdict"] == "inconclusive_missing_evidence"
    assert result["missing"] == ["candidate.parameter_fingerprints"]


def test_lm_head_boundary_precedes_shared_expert_boundary():
    result = classify_pair(
        {"quality_ok": True},
        {"quality_ok": False},
        {
            "required_complete": True,
            "lm_head_bad": True,
            "shared_expert_bad": True,
        },
    )

    assert result["verdict"] == "lm_head_boundary"


def test_shared_expert_boundary_selected_when_lm_head_is_clean():
    result = classify_pair(
        {"quality_ok": True},
        {"quality_ok": False},
        {
            "required_complete": True,
            "lm_head_bad": False,
            "shared_expert_bad": True,
        },
    )

    assert result["verdict"] == "shared_expert_boundary"


def test_clean_primary_boundaries_select_attention_indexer():
    result = classify_pair(
        {"quality_ok": True},
        {"quality_ok": False},
        {
            "required_complete": True,
            "lm_head_bad": False,
            "shared_expert_bad": False,
        },
    )

    assert result["verdict"] == "attention_indexer_boundary"
