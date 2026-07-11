"""CPU-only tests for MiniMax-M3 smoke quality and evidence decisions."""

from __future__ import annotations

import json
from pathlib import Path
from tempfile import TemporaryDirectory

from pipeline.m3_quality_evidence import (
    M3_QUALITY_CASES,
    assess_output,
    assess_quality_outputs,
    bundle_run,
    extract_log_evidence,
    classify_pair,
    main,
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


def test_assess_output_rejects_repeated_digit_output():
    result = assess_output("444444444444", ("4", "four"))

    assert result["repetitive"] is True
    assert result["passed"] is False


def test_assess_output_does_not_accept_expected_digit_inside_wrong_number():
    result = assess_output("14", ("4", "four"))

    assert result["expected_match"] is False
    assert result["passed"] is False


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


def test_extract_log_evidence_preserves_json_and_probe_signals():
    log = """
WARNING M3_PARAM_FINGERPRINT# {"case":"candidate","category":"lm_head","finite_fraction":1.0,"sample_abs_max":2.0}
WARNING M3_PARAM_FINGERPRINT_SUMMARY# {"case":"candidate","found":["lm_head","shared_expert"],"missing":[],"errors":[]}
WARNING M3_LOAD_AUDIT# seen=10 matched=10 unmatched_this_rank=0
WARNING M3_MOE_PROBE#1 tokens=8 in_norm=1.0 shared_present=False shared_norm=-1.0
"""

    evidence = extract_log_evidence(log)

    assert evidence["fingerprints"][0]["category"] == "lm_head"
    assert evidence["fingerprint_summaries"][0]["missing"] == []
    assert evidence["shared_expert_bad"] is True
    assert len(evidence["loader_audit_lines"]) == 1


def test_bundle_copies_provenance_and_indexes_full_logs():
    healthy_log = """
M3_PARAM_FINGERPRINT# {"category":"lm_head","finite_fraction":1.0,"sample_abs_max":2.0}
M3_PARAM_FINGERPRINT# {"category":"shared_expert","finite_fraction":1.0,"sample_abs_max":2.0}
M3_LOAD_AUDIT# seen=2 matched=2 unmatched_this_rank=0
M3_MOE_PROBE#1 shared_present=True shared_norm=2.0
"""
    with TemporaryDirectory() as raw_tmp:
        root = Path(raw_tmp)
        run_dir = root / "run"
        evidence_dir = root / "evidence"
        run_dir.mkdir()
        (run_dir / "run_manifest.json").write_text(
            json.dumps({"dry_run": False}),
            encoding="utf-8",
        )
        (run_dir / "software_versions.txt").write_text(
            "vllm 0.24.0\n",
            encoding="utf-8",
        )
        for case_name, quality_ok in (
            ("cyankiwi_reference", True),
            ("portable_awq_w4a8", False),
        ):
            case_dir = run_dir / case_name
            case_dir.mkdir()
            (case_dir / "serve_report.json").write_text(
                json.dumps({"quality_ok": quality_ok}),
                encoding="utf-8",
            )
            (case_dir / "serve.log").write_text(healthy_log, encoding="utf-8")

        comparison = bundle_run(run_dir, evidence_dir)

        assert comparison["verdict"] == "attention_indexer_boundary"
        assert (evidence_dir / "software_versions.txt").read_text() == "vllm 0.24.0\n"
        artifacts = json.loads(
            (evidence_dir / "artifact_index.json").read_text(encoding="utf-8")
        )
        assert {item["case"] for item in artifacts} == {
            "cyankiwi_reference",
            "portable_awq_w4a8",
        }
        assert all(item["sha256"] for item in artifacts)


def test_bundle_cli_uses_manifest_evidence_dir_when_omitted():
    with TemporaryDirectory() as raw_tmp:
        root = Path(raw_tmp)
        run_dir = root / "run"
        evidence_dir = root / "evidence"
        run_dir.mkdir()
        (run_dir / "run_manifest.json").write_text(
            json.dumps(
                {
                    "dry_run": True,
                    "evidence_dir": str(evidence_dir),
                }
            ),
            encoding="utf-8",
        )

        assert main(["bundle", "--run-dir", str(run_dir)]) == 0
        assert json.loads(
            (evidence_dir / "comparison.json").read_text(encoding="utf-8")
        )["verdict"] == "dry_run"
