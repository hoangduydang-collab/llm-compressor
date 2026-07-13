"""CPU-only tests for canonical MiniMax-M3 chat matrix evidence."""

from __future__ import annotations

import json
from pathlib import Path
from tempfile import TemporaryDirectory

from pipeline.m3_chat_quality import (
    EXPECTED_ARMS,
    bundle_arm,
    classify_matrix,
    normalize_http_responses,
)


def _http(text: str, finish_reason: str = "stop") -> dict:
    return {
        "id": "chatcmpl-test",
        "choices": [
            {
                "finish_reason": finish_reason,
                "message": {"role": "assistant", "content": text},
            }
        ],
        "usage": {"completion_tokens": 1},
    }


def _arm(role: str, interface: str, quality_ok: bool = True) -> dict:
    return {
        "checkpoint_role": role,
        "interface": interface,
        "infrastructure_ok": True,
        "quality_ok": quality_ok,
    }


def test_normalize_http_responses_preserves_raw_and_assesses_quality():
    responses = [_http("Paris"), _http("4")]

    report = normalize_http_responses(responses)

    assert report["infrastructure_ok"] is True
    assert report["quality_ok"] is True
    assert report["prompt_mode"] == "chat_completions"
    assert report["quality_cases"][0]["finish_reason"] == "stop"
    assert report["quality_cases"][0]["raw_response"] == responses[0]


def test_normalize_http_responses_keeps_protocol_error_as_infrastructure_failure():
    report = normalize_http_responses([_http("Paris"), {"error": {"message": "bad"}}])

    assert report["infrastructure_ok"] is False
    assert report["quality_ok"] is False
    assert report["errors"]


def test_complete_matrix_passes_only_when_both_pairs_pass():
    arms = {
        "reference_offline_chat": _arm("reference", "offline"),
        "candidate_offline_chat": _arm("candidate", "offline"),
        "reference_http_chat": _arm("reference", "http"),
        "candidate_http_chat": _arm("candidate", "http"),
    }

    result = classify_matrix(arms)

    assert result["verdict"] == "candidate_quality_pass"
    assert result["complete"] is True


def test_matrix_reports_candidate_quality_failure_on_both_interfaces():
    arms = {
        "reference_offline_chat": _arm("reference", "offline"),
        "candidate_offline_chat": _arm("candidate", "offline", False),
        "reference_http_chat": _arm("reference", "http"),
        "candidate_http_chat": _arm("candidate", "http", False),
    }

    result = classify_matrix(arms)

    assert result["verdict"] == "candidate_quality_fail"
    assert result["failed_arms"] == [
        "candidate_offline_chat",
        "candidate_http_chat",
    ]


def test_matrix_reports_infrastructure_before_semantic_quality():
    arms = {
        "reference_offline_chat": _arm("reference", "offline"),
        "candidate_offline_chat": _arm("candidate", "offline", False),
        "reference_http_chat": _arm("reference", "http"),
        "candidate_http_chat": _arm("candidate", "http", False),
    }
    arms["candidate_http_chat"]["infrastructure_ok"] = False

    result = classify_matrix(arms)

    assert result["verdict"] == "infrastructure_failure"
    assert result["failed_arms"] == ["candidate_http_chat"]


def test_matrix_reports_candidate_interface_disagreement():
    arms = {
        "reference_offline_chat": _arm("reference", "offline"),
        "candidate_offline_chat": _arm("candidate", "offline"),
        "reference_http_chat": _arm("reference", "http"),
        "candidate_http_chat": _arm("candidate", "http", False),
    }

    result = classify_matrix(arms)

    assert result["verdict"] == "candidate_interface_disagreement"
    assert result["failed_arms"] == ["candidate_http_chat"]


def test_matrix_rejects_failed_reference_before_candidate_diagnosis():
    arms = {
        "reference_offline_chat": _arm("reference", "offline", False),
        "candidate_offline_chat": _arm("candidate", "offline", False),
        "reference_http_chat": _arm("reference", "http"),
        "candidate_http_chat": _arm("candidate", "http", False),
    }

    result = classify_matrix(arms)

    assert result["verdict"] == "invalid_reference"
    assert result["failed_arms"] == ["reference_offline_chat"]


def test_matrix_reports_missing_arms_without_guessing():
    result = classify_matrix({"reference_http_chat": _arm("reference", "http")})

    assert result["verdict"] == "inconclusive_missing_arms"
    assert set(result["missing_arms"]) == set(EXPECTED_ARMS) - {"reference_http_chat"}


def test_bundle_http_arm_writes_normalized_report():
    with TemporaryDirectory() as raw_tmp:
        root = Path(raw_tmp)
        run_dir = root / "run"
        evidence_dir = root / "evidence"
        run_dir.mkdir()
        (run_dir / "arm_manifest.json").write_text(
            json.dumps(
                {
                    "arm": "reference_http_chat",
                    "checkpoint_role": "reference",
                    "interface": "http",
                }
            )
        )
        for index, response in enumerate((_http("Paris"), _http("4"))):
            (run_dir / f"http_response_{index}.json").write_text(json.dumps(response))

        report = bundle_arm(run_dir, evidence_dir)

        assert report["quality_ok"] is True
        assert json.loads((evidence_dir / "arm_report.json").read_text())["quality_ok"] is True
