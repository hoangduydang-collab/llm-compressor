import json
from pathlib import Path

import pytest

from pipeline import aa_lcr_v11 as A


def checkpoint_with_candidates_and_judgments(
    path: Path, *, count: int, correct: int = 0
) -> A.Checkpoint:
    checkpoint = A.Checkpoint(
        path / "run.sqlite",
        A.RunContract(
            dataset_revision="revision",
            dataset_file_sha256={"dataset.csv": "a" * 64},
            prompt_sha256="b" * 64,
            judge_system_prompt_sha256=A.sha256_text(A.JUDGE_SYSTEM_PROMPT),
            judge_user_prompt_sha256=A.sha256_text(A.JUDGE_USER_PROMPT_TEMPLATE),
            candidate_model="candidate-model",
            served_model="served-model",
            endpoint_deployment_identity={"deployment": "candidate-v1"},
            candidate_temperature=0.6,
            candidate_top_p=1.0,
            candidate_max_tokens=131_072,
            candidate_concurrency=2,
            candidate_reasoning_enabled=True,
            candidate_max_attempts=30,
            judge_model=A.JUDGE_MODEL,
            judge_reasoning_effort=A.JUDGE_REASONING_EFFORT,
            judge_max_attempts=A.MAX_ATTEMPTS,
            repeats=3,
            code_revision="test",
            question_ids=tuple(range(1, 101)),
        ),
    )
    judge_contract_hash = A.validate_judge_contract(checkpoint.contract)
    for unit in range(count):
        question_id = unit // 3 + 1
        repeat_index = unit % 3
        checkpoint.record_candidate(
            A.CandidateRecord(
                question_id=question_id,
                repeat_index=repeat_index,
                http_status=200,
                raw_response={"usage": {"prompt_tokens": 10, "completion_tokens": 2}},
                usage={"prompt_tokens": 10, "completion_tokens": 2},
                finish_reason="length" if unit == 0 else "stop",
                content="" if unit == 1 else f"answer {unit}",
                reasoning_content=f"reasoning {unit}",
                retry_count=1 if unit == 2 else 0,
                started_at_utc="2026-09-13T00:00:00Z",
                completed_at_utc="2026-09-13T00:00:01Z",
                error=None,
            )
        )
        checkpoint.record_judgment(
            A.JudgmentRecord(
                question_id=question_id,
                repeat_index=repeat_index,
                judge_contract_hash=judge_contract_hash,
                raw_response={"verdict": "CORRECT"},
                verdict="CORRECT" if unit < correct else "INCORRECT",
                retry_count=0,
                started_at_utc="2026-09-13T00:00:02Z",
                completed_at_utc="2026-09-13T00:00:03Z",
                error=None,
            )
        )
    return checkpoint


def complete_checkpoint(path: Path, *, correct: int) -> A.Checkpoint:
    return checkpoint_with_candidates_and_judgments(path, count=300, correct=correct)


def test_incomplete_population_has_no_headline(tmp_path):
    checkpoint = checkpoint_with_candidates_and_judgments(tmp_path, count=299)

    with pytest.raises(A.IncompleteRunError, match="expected 300"):
        A.build_summary(checkpoint)


def test_terminal_judge_failure_has_no_headline(tmp_path):
    checkpoint = complete_checkpoint(tmp_path, correct=225)
    checkpoint.record_judge_failure(
        A.JudgeFailureRecord(
            question_id=1,
            repeat_index=0,
            judge_contract_hash=A.validate_judge_contract(checkpoint.contract),
            requested_model=A.JUDGE_MODEL,
            reasoning_effort=A.JUDGE_REASONING_EFFORT,
            reasoning_mode=A.JUDGE_REASONING_MODE,
            openai_sdk_version=None,
            exception_class="TimeoutError",
            http_status=None,
            request_id=None,
            retry_count=29,
            attempt_count=30,
            started_at_utc="2026-09-13T00:00:02Z",
            completed_at_utc="2026-09-13T00:00:03Z",
            raw_output_text=None,
            returned_model=None,
            response_id=None,
            usage=None,
        )
    )

    with pytest.raises(A.IncompleteRunError, match="judge failures"):
        A.build_summary(checkpoint)


def test_candidate_without_final_content_has_no_headline(tmp_path):
    checkpoint = complete_checkpoint(tmp_path, correct=225)
    row = checkpoint._connection.execute(
        "SELECT record_json FROM candidates WHERE question_id = 1 AND repeat_index = 0"
    ).fetchone()
    assert row is not None
    record = json.loads(row[0])
    record["content"] = None
    checkpoint._connection.execute(
        "UPDATE candidates SET record_json = ? WHERE question_id = 1 AND repeat_index = 0",
        (A.canonical_json(record),),
    )
    checkpoint._connection.commit()

    with pytest.raises(A.IncompleteRunError, match="candidate failures"):
        A.build_summary(checkpoint)


def test_summary_recomputes_pass_at_one_and_diagnostics(tmp_path):
    checkpoint = complete_checkpoint(tmp_path, correct=225)

    summary = A.build_summary(checkpoint)

    assert summary["headline"] == {
        "correct": 225,
        "denominator": 300,
        "pass_at_1": 0.75,
    }
    assert summary["benchmark_claim"] == (
        "AA-LCR v1.1 public-methodology reproduction"
    )
    assert summary["finish_reasons"] == {"length": 1, "stop": 299}
    assert summary["candidate_diagnostics"] == {
        "truncation_count": 1,
        "empty_answer_count": 1,
        "retry_count": 1,
        "persistent_failure_count": 0,
    }


def test_summary_uses_checkpointed_document_categories(tmp_path):
    checkpoint = complete_checkpoint(tmp_path, correct=150)

    checkpoint.record_question_categories(
        {
            question_id: "even" if question_id % 2 == 0 else "odd"
            for question_id in checkpoint.contract.question_ids
        }
    )

    summary = A.build_summary(checkpoint)

    assert summary["document_categories"] == {
        "even": {"count": 150, "correct": 75, "accuracy": 0.5},
        "odd": {"count": 150, "correct": 75, "accuracy": 0.5},
    }


def test_publish_is_atomic_includes_digests_and_no_clobber(tmp_path):
    checkpoint = complete_checkpoint(tmp_path, correct=225)
    destination = tmp_path / "published"

    exports = A.publish_results(checkpoint, destination)

    assert set(exports) == {
        "run-manifest.json",
        "candidates.jsonl",
        "judgments.jsonl",
        "summary.json",
        "report.md",
        "files.sha256",
    }
    assert all(path.is_file() for path in exports.values())
    manifest = (destination / "files.sha256").read_text(encoding="utf-8")
    assert {line.split("  ", 1)[1] for line in manifest.splitlines()} == (
        set(exports) - {"files.sha256"}
    )
    with pytest.raises(FileExistsError):
        A.publish_results(checkpoint, destination)
