import errno
import json
from dataclasses import replace
from pathlib import Path

import pytest

from pipeline import aa_lcr_v11 as A
from pipeline.tests.test_aa_lcr_v11_candidate import server_identity
from pipeline.tests.test_aa_lcr_v11_checkpoint import preflight_audit


def checkpoint_with_candidates_and_judgments(
    path: Path,
    *,
    count: int,
    correct: int = 0,
    input_token_discrepancies: dict[int, tuple[int, int]] | None = None,
    question_input_tokens: dict[int, tuple[int, int]] | None = None,
) -> A.Checkpoint:
    checkpoint = A.Checkpoint(
        path / "run.sqlite",
        A.RunContract(
            dataset_revision="revision",
            dataset_file_sha256={"dataset.csv": "a" * 64},
            prompt_sha256="b" * 64,
            judge_system_prompt_sha256=A.sha256_text(A.JUDGE_SYSTEM_PROMPT),
            judge_user_prompt_sha256=A.sha256_text(A.JUDGE_USER_PROMPT_TEMPLATE),
            candidate_model=A.CANDIDATE_MODEL,
            served_model=A.CANDIDATE_MODEL,
            endpoint_deployment_identity=server_identity(),
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
            input_token_discrepancies=input_token_discrepancies or {},
            question_input_tokens=question_input_tokens or {},
        ),
    )
    checkpoint.record_preflight_audit(preflight_audit())
    checkpoint.record_server_snapshot("before", server_identity())
    checkpoint.record_server_snapshot("after", server_identity())
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


def test_summary_exposes_pinned_input_token_discrepancies(tmp_path):
    checkpoint = checkpoint_with_candidates_and_judgments(
        tmp_path,
        count=300,
        input_token_discrepancies={5: (113266, 113264)},
        question_input_tokens={5: (113266, 113264)},
    )

    summary = A.build_summary(checkpoint)

    assert summary["token_provenance"]["input_token_discrepancies"] == {
        "5": {"published_input_tokens": 113266, "actual_prompt_tokens": 113264}
    }
    assert summary["token_provenance"]["question_input_tokens"] == {
        "5": {"published_input_tokens": 113266, "actual_prompt_tokens": 113264}
    }


def test_incomplete_population_has_no_headline(tmp_path):
    checkpoint = checkpoint_with_candidates_and_judgments(tmp_path, count=299)

    with pytest.raises(A.IncompleteRunError, match="expected 300"):
        A.build_summary(checkpoint)


def test_missing_preflight_audit_has_no_headline(tmp_path):
    checkpoint = complete_checkpoint(tmp_path, correct=225)
    checkpoint._connection.execute("DELETE FROM judge_preflight")
    checkpoint._connection.commit()

    with pytest.raises(A.IncompleteRunError, match="preflight"):
        A.build_summary(checkpoint)


def test_wrong_candidate_contract_identity_has_no_headline(tmp_path):
    checkpoint = complete_checkpoint(tmp_path, correct=225)
    checkpoint.contract = replace(
        checkpoint.contract,
        candidate_model="glm-5.3-w4afp8-shadow",
    )

    with pytest.raises(A.IncompleteRunError, match="candidate identity"):
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


def test_alternate_judge_hash_failure_has_no_headline(tmp_path):
    checkpoint = complete_checkpoint(tmp_path, correct=225)
    checkpoint.record_judge_failure(
        A.JudgeFailureRecord(
            question_id=1,
            repeat_index=0,
            judge_contract_hash="a" * 64,
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


def test_alternate_judge_hash_judgment_has_no_headline(tmp_path):
    checkpoint = complete_checkpoint(tmp_path, correct=225)
    checkpoint.record_judgment(
        A.JudgmentRecord(
            question_id=1,
            repeat_index=0,
            judge_contract_hash="a" * 64,
            raw_response={"verdict": "CORRECT"},
            verdict="CORRECT",
            retry_count=0,
            started_at_utc="2026-09-13T00:00:02Z",
            completed_at_utc="2026-09-13T00:00:03Z",
            error=None,
        )
    )

    with pytest.raises(A.IncompleteRunError, match="unexpected judge hash"):
        A.build_summary(checkpoint)


@pytest.mark.parametrize("table", ["candidates", "judgments"])
def test_tampered_payload_units_are_rejected_as_incomplete(tmp_path, table):
    checkpoint = complete_checkpoint(tmp_path, correct=225)
    row = checkpoint._connection.execute(
        f"SELECT record_json FROM {table} WHERE question_id = 1 AND repeat_index = 0"
    ).fetchone()
    assert row is not None
    record = json.loads(row[0])
    record["question_id"] = 2
    checkpoint._connection.execute(
        f"UPDATE {table} SET record_json = ? "
        "WHERE question_id = 1 AND repeat_index = 0",
        (A.canonical_json(record),),
    )
    checkpoint._connection.commit()

    with pytest.raises(A.IncompleteRunError, match="serialized"):
        A.build_summary(checkpoint)


def test_non_integer_serialized_unit_is_rejected_as_incomplete(tmp_path):
    checkpoint = complete_checkpoint(tmp_path, correct=225)
    row = checkpoint._connection.execute(
        "SELECT record_json FROM candidates WHERE question_id = 1 AND repeat_index = 0"
    ).fetchone()
    assert row is not None
    record = json.loads(row[0])
    record["question_id"] = 1.0
    checkpoint._connection.execute(
        "UPDATE candidates SET record_json = ? "
        "WHERE question_id = 1 AND repeat_index = 0",
        (A.canonical_json(record),),
    )
    checkpoint._connection.commit()

    with pytest.raises(A.IncompleteRunError, match="serialized"):
        A.build_summary(checkpoint)


def test_extra_sql_unit_is_rejected_as_incomplete(tmp_path):
    checkpoint = complete_checkpoint(tmp_path, correct=225)
    row = checkpoint._connection.execute(
        "SELECT record_json FROM candidates WHERE question_id = 1 AND repeat_index = 0"
    ).fetchone()
    assert row is not None
    candidate = json.loads(row[0])
    candidate["repeat_index"] = 3
    checkpoint._connection.execute(
        """
        INSERT INTO candidates (question_id, repeat_index, record_json)
        VALUES (1, 3, ?)
        """,
        (A.canonical_json(candidate),),
    )
    checkpoint._connection.commit()

    with pytest.raises(A.IncompleteRunError, match="candidate SQL units"):
        A.build_summary(checkpoint)


def test_malformed_serialized_row_is_rejected_as_incomplete(tmp_path):
    checkpoint = complete_checkpoint(tmp_path, correct=225)
    checkpoint._connection.execute(
        "UPDATE candidates SET record_json = ? "
        "WHERE question_id = 1 AND repeat_index = 0",
        ("not JSON",),
    )
    checkpoint._connection.commit()

    with pytest.raises(A.IncompleteRunError, match="malformed serialized"):
        A.build_summary(checkpoint)


def test_rename_no_replace_maps_existing_directory_errors(tmp_path):
    for error_number in (errno.EEXIST, errno.ENOTEMPTY):
        with pytest.raises(FileExistsError):
            A._raise_rename_no_replace_error(error_number, tmp_path / "published")


def test_publish_never_replaces_destination_appearing_at_rename(tmp_path, monkeypatch):
    checkpoint = complete_checkpoint(tmp_path, correct=225)
    destination = tmp_path / "published"
    original = A._rename_no_replace

    def destination_appears(source, target):
        target.mkdir()
        (target / "sentinel").write_text("keep", encoding="utf-8")
        original(source, target)

    monkeypatch.setattr(A, "_rename_no_replace", destination_appears)

    with pytest.raises(FileExistsError):
        A.publish_results(checkpoint, destination)
    assert (destination / "sentinel").read_text(encoding="utf-8") == "keep"


def test_candidate_without_final_content_has_no_headline(tmp_path):
    checkpoint = complete_checkpoint(tmp_path, correct=225)
    row = checkpoint._connection.execute(
        "SELECT record_json FROM candidates WHERE question_id = 1 AND repeat_index = 0"
    ).fetchone()
    assert row is not None
    record = json.loads(row[0])
    record["content"] = None
    checkpoint._connection.execute(
        "UPDATE candidates SET record_json = ? "
        "WHERE question_id = 1 AND repeat_index = 0",
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
    assert summary["benchmark_claim"] == ("AA-LCR v1.1 public-methodology reproduction")
    assert summary["finish_reasons"] == {"length": 1, "stop": 299}
    assert summary["candidate_diagnostics"] == {
        "truncation_count": 1,
        "empty_answer_count": 1,
        "retry_count": 1,
        "persistent_failure_count": 0,
    }
    assert summary["judge_retry_count"] == 0
    assert summary["judge_terminal_failure_count"] == 0
    assert summary["judge_identity"]["endpoint"] == A.OPENAI_API_BASE_URL
    assert summary["judge_identity"]["preflight"] == preflight_audit()


def test_summary_judge_retry_count_includes_successes_and_preflight(tmp_path):
    checkpoint = complete_checkpoint(tmp_path, correct=225)
    judgment_row = checkpoint._connection.execute(
        "SELECT record_json FROM judgments WHERE question_id = 1 AND repeat_index = 0"
    ).fetchone()
    assert judgment_row is not None
    judgment = json.loads(judgment_row[0])
    judgment["retry_count"] = 3
    checkpoint._connection.execute(
        "UPDATE judgments SET record_json = ? "
        "WHERE question_id = 1 AND repeat_index = 0",
        (A.canonical_json(judgment),),
    )
    audit = preflight_audit()
    audit["retry_count"] = 2
    checkpoint._connection.execute(
        "UPDATE judge_preflight SET record_json = ? WHERE singleton = 1",
        (A.canonical_json(audit),),
    )
    checkpoint._connection.commit()

    summary = A.build_summary(checkpoint)

    assert summary["judge_retry_count"] == 5
    assert summary["judge_terminal_failure_count"] == 0


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
    run_manifest = json.loads(
        (destination / "run-manifest.json").read_text(encoding="utf-8")
    )
    assert run_manifest["judge_preflight"] == preflight_audit()
    with pytest.raises(FileExistsError):
        A.publish_results(checkpoint, destination)
