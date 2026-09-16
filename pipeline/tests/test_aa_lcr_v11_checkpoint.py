from dataclasses import replace
from types import MappingProxyType

import pytest

from pipeline import aa_lcr_v11 as A


def contract() -> A.RunContract:
    return A.RunContract(
        dataset_revision="dataset-revision",
        dataset_file_sha256={"dataset.csv": "a" * 64},
        prompt_sha256="b" * 64,
        judge_system_prompt_sha256="c" * 64,
        judge_user_prompt_sha256="d" * 64,
        candidate_model="candidate-model",
        served_model="served-model",
        endpoint_deployment_identity={"deployment": "candidate-v1"},
        candidate_temperature=1.0,
        candidate_top_p=0.95,
        candidate_max_tokens=131_072,
        candidate_concurrency=2,
        candidate_reasoning_enabled=True,
        candidate_max_attempts=30,
        judge_model="judge-model",
        judge_reasoning_effort="medium",
        judge_max_attempts=30,
        repeats=3,
        code_revision="abc123",
        question_ids=(1,),
    )


def candidate_record(content: str = "answer") -> A.CandidateRecord:
    return A.CandidateRecord(
        question_id=1,
        repeat_index=0,
        http_status=200,
        raw_response={"choices": []},
        usage={"completion_tokens": 1},
        finish_reason="stop",
        content=content,
        reasoning_content="reasoning",
        retry_count=0,
        started_at_utc="2026-09-13T00:00:00Z",
        completed_at_utc="2026-09-13T00:00:01Z",
        error=None,
    )


def judgment_record(judge_contract_hash: str) -> A.JudgmentRecord:
    return A.JudgmentRecord(
        question_id=1,
        repeat_index=0,
        judge_contract_hash=judge_contract_hash,
        raw_response={"verdict": "CORRECT"},
        verdict="CORRECT",
        retry_count=0,
        started_at_utc="2026-09-13T00:00:02Z",
        completed_at_utc="2026-09-13T00:00:03Z",
        error=None,
    )


def preflight_audit() -> dict[str, object]:
    return {
        "requested_model": A.JUDGE_MODEL,
        "retrieved_model": A.JUDGE_MODEL,
        "returned_model": A.JUDGE_MODEL,
        "endpoint": A.OPENAI_API_BASE_URL,
        "reasoning_effort": A.JUDGE_REASONING_EFFORT,
        "reasoning_mode": A.JUDGE_REASONING_MODE,
        "openai_sdk_version": "3.8.0",
        "response_id": "resp_preflight",
        "request_id": "req_preflight",
        "usage": {"input_tokens": 10, "output_tokens": 4, "total_tokens": 14},
        "started_at_utc": "2026-09-13T00:00:00Z",
        "completed_at_utc": "2026-09-13T00:00:01Z",
        "raw_output_text": '{"verdict":"CORRECT"}',
        "verdict": "CORRECT",
        "retry_count": 0,
        "attempt_count": 1,
        "model_identity": {"id": A.JUDGE_MODEL},
    }


def test_fingerprint_changes_for_measurement_fields():
    base = contract()

    assert base.fingerprint != replace(base, candidate_concurrency=1).fingerprint
    assert base.fingerprint != replace(base, judge_model="other").fingerprint
    assert base.fingerprint != replace(base, prompt_sha256="0" * 64).fingerprint


def test_candidate_is_insert_once(tmp_path):
    db = A.Checkpoint(tmp_path / "run.sqlite", contract())
    record = candidate_record(content="answer")

    db.record_candidate(record)
    db.record_candidate(record)

    with pytest.raises(A.CheckpointConflictError):
        db.record_candidate(replace(record, content="changed"))


def test_missing_candidates_include_transport_errors_not_http_400(tmp_path):
    db = A.Checkpoint(tmp_path / "run.sqlite", replace(contract(), repeats=1))
    timeout = replace(
        candidate_record(),
        http_status=0,
        content=None,
        finish_reason=None,
        error="transport error after 30 attempts: TimeoutError",
    )
    bad_request = replace(
        candidate_record(),
        http_status=400,
        content=None,
        error="HTTP 400 response",
    )

    assert db.missing_candidates() == [(1, 0)]
    db.record_candidate(timeout)
    assert db.missing_candidates() == [(1, 0)]
    db.record_candidate(replace(timeout, http_status=200, content="ok", error=None, finish_reason="stop"))
    assert db.missing_candidates() == []

    other = A.Checkpoint(tmp_path / "http400.sqlite", replace(contract(), repeats=1))
    other.record_candidate(bad_request)
    assert other.missing_candidates() == []


def test_record_candidate_replaces_transport_error_only(tmp_path):
    db = A.Checkpoint(tmp_path / "run.sqlite", contract())
    timeout = replace(
        candidate_record(),
        http_status=0,
        content=None,
        finish_reason=None,
        error="transport error after 30 attempts: TimeoutError",
    )
    success = replace(timeout, http_status=200, content="ok", error=None, finish_reason="stop")

    db.record_candidate(timeout)
    db.record_candidate(success)
    assert db.missing_candidates() == [(1, 1), (1, 2)]
    with pytest.raises(A.CheckpointConflictError):
        db.record_candidate(replace(success, content="changed"))


def test_judgment_key_includes_judge_contract(tmp_path):
    db = A.Checkpoint(tmp_path / "run.sqlite", contract())
    db.record_candidate(candidate_record())
    db.record_judgment(judgment_record(judge_contract_hash="a" * 64))

    assert db.missing_judgments("b" * 64) == [(1, 0)]


def test_checkpoint_rejects_another_run_fingerprint(tmp_path):
    db_path = tmp_path / "run.sqlite"
    A.Checkpoint(db_path, contract())

    with pytest.raises(A.CheckpointConflictError):
        A.Checkpoint(db_path, replace(contract(), code_revision="different"))


def test_build_run_contract_accepts_prepared_dataset_mapping_proxy():
    prepared = A.PreparedDataset(
        revision="dataset-revision",
        questions=(
            A.Question(
                question_id=1,
                category="category",
                document_set_id="set",
                question="Question?",
                official_answer="Answer",
                document_filenames=("document.txt",),
                prompt="prompt",
                cl100k_tokens=1,
            ),
        ),
        file_sha256=MappingProxyType({"dataset.csv": "a" * 64}),
        prompt_sha256="b" * 64,
    )

    run_contract = A.build_run_contract(
        prepared,
        candidate_model="candidate-model",
        served_model="served-model",
        endpoint_deployment_identity={"deployment": "candidate-v1"},
        code_revision="abc123",
    )

    assert len(run_contract.fingerprint) == 64


@pytest.mark.parametrize("repeat_index", [-1, 3])
def test_record_candidate_rejects_repeat_outside_contract(tmp_path, repeat_index):
    db = A.Checkpoint(tmp_path / "run.sqlite", contract())

    with pytest.raises(A.CheckpointConflictError, match="outside run contract"):
        db.record_candidate(replace(candidate_record(), repeat_index=repeat_index))


def test_record_candidate_rejects_unknown_contract_question(tmp_path):
    db = A.Checkpoint(tmp_path / "run.sqlite", contract())

    with pytest.raises(A.CheckpointConflictError, match="outside run contract"):
        db.record_candidate(replace(candidate_record(), question_id=2))


@pytest.mark.parametrize("repeat_index", [-1, 3])
def test_record_judgment_rejects_repeat_outside_contract(tmp_path, repeat_index):
    db = A.Checkpoint(tmp_path / "run.sqlite", contract())
    db.record_candidate(candidate_record())

    with pytest.raises(A.CheckpointConflictError, match="outside run contract"):
        db.record_judgment(
            replace(judgment_record("a" * 64), repeat_index=repeat_index)
        )


def test_record_judgment_rejects_unknown_contract_question(tmp_path):
    db = A.Checkpoint(tmp_path / "run.sqlite", contract())

    with pytest.raises(A.CheckpointConflictError, match="outside run contract"):
        db.record_judgment(replace(judgment_record("a" * 64), question_id=2))


def test_missing_judgments_ignores_out_of_contract_candidate_rows(tmp_path):
    db = A.Checkpoint(tmp_path / "run.sqlite", contract())
    db.record_candidate(candidate_record())
    db._connection.execute("PRAGMA foreign_keys=OFF")
    db._connection.execute("INSERT INTO questions (question_id) VALUES (2)")
    db._connection.execute(
        """
        INSERT INTO candidates (question_id, repeat_index, record_json)
        VALUES (2, 99, '{}')
        """
    )
    db._connection.commit()

    assert db.missing_judgments("a" * 64) == [(1, 0)]


def test_preflight_audit_is_durable_idempotent_and_immutable(tmp_path):
    db = A.Checkpoint(tmp_path / "run.sqlite", contract())
    audit = preflight_audit()

    db.record_preflight_audit(audit)
    db.record_preflight_audit(audit)
    reopened = A.Checkpoint(db.path, db.contract)

    assert reopened.preflight_audit() == audit
    with pytest.raises(A.CheckpointConflictError, match="conflicting immutable"):
        reopened.record_preflight_audit(
            {**audit, "returned_model": "gpt-5.6-luna-shadow"}
        )


def test_preflight_audit_requires_complete_identity_and_request_metadata(tmp_path):
    db = A.Checkpoint(tmp_path / "run.sqlite", contract())
    incomplete = preflight_audit()
    del incomplete["openai_sdk_version"]

    with pytest.raises(A.JudgeProtocolError, match="missing required fields"):
        db.record_preflight_audit(incomplete)

    assert db.preflight_audit() is None
