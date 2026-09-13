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
        candidate_temperature=0.6,
        candidate_top_p=1.0,
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
