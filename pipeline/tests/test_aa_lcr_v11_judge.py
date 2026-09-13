import json
import sqlite3
import sys
from dataclasses import replace
from types import ModuleType
from types import SimpleNamespace

import pytest

from pipeline import aa_lcr_v11 as A
from pipeline.tests.test_aa_lcr_v11_candidate import (
    questions,
    seeded_checkpoint,
)


class FakeAPIError(Exception):
    def __init__(self, status_code: int | None = None, request_id: str | None = None):
        self.status_code = status_code
        self.request_id = request_id


class FakeResponses:
    def __init__(self, outcomes: list[object]) -> None:
        self.outcomes = list(outcomes)
        self.calls: list[dict[str, object]] = []

    def create(self, **kwargs: object) -> object:
        self.calls.append(kwargs)
        outcome = self.outcomes.pop(0)
        if isinstance(outcome, BaseException):
            raise outcome
        return outcome


class FakeOpenAI:
    def __init__(self, outcomes: list[object], *, model_id: str = A.JUDGE_MODEL) -> None:
        self.responses = FakeResponses(outcomes)
        self.models = SimpleNamespace(
            retrieve=lambda model: SimpleNamespace(
                id=model_id, created=123, object="model", owned_by="openai"
            )
        )


def judge_response(
    output_text: str = '{"verdict":"CORRECT"}',
    *,
    model: str = A.JUDGE_MODEL,
) -> object:
    return SimpleNamespace(
        id="resp_123",
        _request_id="req_123",
        model=model,
        output_text=output_text,
        usage=SimpleNamespace(input_tokens=10, output_tokens=4, total_tokens=14),
    )


def candidate_record(content: str | None = "answer") -> A.CandidateRecord:
    return A.CandidateRecord(
        question_id=1,
        repeat_index=0,
        http_status=200,
        raw_response={"choices": []},
        usage={"completion_tokens": 1},
        finish_reason="stop",
        content=content,
        reasoning_content="private reasoning",
        retry_count=0,
        started_at_utc="2026-09-13T00:00:00Z",
        completed_at_utc="2026-09-13T00:00:01Z",
        error=None,
    )


@pytest.fixture
def fake_openai() -> FakeOpenAI:
    return FakeOpenAI([judge_response()])


def test_judge_uses_responses_api_medium_effort(fake_openai):
    record = A.judge_one(fake_openai, questions()[0], candidate_record())

    call = fake_openai.responses.calls[0]
    assert call["model"] == "gpt-5.6-luna"
    assert call["reasoning"] == {"effort": "medium", "mode": "standard"}
    assert call["store"] is False
    assert call["input"][0] == {
        "role": "system",
        "content": A.JUDGE_SYSTEM_PROMPT,
    }
    assert call["input"][1] == {
        "role": "user",
        "content": A.build_judge_user_prompt("Question 1?", "Answer", "answer"),
    }
    assert record.verdict == "CORRECT"
    assert record.returned_model == A.JUDGE_MODEL
    assert record.request_id == "req_123"
    assert record.response_id == "resp_123"
    assert record.raw_output_text == '{"verdict":"CORRECT"}'


def test_build_client_requires_environment_key(monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)

    with pytest.raises(A.CredentialError, match="OPENAI_API_KEY is not set"):
        A.build_openai_client()


def test_build_client_uses_official_sdk_when_installed(monkeypatch):
    created: dict[str, object] = {}

    class FakeSDKClient:
        def __init__(self, **kwargs: object) -> None:
            created.update(kwargs)

    fake_openai_module = ModuleType("openai")
    fake_openai_module.OpenAI = FakeSDKClient  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "openai", fake_openai_module)
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test-only")

    client = A.build_openai_client()

    assert isinstance(client, FakeSDKClient)
    assert created == {
        "api_key": "sk-test-only",
        "timeout": 300.0,
        "max_retries": 0,
    }


def test_preflight_retrieves_exact_model_and_requires_correct_verdict():
    client = FakeOpenAI([judge_response()])

    result = A.preflight_judge(client)

    assert result["model_identity"] == {
        "id": A.JUDGE_MODEL,
        "created": 123,
        "object": "model",
        "owned_by": "openai",
    }
    judgment = result["judgment"]
    assert judgment["requested_model"] == A.JUDGE_MODEL
    assert judgment["retrieved_model"] == A.JUDGE_MODEL
    assert judgment["returned_model"] == A.JUDGE_MODEL
    assert judgment["usage"] == {
        "input_tokens": 10,
        "output_tokens": 4,
        "total_tokens": 14,
    }
    assert judgment["raw_output_text"] == '{"verdict":"CORRECT"}'
    assert judgment["verdict"] == "CORRECT"
    assert judgment["reasoning_effort"] == A.JUDGE_REASONING_EFFORT
    assert judgment["reasoning_mode"] == A.JUDGE_REASONING_MODE
    assert judgment["started_at_utc"].endswith("Z")
    assert judgment["completed_at_utc"].endswith("Z")
    assert judgment["retry_count"] == 0
    assert client.responses.calls[0]["input"][1]["content"].find("What is 2 + 2?") >= 0


def test_preflight_rejects_unrelated_retrieved_model():
    client = FakeOpenAI([judge_response()], model_id="gpt-4.1")

    with pytest.raises(A.JudgeProtocolError, match="unrelated"):
        A.preflight_judge(client)

    assert client.responses.calls == []


def test_malformed_judge_output_retries_then_records_normalized_verdict(monkeypatch):
    client = FakeOpenAI([judge_response("not json"), judge_response('{"verdict":"incorrect"}')])
    monkeypatch.setattr(A.time, "sleep", lambda _seconds: None)

    record = A.judge_one(client, questions()[0], candidate_record())

    assert len(client.responses.calls) == 2
    assert record.retry_count == 1
    assert record.verdict == "INCORRECT"


def test_exhausted_malformed_output_persists_raw_output(monkeypatch):
    client = FakeOpenAI([judge_response("not json")] * A.MAX_ATTEMPTS)
    monkeypatch.setattr(A.time, "sleep", lambda _seconds: None)

    record = A.judge_one(client, questions()[0], candidate_record())

    assert record.verdict is None
    assert record.raw_output_text == "not json"
    assert record.raw_response == {"output_text": "not json"}


def test_malformed_output_failure_retains_received_response_metadata(monkeypatch):
    client = FakeOpenAI([judge_response("not json")] * A.MAX_ATTEMPTS)
    monkeypatch.setattr(A.time, "sleep", lambda _seconds: None)

    record = A.judge_one(client, questions()[0], candidate_record())
    failure = A._judge_failure_record(record)

    assert failure.verdict is None
    assert failure.returned_model == A.JUDGE_MODEL
    assert failure.response_id == "resp_123"
    assert failure.request_id == "req_123"
    assert failure.usage == {
        "input_tokens": 10,
        "output_tokens": 4,
        "total_tokens": 14,
    }
    assert failure.attempt_count == A.MAX_ATTEMPTS


@pytest.mark.parametrize(
    "error",
    [
        FakeAPIError(status_code=429, request_id="rate_123"),
        FakeAPIError(status_code=500, request_id="server_123"),
        TimeoutError(),
    ],
)
def test_retryable_judge_failures_retry(error, monkeypatch):
    client = FakeOpenAI([error, judge_response()])
    monkeypatch.setattr(A.time, "sleep", lambda _seconds: None)

    record = A.judge_one(client, questions()[0], candidate_record())

    assert len(client.responses.calls) == 2
    assert record.verdict == "CORRECT"
    assert record.retry_count == 1


@pytest.mark.parametrize(
    "error",
    [
        FakeAPIError(status_code=401, request_id="auth-request"),
        FakeAPIError(status_code=404, request_id="not-found-request"),
    ],
)
def test_non_retryable_judge_failures_stop_immediately(error, monkeypatch):
    client = FakeOpenAI([error, judge_response()])
    monkeypatch.setattr(A.time, "sleep", lambda _seconds: None)

    record = A.judge_one(client, questions()[0], candidate_record())

    assert len(client.responses.calls) == 1
    assert record.exception_class == type(error).__name__
    assert record.http_status == getattr(error, "status_code", None)


def test_unknown_judge_exception_propagates_without_retry(monkeypatch):
    client = FakeOpenAI([AttributeError("programming error"), judge_response()])
    monkeypatch.setattr(A.time, "sleep", lambda _seconds: None)

    with pytest.raises(AttributeError, match="programming error"):
        A.judge_one(client, questions()[0], candidate_record())

    assert len(client.responses.calls) == 1


def test_sdk_transport_exception_retries_without_status(monkeypatch):
    fake_openai_module = ModuleType("openai")

    class APIConnectionError(Exception):
        pass

    fake_openai_module.APIConnectionError = APIConnectionError  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "openai", fake_openai_module)
    client = FakeOpenAI([APIConnectionError(), judge_response()])
    monkeypatch.setattr(A.time, "sleep", lambda _seconds: None)

    record = A.judge_one(client, questions()[0], candidate_record())

    assert len(client.responses.calls) == 2
    assert record.verdict == "CORRECT"


def test_judge_exhaustion_records_only_safe_error(monkeypatch):
    secret = "sk-do-not-record"
    client = FakeOpenAI([FakeAPIError(status_code=429, request_id=secret)] * A.MAX_ATTEMPTS)
    monkeypatch.setattr(A.time, "sleep", lambda _seconds: None)

    record = A.judge_one(client, questions()[0], candidate_record())

    assert len(client.responses.calls) == A.MAX_ATTEMPTS
    assert record.verdict is None
    assert record.retry_count == A.MAX_ATTEMPTS - 1
    assert record.error == "judge failed after 30 attempts: FakeAPIError HTTP 429"
    assert secret not in A._canonical_dataclass(record)


def test_missing_final_candidate_content_is_not_sent(fake_openai):
    record = A.judge_one(fake_openai, questions()[0], candidate_record(content=None))

    assert fake_openai.responses.calls == []
    assert record.verdict is None
    assert record.error == "candidate has no final content"
    assert record.attempt_count == 0


def test_judge_missing_resumes_without_duplicate_calls(tmp_path):
    checkpoint = seeded_checkpoint(tmp_path)
    checkpoint.record_candidate(candidate_record())
    client = FakeOpenAI([judge_response()])

    A.judge_missing(checkpoint, questions(), client)
    A.judge_missing(checkpoint, questions(), client)

    assert len(client.responses.calls) == 1


@pytest.mark.parametrize(
    "changes",
    [
        {"judge_model": "not-luna"},
        {"judge_reasoning_effort": "low"},
        {"judge_reasoning_mode": "other"},
        {"judge_system_prompt_sha256": "0" * 64},
        {"judge_user_prompt_sha256": "0" * 64},
    ],
)
def test_judge_missing_rejects_invalid_request_contract_before_api_call(
    tmp_path, changes
):
    checkpoint = seeded_checkpoint(tmp_path)
    checkpoint.contract = replace(checkpoint.contract, **changes)
    checkpoint.record_candidate(candidate_record())
    client = FakeOpenAI([judge_response()])

    with pytest.raises(A.JudgeContractError):
        A.judge_missing(checkpoint, questions(), client)

    assert client.responses.calls == []


def test_failed_judgment_is_incomplete_and_failure_audit_is_append_only(
    tmp_path, monkeypatch
):
    checkpoint = seeded_checkpoint(tmp_path)
    checkpoint.record_candidate(candidate_record())
    client = FakeOpenAI([FakeAPIError(status_code=429, request_id="req_failure")] * A.MAX_ATTEMPTS)
    monkeypatch.setattr(A.time, "sleep", lambda _seconds: None)

    with pytest.raises(A.IncompleteJudgmentError, match="incomplete"):
        A.judge_missing(checkpoint, questions(), client)

    contract_hash = A._judge_contract_hash()
    assert checkpoint.missing_judgments(contract_hash) == [(1, 0)]
    failures = checkpoint.judge_failures()
    assert len(failures) == 1
    assert failures[0].attempt_count == A.MAX_ATTEMPTS
    assert failures[0].exception_class == "FakeAPIError"
    assert failures[0].http_status == 429
    assert failures[0].request_id == "req_failure"
    assert failures[0].requested_model == A.JUDGE_MODEL
    assert failures[0].reasoning_effort == A.JUDGE_REASONING_EFFORT
    assert failures[0].reasoning_mode == A.JUDGE_REASONING_MODE

    client.responses.outcomes = [judge_response()]
    with pytest.raises(A.IncompleteJudgmentError, match="incomplete"):
        A.judge_missing(checkpoint, questions(), client)

    assert len(client.responses.calls) == A.MAX_ATTEMPTS
    assert checkpoint.missing_judgments(contract_hash) == [(1, 0)]
    assert len(checkpoint.judge_failures()) == 1


def test_failed_judgment_cannot_occupy_immutable_success_key(tmp_path):
    checkpoint = seeded_checkpoint(tmp_path)
    checkpoint.record_candidate(candidate_record())
    record = A.judge_one(FakeOpenAI([]), questions()[0], candidate_record(content=None))

    with pytest.raises(A.CheckpointConflictError, match="valid CORRECT/INCORRECT"):
        checkpoint.record_judgment(record)


def test_secret_never_enters_judge_artifacts(tmp_path, monkeypatch):
    secret = "sk-test-do-not-serialize"
    monkeypatch.setenv("OPENAI_API_KEY", secret)
    checkpoint = seeded_checkpoint(tmp_path)
    checkpoint.record_candidate(candidate_record())
    client = FakeOpenAI(
        [FakeAPIError(status_code=429, request_id=secret)] * A.MAX_ATTEMPTS
    )

    monkeypatch.setattr(A.time, "sleep", lambda _seconds: None)
    with pytest.raises(A.IncompleteJudgmentError):
        A.judge_missing(checkpoint, questions(), client)

    with sqlite3.connect(checkpoint.path) as connection:
        payload = "\n".join(
            row[0]
            for row in connection.execute(
                "SELECT record_json FROM judgments "
                "UNION ALL SELECT record_json FROM judge_failures"
            )
        )
    assert secret not in payload
