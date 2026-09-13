import json
import sqlite3
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import pytest

from pipeline import aa_lcr_v11 as A


def questions(count: int = 1) -> tuple[A.Question, ...]:
    return tuple(
        A.Question(
            question_id=question_id,
            category="category",
            document_set_id="set",
            question=f"Question {question_id}?",
            official_answer="Answer",
            document_filenames=("document.txt",),
            prompt=f"candidate prompt {question_id}",
            cl100k_tokens=2,
        )
        for question_id in range(1, count + 1)
    )


def server_identity(version: int = 1) -> dict[str, object]:
    return {
        "model_path": f"/models/glm-{version}",
        "served_model": "glm-5.3-w4afp8",
        "tp_size": 2,
        "max_total_num_tokens": 32768,
        "context_length": 32768,
        "quantization": "w4afp8",
        "kv_cache_dtype": "fp8_e4m3fn",
        "reasoning_parser": "glm45",
        "speculative_settings": {
            "draft_model_path": "/models/draft",
            "eagle_topk": 8,
            "num_draft_tokens": 4,
            "num_steps": 3,
            "speculative_algorithm": "EAGLE",
            "speculative_extra_knob": "enabled",
        },
    }


def seeded_checkpoint(
    path: Path,
    *,
    repeats: int = 1,
    question_count: int = 1,
    endpoint_identity: dict[str, object] | None = None,
) -> A.Checkpoint:
    return A.Checkpoint(
        path / "run.sqlite",
        A.RunContract(
            dataset_revision="revision",
            dataset_file_sha256={"dataset.csv": "a" * 64},
            prompt_sha256="b" * 64,
            judge_system_prompt_sha256=A.sha256_text(A.JUDGE_SYSTEM_PROMPT),
            judge_user_prompt_sha256=A.sha256_text(A.JUDGE_USER_PROMPT_TEMPLATE),
            candidate_model="glm-5.3-w4afp8",
            served_model="glm-5.3-w4afp8",
            endpoint_deployment_identity=(
                server_identity() if endpoint_identity is None else endpoint_identity
            ),
            candidate_temperature=0.6,
            candidate_top_p=1.0,
            candidate_max_tokens=131_072,
            candidate_concurrency=2,
            candidate_reasoning_enabled=True,
            candidate_max_attempts=30,
            judge_model=A.JUDGE_MODEL,
            judge_reasoning_effort=A.JUDGE_REASONING_EFFORT,
            judge_max_attempts=30,
            repeats=repeats,
            code_revision="test",
            question_ids=tuple(range(1, question_count + 1)),
        ),
    )


class FakeServer:
    def __init__(self, responses: list[tuple[int, object]] | None = None) -> None:
        self.responses = list(responses or [(200, successful_response())])
        self.chat_calls = 0
        self.requests: list[dict[str, object]] = []
        self.identity_version = 1
        self.active_requests = 0
        self.max_active_requests = 0
        self.block_requests = False
        self.release_requests = threading.Event()
        self._active_lock = threading.Lock()
        owner = self

        class Handler(BaseHTTPRequestHandler):
            def do_GET(self) -> None:
                if self.path == "/get_server_info":
                    self._send_json(
                        200,
                        {
                            "model_path": f"/models/glm-{owner.identity_version}",
                            "tp_size": 2,
                            "max_total_num_tokens": 32768,
                            "context_length": 32768,
                            "quantization": "w4afp8",
                            "kv_cache_dtype": "fp8_e4m3fn",
                            "reasoning_parser": "glm45",
                            "speculative_algorithm": "EAGLE",
                            "num_steps": 3,
                            "eagle_topk": 8,
                            "num_draft_tokens": 4,
                            "draft_model_path": "/models/draft",
                            "speculative_extra_knob": "enabled",
                        },
                    )
                elif self.path == "/v1/models":
                    self._send_json(
                        200,
                        {"data": [{"id": "glm-5.3-w4afp8"}]},
                    )
                else:
                    self.send_error(404)

            def do_POST(self) -> None:
                assert self.path == "/v1/chat/completions"
                length = int(self.headers["Content-Length"])
                owner.requests.append(
                    {
                        "body": json.loads(self.rfile.read(length)),
                        "headers": dict(self.headers.items()),
                    }
                )
                owner.chat_calls += 1
                with owner._active_lock:
                    owner.active_requests += 1
                    owner.max_active_requests = max(
                        owner.max_active_requests, owner.active_requests
                    )
                try:
                    if owner.block_requests:
                        owner.release_requests.wait(timeout=2)
                    status, body = owner.responses.pop(0)
                    if status == 0:
                        self.connection.close()
                        return
                    self._send_json(status, body)
                finally:
                    with owner._active_lock:
                        owner.active_requests -= 1

            def _send_json(self, status: int, body: object) -> None:
                encoded = json.dumps(body).encode()
                self.send_response(status)
                self.send_header("Content-Type", "application/json")
                self.send_header("X-Secret", "response-secret")
                self.send_header("Content-Length", str(len(encoded)))
                self.end_headers()
                self.wfile.write(encoded)

            def log_message(self, _format: str, *_args: object) -> None:
                return

        self.server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)

    @property
    def url(self) -> str:
        host, port = self.server.server_address
        return f"http://{host}:{port}"

    def start(self) -> None:
        self.thread.start()

    def close(self) -> None:
        self.server.shutdown()
        self.thread.join()
        self.server.server_close()


@pytest.fixture
def fake_server():
    server = FakeServer()
    server.start()
    try:
        yield server
    finally:
        server.close()


def successful_response() -> dict[str, object]:
    return {
        "choices": [
            {
                "finish_reason": "stop",
                "message": {
                    "reasoning_content": "private reasoning",
                    "content": "final answer",
                },
            }
        ],
        "usage": {"prompt_tokens": 100, "completion_tokens": 20},
    }


def test_candidate_request_is_glm_max_contract():
    body = A.candidate_request(questions()[0])

    assert body == {
        "model": "glm-5.3-w4afp8",
        "messages": [{"role": "user", "content": "candidate prompt 1"}],
        "max_tokens": 131072,
        "temperature": 0.6,
        "top_p": 1.0,
        "chat_template_kwargs": {"enable_thinking": True},
    }


def test_only_final_content_is_selected():
    record = A.candidate_from_response(1, 0, 200, successful_response())

    assert record.content == "final answer"
    assert record.reasoning_content == "private reasoning"
    assert record.finish_reason == "stop"
    assert record.usage == {"prompt_tokens": 100, "completion_tokens": 20}


def test_resume_does_not_regenerate_terminal_candidate(tmp_path, fake_server):
    checkpoint = seeded_checkpoint(tmp_path)

    A.generate_missing(checkpoint, questions(), fake_server.url, repeats=1)
    A.generate_missing(checkpoint, questions(), fake_server.url, repeats=1)

    assert fake_server.chat_calls == 1


@pytest.mark.parametrize("status", [408, 409, 429, 500])
def test_retryable_http_statuses_are_retried(status, monkeypatch):
    server = FakeServer(
        [(status, {"error": {"message": "retry"}}), (200, successful_response())]
    )
    server.start()
    monkeypatch.setattr(A.time, "sleep", lambda _seconds: None)
    try:
        record = A.generate_one(questions()[0], 0, server.url)
    finally:
        server.close()

    assert server.chat_calls == 2
    assert record.http_status == 200
    assert record.content == "final answer"
    assert record.retry_count == 1


def test_http_400_is_terminal_and_redacts_headers_and_environment(monkeypatch):
    server = FakeServer([(400, {"error": {"message": "bad request"}})])
    server.start()
    monkeypatch.setenv("CANDIDATE_SECRET", "environment-secret")
    try:
        record = A.generate_one(questions()[0], 0, server.url)
    finally:
        server.close()

    assert server.chat_calls == 1
    assert record.http_status == 400
    assert record.raw_response == {"error": {"message": "bad request"}}
    assert record.error is not None
    assert "response-secret" not in record.error
    assert "environment-secret" not in record.error
    assert "X-Secret" not in record.error


def test_exhausted_transport_attempts_create_terminal_record(monkeypatch):
    server = FakeServer([(0, None)] * A.MAX_ATTEMPTS)
    server.start()
    monkeypatch.setattr(A.time, "sleep", lambda _seconds: None)
    try:
        record = A.generate_one(questions()[0], 0, server.url)
    finally:
        server.close()

    assert server.chat_calls == A.MAX_ATTEMPTS
    assert record.http_status == 0
    assert record.raw_response is None
    assert record.retry_count == A.MAX_ATTEMPTS - 1
    assert record.error is not None
    assert record.error.startswith("transport error after 30 attempts: ")


def test_malformed_http_200_is_terminal_record():
    server = FakeServer([(200, {"choices": []})])
    server.start()
    try:
        record = A.generate_one(questions()[0], 0, server.url)
    finally:
        server.close()

    assert server.chat_calls == 1
    assert record.http_status == 200
    assert record.error == "malformed model response: missing choices[0].message"


def test_length_finish_reason_is_terminal_model_record():
    response = successful_response()
    response["choices"][0]["finish_reason"] = "length"
    server = FakeServer([(200, response)])
    server.start()
    try:
        record = A.generate_one(questions()[0], 0, server.url)
    finally:
        server.close()

    assert server.chat_calls == 1
    assert record.http_status == 200
    assert record.finish_reason == "length"
    assert record.content == "final answer"
    assert record.error is None


def test_server_identity_contains_bound_fields(fake_server):
    identity = A.fetch_server_identity(fake_server.url)

    assert identity == server_identity()


def test_identity_change_invalidates_generation(tmp_path, fake_server):
    checkpoint = seeded_checkpoint(tmp_path)
    original_record_candidate = checkpoint.record_candidate

    def record_then_change_identity(record):
        original_record_candidate(record)
        fake_server.identity_version = 2

    checkpoint.record_candidate = record_then_change_identity  # type: ignore[method-assign]

    with pytest.raises(A.CheckpointConflictError, match="server identity changed"):
        A.generate_missing(checkpoint, questions(), fake_server.url, repeats=1)

    assert checkpoint.server_snapshot("before") == server_identity()
    assert checkpoint.server_snapshot("after") == server_identity(version=2)


def test_generation_requires_identity_matching_run_contract(tmp_path, fake_server):
    checkpoint = seeded_checkpoint(
        tmp_path, endpoint_identity=server_identity(version=2)
    )

    with pytest.raises(A.CheckpointConflictError, match="does not match"):
        A.generate_missing(checkpoint, questions(), fake_server.url, repeats=1)

    assert checkpoint.server_snapshot("before") == server_identity()
    assert checkpoint.server_snapshot("after") is None
    assert fake_server.chat_calls == 0


def test_generation_rejects_empty_contract_identity(tmp_path, fake_server):
    checkpoint = seeded_checkpoint(tmp_path, endpoint_identity={})

    with pytest.raises(A.CheckpointConflictError, match="must not be empty"):
        A.generate_missing(checkpoint, questions(), fake_server.url, repeats=1)

    assert fake_server.chat_calls == 0


def test_server_snapshots_are_durable_and_immutable(tmp_path):
    checkpoint = seeded_checkpoint(tmp_path)
    identity = server_identity()

    checkpoint.record_server_snapshot("before", identity)
    reopened = A.Checkpoint(checkpoint.path, checkpoint.contract)

    assert reopened.server_snapshot("before") == identity
    with pytest.raises(A.CheckpointConflictError, match="conflicting immutable"):
        reopened.record_server_snapshot("before", server_identity(version=2))


def test_generation_uses_at_most_two_simultaneous_requests(tmp_path, fake_server):
    checkpoint = seeded_checkpoint(tmp_path, question_count=2)
    fake_server.responses = [(200, successful_response()), (200, successful_response())]
    fake_server.block_requests = True
    releaser = threading.Timer(0.2, fake_server.release_requests.set)
    releaser.start()
    try:
        A.generate_missing(checkpoint, questions(2), fake_server.url, repeats=1)
    finally:
        releaser.cancel()

    assert fake_server.max_active_requests == 2


def test_sqlite_payloads_exclude_response_headers_and_environment(
    tmp_path, fake_server, monkeypatch
):
    checkpoint = seeded_checkpoint(tmp_path)
    monkeypatch.setenv("CANDIDATE_SECRET", "environment-secret")

    A.generate_missing(checkpoint, questions(), fake_server.url, repeats=1)

    with sqlite3.connect(checkpoint.path) as connection:
        persisted = "\n".join(
            row[0]
            for row in connection.execute(
                "SELECT record_json FROM candidates UNION ALL "
                "SELECT payload_json FROM server_snapshots"
            )
        )
    assert "response-secret" not in persisted
    assert "environment-secret" not in persisted
    assert "X-Secret" not in persisted
    assert "Content-Type" not in persisted


def test_post_snapshot_is_recorded_after_worker_failure(tmp_path, fake_server):
    checkpoint = seeded_checkpoint(tmp_path)

    def fail_record(_record):
        raise RuntimeError("checkpoint write failed")

    checkpoint.record_candidate = fail_record  # type: ignore[method-assign]

    with pytest.raises(RuntimeError, match="checkpoint write failed"):
        A.generate_missing(checkpoint, questions(), fake_server.url, repeats=1)

    assert checkpoint.server_snapshot("before") == server_identity()
    assert checkpoint.server_snapshot("after") == server_identity()
