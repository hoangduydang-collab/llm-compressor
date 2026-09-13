"""Pinned AA-LCR v1.1 identity and prompt contract."""

import csv
import hashlib
import importlib
import json
import os
import sqlite3
import stat
import sys
import tempfile
import threading
import time
import zipfile
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath
from types import MappingProxyType
from typing import TYPE_CHECKING, Callable, Collection, Mapping
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

import tiktoken

if TYPE_CHECKING:
    from openai import OpenAI


AA_LCR_VERSION = "1.1"
DATASET_REPO = "ArtificialAnalysis/AA-LCR"
DATASET_REVISION = "9a77ef56b717057ade24ceab4d273712a0b4f19e"
CSV_FILENAME = "AA-LCR_Dataset.csv"
ZIP_FILENAME = "extracted_text/AA-LCR_extracted-text.zip"
ZIP_SHA256 = "5e839249826f6b9bd5324f0d139089c9dc481ccb3f212a6dfad00c51045d9d8a"
HF_RESOLVE = (
    "https://huggingface.co/datasets/"
    f"{DATASET_REPO}/resolve/{DATASET_REVISION}"
)
REPEATS = 3
CANDIDATE_TEMPERATURE = 0.6
CANDIDATE_TOP_P = 1.0
CANDIDATE_MAX_TOKENS = 131_072
CANDIDATE_CONCURRENCY = 2
JUDGE_MODEL = "gpt-5.6-luna"
JUDGE_REASONING_EFFORT = "medium"
JUDGE_REASONING_MODE = "standard"
MAX_ATTEMPTS = 30

CANDIDATE_PROMPT_TEMPLATE = """BEGIN INPUT DOCUMENTS

{documents_text}

END INPUT DOCUMENTS

Answer the following question using the input documents provided above.

START QUESTION

{question}

END QUESTION
"""

JUDGE_SYSTEM_PROMPT = """Decide whether the CANDIDATE ANSWER is correct or incorrect against the OFFICIAL ANSWER.
Note the following points when assessing correctness:

- Numbers should still match when they are the same value written differently, e.g., a
  percentage, a count of percentage points, and the equivalent decimal fraction are the same
  value: 0.675, "67.5%" and "67.5 percentage points" all match. So do different scales
  (thousand, million, bn) and different notations (thousands separators, currency symbols,
  LaTeX markup, and numbers written as words).
- Where the question asks for a particular format (e.g., a percentage, a number of decimal
  places, a unit, a rounding, or an ordering) the CANDIDATE ANSWER must meet it. If the
  question asks for no particular format, accept any equivalent form.
- In cases where the question asks for an ordered list, a title, honorific or article added
  to an entry in the CANDIDATE ANSWER can change where that entry sorts. Accept the ordering
  if it is correct either with those additions or without them.
- Grade the value the CANDIDATE ANSWER finally commits to, and it must commit to one. Values
  reached while working, and alternatives it considers and sets aside, do not count. If it
  offers several values without selecting one, it is incorrect even if one of them is right.
  Hedging is fine as long as one clearly definitive answer is given.
"""

JUDGE_USER_PROMPT_TEMPLATE = """Assess whether the following CANDIDATE ANSWER is CORRECT or INCORRECT.
For the CANDIDATE ANSWER to be correct, it must be consistent with the OFFICIAL ANSWER.

The question, for reference only: START QUESTION {question}

END QUESTION

The OFFICIAL ANSWER: {official_answer}

END OFFICIAL ANSWER

BEGIN CANDIDATE ANSWER TO ASSESS

{candidate_answer}

END CANDIDATE ANSWER TO ASSESS

Reply as JSON, with a verdict of CORRECT or INCORRECT."""


class JudgeProtocolError(ValueError):
    """Raised when the judge response violates the expected JSON protocol."""


class CredentialError(ValueError):
    """Raised when required judge credentials are unavailable."""


class IncompleteJudgmentError(RuntimeError):
    """Raised when a run contains a terminal, unscored judge failure."""


class JudgeContractError(ValueError):
    """Raised when a run's judge request contract differs from AA-LCR v1.1."""


class DatasetIntegrityError(ValueError):
    """Raised when the pinned AA-LCR dataset cannot be verified."""


class CheckpointConflictError(ValueError):
    """Raised when immutable checkpoint data conflicts with existing data."""


@dataclass(frozen=True)
class Question:
    question_id: int
    category: str
    document_set_id: str
    question: str
    official_answer: str
    document_filenames: tuple[str, ...]
    prompt: str
    cl100k_tokens: int


@dataclass(frozen=True)
class PreparedDataset:
    revision: str
    questions: tuple[Question, ...]
    file_sha256: Mapping[str, str]
    prompt_sha256: str


@dataclass(frozen=True)
class RunContract:
    """Every input that defines an AA-LCR measurement."""

    dataset_revision: str
    dataset_file_sha256: Mapping[str, str]
    prompt_sha256: str
    judge_system_prompt_sha256: str
    judge_user_prompt_sha256: str
    candidate_model: str
    served_model: str
    endpoint_deployment_identity: Mapping[str, object]
    candidate_temperature: float
    candidate_top_p: float
    candidate_max_tokens: int
    candidate_concurrency: int
    candidate_reasoning_enabled: bool
    candidate_max_attempts: int
    judge_model: str
    judge_reasoning_effort: str
    judge_max_attempts: int
    repeats: int
    code_revision: str
    question_ids: tuple[int, ...]
    judge_reasoning_mode: str = JUDGE_REASONING_MODE

    @property
    def fingerprint(self) -> str:
        return sha256_text(_canonical_dataclass(self))


@dataclass(frozen=True)
class CandidateRecord:
    question_id: int
    repeat_index: int
    http_status: int
    raw_response: object
    usage: object
    finish_reason: str | None
    content: str | None
    reasoning_content: str | None
    retry_count: int
    started_at_utc: str
    completed_at_utc: str
    error: str | None


@dataclass(frozen=True)
class JudgmentRecord:
    question_id: int
    repeat_index: int
    judge_contract_hash: str
    raw_response: object
    verdict: str | None
    retry_count: int
    started_at_utc: str
    completed_at_utc: str
    error: str | None
    requested_model: str | None = None
    retrieved_model: str | None = None
    returned_model: str | None = None
    reasoning_effort: str | None = None
    reasoning_mode: str | None = None
    openai_sdk_version: str | None = None
    response_id: str | None = None
    request_id: str | None = None
    usage: object = None
    raw_output_text: str | None = None
    exception_class: str | None = None
    http_status: int | None = None
    attempt_count: int = 0


@dataclass(frozen=True)
class JudgeFailureRecord:
    question_id: int
    repeat_index: int
    judge_contract_hash: str
    requested_model: str
    reasoning_effort: str
    reasoning_mode: str
    openai_sdk_version: str | None
    exception_class: str
    http_status: int | None
    request_id: str | None
    retry_count: int
    attempt_count: int
    started_at_utc: str
    completed_at_utc: str
    raw_output_text: str | None
    returned_model: str | None
    response_id: str | None
    usage: object
    verdict: None = None


def build_run_contract(
    prepared_dataset: PreparedDataset,
    *,
    candidate_model: str,
    served_model: str,
    endpoint_deployment_identity: Mapping[str, object],
    code_revision: str,
    candidate_temperature: float = CANDIDATE_TEMPERATURE,
    candidate_top_p: float = CANDIDATE_TOP_P,
    candidate_max_tokens: int = CANDIDATE_MAX_TOKENS,
    candidate_concurrency: int = CANDIDATE_CONCURRENCY,
    candidate_reasoning_enabled: bool = True,
    candidate_max_attempts: int = MAX_ATTEMPTS,
    judge_model: str = JUDGE_MODEL,
    judge_reasoning_effort: str = JUDGE_REASONING_EFFORT,
    judge_reasoning_mode: str = JUDGE_REASONING_MODE,
    judge_max_attempts: int = MAX_ATTEMPTS,
    repeats: int = REPEATS,
) -> RunContract:
    """Build the immutable identity contract before any endpoint traffic."""
    _validate_judge_request_contract(
        judge_model=judge_model,
        judge_reasoning_effort=judge_reasoning_effort,
        judge_reasoning_mode=judge_reasoning_mode,
        judge_system_prompt_sha256=sha256_text(JUDGE_SYSTEM_PROMPT),
        judge_user_prompt_sha256=sha256_text(JUDGE_USER_PROMPT_TEMPLATE),
        judge_max_attempts=judge_max_attempts,
    )
    return RunContract(
        dataset_revision=prepared_dataset.revision,
        dataset_file_sha256=prepared_dataset.file_sha256,
        prompt_sha256=prepared_dataset.prompt_sha256,
        judge_system_prompt_sha256=sha256_text(JUDGE_SYSTEM_PROMPT),
        judge_user_prompt_sha256=sha256_text(JUDGE_USER_PROMPT_TEMPLATE),
        candidate_model=candidate_model,
        served_model=served_model,
        endpoint_deployment_identity=endpoint_deployment_identity,
        candidate_temperature=candidate_temperature,
        candidate_top_p=candidate_top_p,
        candidate_max_tokens=candidate_max_tokens,
        candidate_concurrency=candidate_concurrency,
        candidate_reasoning_enabled=candidate_reasoning_enabled,
        candidate_max_attempts=candidate_max_attempts,
        judge_model=judge_model,
        judge_reasoning_effort=judge_reasoning_effort,
        judge_max_attempts=judge_max_attempts,
        repeats=repeats,
        code_revision=code_revision,
        question_ids=tuple(question.question_id for question in prepared_dataset.questions),
        judge_reasoning_mode=judge_reasoning_mode,
    )


class Checkpoint:
    """Durable, immutable SQLite storage for a single run contract."""

    def __init__(self, path: Path, contract: RunContract) -> None:
        self.path = Path(path)
        self.contract = contract
        self._expected_question_ids = frozenset(contract.question_ids)
        self._lock = threading.Lock()
        self._connection = sqlite3.connect(self.path, check_same_thread=False)
        self._connection.execute("PRAGMA journal_mode=WAL")
        self._connection.execute("PRAGMA synchronous=FULL")
        self._connection.execute("PRAGMA foreign_keys=ON")
        self._initialize()

    def _initialize(self) -> None:
        with self._lock, self._connection:
            self._connection.executescript(
                """
                CREATE TABLE IF NOT EXISTS run (
                    singleton INTEGER PRIMARY KEY CHECK (singleton = 1),
                    fingerprint TEXT NOT NULL,
                    contract_json TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS questions (
                    question_id INTEGER PRIMARY KEY
                );
                CREATE TABLE IF NOT EXISTS candidates (
                    question_id INTEGER NOT NULL REFERENCES questions(question_id),
                    repeat_index INTEGER NOT NULL CHECK (repeat_index >= 0),
                    record_json TEXT NOT NULL,
                    PRIMARY KEY (question_id, repeat_index)
                );
                CREATE TABLE IF NOT EXISTS judgments (
                    question_id INTEGER NOT NULL,
                    repeat_index INTEGER NOT NULL CHECK (repeat_index >= 0),
                    judge_contract_hash TEXT NOT NULL,
                    record_json TEXT NOT NULL,
                    PRIMARY KEY (question_id, repeat_index, judge_contract_hash),
                    FOREIGN KEY (question_id, repeat_index)
                        REFERENCES candidates(question_id, repeat_index)
                );
                CREATE TABLE IF NOT EXISTS judge_failures (
                    question_id INTEGER NOT NULL,
                    repeat_index INTEGER NOT NULL CHECK (repeat_index >= 0),
                    judge_contract_hash TEXT NOT NULL,
                    record_json TEXT NOT NULL,
                    PRIMARY KEY (question_id, repeat_index, judge_contract_hash),
                    FOREIGN KEY (question_id, repeat_index)
                        REFERENCES candidates(question_id, repeat_index)
                );
                CREATE TABLE IF NOT EXISTS server_snapshots (
                    stage TEXT PRIMARY KEY,
                    payload_json TEXT NOT NULL
                );
                """
            )
            contract_json = _canonical_dataclass(self.contract)
            row = self._connection.execute(
                "SELECT fingerprint, contract_json FROM run WHERE singleton = 1"
            ).fetchone()
            if row is None:
                self._connection.execute(
                    "INSERT INTO run (singleton, fingerprint, contract_json) VALUES (1, ?, ?)",
                    (self.contract.fingerprint, contract_json),
                )
                self._connection.executemany(
                    "INSERT INTO questions (question_id) VALUES (?)",
                    ((question_id,) for question_id in self.contract.question_ids),
                )
            elif row != (self.contract.fingerprint, contract_json):
                raise CheckpointConflictError(
                    "checkpoint belongs to a different immutable run contract"
                )

    def _validate_unit(self, question_id: int, repeat_index: int) -> None:
        if (
            question_id not in self._expected_question_ids
            or not 0 <= repeat_index < self.contract.repeats
        ):
            raise CheckpointConflictError(
                "record unit is outside run contract: "
                f"({question_id}, {repeat_index})"
            )

    def _record(
        self,
        table: str,
        key: tuple[object, ...],
        record_json: str,
        *,
        judge_contract_hash: str | None = None,
    ) -> None:
        if table == "candidates":
            query = (
                "SELECT record_json FROM candidates "
                "WHERE question_id = ? AND repeat_index = ?"
            )
            insert = (
                "INSERT INTO candidates (question_id, repeat_index, record_json) "
                "VALUES (?, ?, ?)"
            )
            parameters = (*key, record_json)
        elif table in {"judgments", "judge_failures"}:
            query = (
                f"SELECT record_json FROM {table} WHERE question_id = ? "
                "AND repeat_index = ? AND judge_contract_hash = ?"
            )
            insert = (
                f"INSERT INTO {table} "
                "(question_id, repeat_index, judge_contract_hash, record_json) "
                "VALUES (?, ?, ?, ?)"
            )
            assert judge_contract_hash is not None
            parameters = (*key, judge_contract_hash, record_json)
            key = (*key, judge_contract_hash)
        else:
            raise ValueError(f"unknown immutable record table: {table}")
        with self._lock, self._connection:
            row = self._connection.execute(query, key).fetchone()
            if row is None:
                try:
                    self._connection.execute(insert, parameters)
                except sqlite3.IntegrityError as exc:
                    raise CheckpointConflictError(
                        f"invalid immutable {table[:-1]} key: {key!r}"
                    ) from exc
            elif row[0] != record_json:
                raise CheckpointConflictError(
                    f"conflicting immutable {table[:-1]} record for {key!r}"
                )

    def missing_candidates(self) -> list[tuple[int, int]]:
        with self._lock:
            return [
                (question_id, repeat_index)
                for question_id in self.contract.question_ids
                for repeat_index in range(self.contract.repeats)
                if self._connection.execute(
                    "SELECT 1 FROM candidates WHERE question_id = ? AND repeat_index = ?",
                    (question_id, repeat_index),
                ).fetchone()
                is None
            ]

    def record_candidate(self, record: CandidateRecord) -> None:
        self._validate_unit(record.question_id, record.repeat_index)
        self._record(
            "candidates",
            (record.question_id, record.repeat_index),
            _canonical_dataclass(record),
        )

    def record_server_snapshot(self, stage: str, identity: Mapping[str, object]) -> None:
        """Persist an immutable, redacted server-identity snapshot."""
        payload_json = canonical_json(identity)
        with self._lock, self._connection:
            row = self._connection.execute(
                "SELECT payload_json FROM server_snapshots WHERE stage = ?", (stage,)
            ).fetchone()
            if row is None:
                self._connection.execute(
                    "INSERT INTO server_snapshots (stage, payload_json) VALUES (?, ?)",
                    (stage, payload_json),
                )
            elif row[0] != payload_json:
                raise CheckpointConflictError(
                    f"conflicting immutable server snapshot for {stage!r}"
                )

    def server_snapshot(self, stage: str) -> dict[str, object] | None:
        """Return a persisted server-identity snapshot, if one exists."""
        with self._lock:
            row = self._connection.execute(
                "SELECT payload_json FROM server_snapshots WHERE stage = ?", (stage,)
            ).fetchone()
        return json.loads(row[0]) if row is not None else None

    def missing_judgments(self, judge_contract_hash: str) -> list[tuple[int, int]]:
        with self._lock:
            return [
                (question_id, repeat_index)
                for question_id in self.contract.question_ids
                for repeat_index in range(self.contract.repeats)
                if self._connection.execute(
                    "SELECT 1 FROM candidates WHERE question_id = ? AND repeat_index = ?",
                    (question_id, repeat_index),
                ).fetchone()
                is not None
                and self._connection.execute(
                    """
                    SELECT 1 FROM judgments
                    WHERE question_id = ? AND repeat_index = ? AND judge_contract_hash = ?
                    """,
                    (question_id, repeat_index, judge_contract_hash),
                ).fetchone()
                is None
            ]

    def record_judgment(self, record: JudgmentRecord) -> None:
        if record.verdict not in {"CORRECT", "INCORRECT"}:
            raise CheckpointConflictError(
                "only a valid CORRECT/INCORRECT record may occupy a judgment key"
            )
        self._validate_unit(record.question_id, record.repeat_index)
        self._record(
            "judgments",
            (record.question_id, record.repeat_index),
            _canonical_dataclass(record),
            judge_contract_hash=record.judge_contract_hash,
        )

    def record_judge_failure(self, record: JudgeFailureRecord) -> None:
        """Append one immutable terminal failure event for an unscored unit."""
        self._validate_unit(record.question_id, record.repeat_index)
        self._record(
            "judge_failures",
            (record.question_id, record.repeat_index),
            _canonical_dataclass(record),
            judge_contract_hash=record.judge_contract_hash,
        )

    def judge_failures(self) -> list[JudgeFailureRecord]:
        """Return terminal judge failures which leave the run incomplete."""
        with self._lock:
            rows = self._connection.execute(
                "SELECT record_json FROM judge_failures ORDER BY question_id, repeat_index"
            ).fetchall()
        return [JudgeFailureRecord(**json.loads(row[0])) for row in rows]

    def incomplete_judgments(self, judge_contract_hash: str) -> list[tuple[int, int]]:
        """Return terminal failures for this judge contract."""
        with self._lock:
            return [
                (question_id, repeat_index)
                for question_id, repeat_index in self._connection.execute(
                    """
                    SELECT question_id, repeat_index FROM judge_failures
                    WHERE judge_contract_hash = ?
                    ORDER BY question_id, repeat_index
                    """,
                    (judge_contract_hash,),
                )
            ]

    def candidate_record(self, question_id: int, repeat_index: int) -> CandidateRecord:
        """Return one persisted candidate required for a missing judgment."""
        self._validate_unit(question_id, repeat_index)
        with self._lock:
            row = self._connection.execute(
                "SELECT record_json FROM candidates WHERE question_id = ? AND repeat_index = ?",
                (question_id, repeat_index),
            ).fetchone()
        if row is None:
            raise CheckpointConflictError(
                f"candidate is absent for judgment unit: ({question_id}, {repeat_index})"
            )
        payload = json.loads(row[0])
        if not isinstance(payload, dict):
            raise CheckpointConflictError("persisted candidate record is malformed")
        return CandidateRecord(**payload)


def canonical_json(value: object) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _canonical_dataclass(value: object) -> str:
    fields = getattr(value, "__dataclass_fields__")
    return canonical_json(
        {
            name: _json_value(getattr(value, name))
            for name in fields
        }
    )


def _json_value(value: object) -> object:
    if isinstance(value, Mapping):
        return {str(key): _json_value(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return [_json_value(item) for item in value]
    if isinstance(value, list):
        return [_json_value(item) for item in value]
    return value


def sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def build_candidate_prompt(documents: list[str], question: str) -> str:
    documents_text = "\n\n".join(
        f"BEGIN DOCUMENT {i + 1}:\n{doc}\nEND DOCUMENT {i + 1}"
        for i, doc in enumerate(documents)
    )
    return CANDIDATE_PROMPT_TEMPLATE.format(
        documents_text=documents_text,
        question=question,
    )


def build_judge_user_prompt(
    question: str,
    official_answer: str,
    candidate_answer: str,
) -> str:
    return JUDGE_USER_PROMPT_TEMPLATE.format(
        question=question,
        official_answer=official_answer,
        candidate_answer=candidate_answer,
    )


def parse_judge_verdict(text: str) -> str:
    try:
        value = json.loads(text)
    except json.JSONDecodeError as exc:
        raise JudgeProtocolError("judge output is not JSON") from exc
    if not isinstance(value, dict) or set(value) != {"verdict"}:
        raise JudgeProtocolError("judge JSON must contain only verdict")
    verdict = value["verdict"]
    if not isinstance(verdict, str) or verdict.upper() not in {
        "CORRECT",
        "INCORRECT",
    }:
        raise JudgeProtocolError("judge verdict must be CORRECT or INCORRECT")
    return verdict.upper()


def build_openai_client() -> "OpenAI":
    """Construct the official OpenAI client only when a credential is configured."""
    key = os.environ.get("OPENAI_API_KEY")
    if not key:
        raise CredentialError("OPENAI_API_KEY is not set")
    try:
        openai = importlib.import_module("openai")
    except ImportError as exc:
        raise CredentialError("openai package is required to run the judge") from exc
    return openai.OpenAI(api_key=key, timeout=300.0, max_retries=0)


def _judge_contract_hash(
) -> str:
    return sha256_text(
        canonical_json(
            {
                "model": JUDGE_MODEL,
                "reasoning_effort": JUDGE_REASONING_EFFORT,
                "reasoning_mode": JUDGE_REASONING_MODE,
                "system_prompt_sha256": sha256_text(JUDGE_SYSTEM_PROMPT),
                "user_prompt_sha256": sha256_text(JUDGE_USER_PROMPT_TEMPLATE),
            }
        )
    )


def _validate_judge_request_contract(
    *,
    judge_model: str,
    judge_reasoning_effort: str,
    judge_reasoning_mode: str,
    judge_system_prompt_sha256: str,
    judge_user_prompt_sha256: str,
    judge_max_attempts: int,
) -> None:
    expected = {
        "judge_model": JUDGE_MODEL,
        "judge_reasoning_effort": JUDGE_REASONING_EFFORT,
        "judge_reasoning_mode": JUDGE_REASONING_MODE,
        "judge_system_prompt_sha256": sha256_text(JUDGE_SYSTEM_PROMPT),
        "judge_user_prompt_sha256": sha256_text(JUDGE_USER_PROMPT_TEMPLATE),
        "judge_max_attempts": MAX_ATTEMPTS,
    }
    actual = {
        "judge_model": judge_model,
        "judge_reasoning_effort": judge_reasoning_effort,
        "judge_reasoning_mode": judge_reasoning_mode,
        "judge_system_prompt_sha256": judge_system_prompt_sha256,
        "judge_user_prompt_sha256": judge_user_prompt_sha256,
        "judge_max_attempts": judge_max_attempts,
    }
    if actual != expected:
        mismatched = ", ".join(
            name for name in expected if actual[name] != expected[name]
        )
        raise JudgeContractError(
            "judge run contract differs from pinned AA-LCR v1.1: " + mismatched
        )


def validate_judge_contract(contract: RunContract) -> str:
    """Validate the exact judge request contract before any judge API call."""
    _validate_judge_request_contract(
        judge_model=contract.judge_model,
        judge_reasoning_effort=contract.judge_reasoning_effort,
        judge_reasoning_mode=contract.judge_reasoning_mode,
        judge_system_prompt_sha256=contract.judge_system_prompt_sha256,
        judge_user_prompt_sha256=contract.judge_user_prompt_sha256,
        judge_max_attempts=contract.judge_max_attempts,
    )
    return _judge_contract_hash()


def _response_value(response: object, name: str, default: object = None) -> object:
    return getattr(response, name, default)


def _sdk_version() -> str | None:
    return getattr(sys.modules.get("openai"), "__version__", None)


def _safe_request_id(value: object) -> str | None:
    """Keep public request IDs, never a value that resembles an API key."""
    if not isinstance(value, str) or value.startswith("sk-"):
        return None
    return value


def _usage_value(usage: object) -> object:
    if usage is None:
        return None
    if isinstance(usage, Mapping):
        return {str(key): _json_value(value) for key, value in usage.items()}
    model_dump = getattr(usage, "model_dump", None)
    if callable(model_dump):
        return _usage_value(model_dump())
    values = {
        name: getattr(usage, name)
        for name in ("input_tokens", "output_tokens", "total_tokens")
        if hasattr(usage, name)
    }
    return values or None


def _judge_error_record(
    question: Question,
    candidate: CandidateRecord,
    judge_contract_hash: str,
    retry_count: int,
    started_at_utc: str,
    error: str,
    raw_output_text: str | None = None,
    exc: Exception | None = None,
    response: object | None = None,
    attempt_count: int = 0,
) -> JudgmentRecord:
    return JudgmentRecord(
        question_id=question.question_id,
        repeat_index=candidate.repeat_index,
        judge_contract_hash=judge_contract_hash,
        raw_response=(
            {"output_text": raw_output_text} if raw_output_text is not None else None
        ),
        verdict=None,
        retry_count=retry_count,
        started_at_utc=started_at_utc,
        completed_at_utc=_utc_now(),
        error=error,
        requested_model=JUDGE_MODEL,
        returned_model=_response_value(response, "model"),
        reasoning_effort=JUDGE_REASONING_EFFORT,
        reasoning_mode=JUDGE_REASONING_MODE,
        openai_sdk_version=_sdk_version(),
        response_id=_response_value(response, "id"),
        request_id=_safe_request_id(
            getattr(exc, "request_id", _response_value(response, "_request_id"))
        ),
        usage=_usage_value(_response_value(response, "usage")),
        raw_output_text=raw_output_text,
        exception_class=type(exc).__name__ if exc is not None else None,
        http_status=_exception_status(exc) if exc is not None else None,
        attempt_count=attempt_count,
    )


def _exception_status(exc: BaseException) -> int | None:
    status = getattr(exc, "status_code", getattr(exc, "status", None))
    return status if isinstance(status, int) else None


def _is_retryable_judge_exception(exc: Exception) -> bool:
    """Retry only protocol, transport, timeout, and retryable API-status failures."""
    if isinstance(exc, (JudgeProtocolError, OSError, TimeoutError, URLError)):
        return True
    openai = sys.modules.get("openai")
    sdk_transport_types = tuple(
        exception_type
        for name in (
            "APIConnectionError",
            "APITimeoutError",
            "RateLimitError",
            "InternalServerError",
        )
        if isinstance((exception_type := getattr(openai, name, None)), type)
    )
    if sdk_transport_types and isinstance(exc, sdk_transport_types):
        return True
    status = _exception_status(exc)
    return status is not None and _is_retryable_status(status)


def _is_known_api_exception(exc: Exception) -> bool:
    """Recognize status-bearing OpenAI API errors without importing its SDK eagerly."""
    return _exception_status(exc) is not None


def _judge_failure_record(record: JudgmentRecord) -> JudgeFailureRecord:
    assert record.error is not None
    assert record.exception_class is not None
    return JudgeFailureRecord(
        question_id=record.question_id,
        repeat_index=record.repeat_index,
        judge_contract_hash=record.judge_contract_hash,
        requested_model=record.requested_model or JUDGE_MODEL,
        reasoning_effort=record.reasoning_effort or JUDGE_REASONING_EFFORT,
        reasoning_mode=record.reasoning_mode or JUDGE_REASONING_MODE,
        openai_sdk_version=record.openai_sdk_version,
        exception_class=record.exception_class,
        http_status=record.http_status,
        request_id=record.request_id,
        retry_count=record.retry_count,
        attempt_count=record.attempt_count,
        started_at_utc=record.started_at_utc,
        completed_at_utc=record.completed_at_utc,
        raw_output_text=record.raw_output_text,
        returned_model=record.returned_model,
        response_id=record.response_id,
        usage=record.usage,
    )


def judge_one(
    client: object,
    question: Question,
    candidate: CandidateRecord,
    *,
    judge_contract_hash: str | None = None,
    max_attempts: int = MAX_ATTEMPTS,
) -> JudgmentRecord:
    """Judge final candidate content, retrying transient failures independently."""
    if not 1 <= max_attempts <= MAX_ATTEMPTS:
        raise ValueError(f"max_attempts must be between 1 and {MAX_ATTEMPTS}")
    started_at_utc = _utc_now()
    contract_hash = judge_contract_hash or _judge_contract_hash()
    if candidate.content is None:
        error = JudgeProtocolError("candidate has no final content")
        return _judge_error_record(
            question, candidate, contract_hash, 0, started_at_utc,
            "candidate has no final content",
            exc=error,
            attempt_count=0,
        )
    judge_user_prompt = build_judge_user_prompt(
        question.question, question.official_answer, candidate.content
    )
    for attempt in range(max_attempts):
        response: object | None = None
        output_text: str | None = None
        try:
            response = client.responses.create(
                model=JUDGE_MODEL,
                input=[
                    {"role": "system", "content": JUDGE_SYSTEM_PROMPT},
                    {"role": "user", "content": judge_user_prompt},
                ],
                reasoning={
                    "effort": JUDGE_REASONING_EFFORT,
                    "mode": JUDGE_REASONING_MODE,
                },
                store=False,
            )
            output_text = _response_value(response, "output_text")
            if not isinstance(output_text, str):
                raise JudgeProtocolError("judge response has no output text")
            verdict = parse_judge_verdict(output_text)
        except Exception as exc:
            if _is_retryable_judge_exception(exc) and attempt + 1 < max_attempts:
                time.sleep(_retry_delay(attempt))
                continue
            if not isinstance(exc, JudgeProtocolError) and not _is_known_api_exception(exc):
                raise
            status = _exception_status(exc)
            status_text = f" HTTP {status}" if status is not None else ""
            return _judge_error_record(
                question, candidate, contract_hash, attempt, started_at_utc,
                f"judge failed after {attempt + 1} attempts: {type(exc).__name__}{status_text}",
                output_text,
                exc,
                response,
                attempt + 1,
            )
        return JudgmentRecord(
            question_id=question.question_id,
            repeat_index=candidate.repeat_index,
            judge_contract_hash=contract_hash,
            raw_response={"output_text": output_text},
            verdict=verdict,
            retry_count=attempt,
            started_at_utc=started_at_utc,
            completed_at_utc=_utc_now(),
            error=None,
            requested_model=JUDGE_MODEL,
            returned_model=_response_value(response, "model"),
            reasoning_effort=JUDGE_REASONING_EFFORT,
            reasoning_mode=JUDGE_REASONING_MODE,
            openai_sdk_version=_sdk_version(),
            response_id=_response_value(response, "id"),
            request_id=_response_value(response, "_request_id"),
            usage=_usage_value(_response_value(response, "usage")),
            raw_output_text=output_text,
            attempt_count=attempt + 1,
        )
    raise AssertionError("unreachable")


def preflight_judge(client: object) -> dict[str, object]:
    """Confirm the requested judge model is reachable and obeys the verdict protocol."""
    model = client.models.retrieve(JUDGE_MODEL)
    model_id = _response_value(model, "id")
    if model_id != JUDGE_MODEL:
        raise JudgeProtocolError("judge preflight retrieved unrelated model identity")
    question = Question(
        question_id=0,
        category="preflight",
        document_set_id="preflight",
        question="What is 2 + 2?",
        official_answer="4",
        document_filenames=(),
        prompt="",
        cl100k_tokens=0,
    )
    candidate = CandidateRecord(
        question_id=0, repeat_index=0, http_status=200, raw_response=None,
        usage=None, finish_reason=None, content="4", reasoning_content=None,
        retry_count=0, started_at_utc=_utc_now(), completed_at_utc=_utc_now(),
        error=None,
    )
    record = judge_one(client, question, candidate)
    if record.verdict != "CORRECT":
        raise JudgeProtocolError("judge preflight did not return CORRECT")
    judgment = json.loads(_canonical_dataclass(record))
    judgment["retrieved_model"] = model_id
    return {
        "requested_model": JUDGE_MODEL,
        "model_identity": _json_value(vars(model)),
        "judgment": judgment,
    }


def judge_missing(
    checkpoint: Checkpoint,
    questions: Collection[Question],
    client: object,
) -> None:
    """Persist judgments only for currently absent candidates under this judge contract."""
    question_by_id = {question.question_id: question for question in questions}
    judge_contract_hash = validate_judge_contract(checkpoint.contract)
    incomplete = checkpoint.incomplete_judgments(judge_contract_hash)
    if incomplete:
        raise IncompleteJudgmentError(
            f"judge run is incomplete for terminal failures: {incomplete}"
        )
    missing = checkpoint.missing_judgments(judge_contract_hash)
    if any(question_id not in question_by_id for question_id, _ in missing):
        raise CheckpointConflictError("questions do not cover immutable run contract")
    for question_id, repeat_index in missing:
        candidate = checkpoint.candidate_record(question_id, repeat_index)
        record = judge_one(
            client,
            question_by_id[question_id],
            candidate,
            judge_contract_hash=judge_contract_hash,
            max_attempts=checkpoint.contract.judge_max_attempts,
        )
        if record.verdict in {"CORRECT", "INCORRECT"}:
            checkpoint.record_judgment(record)
        else:
            checkpoint.record_judge_failure(_judge_failure_record(record))
            raise IncompleteJudgmentError(
                f"judge run is incomplete for terminal failure: "
                f"({question_id}, {repeat_index})"
            )


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def candidate_request(question: Question) -> dict[str, object]:
    """Return the pinned OpenAI-compatible candidate request."""
    return {
        "model": "glm-5.3-w4afp8",
        "messages": [{"role": "user", "content": question.prompt}],
        "max_tokens": CANDIDATE_MAX_TOKENS,
        "temperature": CANDIDATE_TEMPERATURE,
        "top_p": CANDIDATE_TOP_P,
        "chat_template_kwargs": {"enable_thinking": True},
    }


def _response_json(response: object) -> object:
    payload = response.read()
    try:
        return json.loads(payload.decode("utf-8"))
    except (AttributeError, UnicodeDecodeError, json.JSONDecodeError):
        return None


def _candidate_record(
    question_id: int,
    repeat_index: int,
    http_status: int,
    raw_response: object,
    retry_count: int,
    started_at_utc: str,
    *,
    error: str | None = None,
) -> CandidateRecord:
    completed_at_utc = _utc_now()
    if not isinstance(raw_response, dict):
        return CandidateRecord(
            question_id, repeat_index, http_status, raw_response, None, None, None,
            None, retry_count, started_at_utc, completed_at_utc,
            error or "malformed model response: response is not a JSON object",
        )
    choices = raw_response.get("choices")
    if not isinstance(choices, list) or not choices:
        return CandidateRecord(
            question_id, repeat_index, http_status, raw_response,
            raw_response.get("usage"), None, None, None, retry_count,
            started_at_utc, completed_at_utc,
            error or "malformed model response: missing choices[0].message",
        )
    choice = choices[0]
    message = choice.get("message") if isinstance(choice, dict) else None
    if not isinstance(message, dict):
        return CandidateRecord(
            question_id, repeat_index, http_status, raw_response,
            raw_response.get("usage"), None, None, None, retry_count,
            started_at_utc, completed_at_utc,
            error or "malformed model response: missing choices[0].message",
        )
    content = message.get("content")
    reasoning_content = message.get("reasoning_content")
    if content is not None and not isinstance(content, str):
        error = error or "malformed model response: message.content is not a string"
        content = None
    if reasoning_content is not None and not isinstance(reasoning_content, str):
        error = error or "malformed model response: reasoning_content is not a string"
        reasoning_content = None
    return CandidateRecord(
        question_id=question_id,
        repeat_index=repeat_index,
        http_status=http_status,
        raw_response=raw_response,
        usage=raw_response.get("usage"),
        finish_reason=choice.get("finish_reason") if isinstance(choice, dict) else None,
        content=content,
        reasoning_content=reasoning_content,
        retry_count=retry_count,
        started_at_utc=started_at_utc,
        completed_at_utc=completed_at_utc,
        error=error,
    )


def candidate_from_response(
    question_id: int,
    repeat_index: int,
    http_status: int,
    response: object,
) -> CandidateRecord:
    """Build a terminal record, retaining reasoning apart from final content."""
    return _candidate_record(
        question_id, repeat_index, http_status, response, 0, _utc_now()
    )


def _retry_delay(retry_count: int) -> float:
    return min(0.25 * (2**retry_count), 5.0)


def _is_retryable_status(status: int) -> bool:
    return status in {408, 409, 429} or 500 <= status <= 599


def generate_one(
    question: Question,
    repeat_index: int,
    base_url: str,
    *,
    max_attempts: int = MAX_ATTEMPTS,
) -> CandidateRecord:
    """Generate one candidate, retrying only transport and explicitly retryable HTTP."""
    if max_attempts < 1:
        raise ValueError("max_attempts must be at least one")
    started_at_utc = _utc_now()
    request = Request(
        f"{base_url.rstrip('/')}/v1/chat/completions",
        data=canonical_json(candidate_request(question)).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    for attempt in range(max_attempts):
        try:
            with urlopen(request, timeout=120) as response:
                status = int(getattr(response, "status", 200))
                raw_response = _response_json(response)
        except HTTPError as exc:
            status = exc.code
            raw_response = _response_json(exc)
        except (OSError, TimeoutError, URLError) as exc:
            if attempt + 1 == max_attempts:
                return _candidate_record(
                    question.question_id, repeat_index, 0, None, attempt,
                    started_at_utc,
                    error=(
                        f"transport error after {max_attempts} attempts: "
                        f"{type(exc).__name__}"
                    ),
                )
            time.sleep(_retry_delay(attempt))
            continue

        if _is_retryable_status(status) and attempt + 1 < max_attempts:
            time.sleep(_retry_delay(attempt))
            continue
        error = None
        if status >= 400:
            error = f"HTTP {status} response"
        return _candidate_record(
            question.question_id, repeat_index, status, raw_response, attempt,
            started_at_utc, error=error,
        )
    raise AssertionError("unreachable")


def fetch_server_identity(base_url: str) -> dict[str, object]:
    """Fetch the SGLang deployment fields which define serving identity."""
    def get_json(path: str) -> object:
        with urlopen(f"{base_url.rstrip('/')}{path}", timeout=30) as response:
            if int(getattr(response, "status", 200)) != 200:
                raise CheckpointConflictError(f"server identity endpoint {path} failed")
            payload = _response_json(response)
        if not isinstance(payload, dict):
            raise CheckpointConflictError(f"server identity endpoint {path} was malformed")
        return payload

    info = get_json("/get_server_info")
    models = get_json("/v1/models")
    assert isinstance(info, dict) and isinstance(models, dict)
    data = models.get("data")
    served_model = (
        data[0].get("id")
        if isinstance(data, list) and data and isinstance(data[0], dict)
        else None
    )
    speculative_aliases = {
        "speculative_algorithm",
        "num_steps",
        "eagle_topk",
        "num_draft_tokens",
        "draft_model_path",
    }
    speculative_settings = {
        name: info[name]
        for name in sorted(info)
        if name.startswith("speculative_") or name in speculative_aliases
    }
    return {
        "model_path": info.get("model_path"),
        "served_model": served_model,
        "tp_size": info.get("tp_size"),
        "max_total_num_tokens": info.get("max_total_num_tokens"),
        "context_length": info.get("context_length"),
        "quantization": info.get("quantization"),
        "kv_cache_dtype": info.get("kv_cache_dtype"),
        "reasoning_parser": info.get("reasoning_parser"),
        "speculative_settings": speculative_settings,
    }


def generate_missing(
    checkpoint: Checkpoint,
    questions: Collection[Question],
    base_url: str,
    *,
    repeats: int,
) -> None:
    """Fill only absent candidate units and reject a changing SGLang deployment."""
    if repeats != checkpoint.contract.repeats:
        raise CheckpointConflictError("repeats does not match immutable run contract")
    question_by_id = {question.question_id: question for question in questions}
    missing = checkpoint.missing_candidates()
    if any(question_id not in question_by_id for question_id, _ in missing):
        raise CheckpointConflictError("questions do not cover immutable run contract")
    contract_identity = dict(checkpoint.contract.endpoint_deployment_identity)
    if not contract_identity:
        raise CheckpointConflictError(
            "immutable run contract endpoint deployment identity must not be empty"
        )
    before_identity = fetch_server_identity(base_url)
    checkpoint.record_server_snapshot("before", before_identity)
    if before_identity != contract_identity:
        raise CheckpointConflictError("server identity does not match immutable run contract")

    def generate_and_record(question_id: int, repeat_index: int) -> None:
        record = generate_one(
            question_by_id[question_id],
            repeat_index,
            base_url,
            max_attempts=checkpoint.contract.candidate_max_attempts,
        )
        checkpoint.record_candidate(record)

    worker_error: BaseException | None = None
    try:
        with ThreadPoolExecutor(max_workers=CANDIDATE_CONCURRENCY) as executor:
            futures = [
                executor.submit(generate_and_record, question_id, repeat_index)
                for question_id, repeat_index in missing
            ]
            for future in futures:
                future.result()
    except BaseException as exc:
        worker_error = exc

    try:
        after_identity = fetch_server_identity(base_url)
        checkpoint.record_server_snapshot("after", after_identity)
        if after_identity != before_identity:
            raise CheckpointConflictError("server identity changed during candidate generation")
    except BaseException as post_identity_error:
        if worker_error is not None:
            raise worker_error from post_identity_error
        raise
    if worker_error is not None:
        raise worker_error


def _safe_member_path(name: str, *, is_directory: bool) -> PurePosixPath:
    components = name.split("/")
    if is_directory and components[-1:] == [""]:
        components.pop()
    if (
        not name
        or "\x00" in name
        or "\\" in name
        or name.startswith("/")
        or (len(name) >= 2 and name[0].isalpha() and name[1] == ":")
        or not components
        or any(part in ("", ".", "..") for part in components)
    ):
        raise DatasetIntegrityError(f"unsafe archive member: {name!r}")
    return PurePosixPath(*components)


def _member_path(member: zipfile.ZipInfo) -> PurePosixPath:
    is_directory = member.is_dir()
    path = _safe_member_path(member.filename, is_directory=is_directory)
    unix_type = stat.S_IFMT(member.external_attr >> 16)
    allowed_type = stat.S_IFDIR if is_directory else stat.S_IFREG
    if unix_type not in (0, allowed_type):
        raise DatasetIntegrityError(f"unsafe archive member: {member.filename!r}")
    return path


def _member_has_nul_name(archive: zipfile.ZipFile, member: zipfile.ZipInfo) -> bool:
    """Inspect the local-header filename before zipfile truncates a NUL suffix."""
    assert archive.fp is not None
    archive.fp.seek(member.header_offset + 26)
    filename_length = int.from_bytes(archive.fp.read(2), "little")
    extra_length = int.from_bytes(archive.fp.read(2), "little")
    if filename_length == 0 or extra_length < 0:
        raise DatasetIntegrityError(f"unsafe archive member: {member.filename!r}")
    return b"\x00" in archive.fp.read(filename_length)


def safe_extract_zip(
    zip_path: Path,
    destination: Path,
    *,
    expected_members: Collection[str] | None = None,
) -> list[Path]:
    """Extract a ZIP only after validating every archive member."""
    expected_paths = (
        {
            _safe_member_path(member, is_directory=False).as_posix()
            for member in expected_members
        }
        if expected_members is not None
        else None
    )
    expected_directories = (
        {
            parent.as_posix()
            for path in expected_paths or set()
            for parent in PurePosixPath(path).parents
            if parent != PurePosixPath(".")
        }
        if expected_paths is not None
        else set()
    )
    with zipfile.ZipFile(zip_path) as archive:
        members: list[tuple[zipfile.ZipInfo, PurePosixPath]] = []
        names: set[str] = set()
        extracted_file_names: set[str] = set()
        for member in archive.infolist():
            if _member_has_nul_name(archive, member):
                raise DatasetIntegrityError(f"unsafe archive member: {member.filename!r}")
            path = _member_path(member)
            normalized = path.as_posix()
            if normalized in names:
                raise DatasetIntegrityError(
                    f"duplicate archive member: {member.filename!r}"
                )
            names.add(normalized)
            if expected_paths is not None:
                allowed = expected_directories if member.is_dir() else expected_paths
                if normalized not in allowed:
                    raise DatasetIntegrityError(
                        f"unexpected archive member: {member.filename!r}"
                    )
                if not member.is_dir():
                    extracted_file_names.add(normalized)
            members.append((member, path))

        destination.mkdir(parents=True, exist_ok=True)
        resolved_destination = destination.resolve()
        extracted: list[Path] = []
        for member, path in members:
            target = destination.joinpath(*path.parts)
            resolved_parent = target.parent.resolve()
            resolved_target = target.resolve(strict=False)
            if not (
                resolved_parent.is_relative_to(resolved_destination)
                and resolved_target.is_relative_to(resolved_destination)
            ):
                raise DatasetIntegrityError(f"unsafe archive member: {member.filename!r}")
            target.parent.mkdir(parents=True, exist_ok=True)
            if member.is_dir():
                target.mkdir(exist_ok=True)
            else:
                with archive.open(member) as source, target.open("wb") as output:
                    output.write(source.read())
                extracted.append(target)
    if expected_paths is not None and extracted_file_names != expected_paths:
        raise DatasetIntegrityError("archive is missing expected document members")
    return extracted


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _download_file(
    destination: Path,
    remote_path: str,
    opener: Callable[..., object],
    expected_sha256: str | None = None,
) -> str:
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="wb",
            dir=destination.parent,
            prefix=f".{destination.name}.",
            delete=False,
        ) as temporary:
            temporary_path = Path(temporary.name)
            with opener(f"{HF_RESOLVE}/{remote_path}") as response:
                status = getattr(response, "status", None)
                if status != 200:
                    raise DatasetIntegrityError(
                        f"download failed for {remote_path}: HTTP {status}"
                    )
                while chunk := response.read(1024 * 1024):
                    temporary.write(chunk)
        assert temporary_path is not None
        digest = _sha256_file(temporary_path)
        if expected_sha256 is not None and digest != expected_sha256:
            raise DatasetIntegrityError(f"SHA-256 mismatch for {remote_path}")
        os.replace(temporary_path, destination)
        return digest
    except Exception:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)
        raise


def _row_value(row: Mapping[str, str], *names: str) -> str:
    normalized = {
        key.strip().lower().replace(" ", "_"): value.strip()
        for key, value in row.items()
        if key is not None and value is not None
    }
    for name in names:
        if value := normalized.get(name):
            return value
    raise DatasetIntegrityError(f"CSV is missing a value for {names[0]}")


def _read_dataset_rows(csv_path: Path) -> list[dict[str, str]]:
    with csv_path.open(encoding="utf-8-sig", newline="") as source:
        return list(csv.DictReader(source))


def _document_member_paths(rows: Collection[Mapping[str, str]]) -> set[str]:
    members: set[str] = set()
    for row in rows:
        category = _row_value(row, "document_category", "category")
        document_set_id = _row_value(row, "document_set_id", "data_source_id")
        filenames = _row_value(row, "data_source_filenames").split(";")
        for filename in filenames:
            members.add(
                _safe_member_path(
                    f"lcr/{category}/{document_set_id}/{filename.strip()}",
                    is_directory=False,
                ).as_posix()
            )
    return members


def _load_questions(
    documents_root: Path, rows: Collection[Mapping[str, str]]
) -> tuple[Question, ...]:
    try:
        encoding = tiktoken.get_encoding("cl100k_base")
    except Exception as exc:
        raise DatasetIntegrityError("cl100k_base tokenizer is unavailable") from exc

    questions: list[Question] = []
    for row in rows:
        try:
            question_id = int(_row_value(row, "question_id", "id"))
            document_category = _row_value(row, "document_category", "category")
            document_set_id = _row_value(row, "document_set_id", "data_source_id")
            filenames = tuple(
                filename.strip()
                for filename in _row_value(row, "data_source_filenames").split(";")
                if filename.strip()
            )
            if not filenames:
                raise DatasetIntegrityError("question has no document filenames")
            documents = []
            for filename in filenames:
                member = _safe_member_path(
                    f"lcr/{document_category}/{document_set_id}/{filename}",
                    is_directory=False,
                )
                document_path = documents_root.joinpath(*member.parts)
                if not document_path.is_file():
                    raise DatasetIntegrityError(
                        f"referenced document is missing: {member.as_posix()}"
                    )
                documents.append(document_path.read_text(encoding="utf-8"))
            question_text = _row_value(row, "question")
            prompt = build_candidate_prompt(documents, question_text)
            tokens = len(encoding.encode(prompt))
            expected_tokens = int(_row_value(row, "input_tokens"))
            if tokens != expected_tokens:
                raise DatasetIntegrityError(
                    f"cl100k_base token mismatch for question {question_id}"
                )
            questions.append(
                Question(
                    question_id=question_id,
                    category=_row_value(row, "category"),
                    document_set_id=document_set_id,
                    question=question_text,
                    official_answer=_row_value(
                        row, "official_answer", "answer"
                    ),
                    document_filenames=filenames,
                    prompt=prompt,
                    cl100k_tokens=tokens,
                )
            )
        except (TypeError, ValueError) as exc:
            if isinstance(exc, DatasetIntegrityError):
                raise
            raise DatasetIntegrityError("invalid dataset CSV row") from exc
    return tuple(questions)


def validate_questions(questions: list[Question] | tuple[Question, ...]) -> None:
    question_ids = [question.question_id for question in questions]
    if set(question_ids) != set(range(1, 101)) or len(question_ids) != 100:
        raise DatasetIntegrityError("dataset must contain question IDs 1..100")
    if len({question.document_set_id for question in questions}) != 30:
        raise DatasetIntegrityError("dataset must contain 30 document sets")


def prepare_dataset(
    work_dir: Path, opener: Callable[..., object] = urlopen
) -> PreparedDataset:
    """Download, validate, and prepare the immutable pinned AA-LCR v1.1 dataset."""
    work_dir = Path(work_dir)
    csv_path = work_dir / CSV_FILENAME
    zip_path = work_dir / Path(ZIP_FILENAME).name
    documents_root = work_dir / "extracted_text"
    digests = {
        CSV_FILENAME: _download_file(csv_path, CSV_FILENAME, opener),
        ZIP_FILENAME: _download_file(zip_path, ZIP_FILENAME, opener, ZIP_SHA256),
    }
    rows = _read_dataset_rows(csv_path)
    extracted_paths = safe_extract_zip(
        zip_path,
        documents_root,
        expected_members=_document_member_paths(rows),
    )
    for path in extracted_paths:
        digests[path.relative_to(documents_root).as_posix()] = _sha256_file(path)
    questions = _load_questions(documents_root, rows)
    validate_questions(questions)
    prompt_sha256 = sha256_text(
        canonical_json([question.prompt for question in questions])
    )
    return PreparedDataset(
        revision=DATASET_REVISION,
        questions=questions,
        file_sha256=MappingProxyType(digests),
        prompt_sha256=prompt_sha256,
    )
