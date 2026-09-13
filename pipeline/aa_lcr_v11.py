"""Pinned AA-LCR v1.1 identity and prompt contract."""

import csv
import hashlib
import json
import os
import sqlite3
import stat
import tempfile
import threading
import zipfile
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from types import MappingProxyType
from typing import Callable, Collection, Mapping
from urllib.request import urlopen

import tiktoken


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
    judge_max_attempts: int = MAX_ATTEMPTS,
    repeats: int = REPEATS,
) -> RunContract:
    """Build the immutable identity contract before any endpoint traffic."""
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
    )


class Checkpoint:
    """Durable, immutable SQLite storage for a single run contract."""

    def __init__(self, path: Path, contract: RunContract) -> None:
        self.path = Path(path)
        self.contract = contract
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
                    repeat_index INTEGER NOT NULL,
                    record_json TEXT NOT NULL,
                    PRIMARY KEY (question_id, repeat_index)
                );
                CREATE TABLE IF NOT EXISTS judgments (
                    question_id INTEGER NOT NULL,
                    repeat_index INTEGER NOT NULL,
                    judge_contract_hash TEXT NOT NULL,
                    record_json TEXT NOT NULL,
                    PRIMARY KEY (question_id, repeat_index, judge_contract_hash),
                    FOREIGN KEY (question_id, repeat_index)
                        REFERENCES candidates(question_id, repeat_index)
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
        else:
            query = (
                "SELECT record_json FROM judgments WHERE question_id = ? "
                "AND repeat_index = ? AND judge_contract_hash = ?"
            )
            insert = (
                "INSERT INTO judgments "
                "(question_id, repeat_index, judge_contract_hash, record_json) "
                "VALUES (?, ?, ?, ?)"
            )
            assert judge_contract_hash is not None
            parameters = (*key, judge_contract_hash, record_json)
            key = (*key, judge_contract_hash)
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
        self._record(
            "candidates",
            (record.question_id, record.repeat_index),
            _canonical_dataclass(record),
        )

    def missing_judgments(self, judge_contract_hash: str) -> list[tuple[int, int]]:
        with self._lock:
            return [
                (int(question_id), int(repeat_index))
                for question_id, repeat_index in self._connection.execute(
                    """
                    SELECT c.question_id, c.repeat_index
                    FROM candidates AS c
                    LEFT JOIN judgments AS j
                      ON j.question_id = c.question_id
                     AND j.repeat_index = c.repeat_index
                     AND j.judge_contract_hash = ?
                    WHERE j.question_id IS NULL
                    ORDER BY c.question_id, c.repeat_index
                    """,
                    (judge_contract_hash,),
                )
            ]

    def record_judgment(self, record: JudgmentRecord) -> None:
        self._record(
            "judgments",
            (record.question_id, record.repeat_index),
            _canonical_dataclass(record),
            judge_contract_hash=record.judge_contract_hash,
        )


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
