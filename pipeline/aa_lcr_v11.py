"""Pinned AA-LCR v1.1 identity and prompt contract."""

import csv
import hashlib
import json
import os
import stat
import tempfile
import zipfile
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from types import MappingProxyType
from typing import Callable, Mapping
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


def canonical_json(value: object) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


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


def _member_path(member: zipfile.ZipInfo) -> PurePosixPath:
    path = PurePosixPath(member.filename)
    if (
        not member.filename
        or member.filename.startswith(("/", "\\"))
        or path.is_absolute()
        or ".." in path.parts
        or any(part in ("", ".") for part in path.parts)
    ):
        raise DatasetIntegrityError(f"unsafe archive member: {member.filename!r}")
    if stat.S_IFMT(member.external_attr >> 16) == stat.S_IFLNK:
        raise DatasetIntegrityError(f"unsafe archive member: {member.filename!r}")
    return path


def safe_extract_zip(zip_path: Path, destination: Path) -> list[Path]:
    """Extract a ZIP only after validating every archive member."""
    with zipfile.ZipFile(zip_path) as archive:
        members: list[tuple[zipfile.ZipInfo, PurePosixPath]] = []
        names: set[str] = set()
        for member in archive.infolist():
            path = _member_path(member)
            normalized = path.as_posix()
            if normalized in names:
                raise DatasetIntegrityError(
                    f"duplicate archive member: {member.filename!r}"
                )
            names.add(normalized)
            members.append((member, path))

        destination.mkdir(parents=True, exist_ok=True)
        extracted: list[Path] = []
        for member, path in members:
            target = destination.joinpath(*path.parts)
            target.parent.mkdir(parents=True, exist_ok=True)
            if member.is_dir():
                target.mkdir(exist_ok=True)
            else:
                with archive.open(member) as source, target.open("wb") as output:
                    output.write(source.read())
                extracted.append(target)
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


def _load_questions(csv_path: Path, documents_root: Path) -> tuple[Question, ...]:
    try:
        encoding = tiktoken.get_encoding("cl100k_base")
    except Exception as exc:
        raise DatasetIntegrityError("cl100k_base tokenizer is unavailable") from exc

    document_paths: dict[str, Path] = {}
    for document_path in documents_root.rglob("*"):
        if document_path.is_file():
            if document_path.name in document_paths:
                raise DatasetIntegrityError(
                    f"duplicate extracted document filename: {document_path.name}"
                )
            document_paths[document_path.name] = document_path

    with csv_path.open(encoding="utf-8-sig", newline="") as source:
        rows = list(csv.DictReader(source))
    questions: list[Question] = []
    for row in rows:
        try:
            question_id = int(_row_value(row, "question_id", "id"))
            filenames = tuple(
                filename.strip()
                for filename in _row_value(row, "data_source_filenames").split(";")
                if filename.strip()
            )
            if not filenames:
                raise DatasetIntegrityError("question has no document filenames")
            documents = []
            for filename in filenames:
                document_path = document_paths.get(filename)
                if document_path is None:
                    raise DatasetIntegrityError(f"referenced document is missing: {filename}")
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
                    document_set_id=_row_value(
                        row, "document_set_id", "data_source_id"
                    ),
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
    extracted_paths = safe_extract_zip(zip_path, documents_root)
    for path in extracted_paths:
        digests[path.relative_to(documents_root).as_posix()] = _sha256_file(path)
    questions = _load_questions(csv_path, documents_root)
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
