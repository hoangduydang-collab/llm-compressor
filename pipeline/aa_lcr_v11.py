"""Pinned AA-LCR v1.1 identity and prompt contract."""

import hashlib
import json


AA_LCR_VERSION = "1.1"
DATASET_REPO = "ArtificialAnalysis/AA-LCR"
DATASET_REVISION = "9a77ef56b717057ade24ceab4d273712a0b4f19e"
CSV_FILENAME = "AA-LCR_Dataset.csv"
ZIP_FILENAME = "extracted_text/AA-LCR_extracted-text.zip"
ZIP_SHA256 = "5e839249826f6b9bd5324f0d139089c9dc481ccb3f212a6dfad00c51045d9d8a"
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
