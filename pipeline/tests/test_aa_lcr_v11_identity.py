# ruff: noqa: E501

import pytest

from pipeline import aa_lcr_v11 as A

EXPECTED_CANDIDATE_PROMPT = """BEGIN INPUT DOCUMENTS

{documents_text}

END INPUT DOCUMENTS

Answer the following question using the input documents provided above.

START QUESTION

{question}

END QUESTION
"""

EXPECTED_JUDGE_SYSTEM_PROMPT = """Decide whether the CANDIDATE ANSWER is correct or incorrect against the OFFICIAL ANSWER.
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

EXPECTED_JUDGE_USER_PROMPT = """Assess whether the following CANDIDATE ANSWER is CORRECT or INCORRECT.
For the CANDIDATE ANSWER to be correct, it must be consistent with the OFFICIAL ANSWER.

The question, for reference only: START QUESTION {question}

END QUESTION

The OFFICIAL ANSWER: {official_answer}

END OFFICIAL ANSWER

BEGIN CANDIDATE ANSWER TO ASSESS

{candidate_answer}

END CANDIDATE ANSWER TO ASSESS

Reply as JSON, with a verdict of CORRECT or INCORRECT."""


def test_official_v11_identity_is_immutable():
    assert A.AA_LCR_VERSION == "1.1"
    assert A.DATASET_REPO == "ArtificialAnalysis/AA-LCR"
    assert A.DATASET_REVISION == "9a77ef56b717057ade24ceab4d273712a0b4f19e"
    assert A.CSV_FILENAME == "AA-LCR_Dataset.csv"
    assert A.ZIP_FILENAME == "extracted_text/AA-LCR_extracted-text.zip"
    assert A.ZIP_SHA256 == (
        "5e839249826f6b9bd5324f0d139089c9dc481ccb3f212a6dfad00c51045d9d8a"
    )
    assert A.REPEATS == 3
    assert A.CANDIDATE_TEMPERATURE == 0.6
    assert A.CANDIDATE_TOP_P == 1.0
    assert A.CANDIDATE_MAX_TOKENS == 131_072
    assert A.CANDIDATE_CONCURRENCY == 2
    assert A.CANDIDATE_HTTP_TIMEOUT_SECONDS == 7200
    assert A.JUDGE_MODEL == "gpt-5.6-luna"
    assert A.JUDGE_REASONING_EFFORT == "medium"
    assert A.MAX_ATTEMPTS == 30


def test_v11_prompts_match_pinned_revision_verbatim():
    assert A.CANDIDATE_PROMPT_TEMPLATE == EXPECTED_CANDIDATE_PROMPT
    assert A.JUDGE_SYSTEM_PROMPT == EXPECTED_JUDGE_SYSTEM_PROMPT
    assert A.JUDGE_USER_PROMPT_TEMPLATE == EXPECTED_JUDGE_USER_PROMPT


def test_candidate_prompt_preserves_document_order_and_boundaries():
    prompt = A.build_candidate_prompt(["first", "second"], "Which one?")
    assert (
        prompt
        == """BEGIN INPUT DOCUMENTS

BEGIN DOCUMENT 1:
first
END DOCUMENT 1

BEGIN DOCUMENT 2:
second
END DOCUMENT 2

END INPUT DOCUMENTS

Answer the following question using the input documents provided above.

START QUESTION

Which one?

END QUESTION
"""
    )


def test_judge_user_prompt_interpolates_without_trailing_text():
    prompt = A.build_judge_user_prompt("Q", "A", "candidate")
    assert "START QUESTION Q\n\nEND QUESTION" in prompt
    assert "BEGIN CANDIDATE ANSWER TO ASSESS\n\ncandidate" in prompt
    assert prompt.endswith("Reply as JSON, with a verdict of CORRECT or INCORRECT.")


def test_canonical_json_and_sha256_text_are_deterministic():
    assert A.canonical_json({"z": "é", "a": [1, 2]}) == '{"a":[1,2],"z":"é"}'
    assert A.sha256_text("AA-LCR") == (
        "701af1b36dc7ab0b4f70372aac754d0cf003214cfc94fb7f579167d9eefe86b6"
    )


def test_judge_verdict_is_strict_json():
    assert A.parse_judge_verdict('{"verdict":"CORRECT"}') == "CORRECT"
    with pytest.raises(A.JudgeProtocolError):
        A.parse_judge_verdict("CORRECT")
    with pytest.raises(A.JudgeProtocolError):
        A.parse_judge_verdict('{"verdict":"PARTIAL"}')
