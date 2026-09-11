"""Tests for Lotfi et al. 2026 (arXiv:2606.00206) error categories on AA GPQA.

Paper §4.2 / Table 12: on an incorrect attempt, assign exactly one of
overthinking, logical_error, arithmetic_error, formatting_hallucination_other.
Overthinking is mandatory whenever the gold answer appears in intermediate
reasoning but is not the final answer.
"""

from pipeline.aa_overthinking import (
    committed_letters,
    infer_gold_choice_text,
    parse_mcq_prompt,
    categorize_failure,
    lookup_official_gold,
    screen_records,
)


PROMPT = """Answer the following multiple choice question. The last line of your response should be in the following format: 'Answer: A/B/C/D' (e.g. 'Answer: A').

What is the capital of France?

A) Berlin
B) Paris
C) Madrid
D) Rome
"""


def test_parse_mcq_prompt():
    stem, choices = parse_mcq_prompt(PROMPT)
    assert "capital of France" in stem
    assert choices == {"A": "Berlin", "B": "Paris", "C": "Madrid", "D": "Rome"}


def test_committed_letters_in_order():
    text = (
        "I first think Answer: A is right.\n"
        "Wait, actually Answer: B.\n"
        "Final: Answer: C"
    )
    assert committed_letters(text) == ["A", "B", "C"]


def test_overthinking_when_gold_appears_then_abandoned():
    result = categorize_failure(
        text="First Answer: B then I doubt. Answer: A",
        score=0.0,
        gold_letter="B",
        final_letter="A",
    )
    assert result["category"] == "overthinking"


def test_not_overthinking_when_gold_never_committed():
    result = categorize_failure(
        text="I like C. Answer: C",
        score=0.0,
        gold_letter="B",
        final_letter="C",
    )
    assert result["category"] != "overthinking"
    assert result["category"] == "not_overthinking"


def test_correct_attempt_is_not_a_failure():
    result = categorize_failure(
        text="Answer: B",
        score=1.0,
        gold_letter="B",
        final_letter="B",
    )
    assert result["category"] is None


def test_cap_hit_with_earlier_gold_is_overthinking():
    """Paper: reaches the correct solution but does not commit as final."""
    result = categorize_failure(
        text="Answer: B\nWait, alternatively " + "loop " * 20,
        score=0.0,
        gold_letter="B",
        capped=True,
    )
    assert result["category"] == "overthinking"


def test_infer_gold_from_correct_sibling():
    records = [
        {
            "stem": "What is the capital of France?",
            "choices": {"A": "Berlin", "B": "Paris", "C": "Madrid", "D": "Rome"},
            "score": 1.0,
            "text": "Answer: B",
        },
        {
            "stem": "What is the capital of France?",
            "choices": {"A": "Paris", "B": "Rome", "C": "Berlin", "D": "Madrid"},
            "score": 0.0,
            "text": "Answer: B",
        },
    ]
    gold_text = infer_gold_choice_text(records)
    assert gold_text == "Paris"
    letter = next(
        k for k, v in records[1]["choices"].items() if v == gold_text
    )
    assert letter == "A"
    result = categorize_failure(
        text=records[1]["text"],
        score=0.0,
        gold_letter="A",
        final_letter="B",
    )
    assert result["category"] != "overthinking"


GPQA_ROW = {
    "Question": "What is the capital of France?",
    "Correct Answer": "Paris",
    "Incorrect Answer 1": "Berlin",
    "Incorrect Answer 2": "Madrid",
    "Incorrect Answer 3": "Rome",
}


def test_official_gold_labels_never_scored_stem():
    """Official GPQA gold does not need a score=1 sibling."""
    records = [
        {
            "stem": "What is the capital of France?",
            "stem_key": "never-scored",
            "choices": {"A": "Berlin", "B": "Rome", "C": "Paris", "D": "Madrid"},
            "score": 0.0,
            "text": "Answer: B",
            "completion_tokens": 80,
        }
    ]
    summary = screen_records(records, gpqa_rows=[GPQA_ROW])
    assert summary["n_fail_gold_unknown"] == 0
    row = summary["rows"][0]
    assert row["gold_letter"] == "C"
    assert row["gold_source"] == "gpqa_question"
    assert row["category"] == "not_overthinking"


def test_official_gold_via_choice_set_when_stem_differs():
    records = [
        {
            "stem": "AA wrapper: What is the capital of France?",
            "stem_key": "wrapped",
            "choices": {"A": "Paris", "B": "Rome", "C": "Berlin", "D": "Madrid"},
            "score": 0.0,
            "text": "Answer: B",
            "completion_tokens": 40,
        }
    ]
    summary = screen_records(records, gpqa_rows=[GPQA_ROW])
    row = summary["rows"][0]
    assert row["gold_source"] == "gpqa_choice_set"
    assert row["gold_letter"] == "A"


def test_official_gold_beats_wrong_sibling_vote():
    records = [
        {
            "stem": "What is the capital of France?",
            "stem_key": "fr",
            "choices": {"A": "Berlin", "B": "Paris", "C": "Madrid", "D": "Rome"},
            "score": 1.0,
            "text": "Answer: A",
            "completion_tokens": 20,
        },
        {
            "stem": "What is the capital of France?",
            "stem_key": "fr",
            "choices": {"A": "Paris", "B": "Rome", "C": "Berlin", "D": "Madrid"},
            "score": 0.0,
            "text": "First Answer: A then I switch. Answer: B",
            "completion_tokens": 50,
        },
    ]
    summary = screen_records(records, gpqa_rows=[GPQA_ROW])
    fail = next(r for r in summary["rows"] if r["score"] < 1.0)
    assert fail["gold_letter"] == "A"
    assert fail["gold_source"] == "gpqa_question"
    assert fail["category"] == "overthinking"


def test_lookup_official_gold_normalizes_whitespace():
    from pipeline.aa_overthinking import index_gpqa_diamond

    by_stem, by_choices = index_gpqa_diamond(
        [{**GPQA_ROW, "Question": "What is the\ncapital of   France?"}]
    )
    gold, source = lookup_official_gold(
        "What is the capital of France?",
        {"A": "Paris", "B": "Rome", "C": "Berlin", "D": "Madrid"},
        by_stem,
        by_choices,
    )
    assert source == "gpqa_question"
    assert gold == "Paris"

