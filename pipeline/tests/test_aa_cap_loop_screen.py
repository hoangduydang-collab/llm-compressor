"""Cap-hit loop screen: M3 degeneracy detectors on AA GPQA cache traces.

Baseline: pipeline/sample_output_check.py (distinct-4gram < 0.30 is the
arringarring collapse signature) plus pipeline/evalsuite/health.py
_periodic_suffix. Classification must fire on a late tail collapse, not
only on an all-loop trace — a 131k-token CoT that reasons then loops
would look healthy if scored on the full text alone.
"""

from __future__ import annotations

import hashlib
import json
import sqlite3

from pipeline.aa_cap_loop_screen import (
    CAP_AA,
    assistant_text,
    classify_trace,
    load_cache_records,
    screen_records,
    summarize_screen,
)


def _healthy_reasoning(n: int = 80) -> str:
    """Diverse CoT-shaped prose.

    A shared sentence skeleton is itself a loop to the distinct-4gram
    detector (the first fixtures scored 0.11–0.20). Unique lexical bulk
    has to dominate each step, the way a real GPQA trace does.
    """
    parts = []
    for i in range(n):
        payload = "".join(
            hashlib.sha256(f"gpqa-step-{i}-{k}".encode()).hexdigest()
            for k in range(8)
        )
        parts.append(
            f"Check {i}: payload {payload} rules out option "
            f"{chr(65 + (i % 4))} because the implied coupling cannot "
            f"match experiment E-{i}."
        )
    return " ".join(parts)


def test_assistant_text_prefers_reasoning_and_visible_content():
    convo = [
        {"role": "user", "content": "Q?"},
        {
            "role": "assistant",
            "reasoning_content": "long think",
            "content": "Answer: B",
            "usage": {"completion_tokens": 12},
        },
    ]
    text = assistant_text(convo)
    assert "long think" in text
    assert "Answer: B" in text


def test_assistant_text_does_not_duplicate_nested_reasoning():
    reasoning = "I will pick B after checking C."
    convo = [
        {"role": "user", "content": "Q?"},
        {
            "role": "assistant",
            "reasoning_content": reasoning,
            "content": reasoning + "\nAnswer: B",
        },
    ]
    text = assistant_text(convo)
    assert text.count(reasoning) == 1


def test_arring_collapse_is_a_loop():
    result = classify_trace("arring" * 400)
    assert result["label"] == "loop"
    assert result["reasons"]


def test_repeated_sentence_is_a_loop():
    result = classify_trace(" I don't know. " * 80)
    assert result["label"] == "loop"


def test_healthy_long_reasoning_is_unfinished_not_a_loop():
    result = classify_trace(_healthy_reasoning(120))
    assert result["label"] == "unfinished"
    assert result["reasons"] == []


def test_long_cot_that_reuses_notation_is_unfinished_if_zlib_is_healthy():
    """The 0.30 distinct-4gram gate over-fires on 131k GPQA CoT. A trace that
    restates the same technical sentence with unique numbers must not be
    called a tight loop when zlib stays in the English range."""
    text = " ".join(
        f"The selection rule for NN partial wave {i} forbids the {i + 1} "
        f"transition near threshold, consistent with table row {i * 3}."
        for i in range(400)
    )
    result = classify_trace(text)
    assert result["zlib_tail_8k"] is not None and result["zlib_tail_8k"] > 0.08
    assert result["label"] == "unfinished"


def test_late_collapse_is_a_loop_even_when_prefix_is_healthy():
    """The case a full-text-only score would miss."""
    text = _healthy_reasoning(120) + (" arring" * 4000)
    result = classify_trace(text)
    assert result["label"] == "loop"
    assert result["tail_fired"] is True


def test_load_cache_and_screen_keeps_only_cap_hits(tmp_path):
    db = tmp_path / "cache.db"
    con = sqlite3.connect(db)
    con.execute("create table Cache (key text, value text)")

    def put(key, score, tokens, text):
        rec = {
            "score": score,
            "convo": [
                {"role": "user", "content": f"question {key}"},
                {
                    "role": "assistant",
                    "reasoning_content": text,
                    "content": "",
                    "usage": {"completion_tokens": tokens},
                },
            ],
        }
        con.execute("insert into Cache values (?, ?)", (key, json.dumps(rec)))

    put("short", 1.0, 2000, _healthy_reasoning(40))
    put("loop", 0.0, CAP_AA, "arring" * 400)
    put("unfin", 0.0, CAP_AA, _healthy_reasoning(120))
    con.commit()
    con.close()

    records = load_cache_records(db)
    assert len(records) == 3
    summary = summarize_screen(screen_records(records, cap=CAP_AA))
    assert summary["n_loaded"] == 3
    assert summary["n_capped"] == 2
    assert summary["n_loop"] == 1
    assert summary["n_unfinished"] == 1
    assert summary["capped_unique_questions"] == 2
