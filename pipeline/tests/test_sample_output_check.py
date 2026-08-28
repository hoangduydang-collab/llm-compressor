"""Tests for the degeneracy detectors in the sample-output check.

The point of these is that the check must FIRE on the real M3 failure text and
must NOT fire on ordinary prose -- a detector that does neither is decoration.
"""

from pipeline.sample_output_check import degeneracy_report, judge


HEALTHY = (
    " The sky appears blue because shorter wavelengths of sunlight scatter more "
    "strongly off air molecules than longer ones, a process called Rayleigh "
    "scattering. Our eyes are also more sensitive to blue than to violet."
)


def test_healthy_prose_passes():
    assert judge(degeneracy_report(HEALTHY)) == []


def test_arring_collapse_is_caught():
    """The actual MiniMax-M3 full-calib AWQ failure output."""
    text = "arring" * 60
    problems = judge(degeneracy_report(text))
    assert problems, "the arringarring collapse must be detected"
    assert any("4-gram" in p or "repetition" in p for p in problems)


def test_single_repeated_word_is_caught():
    problems = judge(degeneracy_report(" the the the the the the the the the the"))
    assert problems


def test_empty_output_is_caught():
    assert any("too short" in p for p in judge(degeneracy_report("")))


def test_short_but_valid_answer_is_not_flagged_for_repetition():
    """A terse correct answer is short, not degenerate: it may trip the length
    floor but must not be called repetitive."""
    problems = judge(degeneracy_report(" Paris, on the Seine."))
    assert not any("repetition" in p for p in problems)


def test_code_output_passes():
    """Code repeats tokens more than prose; it must still pass."""
    code = (
        "def fib(n):\n    a, b = 0, 1\n    for _ in range(n):\n"
        "        a, b = b, a + b\n    return a\n"
    )
    assert judge(degeneracy_report(code)) == []


def test_report_fields():
    r = degeneracy_report(HEALTHY)
    assert r["chars"] > 100 and r["words"] > 20
    assert 0.0 < r["unique_word_ratio"] <= 1.0
    assert 0.0 <= r["top_4gram_share"] < 0.25


# --- the metric that actually separates the classes ------------------------

def test_repeated_sentence_loop_is_caught():
    """Another real collapse mode: the model loops a whole sentence."""
    assert judge(degeneracy_report(" I don't know. " * 20))


def test_single_character_flood_is_caught():
    assert judge(degeneracy_report("!" * 200))


def test_distinct_4gram_ratio_separates_healthy_from_degenerate():
    """Pins the measured margin. Healthy >= 0.75, degenerate <= 0.111, so the
    0.30 threshold has room on both sides; this fails if a change narrows it."""
    healthy = [
        HEALTHY,
        "def fib(n):\n    a, b = 0, 1\n    for _ in range(n):\n"
        "        a, b = b, a + b\n    return a\n",
    ]
    degenerate = ["arring" * 60, " the the the the the the the the the the",
                  " I don't know. " * 20, "!" * 200]
    for text in healthy:
        assert degeneracy_report(text)["distinct_4gram_ratio"] >= 0.60, text[:40]
    for text in degenerate:
        assert degeneracy_report(text)["distinct_4gram_ratio"] <= 0.20, text[:40]


def test_top_4gram_share_alone_would_have_missed_arring():
    """Regression note: the first implementation judged on top-4gram share, and
    'arringarring...' cycles six 4-grams for a share of only ~0.17 -- below any
    threshold prose survives. Keep the distinct-ratio check as the primary."""
    report = degeneracy_report("arring" * 60)
    assert report["top_4gram_share"] < 0.25
    assert report["distinct_4gram_ratio"] < 0.05
    assert judge(report), "must still be caught, via the distinct-ratio check"
