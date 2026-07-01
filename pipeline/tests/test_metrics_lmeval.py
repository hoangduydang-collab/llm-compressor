"""Unit tests for lm-eval metric key resolution."""

import pytest

from pipeline.config import EvalTask
from pipeline.metrics_lmeval import resolve_task_metric


def test_resolve_bbh_get_answer():
    task = EvalTask(name="bbh", metric="exact_match,strict-match")
    metrics = {
        "sample_len": 123.0,
        "exact_match,get-answer": 0.42,
        "exact_match_stderr,get-answer": 0.01,
    }
    value, key = resolve_task_metric(task, metrics)
    assert key == "exact_match,get-answer"
    assert value == pytest.approx(0.42)


def test_resolve_gsm8k_strict_match():
    task = EvalTask(name="gsm8k", metric="exact_match,strict-match")
    metrics = {"exact_match,strict-match": 0.75}
    value, key = resolve_task_metric(task, metrics)
    assert value == pytest.approx(0.75)
