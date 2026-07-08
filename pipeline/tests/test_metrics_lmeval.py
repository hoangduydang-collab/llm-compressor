"""Unit tests for lm-eval metric key resolution."""

import pytest

from pipeline.config import EvalTask
from pipeline.metrics_lmeval import (
    aggregate_group_metric,
    require_task_results,
    require_task_results_or_aggregate,
    resolve_task_metric,
    task_results_from_batch,
)


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


def test_task_results_from_batch_leaf():
    batch = {"results": {"gsm8k": {"exact_match,strict-match": 0.5}}}
    assert task_results_from_batch(batch, "gsm8k") == {"exact_match,strict-match": 0.5}


def test_task_results_from_batch_group_under_groups():
    # lm-eval may only expose a group aggregate under ``groups`` (not ``results``).
    batch = {
        "results": {"mmlu_anatomy": {"acc,none": 0.6}},
        "groups": {"mmlu": {"acc,none": 0.65}},
    }
    assert task_results_from_batch(batch, "mmlu") == {"acc,none": 0.65}


def test_require_task_results_raises_when_missing():
    batch = {"results": {"mmlu_anatomy": {"acc,none": 0.6}}, "groups": {}}
    with pytest.raises(KeyError):
        require_task_results(batch, "mmlu")


def test_aggregate_group_metric_macro_average():
    batch = {
        "results": {
            "leaderboard_bbh_boolean_expressions": {"acc_norm,none": 0.8},
            "leaderboard_bbh_causal_judgement": {"acc_norm,none": 0.6},
        },
        "group_subtasks": {
            "leaderboard_bbh": [
                "leaderboard_bbh_boolean_expressions",
                "leaderboard_bbh_causal_judgement",
            ]
        },
    }
    agg = aggregate_group_metric(batch, "leaderboard_bbh", "acc_norm,none")
    assert agg == {"acc_norm,none": pytest.approx(0.7)}


def test_require_task_results_or_aggregate_leaderboard_bbh_shape():
    """Group row N/A; only per-subtask acc_norm — macro-average is used."""
    batch = {
        "results": {
            "leaderboard_bbh_boolean_expressions": {"acc_norm,none": 0.8},
            "leaderboard_bbh_causal_judgement": {"acc_norm,none": 0.6},
        },
        "groups": {"leaderboard_bbh": "N/A"},
        "group_subtasks": {
            "leaderboard_bbh": [
                "leaderboard_bbh_boolean_expressions",
                "leaderboard_bbh_causal_judgement",
            ]
        },
    }
    task = EvalTask(name="leaderboard_bbh", metric="acc_norm,none", num_fewshot=3)
    metrics = require_task_results_or_aggregate(batch, task)
    value, key = resolve_task_metric(task, metrics)
    assert key == "acc_norm,none"
    assert value == pytest.approx(0.7)
