"""Tests for evaluation-suite task shard selection."""

import pytest

from pipeline.config import EvalTask, PipelineConfig
from pipeline.evalsuite.cli import _select_tasks


def test_select_tasks_preserves_requested_order():
    cfg = PipelineConfig()
    cfg.eval.tasks = [EvalTask(name="mmlu_pro"), EvalTask(name="gsm8k")]

    selected = _select_tasks(cfg.eval.tasks, "gsm8k,mmlu_pro")

    assert [task.name for task in selected] == ["gsm8k", "mmlu_pro"]


def test_select_tasks_rejects_unknown_name():
    with pytest.raises(ValueError, match="unknown eval task"):
        _select_tasks([EvalTask(name="gsm8k")], "gpqa_diamond")
