"""Unit tests for per-task lm-eval kwargs (no GPU)."""

from pipeline.config import EvalTask
from pipeline.lmeval_runner import per_task_limit, per_task_num_fewshot


def test_per_task_num_fewshot_scalar_when_uniform():
    tasks = [
        EvalTask(name="a", num_fewshot=5),
        EvalTask(name="b", num_fewshot=5),
    ]
    assert per_task_num_fewshot(tasks) == 5


def test_per_task_num_fewshot_dict_when_mixed():
    tasks = [
        EvalTask(name="wikitext", num_fewshot=0),
        EvalTask(name="mmlu", num_fewshot=5),
    ]
    assert per_task_num_fewshot(tasks) == {"wikitext": 0, "mmlu": 5}


def test_per_task_limit_omitted_when_all_unlimited():
    tasks = [
        EvalTask(name="wikitext", limit=None),
        EvalTask(name="mmlu", limit=None),
    ]
    assert per_task_limit(tasks) is None


def test_per_task_limit_scalar_when_uniform():
    tasks = [
        EvalTask(name="a", limit=250),
        EvalTask(name="b", limit=250),
    ]
    assert per_task_limit(tasks) == 250


def test_per_task_limit_dict_when_mixed():
    tasks = [
        EvalTask(name="wikitext", limit=None),
        EvalTask(name="mmlu", limit=250),
    ]
    assert per_task_limit(tasks) == {"mmlu": 250}
