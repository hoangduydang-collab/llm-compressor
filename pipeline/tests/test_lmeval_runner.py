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
    # Helper still reports dict form; evaluate_tasks uses per-task scalars instead.
    assert per_task_num_fewshot(tasks) == {"wikitext": 0, "mmlu": 5}


def test_merge_eval_results():
    from pipeline.lmeval_runner import _merge_eval_results

    merged: dict = {}
    _merge_eval_results(
        merged,
        {
            "results": {"wikitext": {"word_perplexity,none": 11.0}},
            "samples": {"wikitext": [{"doc_id": 0}]},
            "config": {"model": "vllm"},
        },
    )
    _merge_eval_results(
        merged,
        {
            "results": {"mmlu": {"acc,none": 0.8}},
            "samples": {"mmlu": [{"doc_id": 1}]},
        },
    )
    assert set(merged["results"]) == {"wikitext", "mmlu"}
    assert len(merged["samples"]["wikitext"]) == 1
    assert merged["config"] == {"model": "vllm"}


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
