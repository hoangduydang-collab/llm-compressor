"""Helpers for resolving lm-eval metric keys across task / filter variants."""

from __future__ import annotations

from pipeline.config import EvalTask


def metric_base(metric: str) -> str:
    """``acc,none`` -> ``acc``."""
    return metric.split(",")[0]


def task_results_from_batch(batch: dict, task_name: str) -> dict | None:
    """Resolve an lm-eval task's metrics from a ``simple_evaluate`` batch.

    Group tasks (``mmlu``, ``bbh``, …) may store their aggregate under
    ``groups`` rather than ``results``, so both are checked.
    """
    results = batch.get("results") or {}
    task_results = results.get(task_name)
    if isinstance(task_results, dict):
        return task_results
    groups = batch.get("groups") or {}
    group_results = groups.get(task_name)
    if isinstance(group_results, dict):
        return group_results
    return None


def require_task_results(batch: dict, task_name: str) -> dict:
    """Like :func:`task_results_from_batch` but raises if the task is absent."""
    task_results = task_results_from_batch(batch, task_name)
    if not isinstance(task_results, dict):
        results_keys = sorted((batch.get("results") or {}).keys())
        groups_keys = sorted((batch.get("groups") or {}).keys())
        raise KeyError(
            f"lm-eval batch missing results for task {task_name!r}; "
            f"results keys: {results_keys}; groups keys: {groups_keys}"
        )
    return task_results


def resolve_task_metric(task: EvalTask, task_results: dict) -> tuple[float, str]:
    """Return ``(value, resolved_metric_key)`` from lm-eval task results."""
    base = metric_base(task.metric)
    candidates = [task.metric, base]

    if base == "exact_match":
        candidates.extend(
            [
                "exact_match,strict-match",
                "exact_match,get-answer",
                "exact_match,flexible-extract",
            ]
        )
    elif base == "acc":
        candidates.extend(["acc,none", "acc_norm,none"])
    elif base == "acc_norm":
        candidates.extend(["acc_norm,none", "acc,none"])
    elif base == "word_perplexity":
        candidates.extend(["word_perplexity,none", "perplexity,none"])

    seen: set[str] = set()
    for key in candidates:
        if key in seen:
            continue
        seen.add(key)
        val = task_results.get(key)
        if isinstance(val, (int, float)):
            return float(val), key

    for key, val in task_results.items():
        if (
            isinstance(val, (int, float))
            and key.startswith(f"{base},")
            and "stderr" not in key
        ):
            return float(val), key

    raise KeyError(
        f"metric {task.metric!r} not in lm-eval results for task "
        f"{task.name!r}; available: {list(task_results.keys())}"
    )
