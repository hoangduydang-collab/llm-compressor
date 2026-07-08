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


def aggregate_group_metric(
    batch: dict, task_name: str, metric: str
) -> dict | None:
    """Macro-average ``metric`` across lm-eval group subtasks.

    Some leaderboard groups (e.g. ``leaderboard_bbh``) report ``N/A`` at the
    group level; only per-subtask scores exist under ``results``.
    """
    results = batch.get("results") or {}
    subtasks = (batch.get("group_subtasks") or {}).get(task_name)
    if not subtasks:
        prefix = f"{task_name}_"
        subtasks = sorted(k for k in results if k.startswith(prefix))
    if not subtasks:
        return None

    values: list[float] = []
    resolved_key: str | None = None
    for subtask in subtasks:
        sub_results = results.get(subtask)
        if not isinstance(sub_results, dict):
            continue
        try:
            val, key = resolve_task_metric(
                EvalTask(name=subtask, metric=metric), sub_results
            )
        except KeyError:
            continue
        values.append(val)
        resolved_key = key

    if not values:
        return None

    return {resolved_key or metric: sum(values) / len(values)}


def require_task_results_or_aggregate(batch: dict, task: EvalTask) -> dict:
    """Resolve metrics from group aggregate, subtask macro-average, or raise."""
    direct = task_results_from_batch(batch, task.name)
    if isinstance(direct, dict):
        try:
            resolve_task_metric(task, direct)
            return direct
        except KeyError:
            pass

    aggregated = aggregate_group_metric(batch, task.name, task.metric)
    if aggregated is not None:
        return aggregated

    return require_task_results(batch, task.name)


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
