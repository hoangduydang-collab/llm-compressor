"""Helpers for resolving lm-eval metric keys across task / filter variants."""

from __future__ import annotations

from pipeline.config import EvalTask


def metric_base(metric: str) -> str:
    """``acc,none`` -> ``acc``."""
    return metric.split(",")[0]


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
