"""Shared lm-eval runner: one vLLM load for all configured tasks."""

from __future__ import annotations

from pipeline.config import EvalTask, PipelineConfig


def model_args(cfg: PipelineConfig, model_path: str) -> str:
    s = cfg.serve
    return (
        f"pretrained={model_path},"
        f"tensor_parallel_size={s.tensor_parallel_size},"
        f"max_model_len={s.max_model_len},"
        f"gpu_memory_utilization={s.gpu_memory_utilization},"
        f"trust_remote_code={cfg.model.trust_remote_code},"
        f"enforce_eager={s.enforce_eager},"
        f"dtype=auto,"
    )


def per_task_num_fewshot(tasks: list[EvalTask]) -> int | dict[str, int]:
    """lm-eval ``num_fewshot``: scalar when uniform, else per-task dict."""
    if not tasks:
        return 0
    by_name = {t.name: t.num_fewshot for t in tasks}
    unique = set(by_name.values())
    if len(unique) == 1:
        return next(iter(unique))
    return by_name


def per_task_limit(tasks: list[EvalTask]) -> int | float | dict[str, int | float] | None:
    """lm-eval ``limit``: omitted when all tasks are unlimited, else scalar or dict."""
    if not tasks:
        return None
    by_name = {t.name: t.limit for t in tasks}
    if all(v is None for v in by_name.values()):
        return None
    limited = {name: lim for name, lim in by_name.items() if lim is not None}
    if len(limited) == len(by_name):
        unique = set(limited.values())
        if len(unique) == 1:
            return next(iter(unique))
    return limited


def evaluate_tasks(
    model_path: str,
    cfg: PipelineConfig,
    tasks: list[EvalTask],
    *,
    log_samples: bool = False,
) -> dict:
    """Run lm-eval once over ``tasks``; returns the full ``simple_evaluate`` dict."""
    if not tasks:
        raise ValueError("evaluate_tasks requires at least one task")

    from pipeline._env import ensure_writable_caches

    ensure_writable_caches()

    import lm_eval
    import lm_eval.models  # noqa: F401  (populates the model registry)

    ev = cfg.eval
    kwargs: dict = {
        "model": ev.backend,
        "model_args": model_args(cfg, model_path),
        "tasks": [t.name for t in tasks],
        "num_fewshot": per_task_num_fewshot(tasks),
        "apply_chat_template": ev.apply_chat_template,
    }
    limit = per_task_limit(tasks)
    if limit is not None:
        kwargs["limit"] = limit
    if log_samples:
        kwargs["log_samples"] = True

    names = ", ".join(t.name for t in tasks)
    print(f"[lmeval] evaluating tasks ({names}) with a single model load")
    return lm_eval.simple_evaluate(**kwargs)
