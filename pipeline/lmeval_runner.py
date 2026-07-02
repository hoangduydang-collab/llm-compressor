"""Shared lm-eval runner: one backend load for all configured tasks."""

from __future__ import annotations

from pipeline.config import EvalTask, PipelineConfig


def _arg_parts(parts: list[str]) -> str:
    return ",".join(parts) + ","


def vllm_model_args(cfg: PipelineConfig, model_path: str) -> str:
    s = cfg.serve
    return _arg_parts(
        [
            f"pretrained={model_path}",
            f"tensor_parallel_size={s.tensor_parallel_size}",
            f"max_model_len={s.max_model_len}",
            f"gpu_memory_utilization={s.gpu_memory_utilization}",
            f"trust_remote_code={cfg.model.trust_remote_code}",
            f"enforce_eager={s.enforce_eager}",
            "dtype=auto",
        ]
    )


def sglang_model_args(cfg: PipelineConfig, model_path: str) -> str:
    """lm-eval ``SGLangLM`` arg string (maps ``serve.*`` to SGLang Engine knobs)."""
    s = cfg.serve
    parts = [
        f"pretrained={model_path}",
        f"tp_size={s.tensor_parallel_size}",
        f"trust_remote_code={cfg.model.trust_remote_code}",
        "dtype=auto",
        f"context_length={s.max_model_len}",
        f"mem_fraction_static={s.gpu_memory_utilization}",
    ]
    if s.kv_cache_dtype:
        kv = s.kv_cache_dtype
        if kv == "fp8":
            kv = "fp8_e4m3"
        parts.append(f"kv_cache_dtype={kv}")
    for key, value in s.sglang_kwargs.items():
        parts.append(f"{key}={value}")
    return _arg_parts(parts)


def model_args(cfg: PipelineConfig, model_path: str) -> str:
    backend = cfg.eval.backend
    if backend == "vllm":
        return vllm_model_args(cfg, model_path)
    if backend == "sglang":
        return sglang_model_args(cfg, model_path)
    raise ValueError(
        f"unsupported eval.backend {backend!r}; valid: 'vllm', 'sglang'"
    )


def per_task_num_fewshot(tasks: list[EvalTask]) -> int | dict[str, int]:
    """lm-eval ``num_fewshot``: scalar when uniform, else per-task dict.

    Note: dict form is not safe for group tasks (mmlu, bbh) because lm-eval
    propagates the whole dict to subtasks. ``evaluate_tasks`` always uses a
    scalar per task via a reused model instance instead.
    """
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


def _load_lm_model(cfg: PipelineConfig, model_path: str):
    """Instantiate the lm-eval backend once for reuse across tasks."""
    from lm_eval.api.registry import get_model
    import lm_eval.models  # noqa: F401  (populates the model registry)

    model_cls = get_model(cfg.eval.backend)
    return model_cls.create_from_arg_string(
        model_args(cfg, model_path),
        {"batch_size": "auto"},
    )


def _merge_eval_results(merged: dict, batch: dict) -> None:
    merged.setdefault("results", {}).update(batch.get("results", {}))
    if batch.get("samples"):
        merged.setdefault("samples", {}).update(batch["samples"])
    for key, value in batch.items():
        if key not in ("results", "samples"):
            merged[key] = value


def evaluate_tasks(
    model_path: str,
    cfg: PipelineConfig,
    tasks: list[EvalTask],
    *,
    log_samples: bool = False,
) -> dict:
    """Evaluate ``tasks`` with one model load; returns merged ``simple_evaluate`` dict."""
    if not tasks:
        raise ValueError("evaluate_tasks requires at least one task")

    import os

    from pipeline._env import (
        apply_sglang_compat_env,
        ensure_writable_caches,
        preflight_sglang_deepgemm,
    )

    if cfg.eval.backend == "sglang" and cfg.serve.sglang_compat_fallbacks:
        applied = apply_sglang_compat_env()
        if applied:
            print(f"[lmeval] sglang compat env: {applied}")
        for note in preflight_sglang_deepgemm():
            print(f"[lmeval] WARNING: {note}")

    ensure_writable_caches()

    import lm_eval

    ev = cfg.eval
    names = ", ".join(t.name for t in tasks)
    print(
        f"[lmeval] backend={ev.backend} evaluating tasks ({names}) "
        "with a single model load"
    )

    lm = _load_lm_model(cfg, model_path)
    merged: dict = {}

    try:
        for task in tasks:
            print(
                f"[lmeval] task={task.name} num_fewshot={task.num_fewshot} "
                f"limit={task.limit}"
            )
            kwargs: dict = {
                "model": lm,
                "tasks": [task.name],
                "num_fewshot": task.num_fewshot,
                "apply_chat_template": ev.apply_chat_template,
            }
            if task.limit is not None:
                kwargs["limit"] = task.limit
            if log_samples:
                kwargs["log_samples"] = True

            batch = lm_eval.simple_evaluate(**kwargs)
            _merge_eval_results(merged, batch)
    finally:
        cleanup = getattr(lm, "clean", None) or getattr(lm, "cleanup", None)
        if callable(cleanup):
            cleanup()

    return merged
