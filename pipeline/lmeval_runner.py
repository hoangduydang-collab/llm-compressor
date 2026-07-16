"""Shared lm-eval runner: one backend load for all configured tasks."""

from __future__ import annotations

import json
import os
from collections.abc import Callable
from pathlib import Path

from pipeline.config import EvalTask, PipelineConfig

TaskBatchCallback = Callable[[EvalTask, int | None, dict], None]


def _arg_parts(parts: list[str]) -> str:
    return ",".join(parts) + ","


def _thinking_model_arg_parts(ev) -> list[str]:
    """Optional lm-eval model_args for reasoning / thinking chat models."""
    parts: list[str] = []
    if ev.enable_thinking is not None:
        parts.append(f"enable_thinking={ev.enable_thinking}")
    if ev.think_end_token:
        parts.append(f"think_end_token={ev.think_end_token}")
    return parts


def vllm_model_args(cfg: PipelineConfig, model_path: str) -> str:
    s = cfg.serve
    runtime_kwargs: dict[str, object] = {}
    if s.enable_expert_parallel:
        runtime_kwargs["enable_expert_parallel"] = True
    if s.block_size is not None:
        runtime_kwargs["block_size"] = s.block_size
    if s.kv_cache_dtype is not None:
        runtime_kwargs["kv_cache_dtype"] = s.kv_cache_dtype
    runtime_kwargs.update(s.vllm_kwargs)
    return _arg_parts(
        [
            f"pretrained={model_path}",
            f"tensor_parallel_size={s.tensor_parallel_size}",
            f"max_model_len={s.max_model_len}",
            f"gpu_memory_utilization={s.gpu_memory_utilization}",
            f"trust_remote_code={cfg.model.trust_remote_code}",
            f"enforce_eager={s.enforce_eager}",
            f"disable_custom_all_reduce={s.disable_custom_all_reduce}",
            "dtype=auto",
            *(f"{key}={value}" for key, value in runtime_kwargs.items()),
            *_thinking_model_arg_parts(cfg.eval),
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
    parts.extend(_thinking_model_arg_parts(cfg.eval))
    return _arg_parts(parts)


def model_args(cfg: PipelineConfig, model_path: str) -> str:
    backend = cfg.eval.backend
    if backend == "vllm":
        return vllm_model_args(cfg, model_path)
    if backend == "sglang":
        return sglang_model_args(cfg, model_path)
    raise ValueError(f"unsupported eval.backend {backend!r}; valid: 'vllm', 'sglang'")


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


def per_task_limit(
    tasks: list[EvalTask],
) -> int | float | dict[str, int | float] | None:
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


def _prepare_vllm_runtime(model_path: str, model_source: str) -> dict:
    """Apply the same MiniMax runtime preparation used by the fidelity probe."""
    from pipeline.m3_distributional_probe import _prepare_minimax_runtime

    return _prepare_minimax_runtime(Path(model_path), model_source)


def _load_lm_model(cfg: PipelineConfig, model_path: str):
    """Instantiate the lm-eval backend once for reuse across tasks."""
    if cfg.eval.backend == "vllm":
        runtime = _prepare_vllm_runtime(model_path, cfg.model.id)
        if runtime:
            print(f"[lmeval] vLLM runtime preparation: {runtime}")
    import lm_eval.models  # noqa: F401  (populates the model registry)
    from lm_eval.api.registry import get_model

    model_cls = get_model(cfg.eval.backend)
    batch_size = cfg.eval.lm_eval_batch_size
    lm = model_cls.create_from_arg_string(
        model_args(cfg, model_path),
        {"batch_size": batch_size},
    )
    ready_file = os.environ.get("M3_MODEL_READY_FILE")
    if ready_file:
        ready = Path(ready_file)
        ready.parent.mkdir(parents=True, exist_ok=True)
        temp = ready.with_suffix(ready.suffix + ".tmp")
        temp.write_text(json.dumps({"status": "ready"}) + "\n")
        temp.replace(ready)
    return lm


def _merge_eval_results(merged: dict, batch: dict) -> None:
    merged.setdefault("results", {}).update(batch.get("results", {}))
    if batch.get("samples"):
        merged.setdefault("samples", {}).update(batch["samples"])
    # Group tasks (mmlu, bbh) put their aggregate under ``groups`` and their
    # subtask mapping under ``group_subtasks``; accumulate both so no task's
    # aggregate is lost when several tasks share one model load.
    for accumulated_key in ("groups", "group_subtasks"):
        if batch.get(accumulated_key):
            merged.setdefault(accumulated_key, {}).update(batch[accumulated_key])
    for key, value in batch.items():
        if key not in ("results", "samples", "groups", "group_subtasks"):
            merged[key] = value


def evaluate_tasks(
    model_path: str,
    cfg: PipelineConfig,
    tasks: list[EvalTask],
    *,
    log_samples: bool = False,
    completed_task_seeds: set[tuple[str, int]] | None = None,
    on_task_complete: TaskBatchCallback | None = None,
) -> dict:
    """Evaluate ``tasks`` with one model load; returns merged ``simple_evaluate`` dict.

    Repeated generation seeds reuse the same loaded backend. If
    ``on_task_complete`` is set, it receives the task, generation seed (or
    ``None`` for legacy one-pass evaluation), and raw result batch.
    """
    if not tasks:
        raise ValueError("evaluate_tasks requires at least one task")

    from pipeline._env import (
        apply_sglang_compat_env,
        ensure_writable_caches,
        preflight_sglang_deepgemm,
    )

    if cfg.eval.backend == "sglang":
        if cfg.serve.sglang_compat_fallbacks:
            applied = apply_sglang_compat_env()
            if applied:
                print(f"[lmeval] sglang compat env: {applied}")
            for note in preflight_sglang_deepgemm():
                print(f"[lmeval] WARNING: {note}")
        else:
            from pipeline._env import apply_lm_eval_sglang_compat

            applied = apply_lm_eval_sglang_compat()
            if applied:
                print(f"[lmeval] sglang sampling compat: {applied}")

    ensure_writable_caches()

    import lm_eval

    ev = cfg.eval
    names = ", ".join(t.name for t in tasks)
    print(
        f"[lmeval] backend={ev.backend} evaluating tasks ({names}) "
        "with a single model load"
    )
    if ev.apply_chat_template or ev.fewshot_as_multiturn or ev.enable_thinking:
        print(
            f"[lmeval] harness: apply_chat_template={ev.apply_chat_template} "
            f"fewshot_as_multiturn={ev.fewshot_as_multiturn} "
            f"enable_thinking={ev.enable_thinking} "
            f"think_end_token={ev.think_end_token!r} "
            f"gen_kwargs={ev.gen_kwargs}"
        )

    from pipeline.evalsuite.sampling import (
        load_sample_manifest,
        sample_map_for_task,
    )

    manifest = (
        load_sample_manifest(ev.samples_manifest) if ev.samples_manifest else None
    )
    exact_samples = {
        task.name: sample_map_for_task(manifest, task.name) if manifest else None
        for task in tasks
    }
    for task in tasks:
        if exact_samples[task.name] is not None and task.limit is not None:
            raise ValueError(
                f"task {task.name}: exact samples and limit are mutually exclusive"
            )

    lm = _load_lm_model(cfg, model_path)
    merged: dict = {}
    completed_task_seeds = completed_task_seeds or set()
    generation_seeds: tuple[int | None, ...] = (
        tuple(int(seed) for seed in ev.generation_seeds)
        if ev.generation_seeds
        else (None,)
    )

    try:
        for task in tasks:
            for generation_seed in generation_seeds:
                if (
                    generation_seed is not None
                    and (
                        task.name,
                        generation_seed,
                    )
                    in completed_task_seeds
                ):
                    continue
                print(
                    f"[lmeval] task={task.name} seed={generation_seed} "
                    f"num_fewshot={task.num_fewshot} limit={task.limit}"
                )
                kwargs: dict = {
                    "model": lm,
                    "tasks": [task.name],
                    "num_fewshot": task.num_fewshot,
                    "apply_chat_template": ev.apply_chat_template,
                }
                if ev.fewshot_as_multiturn:
                    kwargs["fewshot_as_multiturn"] = True
                if ev.gen_kwargs:
                    gen_kwargs = dict(ev.gen_kwargs)
                    # Preserve do_sample so the task-level generation defaults
                    # cannot silently turn a paper-grade sampling run greedy.
                    if generation_seed is not None:
                        gen_kwargs["seed"] = generation_seed
                    kwargs["gen_kwargs"] = gen_kwargs
                sample_map = exact_samples[task.name]
                if sample_map is not None:
                    kwargs["samples"] = sample_map
                elif task.limit is not None:
                    kwargs["limit"] = task.limit
                if log_samples:
                    kwargs["log_samples"] = True
                if generation_seed is not None:
                    kwargs.update(
                        random_seed=42,
                        numpy_random_seed=42,
                        torch_random_seed=42,
                        fewshot_random_seed=42,
                    )

                batch = lm_eval.simple_evaluate(**kwargs)
                _merge_eval_results(merged, batch)
                if on_task_complete is not None:
                    on_task_complete(task, generation_seed, batch)
    finally:
        cleanup = getattr(lm, "clean", None) or getattr(lm, "cleanup", None)
        if callable(cleanup):
            cleanup()

    return merged
