"""Static benchmark evaluation via lm-eval with per-sample logging."""

from __future__ import annotations

import json
from pathlib import Path

from pipeline.config import EvalConfig, EvalTask, PipelineConfig
from pipeline.eval_gate import _gate_metric


def _model_args(cfg: PipelineConfig, model_path: str) -> str:
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


def _metric_base(metric: str) -> str:
    """``acc,none`` -> ``acc``."""
    return metric.split(",")[0]


def _extract_sample_row(sample: dict, task: EvalTask) -> dict:
    """Normalize one lm-eval logged sample into a pairable JSONL row."""
    doc_id = sample.get("doc_id")
    if doc_id is None:
        doc_id = sample.get("doc_hash") or sample.get("id")

    base = _metric_base(task.metric)
    candidates = [base, task.metric, "acc", "acc_norm", "exact_match", "perplexity"]

    metric_value = None
    used_metric = None
    for key in candidates:
        if key in sample and isinstance(sample[key], (int, float, bool)):
            metric_value = float(sample[key])
            used_metric = key
            break
    if metric_value is None:
        nested = sample.get("metrics")
        if isinstance(nested, dict):
            for key in candidates:
                if key in nested and isinstance(nested[key], (int, float, bool)):
                    metric_value = float(nested[key])
                    used_metric = key
                    break

    row: dict = {
        "doc_id": doc_id,
        "target": sample.get("target"),
        "response": _first_response(sample),
        "metric": used_metric or base,
        "metric_value": metric_value,
    }

    if task.higher_is_better:
        if metric_value is not None:
            row["correct"] = int(metric_value >= 0.5)
        else:
            row["correct"] = None
    else:
        row["correct"] = None  # perplexity: compare via metric_value only

    return row


def _first_response(sample: dict):
    for key in ("filtered_resps", "resps", "response"):
        val = sample.get(key)
        if isinstance(val, list) and val:
            return val[0]
        if isinstance(val, str):
            return val
    return None


def _write_samples(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = sorted(rows, key=lambda r: (str(r.get("doc_id")),))
    with path.open("w", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")


def _evaluate_task(model_path: str, cfg: PipelineConfig, task: EvalTask, log_samples: bool) -> dict:
    from pipeline._env import ensure_writable_caches

    ensure_writable_caches()

    import lm_eval
    import lm_eval.models  # noqa: F401

    ev = cfg.eval
    return lm_eval.simple_evaluate(
        model=ev.backend,
        model_args=_model_args(cfg, model_path),
        tasks=[task.name],
        num_fewshot=task.num_fewshot,
        limit=task.limit,
        apply_chat_template=ev.apply_chat_template,
        log_samples=log_samples,
    )


def _build_gate_report(
    cfg: PipelineConfig,
    ckpt: Path,
    aggregate: dict[str, dict[str, float]],
    ev: EvalConfig,
) -> dict:
    baseline: dict = {}
    if ev.baseline:
        with Path(ev.baseline).open("r", encoding="utf-8") as fh:
            baseline = json.load(fh)

    report: dict = {"checkpoint": str(ckpt), "tasks": {}, "passed": True}
    task_by_name = {t.name: t for t in ev.tasks}

    for task_name, metrics in aggregate.items():
        task = task_by_name[task_name]
        value = metrics.get(task.metric)
        if value is None:
            value = metrics.get(_metric_base(task.metric))
        if value is None:
            raise KeyError(
                f"metric {task.metric!r} missing for task {task_name!r}; "
                f"available: {list(metrics.keys())}"
            )
        base_val = baseline.get(task_name, {}).get(task.metric)
        if base_val is None and task_name in baseline:
            base_val = baseline[task_name].get(_metric_base(task.metric))
        entry = _gate_metric(task, float(value), base_val, ev)
        report["tasks"][task_name] = entry
        if entry["passed"] is False:
            report["passed"] = False

    if not baseline:
        report["passed"] = None
    return report


def run_static_eval(
    cfg: PipelineConfig,
    model_path: str | Path,
    out_dir: str | Path,
) -> dict:
    """Run all static lm-eval tasks; write aggregate + per-sample JSONL.

    Returns ``{"gate": eval_report, "aggregate": {...}, "samples_dir": Path}``.
    """
    model_path = Path(model_path)
    out_dir = Path(out_dir)
    samples_dir = Path(cfg.eval.samples_dir) if cfg.eval.samples_dir else out_dir / "samples"
    out_dir.mkdir(parents=True, exist_ok=True)

    ev = cfg.eval
    log_samples = ev.log_samples
    aggregate: dict[str, dict[str, float]] = {}
    all_samples: dict[str, list[dict]] = {}

    for task in ev.tasks:
        print(f"[evalsuite] static task: {task.name} (limit={task.limit})")
        results = _evaluate_task(str(model_path), cfg, task, log_samples=log_samples)
        task_results = results["results"][task.name]
        aggregate[task.name] = {
            k: float(v) if isinstance(v, (int, float)) else v
            for k, v in task_results.items()
            if isinstance(v, (int, float))
        }

        if log_samples and results.get("samples"):
            raw = results["samples"].get(task.name) or []
            rows = [_extract_sample_row(s, task) for s in raw]
            all_samples[task.name] = rows
            _write_samples(samples_dir / f"{task.name}.jsonl", rows)

    with (out_dir / "aggregate.json").open("w", encoding="utf-8") as fh:
        json.dump(aggregate, fh, indent=2)

    meta = {
        "model_path": str(model_path),
        "backend": ev.backend,
        "tasks": [t.name for t in ev.tasks],
        "log_samples": log_samples,
    }
    with (out_dir / "eval_meta.json").open("w", encoding="utf-8") as fh:
        json.dump(meta, fh, indent=2)

    gate = _build_gate_report(cfg, model_path, aggregate, ev)

    print("\n========== STATIC EVAL (evalsuite) ==========")
    for name, entry in gate["tasks"].items():
        status = {True: "PASS", False: "FAIL", None: "N/A "}[entry["passed"]]
        print(
            f"[{status}] {name}: {entry['metric']}={entry['value']:.4f} "
            f"(baseline={entry['baseline']}, threshold={entry.get('threshold')})"
        )
    print(f"overall gate: {gate['passed']}")
    print("============================================\n")

    return {
        "gate": gate,
        "aggregate": aggregate,
        "samples_dir": samples_dir,
        "sample_counts": {k: len(v) for k, v in all_samples.items()},
    }
