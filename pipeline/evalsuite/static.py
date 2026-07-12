"""Static benchmark evaluation via lm-eval with per-sample logging."""

from __future__ import annotations

import json
from pathlib import Path

from pipeline.config import EvalConfig, EvalTask, PipelineConfig
from pipeline.evalsuite.health import (
    analyze_generation,
    summarize_generation_health,
)
from pipeline.evalsuite.sampling import stable_sample_uid
from pipeline.eval_gate import _gate_metric
from pipeline.lmeval_runner import evaluate_tasks
from pipeline.metrics_lmeval import (
    metric_base,
    require_task_results_or_aggregate,
    resolve_task_metric,
)


def _extract_sample_row(sample: dict, task: EvalTask) -> dict:
    """Normalize one lm-eval logged sample into a pairable JSONL row."""
    doc_id = sample.get("doc_id")
    if doc_id is None:
        doc_id = sample.get("doc_hash") or sample.get("id")

    base = metric_base(task.metric)
    candidates = [base, task.metric, "acc", "acc_norm", "exact_match", "perplexity"]
    if base == "exact_match":
        candidates.extend(
            [
                "exact_match,strict-match",
                "exact_match,get-answer",
                "exact_match,flexible-extract",
            ]
        )

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
    if metric_value is None:
        for key, val in sample.items():
            if (
                isinstance(val, (int, float, bool))
                and isinstance(key, str)
                and key.startswith(f"{base},")
                and "stderr" not in key
            ):
                metric_value = float(val)
                used_metric = key
                break

    subtask = str(sample.get("_eval_subtask") or task.name)
    response = _first_response(sample)
    extracted_answer = _first_filtered_response(sample)
    response_token_ids = sample.get("response_token_ids")
    if not isinstance(response_token_ids, list):
        response_token_ids = None
    max_gen_toks = sample.get("max_gen_toks")
    if not isinstance(max_gen_toks, int):
        max_gen_toks = None
    row: dict = {
        "sample_uid": stable_sample_uid(task.name, subtask, doc_id),
        "task": task.name,
        "subtask": subtask,
        "doc_id": doc_id,
        "target": sample.get("target"),
        "response": response,
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

    if isinstance(response, str) or response_token_ids is not None:
        row["health"] = analyze_generation(
            response if isinstance(response, str) else None,
            token_ids=response_token_ids,
            max_gen_toks=max_gen_toks,
            extracted_answer=extracted_answer,
        )
    else:
        row["health"] = {"applicable": False}
    if response_token_ids is not None:
        row["response_token_ids"] = response_token_ids
    return row


def _first_response(sample: dict):
    for key in ("resps", "response", "filtered_resps"):
        val = sample.get(key)
        if isinstance(val, list) and val:
            return val[0]
        if isinstance(val, str):
            return val
    return None


def _first_filtered_response(sample: dict):
    value = sample.get("filtered_resps")
    if isinstance(value, list) and value:
        return value[0]
    return None


def _write_samples(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = sorted(
        rows,
        key=lambda r: (str(r.get("sample_uid") or ""), str(r.get("doc_id"))),
    )
    with path.open("w", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")


def _write_json_atomic(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(data, indent=2), encoding="utf-8")
    temporary.replace(path)


def _numeric_metrics(task_results: dict) -> dict[str, float]:
    return {
        k: float(v)
        for k, v in task_results.items()
        if isinstance(v, (int, float)) and "stderr" not in k
    }


def load_aggregate_checkpoint(path: Path) -> dict[str, dict[str, float]]:
    """Load per-task metrics from a prior partial run, if present."""
    if not path.is_file():
        return {}
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        return {}
    out: dict[str, dict[str, float]] = {}
    for task_name, metrics in data.items():
        if isinstance(metrics, dict):
            out[task_name] = _numeric_metrics(metrics)
    return out


def pending_eval_tasks(
    tasks: list[EvalTask], completed: set[str]
) -> list[EvalTask]:
    return [t for t in tasks if t.name not in completed]


def _tag_samples(samples: list[dict], subtask: str) -> list[dict]:
    return [{**sample, "_eval_subtask": subtask} for sample in samples]


def _collect_task_samples(batch: dict, task_name: str) -> list[dict]:
    """Collect samples while preserving their leaf-task namespace."""
    samples_map = batch.get("samples") or {}
    direct = samples_map.get(task_name)
    if direct:
        return _tag_samples(list(direct), task_name)

    subtasks = (batch.get("group_subtasks") or {}).get(task_name)
    if subtasks:
        merged: list[dict] = []
        for subtask in subtasks:
            merged.extend(
                _tag_samples(list(samples_map.get(subtask) or []), subtask)
            )
        if merged:
            return merged

    prefix = f"{task_name}_"
    merged = []
    for subtask, values in samples_map.items():
        if subtask.startswith(prefix) and values:
            merged.extend(_tag_samples(list(values), subtask))
    return merged


def checkpoint_task_result(
    *,
    task: EvalTask,
    batch: dict,
    aggregate: dict[str, dict[str, float]],
    aggregate_path: Path,
    samples_dir: Path,
    log_samples: bool,
) -> list[dict]:
    """Persist one task's metrics (and optional samples) immediately after eval."""
    task_results = require_task_results_or_aggregate(batch, task)
    aggregate[task.name] = _numeric_metrics(task_results)
    aggregate_path.parent.mkdir(parents=True, exist_ok=True)
    aggregate_path.write_text(json.dumps(aggregate, indent=2), encoding="utf-8")

    rows: list[dict] = []
    if log_samples:
        raw = _collect_task_samples(batch, task.name)
        if raw:
            rows = [_extract_sample_row(s, task) for s in raw]
            _write_samples(samples_dir / f"{task.name}.jsonl", rows)
            _write_json_atomic(
                samples_dir.parent / "generation_health" / f"{task.name}.json",
                summarize_generation_health(rows),
            )

    print(
        f"[evalsuite] checkpoint: {task.name} "
        f"({len(rows)} samples) -> {aggregate_path}"
    )
    return rows


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

    for task in ev.tasks:
        metrics = aggregate.get(task.name)
        if metrics is None:
            continue
        value, resolved = resolve_task_metric(task, metrics)
        base_val = baseline.get(task.name, {}).get(task.metric)
        if base_val is None and task.name in baseline:
            base_val = baseline[task.name].get(resolved)
            if base_val is None:
                base_val = baseline[task.name].get(metric_base(task.metric))
        entry = _gate_metric(task, value, base_val, ev)
        entry["resolved_metric"] = resolved
        report["tasks"][task.name] = entry
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
    """Run static lm-eval tasks with per-task checkpointing.

    After each task, writes ``aggregate.json`` and ``samples/<task>.jsonl``
    (when ``log_samples`` is enabled). Re-running the same ``out_dir`` skips
    tasks already present in ``aggregate.json``.

    Returns ``{"gate": eval_report, "aggregate": {...}, "samples_dir": Path}``.
    """
    model_path = Path(model_path)
    out_dir = Path(out_dir)
    samples_dir = Path(cfg.eval.samples_dir) if cfg.eval.samples_dir else out_dir / "samples"
    out_dir.mkdir(parents=True, exist_ok=True)

    ev = cfg.eval
    log_samples = ev.log_samples
    aggregate_path = out_dir / "aggregate.json"
    aggregate = load_aggregate_checkpoint(aggregate_path)
    completed = set(aggregate)
    pending = pending_eval_tasks(ev.tasks, completed)
    all_samples: dict[str, list[dict]] = {}

    if completed:
        print(
            f"[evalsuite] resuming: {len(completed)} task(s) already in "
            f"{aggregate_path}: {', '.join(sorted(completed))}"
        )
    if pending:
        pending_names = ", ".join(t.name for t in pending)
        print(f"[evalsuite] pending: {pending_names}")

    meta = {
        "model_path": str(model_path),
        "backend": ev.backend,
        "tasks": [t.name for t in ev.tasks],
        "completed_tasks": sorted(completed),
        "pending_tasks": [t.name for t in pending],
        "log_samples": log_samples,
        "checkpoint_after_each_task": True,
    }
    with (out_dir / "eval_meta.json").open("w", encoding="utf-8") as fh:
        json.dump(meta, fh, indent=2)

    if pending:

        def _on_task_complete(task: EvalTask, batch: dict) -> None:
            rows = checkpoint_task_result(
                task=task,
                batch=batch,
                aggregate=aggregate,
                aggregate_path=aggregate_path,
                samples_dir=samples_dir,
                log_samples=log_samples,
            )
            all_samples[task.name] = rows

        evaluate_tasks(
            str(model_path),
            cfg,
            pending,
            log_samples=log_samples,
            on_task_complete=_on_task_complete,
        )

    for task in ev.tasks:
        if task.name not in all_samples and log_samples:
            sample_path = samples_dir / f"{task.name}.jsonl"
            if sample_path.is_file():
                with sample_path.open(encoding="utf-8") as fh:
                    all_samples[task.name] = [
                        json.loads(line) for line in fh if line.strip()
                    ]

    meta["completed_tasks"] = sorted(aggregate)
    meta["pending_tasks"] = [t.name for t in ev.tasks if t.name not in aggregate]
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
