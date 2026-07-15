"""Static benchmark evaluation via lm-eval with per-sample logging."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

from pipeline.config import EvalConfig, EvalTask, PipelineConfig
from pipeline.eval_gate import _gate_metric
from pipeline.evalsuite.health import (
    analyze_generation,
    summarize_generation_health,
    unwrap_singleton,
)
from pipeline.evalsuite.sampling import stable_sample_uid
from pipeline.lmeval_runner import evaluate_tasks
from pipeline.metrics_lmeval import (
    metric_base,
    require_task_results_or_aggregate,
    resolve_task_metric,
)


def _stable_attempt_uid(sample_uid: str, generation_seed: int) -> str:
    payload = f"{sample_uid}\0{generation_seed}".encode()
    return hashlib.sha256(payload).hexdigest()


def _extract_sample_row(
    sample: dict,
    task: EvalTask,
    generation_seed: int | None = None,
) -> dict:
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
    if generation_seed is not None:
        row.update(
            generation_seed=generation_seed,
            attempt_uid=_stable_attempt_uid(row["sample_uid"], generation_seed),
            source_doc=sample.get("doc"),
            generation_arguments=sample.get("arguments"),
            extracted_answer=extracted_answer,
        )

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
        value = sample.get(key)
        if isinstance(value, str):
            return value
        if isinstance(value, (list, tuple)) and value:
            return unwrap_singleton(value)
    return None


def _first_filtered_response(sample: dict):
    value = sample.get("filtered_resps")
    if isinstance(value, (list, tuple)) and value:
        return unwrap_singleton(value)
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


def _deduplicate_sample_rows(rows: list[dict]) -> list[dict]:
    """Collapse repeated lm-eval records without hiding conflicting evidence."""
    unique: dict[str, dict] = {}
    for row in rows:
        uid = str(row.get("attempt_uid") or row.get("sample_uid") or "")
        previous = unique.get(uid)
        if previous is None:
            unique[uid] = row
        elif previous != row:
            label = "attempt_uid" if row.get("attempt_uid") else "sample_uid"
            raise ValueError(f"conflicting duplicate {label}: {uid}")
    return list(unique.values())


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


def pending_eval_tasks(tasks: list[EvalTask], completed: set[str]) -> list[EvalTask]:
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
            merged.extend(_tag_samples(list(samples_map.get(subtask) or []), subtask))
        if merged:
            return merged

    prefix = f"{task_name}_"
    merged = []
    for subtask, values in samples_map.items():
        if subtask.startswith(prefix) and values:
            merged.extend(_tag_samples(list(values), subtask))
    return merged


def _select_task_filter_samples(samples: list[dict], task: EvalTask) -> list[dict]:
    """Select the lm-eval filter pipeline named by the configured metric.

    lm-eval logs one sample record per filter pipeline. Those records describe
    the same model attempt and therefore intentionally share an attempt UID;
    only the filter used by ``task.metric`` belongs in the checkpoint.
    """
    _, separator, expected_filter = task.metric.partition(",")
    if not separator or not expected_filter:
        return samples

    filter_names = {
        str(sample["filter"])
        for sample in samples
        if sample.get("filter") is not None
    }
    if not filter_names:
        # Older/synthetic lm-eval sample rows do not identify their filter.
        return samples

    selected = [
        sample for sample in samples if sample.get("filter") == expected_filter
    ]
    if not selected:
        available = ", ".join(sorted(filter_names))
        raise ValueError(
            f"configured filter {expected_filter!r} missing from logged samples; "
            f"available filters: {available}"
        )
    return selected


def checkpoint_task_result(
    *,
    task: EvalTask,
    batch: dict,
    aggregate: dict[str, dict[str, float]],
    aggregate_path: Path,
    samples_dir: Path,
    log_samples: bool,
    generation_seed: int | None = None,
    expected_generation_seeds: list[int] | None = None,
    progress_path: Path | None = None,
) -> list[dict]:
    """Persist one task's metrics (and optional samples) immediately after eval."""
    if generation_seed is not None and (
        expected_generation_seeds is None
        or generation_seed not in expected_generation_seeds
    ):
        raise ValueError(f"unexpected generation seed: {generation_seed}")
    task_results = require_task_results_or_aggregate(batch, task)
    rows: list[dict] = []
    if log_samples:
        raw = _select_task_filter_samples(
            _collect_task_samples(batch, task.name), task
        )
        if raw:
            rows = _deduplicate_sample_rows(
                [_extract_sample_row(s, task, generation_seed) for s in raw]
            )
            sample_path = samples_dir / f"{task.name}.jsonl"
            if generation_seed is not None and sample_path.is_file():
                existing = [
                    json.loads(line)
                    for line in sample_path.read_text(encoding="utf-8").splitlines()
                    if line.strip()
                ]
                rows = _deduplicate_sample_rows(existing + rows)
            _write_samples(sample_path, rows)
            _write_json_atomic(
                samples_dir.parent / "generation_health" / f"{task.name}.json",
                summarize_generation_health(rows),
            )

    if generation_seed is None:
        aggregate[task.name] = _numeric_metrics(task_results)
    else:
        task_metrics = {
            key: value
            for key, value in aggregate.get(task.name, {}).items()
            if key.startswith("pass_at_1_seed_")
        }
        correctness = [
            float(row["correct"])
            for row in rows
            if row.get("generation_seed") == generation_seed
            and row.get("correct") is not None
        ]
        task_metrics[f"pass_at_1_seed_{generation_seed}"] = (
            sum(correctness) / len(correctness) if correctness else 0.0
        )
        all_correctness = [
            float(row["correct"]) for row in rows if row.get("correct") is not None
        ]
        task_metrics["mean_pass_at_1"] = (
            sum(all_correctness) / len(all_correctness) if all_correctness else 0.0
        )
        task_metrics[task.metric] = task_metrics["mean_pass_at_1"]
        aggregate[task.name] = task_metrics
    aggregate_path.parent.mkdir(parents=True, exist_ok=True)
    _write_json_atomic(aggregate_path, aggregate)

    if generation_seed is not None and progress_path is not None:
        progress = (
            json.loads(progress_path.read_text(encoding="utf-8"))
            if progress_path.is_file()
            else {"schema_version": 1, "tasks": {}}
        )
        completed = set(progress.setdefault("tasks", {}).get(task.name, []))
        completed.add(generation_seed)
        progress["tasks"][task.name] = [
            seed for seed in expected_generation_seeds or [] if seed in completed
        ]
        _write_json_atomic(progress_path, progress)

    print(
        f"[evalsuite] checkpoint: {task.name} ({len(rows)} samples) -> {aggregate_path}"
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
    samples_dir = (
        Path(cfg.eval.samples_dir) if cfg.eval.samples_dir else out_dir / "samples"
    )
    out_dir.mkdir(parents=True, exist_ok=True)

    ev = cfg.eval
    log_samples = ev.log_samples
    aggregate_path = out_dir / "aggregate.json"
    aggregate = load_aggregate_checkpoint(aggregate_path)
    progress_path = out_dir / "seed_progress.json"
    progress = (
        json.loads(progress_path.read_text(encoding="utf-8"))
        if progress_path.is_file()
        else {"schema_version": 1, "tasks": {}}
    )
    completed_task_seeds = {
        (str(task_name), int(seed))
        for task_name, seeds in (progress.get("tasks") or {}).items()
        for seed in seeds
    }
    if ev.generation_seeds:
        expected_seeds = set(ev.generation_seeds)
        completed = {
            task.name
            for task in ev.tasks
            if {seed for name, seed in completed_task_seeds if name == task.name}
            == expected_seeds
        }
        pending = pending_eval_tasks(ev.tasks, completed)
    else:
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

        def _on_task_complete(
            task: EvalTask,
            generation_seed: int | None,
            batch: dict,
        ) -> None:
            rows = checkpoint_task_result(
                task=task,
                batch=batch,
                aggregate=aggregate,
                aggregate_path=aggregate_path,
                samples_dir=samples_dir,
                log_samples=log_samples,
                generation_seed=generation_seed,
                expected_generation_seeds=ev.generation_seeds or None,
                progress_path=progress_path,
            )
            all_samples[task.name] = rows

        evaluate_tasks(
            str(model_path),
            cfg,
            pending,
            log_samples=log_samples,
            completed_task_seeds=completed_task_seeds,
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
