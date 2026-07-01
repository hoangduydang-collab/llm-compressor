"""Accuracy gate.

Runs a small eval suite (Wikitext PPL + an MMLU/task slice by default) on the
quantized checkpoint via lm-eval over the vLLM backend, compares against a
baseline metrics JSON, and emits a pass/fail report.

Baseline JSON format (produced by ``--make-baseline`` on an unquantized model):
    {"wikitext": {"word_perplexity,none": 9.1}, "mmlu": {"acc,none": 0.65}}
"""

import json
from pathlib import Path

from pipeline.config import EvalConfig, EvalTask, PipelineConfig
from pipeline.lmeval_runner import evaluate_tasks


def _gate_metric(task: EvalTask, value: float, baseline: float | None, ev: EvalConfig) -> dict:
    entry = {
        "metric": task.metric,
        "value": value,
        "baseline": baseline,
        "higher_is_better": task.higher_is_better,
    }
    if baseline is None:
        entry["passed"] = None  # recorded, but no gate without a baseline
        return entry

    if task.higher_is_better:
        threshold = baseline * ev.recovery_threshold
        entry["threshold"] = threshold
        entry["passed"] = value >= threshold
    else:
        # perplexity-style: lower is better; cap relative increase.
        threshold = baseline * (1.0 + ev.max_ppl_increase)
        entry["threshold"] = threshold
        entry["passed"] = value <= threshold
    return entry


def _task_metric_value(task: EvalTask, task_results: dict) -> float:
    value = task_results.get(task.metric)
    if value is None:
        available = list(task_results.keys())
        raise KeyError(
            f"metric {task.metric!r} not in lm-eval results for task "
            f"{task.name!r}; available: {available}"
        )
    return float(value)


def _build_report(
    cfg: PipelineConfig,
    ckpt: Path,
    task_results_by_name: dict[str, dict],
) -> dict:
    ev = cfg.eval
    baseline: dict = {}
    if ev.baseline:
        with Path(ev.baseline).open("r", encoding="utf-8") as fh:
            baseline = json.load(fh)

    report: dict = {"checkpoint": str(ckpt), "tasks": {}, "passed": True}

    for task in ev.tasks:
        task_results = task_results_by_name[task.name]
        value = _task_metric_value(task, task_results)
        base_val = baseline.get(task.name, {}).get(task.metric)
        entry = _gate_metric(task, value, base_val, ev)
        report["tasks"][task.name] = entry
        if entry["passed"] is False:
            report["passed"] = False

    if not baseline:
        report["passed"] = None

    print("\n========== ACCURACY GATE ==========")
    for name, entry in report["tasks"].items():
        status = {True: "PASS", False: "FAIL", None: "N/A "}[entry["passed"]]
        print(
            f"[{status}] {name}: {entry['metric']}={entry['value']:.4f} "
            f"(baseline={entry['baseline']}, threshold={entry.get('threshold')})"
        )
    print(f"overall: {report['passed']}")
    print("===================================\n")
    return report


def run_eval_gate(cfg: PipelineConfig, ckpt: Path) -> dict:
    """Evaluate ``ckpt`` and gate against the baseline. Returns a report dict."""
    results = evaluate_tasks(str(ckpt), cfg, cfg.eval.tasks)
    task_results_by_name = {task.name: results["results"][task.name] for task in cfg.eval.tasks}
    return _build_report(cfg, ckpt, task_results_by_name)


def make_baseline(cfg: PipelineConfig, model_path: str, out_path: Path) -> dict:
    """Evaluate an (unquantized) model and write a baseline metrics JSON."""
    results = evaluate_tasks(model_path, cfg, cfg.eval.tasks)
    baseline: dict = {}
    for task in cfg.eval.tasks:
        task_results = results["results"][task.name]
        baseline[task.name] = {task.metric: _task_metric_value(task, task_results)}
    with out_path.open("w", encoding="utf-8") as fh:
        json.dump(baseline, fh, indent=2)
    print(f"wrote baseline -> {out_path}")
    return baseline
