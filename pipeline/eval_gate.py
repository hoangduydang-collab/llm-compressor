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


def _evaluate_task(model_path: str, cfg: PipelineConfig, task: EvalTask) -> dict:
    from pipeline._env import ensure_writable_caches

    ensure_writable_caches()

    import lm_eval
    import lm_eval.models  # noqa: F401  (populates the model registry)

    s = cfg.serve
    model_args = (
        f"pretrained={model_path},"
        f"tensor_parallel_size={s.tensor_parallel_size},"
        f"max_model_len={s.max_model_len},"
        f"gpu_memory_utilization={s.gpu_memory_utilization},"
        f"trust_remote_code={cfg.model.trust_remote_code},"
        f"dtype=auto,"
    )

    results = lm_eval.simple_evaluate(
        model=cfg.eval.backend,
        model_args=model_args,
        tasks=[task.name],
        num_fewshot=task.num_fewshot,
        limit=task.limit,
        apply_chat_template=cfg.eval.apply_chat_template,
    )
    return results["results"][task.name]


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


def run_eval_gate(cfg: PipelineConfig, ckpt: Path) -> dict:
    """Evaluate ``ckpt`` and gate against the baseline. Returns a report dict."""
    ev = cfg.eval
    baseline: dict = {}
    if ev.baseline:
        with Path(ev.baseline).open("r", encoding="utf-8") as fh:
            baseline = json.load(fh)

    report: dict = {"checkpoint": str(ckpt), "tasks": {}, "passed": True}

    for task in ev.tasks:
        task_results = _evaluate_task(str(ckpt), cfg, task)
        value = task_results.get(task.metric)
        if value is None:
            available = list(task_results.keys())
            raise KeyError(
                f"metric {task.metric!r} not in lm-eval results for task "
                f"{task.name!r}; available: {available}"
            )
        base_val = baseline.get(task.name, {}).get(task.metric)
        entry = _gate_metric(task, float(value), base_val, ev)
        report["tasks"][task.name] = entry
        if entry["passed"] is False:
            report["passed"] = False

    # If no baseline at all, the gate is "informational only".
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


def make_baseline(cfg: PipelineConfig, model_path: str, out_path: Path) -> dict:
    """Evaluate an (unquantized) model and write a baseline metrics JSON."""
    baseline: dict = {}
    for task in cfg.eval.tasks:
        task_results = _evaluate_task(model_path, cfg, task)
        baseline[task.name] = {task.metric: float(task_results[task.metric])}
    with out_path.open("w", encoding="utf-8") as fh:
        json.dump(baseline, fh, indent=2)
    print(f"wrote baseline -> {out_path}")
    return baseline
