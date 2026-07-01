"""Render compare.json into a human-readable Markdown report."""

from __future__ import annotations

import json
from pathlib import Path


def _pct(x: float | None) -> str:
    if x is None:
        return "n/a"
    return f"{100 * x:.2f}%"


def _num(x: float | None, digits: int = 4) -> str:
    if x is None:
        return "n/a"
    return f"{x:.{digits}f}"


def render_report(compare: dict, out_path: str | Path) -> Path:
    out_path = Path(out_path)
    la = compare.get("label_a", "A")
    lb = compare.get("label_b", "B")
    lines: list[str] = [
        "# Quantized vs Original Evaluation Report",
        "",
        f"- **{la}**: `{compare.get('dir_a')}`",
        f"- **{lb}**: `{compare.get('dir_b')}`",
        "",
    ]

    summary = compare.get("summary") or {}
    if summary:
        lines.extend(
            [
                "## Summary",
                "",
                f"| Metric | Value |",
                f"|--------|-------|",
                f"| Tasks compared | {summary.get('tasks_compared', 'n/a')} |",
                f"| Samples paired (micro) | {summary.get('samples_paired', 'n/a')} |",
                f"| Micro flip-rate | {_pct(summary.get('micro_flip_rate'))} |",
                f"| Micro regression rate ({la} ok, {lb} fail) | {_pct(summary.get('micro_regression_rate'))} |",
                f"| Micro recovery rate ({la} fail, {lb} ok) | {_pct(summary.get('micro_recovery_rate'))} |",
                f"| Macro flip-rate (task mean) | {_pct(summary.get('macro_flip_rate'))} |",
                f"| Macro delta acc ({lb} - {la}) | {_num(summary.get('macro_delta_acc'))} |",
                "",
            ]
        )

    lines.extend(["## Static tasks", ""])
    lines.append(
        f"| Task | N | acc {la} | acc {lb} | delta | flip | regress | recover | kappa | McNemar p |"
    )
    lines.append("|------|---|---------|---------|-------|------|---------|---------|-------|-----------|")

    for task, res in sorted((compare.get("tasks") or {}).items()):
        if res.get("kind") == "perplexity":
            continue
        if res.get("n_paired", 0) == 0:
            continue
        mcnemar_p = res.get("mcnemar", {}).get("p_value")
        lines.append(
            f"| {task} | {res['n_paired']} | {_pct(res.get('acc_a'))} | {_pct(res.get('acc_b'))} | "
            f"{_num(res.get('delta'))} | {_pct(res.get('flip_rate'))} | "
            f"{res.get('regressions_a_correct_b_wrong', 0)} | {res.get('recoveries_a_wrong_b_correct', 0)} | "
            f"{_num(res.get('cohens_kappa'))} | {_num(mcnemar_p, 3)} |"
        )

    lines.append("")
    ppl_tasks = [t for t, r in (compare.get("tasks") or {}).items() if r.get("kind") == "perplexity"]
    if ppl_tasks:
        lines.extend(["## Perplexity", ""])
        lines.append(f"| Task | N | PPL {la} | PPL {lb} | delta | rel increase |")
        lines.append("|------|---|--------|--------|-------|--------------|")
        for task in ppl_tasks:
            res = compare["tasks"][task]
            lines.append(
                f"| {task} | {res.get('n_paired', 0)} | {_num(res.get('mean_a'))} | "
                f"{_num(res.get('mean_b'))} | {_num(res.get('delta'))} | "
                f"{_pct(res.get('rel_increase'))} |"
            )
        lines.append("")

    agentic = compare.get("agentic")
    if agentic and agentic.get("n_paired", 0) > 0:
        mcnemar_p = agentic.get("mcnemar", {}).get("p_value")
        lines.extend(
            [
                "## Agentic (tau2)",
                "",
                f"| N | success {la} | success {lb} | delta | flip | regress | recover | kappa | McNemar p |",
                f"|---|-------------|-------------|-------|------|---------|---------|-------|-----------|",
                f"| {agentic['n_paired']} | {_pct(agentic.get('acc_a'))} | {_pct(agentic.get('acc_b'))} | "
                f"{_num(agentic.get('delta'))} | {_pct(agentic.get('flip_rate'))} | "
                f"{agentic.get('regressions_a_correct_b_wrong', 0)} | "
                f"{agentic.get('recoveries_a_wrong_b_correct', 0)} | "
                f"{_num(agentic.get('cohens_kappa'))} | {_num(mcnemar_p, 3)} |",
                "",
                "> Agentic scores use tau2 task reward >= threshold as success. "
                "User-simulator LLM noise may require `num_trials > 1`.",
                "",
            ]
        )

    lines.extend(
        [
            "## Deferred: serving performance (concurrency / speed)",
            "",
            "Throughput, TTFT, and ITL benchmarks are not run by this pipeline stage.",
            "Use the aiperf suite in `benchmarks/llm-perf-benchmarks/`:",
            "",
            "```bash",
            "PROFILE=<model> ENGINE=vllm bash scripts/preflight.sh",
            "PROFILE=<model> ENGINE=vllm bash scripts/run_all.sh",
            "```",
            "",
        ]
    )

    out_path.write_text("\n".join(lines), encoding="utf-8")
    return out_path


def write_report_from_compare_json(compare_json: str | Path, report_path: str | Path | None = None) -> Path:
    compare_json = Path(compare_json)
    with compare_json.open("r", encoding="utf-8") as fh:
        data = json.load(fh)
    report_path = report_path or compare_json.parent / "report.md"
    return render_report(data, report_path)
