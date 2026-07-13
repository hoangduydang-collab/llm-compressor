"""CLI for the evaluation suite: run evals and compare quant vs original."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from pipeline.config import EvalTask, PipelineConfig, load_config
from pipeline.evalsuite.agentic import run_agentic_eval
from pipeline.evalsuite.compare import compare_eval_dirs
from pipeline.evalsuite.report import render_report
from pipeline.evalsuite.static import run_static_eval


def _select_tasks(tasks: list[EvalTask], requested: str) -> list[EvalTask]:
    names = [name.strip() for name in requested.split(",") if name.strip()]
    if not names:
        raise ValueError("--tasks must contain at least one task name")
    by_name = {task.name: task for task in tasks}
    unknown = [name for name in names if name not in by_name]
    if unknown:
        raise ValueError(
            f"unknown eval task(s) {unknown}; configured: {sorted(by_name)}"
        )
    if len(set(names)) != len(names):
        raise ValueError("--tasks contains duplicate task names")
    return [by_name[name] for name in names]


def _cmd_run(args: argparse.Namespace) -> int:
    cfg = load_config(args.config)
    if args.overrides:
        from pipeline.run import _apply_overrides

        _apply_overrides(cfg, args.overrides)
        cfg.validate()
    if args.agentic:
        cfg.agentic.enabled = True
    if args.tasks:
        cfg.eval.tasks = _select_tasks(cfg.eval.tasks, args.tasks)
    if args.samples_manifest:
        cfg.eval.samples_manifest = args.samples_manifest

    model_path = args.model or cfg.model.id
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    static_result = run_static_eval(cfg, model_path, out_dir)

    agentic_result = None
    if cfg.agentic.enabled:
        agentic_result = run_agentic_eval(
            cfg,
            model_path,
            out_dir,
            agent_base=args.agent_base,
            agent_model=args.agent_model,
        )

    manifest = {
        "model": str(model_path),
        "out_dir": str(out_dir),
        "static": {
            "gate_passed": static_result["gate"].get("passed"),
            "sample_counts": static_result.get("sample_counts"),
        },
        "agentic": agentic_result.get("aggregate") if agentic_result else None,
    }
    import json

    with (out_dir / "manifest.json").open("w", encoding="utf-8") as fh:
        json.dump(manifest, fh, indent=2)

    print(f"[evalsuite] wrote results -> {out_dir}")
    return 0


def _cmd_compare(args: argparse.Namespace) -> int:
    cfg = None
    if args.config:
        cfg = load_config(args.config)

    report = compare_eval_dirs(
        args.a,
        args.b,
        out_dir=args.out,
        cfg=cfg,
        label_a=args.label_a,
        label_b=args.label_b,
    )

    out_dir = Path(args.out)
    render_report(report, out_dir / "report.md")
    print(f"[evalsuite] compare.json + report.md -> {out_dir}")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Evaluation suite for quant vs original comparison")
    sub = parser.add_subparsers(dest="command", required=True)

    run_p = sub.add_parser("run", help="Run static (+ optional agentic) eval on one model")
    run_p.add_argument("--config", required=True, help="pipeline YAML config")
    run_p.add_argument("--model", help="model path or HF id (default: config.model.id)")
    run_p.add_argument("--out", required=True, help="output directory for eval artifacts")
    run_p.add_argument(
        "--tasks",
        help="comma-separated configured task names for this shard",
    )
    run_p.add_argument(
        "--samples-manifest",
        help="exact shared sample-index manifest for paired evaluation",
    )
    run_p.add_argument("--agent-base", help="OpenAI-compatible base URL for agentic (tau2)")
    run_p.add_argument("--agent-model", help="model name served at agent-base")
    run_p.add_argument(
        "--agentic",
        action="store_true",
        help="enable tau2 agentic benchmark eval (also requires agentic.* in config)",
    )
    run_p.add_argument(
        "--set", dest="overrides", action="append", default=[],
        help="override config field, e.g. --set eval.tasks[0].limit=8",
    )
    run_p.set_defaults(func=_cmd_run)

    cmp_p = sub.add_parser("compare", help="Post-hoc compare two eval output dirs")
    cmp_p.add_argument("--a", required=True, help="original model eval dir")
    cmp_p.add_argument("--b", required=True, help="quantized model eval dir")
    cmp_p.add_argument("--out", required=True, help="comparison output dir")
    cmp_p.add_argument("--config", help="optional config for compare thresholds")
    cmp_p.add_argument("--label-a", default="original")
    cmp_p.add_argument("--label-b", default="quantized")
    cmp_p.set_defaults(func=_cmd_compare)

    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
