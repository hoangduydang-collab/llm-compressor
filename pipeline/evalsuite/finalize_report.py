"""Finalize eval_report.json from an existing evalsuite/ aggregate (no GPU)."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from pipeline.config import load_config
from pipeline.evalsuite.static import _build_gate_report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Build eval_report.json from evalsuite/aggregate.json"
    )
    parser.add_argument(
        "--run-dir",
        required=True,
        help="artifact run dir containing evalsuite/aggregate.json",
    )
    parser.add_argument(
        "--config",
        default="pipeline/configs/qwen3_30b_a3b.yaml",
        help="pipeline config (for task metric definitions)",
    )
    args = parser.parse_args(argv)

    run_dir = Path(args.run_dir)
    evalsuite = run_dir / "evalsuite"
    agg_path = evalsuite / "aggregate.json"
    if not agg_path.exists():
        print(f"missing {agg_path}", file=sys.stderr)
        return 1

    cfg = load_config(args.config)
    aggregate = json.loads(agg_path.read_text(encoding="utf-8"))
    ckpt = run_dir / "checkpoint"
    if not ckpt.is_dir():
        ckpt = run_dir

    gate = _build_gate_report(cfg, ckpt, aggregate, cfg.eval)
    out = run_dir / "eval_report.json"
    out.write_text(json.dumps(gate, indent=2), encoding="utf-8")

    print(f"wrote {out}")
    print(f"overall gate: {gate.get('passed')}")
    for name, entry in gate.get("tasks", {}).items():
        print(f"  {name}: {entry.get('resolved_metric', entry['metric'])}={entry['value']:.4f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
