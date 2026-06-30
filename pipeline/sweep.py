"""Method x format sweep driver (M2 Phase 3 comparison matrix).

Runs the same base config across a grid of (method, scheme) cells, then collects
each run's eval_report.json into a single comparison table (CSV + JSON).

    python -m pipeline.sweep --config pipeline/configs/minimax_m3.yaml \
        --methods gptq awq smoothquant+gptq smoothquant+awq \
        --schemes W4A8 W4AFP8

Tier-2 cells (AutoRound, SpinQuant rotations) are added by extending --methods.
"""

import argparse
import csv
import json
from pathlib import Path

from pipeline.config import load_config
from pipeline.run import main as run_main

CORE_METHODS = ["gptq", "awq", "smoothquant+gptq", "smoothquant+awq"]
CORE_SCHEMES = ["W4A8", "W4AFP8"]


def _latest_run_dir(output_dir: Path, run_slug: str) -> Path | None:
    base = output_dir / run_slug
    if not base.exists():
        return None
    runs = sorted(p for p in base.iterdir() if p.is_dir())
    return runs[-1] if runs else None


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="method x format sweep")
    parser.add_argument("--config", required=True)
    parser.add_argument("--methods", nargs="+", default=CORE_METHODS)
    parser.add_argument("--schemes", nargs="+", default=CORE_SCHEMES)
    parser.add_argument(
        "--stage", default="all", help="passed through to pipeline.run --stage"
    )
    parser.add_argument("--out", default="sweep_results", help="output prefix")
    args = parser.parse_args(argv)

    base_cfg = load_config(args.config)
    output_dir = Path(base_cfg.output_dir)

    rows: list[dict] = []
    for method in args.methods:
        for scheme in args.schemes:
            print(f"\n##### SWEEP CELL: method={method} scheme={scheme} #####")
            rc = run_main(
                [
                    "--config", args.config,
                    "--stage", args.stage,
                    "--set", f"quantization.method={method}",
                    "--set", f"quantization.scheme={scheme}",
                ]
            )
            # Reconstruct the run_slug to locate the produced artifacts.
            cfg = load_config(args.config)
            cfg.quantization.method = method
            cfg.quantization.scheme = scheme
            run_dir = _latest_run_dir(output_dir, cfg.run_slug)

            row: dict = {"method": method, "scheme": scheme, "exit_code": rc}
            report_path = run_dir / "eval_report.json" if run_dir else None
            if report_path and report_path.exists():
                report = json.loads(report_path.read_text())
                row["gate_passed"] = report.get("passed")
                for tname, entry in report.get("tasks", {}).items():
                    row[f"{tname}:{entry['metric']}"] = entry["value"]
            rows.append(row)

    # Write the comparison table.
    json_path = Path(f"{args.out}.json")
    json_path.write_text(json.dumps(rows, indent=2))

    if rows:
        fieldnames = sorted({k for r in rows for k in r})
        csv_path = Path(f"{args.out}.csv")
        with csv_path.open("w", newline="", encoding="utf-8") as fh:
            writer = csv.DictWriter(fh, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)
        print(f"\n[sweep] wrote {json_path} and {csv_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
