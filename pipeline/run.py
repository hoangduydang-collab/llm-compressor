"""One-command pipeline CLI.

Examples
--------
Quantize + verify-serve + gate, all stages::

    python -m pipeline.run --config pipeline/configs/qwen3_30b_a3b_w4afp8_gptq.yaml

Only quantize::

    python -m pipeline.run --config <cfg.yaml> --stage quantize

Produce a baseline metrics JSON from the unquantized model::

    python -m pipeline.run --config <cfg.yaml> --make-baseline baseline.json

Override config fields from the command line (dotted keys)::

    python -m pipeline.run --config <cfg.yaml> --set quantization.scheme=W4A8
"""

import argparse
import sys
from pathlib import Path

from pipeline import versioning
from pipeline.config import PipelineConfig, load_config

STAGES = ("quantize", "serve", "eval", "all")


def _apply_overrides(cfg: PipelineConfig, overrides: list[str]) -> None:
    """Apply ``a.b.c=value`` overrides onto the (nested dataclass) config."""
    import yaml

    for item in overrides:
        if "=" not in item:
            raise ValueError(f"--set expects key=value, got {item!r}")
        key, raw = item.split("=", 1)
        value = yaml.safe_load(raw)  # parse ints/floats/bools/lists
        obj = cfg
        parts = key.split(".")
        for p in parts[:-1]:
            obj = getattr(obj, p)
        if not hasattr(obj, parts[-1]):
            raise AttributeError(f"unknown config field: {key}")
        setattr(obj, parts[-1], value)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Quantization pipeline runner")
    parser.add_argument("--config", required=True, help="path to a pipeline YAML config")
    parser.add_argument(
        "--stage", default="all", choices=STAGES, help="which stage(s) to run"
    )
    parser.add_argument(
        "--set", dest="overrides", action="append", default=[],
        help="override a config field, e.g. --set quantization.scheme=W4A8",
    )
    parser.add_argument(
        "--make-baseline", metavar="OUT.json",
        help="evaluate the (unquantized) model.id and write a baseline JSON, then exit",
    )
    parser.add_argument(
        "--checkpoint", help="for --stage serve/eval: use an existing checkpoint dir"
    )
    args = parser.parse_args(argv)

    cfg = load_config(args.config)
    if args.overrides:
        _apply_overrides(cfg, args.overrides)
        cfg.validate()

    if args.make_baseline:
        from pipeline.eval_gate import make_baseline

        make_baseline(cfg, cfg.model.id, Path(args.make_baseline))
        return 0

    # Resolve checkpoint + run dir.
    if args.stage in ("serve", "eval") and args.checkpoint:
        ckpt = Path(args.checkpoint)
        run_dir = ckpt.parent if ckpt.name == "checkpoint" else ckpt
    else:
        run_dir = versioning.create_run_dir(cfg)
        versioning.write_config(run_dir, cfg)
        ckpt = versioning.checkpoint_dir(run_dir)

    print(f"[pipeline] run dir: {run_dir}")
    versioning.write_metadata(run_dir, cfg)

    overall_ok = True

    if args.stage in ("quantize", "all"):
        from pipeline.quantize import run_quantize

        ckpt = run_quantize(cfg, run_dir)
        print(f"[pipeline] checkpoint saved -> {ckpt}")

    if args.stage in ("serve", "all") and cfg.serve.enabled:
        import json

        from pipeline.serve_verify import verify_serve

        report = verify_serve(cfg, ckpt)
        (run_dir / "serve_report.json").write_text(json.dumps(report, indent=2))
        if report.get("loaded") and not report.get("sane_output", True):
            overall_ok = False

    if args.stage in ("eval", "all") and cfg.eval.enabled:
        from pipeline.eval_gate import run_eval_gate

        report = run_eval_gate(cfg, ckpt)
        versioning.write_eval_report(run_dir, report)
        if report.get("passed") is False:
            overall_ok = False

    print(f"[pipeline] done. overall_ok={overall_ok}")
    return 0 if overall_ok else 1


if __name__ == "__main__":
    sys.exit(main())
