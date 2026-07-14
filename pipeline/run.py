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
from pipeline.distributed import DistributedContext

STAGES = ("quantize", "serve", "eval", "all")


def _create_distributed_run_dir(
    cfg: PipelineConfig, dist_ctx: DistributedContext
) -> Path:
    """Create a run directory on rank zero and share it with every rank."""
    local_path = versioning.create_run_dir(cfg) if dist_ctx.is_source else None
    return dist_ctx.broadcast_path(local_path)


def _apply_overrides(cfg: PipelineConfig, overrides: list[str]) -> None:
    """Apply ``a.b.c=value`` overrides onto the (nested dataclass) config.

    Intermediate dict fields (e.g. ``serve.sglang_kwargs.disable_cuda_graph``)
    are supported; the leaf key is written into the dict.
    """
    import yaml

    for item in overrides:
        if "=" not in item:
            raise ValueError(f"--set expects key=value, got {item!r}")
        key, raw = item.split("=", 1)
        value = yaml.safe_load(raw)  # parse ints/floats/bools/lists
        obj = cfg
        parts = key.split(".")
        for p in parts[:-1]:
            nxt = getattr(obj, p) if not isinstance(obj, dict) else obj[p]
            obj = nxt
        leaf = parts[-1]
        if isinstance(obj, dict):
            obj[leaf] = value
        elif hasattr(obj, leaf):
            setattr(obj, leaf, value)
        else:
            raise AttributeError(f"unknown config field: {key}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Quantization pipeline runner")
    parser.add_argument(
        "--config", required=True, help="path to a pipeline YAML config"
    )
    parser.add_argument(
        "--stage", default="all", choices=STAGES, help="which stage(s) to run"
    )
    parser.add_argument(
        "--set",
        dest="overrides",
        action="append",
        default=[],
        help="override a config field, e.g. --set quantization.scheme=W4A8",
    )
    parser.add_argument(
        "--make-baseline",
        metavar="OUT.json",
        help="evaluate the (unquantized) model.id and write a baseline JSON, then exit",
    )
    parser.add_argument(
        "--checkpoint", help="for --stage serve/eval: use an existing checkpoint dir"
    )
    parser.add_argument(
        "--agentic",
        action="store_true",
        help="enable tau2 agentic benchmark eval (also requires agentic.* in config)",
    )
    parser.add_argument(
        "--agent-base",
        help=(
            "OpenAI-compatible base URL for the agent under test "
            "(overrides agentic.agent_base)"
        ),
    )
    parser.add_argument(
        "--agent-model",
        help="model name served at --agent-base (overrides agentic.agent_model)",
    )
    parser.add_argument(
        "--evidence-only",
        action="store_true",
        help=(
            "run quantization/calibration but do not save a checkpoint; required "
            "for partial-layer speed smokes"
        ),
    )
    args = parser.parse_args(argv)

    dist_ctx = DistributedContext.from_environment()
    try:
        return _run(args, dist_ctx)
    finally:
        dist_ctx.close()


def _run(args: argparse.Namespace, dist_ctx: DistributedContext) -> int:
    """Run parsed pipeline arguments inside an initialized process context."""

    if args.evidence_only and args.stage != "quantize":
        raise ValueError("--evidence-only requires --stage quantize")
    if dist_ctx.enabled and args.stage != "quantize":
        raise ValueError("distributed pipeline runs currently require --stage quantize")

    cfg = load_config(args.config)
    if args.overrides:
        _apply_overrides(cfg, args.overrides)
        cfg.validate()
    if args.agentic:
        cfg.agentic.enabled = True

    if args.make_baseline:
        from pipeline.eval_gate import make_baseline

        make_baseline(cfg, cfg.model.id, Path(args.make_baseline))
        return 0

    # Resolve checkpoint + run dir.
    if args.stage in ("serve", "eval") and args.checkpoint:
        ckpt = Path(args.checkpoint)
        run_dir = ckpt.parent if ckpt.name == "checkpoint" else ckpt
    else:
        run_dir = _create_distributed_run_dir(cfg, dist_ctx)
        if dist_ctx.is_source:
            versioning.write_config(run_dir, cfg)
        ckpt = versioning.checkpoint_dir(run_dir)

    print(f"[pipeline] run dir: {run_dir}")
    if dist_ctx.is_source:
        versioning.write_metadata(
            run_dir, cfg, extra={"distributed": dist_ctx.snapshot()}
        )
    dist_ctx.barrier()

    overall_ok = True

    if args.stage in ("quantize", "all"):
        from pipeline.quantize import run_quantize

        ckpt = run_quantize(
            cfg,
            run_dir,
            dist_ctx,
            save_checkpoint=not args.evidence_only,
        )
        if args.evidence_only:
            print(f"[pipeline] evidence-only quantization complete -> {run_dir}")
        else:
            print(f"[pipeline] checkpoint saved -> {ckpt}")

    if args.stage in ("serve", "all") and cfg.serve.enabled:
        import json

        from pipeline.serve_verify import verify_serve

        report = verify_serve(cfg, ckpt)
        (run_dir / "serve_report.json").write_text(json.dumps(report, indent=2))
        if not report.get("ok", False):
            overall_ok = False

    if args.stage in ("eval", "all") and cfg.eval.enabled:
        eval_out = run_dir / "evalsuite"
        if cfg.eval.log_samples:
            from pipeline.evalsuite.static import run_static_eval

            static_result = run_static_eval(cfg, ckpt, eval_out)
            report = static_result["gate"]
        else:
            from pipeline.eval_gate import run_eval_gate

            report = run_eval_gate(cfg, ckpt)

        if cfg.agentic.enabled:
            from pipeline.evalsuite.agentic import run_agentic_eval

            run_agentic_eval(
                cfg,
                ckpt,
                eval_out,
                agent_base=args.agent_base,
                agent_model=args.agent_model,
            )

        versioning.write_eval_report(run_dir, report)
        if report.get("passed") is False:
            overall_ok = False

    print(f"[pipeline] done. overall_ok={overall_ok}")
    return 0 if overall_ok else 1


if __name__ == "__main__":
    sys.exit(main())
