"""Trace-only discriminator for MiniMax-M3 sequential calibration targets.

This command deliberately stops before modifier construction, calibration forwards,
AWQ, quantization, and evaluation. It observes the production sequential FX tracer at
the full multimodal wrapper and live language-model subtree boundaries.
"""

from __future__ import annotations

import argparse
import contextlib
import hashlib
import inspect
import json
import subprocess
import sys
import traceback
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

DEFAULT_CONFIG = Path("pipeline/configs/minimax_m3_full_calib.yaml")


def _write_json_atomic(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def _write_text_atomic(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(value)
    temporary.replace(path)


def filter_sample_for_model(model: Any, sample: dict[str, Any]) -> dict[str, Any]:
    """Keep collated keys explicitly accepted by ``model.forward`` in input order."""
    parameters = inspect.signature(model.forward).parameters.values()
    accepted = {
        parameter.name
        for parameter in parameters
        if parameter.kind
        in (
            inspect.Parameter.POSITIONAL_ONLY,
            inspect.Parameter.POSITIONAL_OR_KEYWORD,
            inspect.Parameter.KEYWORD_ONLY,
        )
    }
    return {key: value for key, value in sample.items() if key in accepted}


def find_language_model_root(model: Any) -> Any:
    """Return the live MiniMax language subtree without reconstructing the model."""
    for path in (
        "model.model.language_model",
        "model.language_model",
        "language_model",
    ):
        current = model
        try:
            for component in path.split("."):
                current = getattr(current, component)
        except AttributeError:
            continue
        if current is not model:
            return current

    for name, module in model.named_modules():
        if name.endswith("language_model") and module is not model:
            return module
    raise ValueError("could not locate a live language_model subtree")


def classify_trace_reports(reports: dict[str, dict[str, Any]]) -> str:
    """Classify structural evidence without claiming an AWQ numerical root cause."""
    full = reports.get("full_wrapper", {})
    language = reports.get("language_model", {})
    if full.get("status") != "ok" or language.get("status") != "ok":
        return "trace_error"
    full_targets = int(full.get("target_node_count", 0))
    language_targets = int(language.get("target_node_count", 0))
    if full_targets == 0 and language_targets > 0:
        return "multimodal_wrapper_boundary"
    if full_targets == 0 and language_targets == 0:
        return "targets_absent_from_both_fx_graphs"
    if (full_targets > 0 and int(full.get("partition_count", 0)) <= 1) or (
        language_targets > 0 and int(language.get("partition_count", 0)) <= 1
    ):
        return "partition_collapse"
    return "trace_structure_healthy"


def persist_root_artifacts(output_dir: Path, report: dict[str, Any]) -> None:
    """Atomically persist compact metadata and separately inspectable FX evidence."""
    compact = dict(report)
    graph_code = str(compact.pop("graph_code", ""))
    nodes = compact.pop("nodes", [])
    _write_text_atomic(output_dir / "graph.py", graph_code)
    _write_json_atomic(output_dir / "nodes.json", nodes)
    _write_json_atomic(output_dir / "report.json", compact)


def trace_root(
    *,
    label: str,
    model: Any,
    sample: dict[str, Any],
    sequential_targets: list[str],
    ignore: list[str],
    output_dir: Path,
    trace_fn: Callable[..., Any] | None = None,
) -> dict[str, Any]:
    """Trace one root, retaining partial diagnostics when the tracer raises."""
    if trace_fn is None:
        from llmcompressor.pipelines.sequential.helpers import trace_subgraphs

        trace_fn = trace_subgraphs

    report: dict[str, Any] = {
        "schema_version": 1,
        "label": label,
        "model_class": type(model).__name__,
        "model_module": type(model).__module__,
        "sample_keys": list(sample),
        "sequential_targets": list(sequential_targets),
        "trace_function_module": getattr(trace_fn, "__module__", None),
        "trace_function_file": inspect.getsourcefile(trace_fn),
    }
    try:
        subgraphs = trace_fn(
            model,
            sample,
            sequential_targets=sequential_targets,
            ignore=ignore,
            diagnostics=report,
        )
        report["status"] = "ok"
        report["returned_subgraph_count"] = len(subgraphs)
    except Exception as exc:  # noqa: BLE001 - diagnostic must retain all failures
        report["status"] = "error"
        report["error"] = {
            "type": type(exc).__name__,
            "message": str(exc),
            "traceback": traceback.format_exc(),
        }
    persist_root_artifacts(output_dir, report)
    return report


def _git_revision() -> str | None:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True, stderr=subprocess.DEVNULL
        ).strip()
    except Exception:
        return None


def run_diagnostic(
    *, config_path: Path, output_dir: Path, model_id: str | None = None
) -> dict[str, Any]:
    """Load once, collate one production batch, and trace two roots."""
    from llmcompressor.args import DatasetArguments
    from llmcompressor.datasets import get_calibration_dataloader
    from llmcompressor.modeling.moe.context import moe_calibration_context
    from llmcompressor.modeling.offset_norm import norm_calibration_context
    from pipeline.calibration import build_calibration_dataset
    from pipeline.config import load_config
    from pipeline.minimax_m3_config import patch_minimax_m3_for_text_calibration
    from pipeline.provenance import collect_model_provenance
    from pipeline.quantize import _load_model_and_tokenizer

    config = load_config(config_path)
    if model_id is not None:
        config.model.id = model_id
    if not config.calibration.sequential_targets:
        raise ValueError("trace diagnostic requires calibration.sequential_targets")

    output_dir.mkdir(parents=True, exist_ok=True)
    model, tokenizer = _load_model_and_tokenizer(config)
    if not patch_minimax_m3_for_text_calibration(model):
        raise RuntimeError("loaded model is not recognized as MiniMax-M3")

    dataset = build_calibration_dataset(config.calibration, tokenizer)
    dataset_args = DatasetArguments(
        dataset=dataset,
        max_seq_length=config.calibration.max_seq_length,
        num_calibration_samples=config.calibration.num_samples,
        shuffle_calibration_samples=False,
        moe_calibrate_all_experts=config.calibration.moe_calibrate_all_experts,
        sequential_targets=config.calibration.sequential_targets,
    )
    dataloader = get_calibration_dataloader(dataset_args, tokenizer)
    if dataloader is None:
        raise RuntimeError("production calibration dataloader was not constructed")
    sample = next(iter(dataloader))
    language_model = find_language_model_root(model)

    reports: dict[str, dict[str, Any]] = {}
    with contextlib.ExitStack() as stack:
        stack.enter_context(norm_calibration_context(model))
        if config.calibration.moe_calibrate_all_experts:
            stack.enter_context(moe_calibration_context())
        reports["full_wrapper"] = trace_root(
            label="full_wrapper",
            model=model,
            sample=dict(sample),
            sequential_targets=list(config.calibration.sequential_targets),
            ignore=dataset_args.tracing_ignore,
            output_dir=output_dir / "full_wrapper",
        )
        reports["language_model"] = trace_root(
            label="language_model",
            model=language_model,
            sample=filter_sample_for_model(language_model, dict(sample)),
            sequential_targets=list(config.calibration.sequential_targets),
            ignore=dataset_args.tracing_ignore,
            output_dir=output_dir / "language_model",
        )

    statuses = {label: report["status"] for label, report in reports.items()}
    aggregate = {
        "schema_version": 1,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "status": "ok"
        if all(value == "ok" for value in statuses.values())
        else "error",
        "classification": classify_trace_reports(reports),
        "root_statuses": statuses,
        "root_summaries": {
            label: {
                key: report.get(key)
                for key in (
                    "matched_target_count",
                    "target_node_count",
                    "partition_count",
                    "subgraph_count",
                    "returned_subgraph_count",
                )
            }
            for label, report in reports.items()
        },
        "config_path": str(config_path),
        "config_sha256": hashlib.sha256(config_path.read_bytes()).hexdigest(),
        "model_id": config.model.id,
        "git_revision": _git_revision(),
        "command": [sys.executable, *sys.argv],
        "provenance": collect_model_provenance(
            model, config.calibration.sequential_targets
        ),
    }
    _write_json_atomic(output_dir / "report.json", aggregate)
    return aggregate


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--model-id")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    aggregate = run_diagnostic(
        config_path=args.config, output_dir=args.output_dir, model_id=args.model_id
    )
    print(json.dumps({key: aggregate[key] for key in ("status", "classification")}))
    return 0 if aggregate["status"] == "ok" else 1


if __name__ == "__main__":
    raise SystemExit(main())
