"""Evidence model for the MiniMax-M3 AWQ/GPTQ repair matrix."""

from __future__ import annotations

import argparse
import os
import platform
import socket
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from pipeline.m3_layer_boundary_diagnostics import (
    _first_explosive_boundary,
    bundle_arm,
)
from pipeline.m3_routed_diagnostics import _git_output, _read_json, _sha256, _write_json


@dataclass(frozen=True)
class RepairArmSpec:
    checkpoint_role: str
    scheme: str
    interface: str = "offline"

    @property
    def disable_activations(self) -> bool:
        return self.scheme == "w4a16" and self.checkpoint_role != "reference"


ARM_SPECS = {
    "reference_w4a16": RepairArmSpec("reference", "w4a16"),
    "awq_control_w4a8": RepairArmSpec("awq", "w4a8"),
    "gptq_w4a8": RepairArmSpec("gptq", "w4a8"),
    "gptq_w4a16": RepairArmSpec("gptq", "w4a16"),
    "gptq_http": RepairArmSpec("gptq", "w4a8", "http"),
    "awq_offsetfix_w4a8": RepairArmSpec("awq_offsetfix", "w4a8"),
    "awq_offsetfix_w4a16": RepairArmSpec("awq_offsetfix", "w4a16"),
    "awq_offsetfix_http": RepairArmSpec("awq_offsetfix", "w4a8", "http"),
    "awq_nosmooth_w4a8": RepairArmSpec("awq_nosmooth", "w4a8"),
    "awq_nosmooth_w4a16": RepairArmSpec("awq_nosmooth", "w4a16"),
    "awq_nosmooth_http": RepairArmSpec("awq_nosmooth", "w4a8", "http"),
}
EXPECTED_ARMS = tuple(ARM_SPECS)


def _boundary(arms: dict[str, dict[str, Any]], name: str) -> dict[str, Any] | None:
    return _first_explosive_boundary(
        arms["reference_w4a16"].get("layer_boundary_records") or [],
        [arms[name].get("layer_boundary_records") or []],
    )


def classify_matrix(arms: dict[str, dict[str, Any]]) -> dict[str, Any]:
    missing = [name for name in EXPECTED_ARMS if name not in arms]
    if missing:
        return {"verdict": "inconclusive_missing_arms", "complete": False,
                "missing_arms": missing}
    failed = [name for name in EXPECTED_ARMS
              if arms[name].get("infrastructure_ok") is not True]
    if failed:
        return {"verdict": "infrastructure_failure", "complete": True,
                "failed_arms": failed}
    if arms["reference_w4a16"].get("quality_ok") is not True:
        return {"verdict": "invalid_reference", "complete": True}

    details = {
        "boundaries": {
            name: _boundary(arms, name)
            for name in (
                "awq_control_w4a8", "gptq_w4a8", "gptq_w4a16",
                "awq_offsetfix_w4a8", "awq_offsetfix_w4a16",
                "awq_nosmooth_w4a8", "awq_nosmooth_w4a16",
            )
        }
    }
    if all(arms[name].get("quality_ok") is True for name in (
        "awq_offsetfix_w4a8", "awq_offsetfix_w4a16", "awq_offsetfix_http"
    )):
        return {"verdict": "minimax_offset_norm_root_cause", "complete": True,
                **details}
    if any(arms[name].get("quality_ok") is True for name in (
        "awq_offsetfix_w4a8", "awq_offsetfix_w4a16", "awq_offsetfix_http"
    )):
        return {"verdict": "offset_norm_partial_recovery", "complete": True,
                **details}

    if all(arms[name].get("quality_ok") is True for name in (
        "awq_nosmooth_w4a8", "awq_nosmooth_w4a16", "awq_nosmooth_http"
    )):
        return {"verdict": "awq_mlp_input_smoothing_root_cause", "complete": True,
                **details}
    if any(arms[name].get("quality_ok") is True for name in (
        "awq_nosmooth_w4a8", "awq_nosmooth_w4a16", "awq_nosmooth_http"
    )):
        return {"verdict": "no_smoothing_partial_recovery", "complete": True,
                **details}

    gptq_pass = all(arms[name].get("quality_ok") is True for name in (
        "gptq_w4a8", "gptq_w4a16", "gptq_http"
    ))
    if gptq_pass:
        return {"verdict": "awq_specific_unresolved", "complete": True, **details}

    awq_boundary = details["boundaries"]["awq_control_w4a8"]
    gptq_boundaries = [details["boundaries"][name]
                       for name in ("gptq_w4a8", "gptq_w4a16")]
    if awq_boundary and all(
        boundary and boundary.get("layer") == awq_boundary.get("layer")
        and boundary.get("boundary") == awq_boundary.get("boundary")
        for boundary in gptq_boundaries
    ):
        verdict = "shared_compression_export_boundary"
    else:
        verdict = "gptq_distinct_boundary"
    return {"verdict": verdict, "complete": True, **details}


def write_manifest(
    arm: str,
    run_dir: Path,
    evidence_dir: Path,
    source_checkpoint: Path,
    overlay_checkpoint: Path,
    model_id: str,
    dry_run: bool,
) -> dict[str, Any]:
    spec = ARM_SPECS[arm]
    manifest = {
        "schema_version": 1,
        "matrix_id": run_dir.parent.name,
        "arm": arm,
        **asdict(spec),
        "source_checkpoint": str(source_checkpoint.resolve()),
        "overlay_checkpoint": str(overlay_checkpoint.resolve()),
        "source_config_sha256": _sha256(source_checkpoint / "config.json"),
        "source_index_sha256": _sha256(
            source_checkpoint / "model.safetensors.index.json"
        ),
        "model_id": model_id,
        "dry_run": dry_run,
        "started_at": datetime.now(timezone.utc).isoformat(),
        "finished_at": None,
        "git_commit": _git_output("rev-parse", "HEAD"),
        "git_status_short": _git_output("status", "--short"),
        "host": socket.gethostname(),
        "platform": platform.platform(),
        "python": platform.python_version(),
        "scheduler": {
            key: os.environ.get(key)
            for key in ("SLURM_JOB_ID", "SLURM_JOB_NODELIST", "SLURM_JOB_GPUS")
            if os.environ.get(key)
        },
        "scheduler_log": os.environ.get("M3_REPAIR_SRUN_LOG"),
        "interface": spec.interface,
        "diagnostic_layers": (
            os.environ.get("M3_DIAGNOSTIC_LAYERS", "")
            if spec.interface == "offline" else ""
        ),
        "deviations": [],
        "retries": [],
        "evidence_dir": str(evidence_dir.resolve()),
    }
    _write_json(run_dir / "arm_manifest.json", manifest)
    _write_json(evidence_dir / "arm_manifest.json", manifest)
    return manifest


def aggregate(evidence_root: Path) -> dict[str, Any]:
    arms = {
        name: _read_json(evidence_root / name / "arm_report.json")
        for name in EXPECTED_ARMS
        if (evidence_root / name / "arm_report.json").is_file()
    }
    result = {**classify_matrix(arms), "arms": arms}
    audit = _read_json(evidence_root / "checkpoint_scale_audit.json")
    if audit:
        result["checkpoint_scale_audit"] = audit
    _write_json(evidence_root / "comparison.json", result)
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    manifest = sub.add_parser("manifest")
    manifest.add_argument("--arm", choices=EXPECTED_ARMS, required=True)
    manifest.add_argument("--run-dir", type=Path, required=True)
    manifest.add_argument("--evidence-dir", type=Path, required=True)
    manifest.add_argument("--source-checkpoint", type=Path, required=True)
    manifest.add_argument("--overlay-checkpoint", type=Path, required=True)
    manifest.add_argument("--model-id", required=True)
    manifest.add_argument("--dry-run", action="store_true")
    bundle = sub.add_parser("bundle-arm")
    bundle.add_argument("--run-dir", type=Path, required=True)
    bundle.add_argument("--evidence-dir", type=Path, required=True)
    aggregate_parser = sub.add_parser("aggregate")
    aggregate_parser.add_argument("--evidence-root", type=Path, required=True)
    args = parser.parse_args()
    if args.command == "manifest":
        write_manifest(args.arm, args.run_dir, args.evidence_dir,
                       args.source_checkpoint, args.overlay_checkpoint,
                       args.model_id, args.dry_run)
    elif args.command == "bundle-arm":
        bundle_arm(args.run_dir, args.evidence_dir)
    else:
        aggregate(args.evidence_root)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
