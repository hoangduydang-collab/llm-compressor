"""Evidence model for the parallel MiniMax-M3 layer-boundary matrix."""

from __future__ import annotations

import argparse
import os
import platform
import shutil
import socket
from dataclasses import asdict, dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

from pipeline.m3_chat_quality import normalize_http_responses, normalize_offline_report
from pipeline.m3_quality_evidence import _notable_log_excerpt, extract_log_evidence
from pipeline.m3_routed_diagnostics import (
    VLLM_ROUTER_IGNORE,
    VLLM_SHARED_EXPERT_IGNORE,
    _git_output,
    _read_json,
    _sha256,
    _write_json,
    _write_jsonl,
)


@dataclass(frozen=True)
class ArmSpec:
    checkpoint_role: str
    scheme: str
    interface: str = "offline"
    enable_ep: bool = True
    kv_cache_dtype: str = "fp8"
    router_alias: bool = False

    @property
    def disable_activations(self) -> bool:
        return self.checkpoint_role == "candidate" and self.scheme == "w4a16"


ARM_SPECS = {
    "reference_w4a16_ep_fp8kv": ArmSpec("reference", "w4a16"),
    "candidate_w4a8_ep_fp8kv": ArmSpec("candidate", "w4a8"),
    "candidate_w4a16_ep_fp8kv": ArmSpec("candidate", "w4a16"),
    "candidate_w4a8_router_alias": ArmSpec(
        "candidate", "w4a8", router_alias=True
    ),
    "candidate_w4a16_router_alias": ArmSpec(
        "candidate", "w4a16", router_alias=True
    ),
    "reference_w4a16_tp_fp8kv": ArmSpec(
        "reference", "w4a16", enable_ep=False
    ),
    "candidate_w4a8_tp_fp8kv": ArmSpec(
        "candidate", "w4a8", enable_ep=False
    ),
    "candidate_w4a16_tp_fp8kv": ArmSpec(
        "candidate", "w4a16", enable_ep=False
    ),
    "candidate_w4a8_ep_autokv": ArmSpec(
        "candidate", "w4a8", kv_cache_dtype="auto"
    ),
    "candidate_w4a16_ep_autokv": ArmSpec(
        "candidate", "w4a16", kv_cache_dtype="auto"
    ),
    "candidate_w4a8_router_http": ArmSpec(
        "candidate", "w4a8", interface="http", router_alias=True
    ),
}
EXPECTED_ARMS = tuple(ARM_SPECS)

BOUNDARY_ORDER = {
    name: index
    for index, name in enumerate(
        (
            "decoder_input_hidden",
            "decoder_input_residual",
            "attention_input",
            "attention_output",
            "moe_input",
            "moe_output",
            "decoder_output_hidden",
            "decoder_output_residual",
        )
    )
}


def _record_key(record: dict[str, Any]) -> tuple[int, int, str] | None:
    try:
        return (
            int(record["rank"]),
            int(record["layer"]),
            str(record["boundary"]),
        )
    except (KeyError, TypeError, ValueError):
        return None


def _first_explosive_boundary(
    reference: list[dict[str, Any]], candidates: list[list[dict[str, Any]]]
) -> dict[str, Any] | None:
    ref = {
        key: record
        for record in reference
        if (key := _record_key(record)) is not None
    }
    hits: list[dict[str, Any]] = []
    for records in candidates:
        for record in records:
            key = _record_key(record)
            if key is None or key not in ref:
                continue
            try:
                norm = float(record.get("norm", 0.0))
                ref_norm = float(ref[key].get("norm", 0.0))
                finite = float(record.get("finite_fraction", 1.0))
            except (TypeError, ValueError):
                continue
            ratio = norm / ref_norm if ref_norm > 0 else float("inf")
            if finite < 1.0 or ratio >= 50.0:
                hits.append(
                    {
                        "rank": key[0],
                        "layer": key[1],
                        "boundary": key[2],
                        "norm": norm,
                        "reference_norm": ref_norm,
                        "norm_ratio": ratio,
                        "finite_fraction": finite,
                    }
                )
    if not hits:
        return None
    return min(
        hits,
        key=lambda item: (
            item["layer"], BOUNDARY_ORDER.get(item["boundary"], 999), item["rank"]
        ),
    )


def _router_health(arm: dict[str, Any]) -> dict[str, Any]:
    routers = [
        record
        for record in arm.get("fingerprints") or []
        if record.get("category") == "moe_router"
    ]
    ranks = {record.get("rank") for record in routers}
    return {
        "records": len(routers),
        "ranks": sorted(rank for rank in ranks if isinstance(rank, int)),
        "complete": ranks == set(range(8)),
        "fp32_nonzero": bool(routers)
        and all(
            record.get("dtype") == "torch.float32"
            and float(record.get("sample_abs_max", 0.0)) > 0.0
            for record in routers
        ),
    }


def classify_matrix(arms: dict[str, dict[str, Any]]) -> dict[str, Any]:
    missing = [name for name in EXPECTED_ARMS if name not in arms]
    if missing:
        return {
            "verdict": "inconclusive_missing_arms",
            "complete": False,
            "missing_arms": missing,
        }
    failed = [
        name for name in EXPECTED_ARMS
        if arms[name].get("infrastructure_ok") is not True
    ]
    if failed:
        return {
            "verdict": "infrastructure_failure",
            "complete": True,
            "failed_arms": failed,
        }
    if arms["reference_w4a16_ep_fp8kv"].get("quality_ok") is not True:
        return {"verdict": "invalid_reference", "complete": True}

    details = {
        "router_health": {
            name: _router_health(arms[name])
            for name in (
                "reference_w4a16_ep_fp8kv",
                "candidate_w4a8_ep_fp8kv",
                "candidate_w4a16_ep_fp8kv",
                "candidate_w4a8_router_alias",
                "candidate_w4a16_router_alias",
            )
        }
    }
    if any(
        arms[name].get("quality_ok") is True
        for name in (
            "candidate_w4a8_router_alias",
            "candidate_w4a16_router_alias",
        )
    ):
        return {"verdict": "router_alias_boundary", "complete": True, **details}
    if any(
        arms[name].get("quality_ok") is True
        for name in (
            "candidate_w4a8_tp_fp8kv",
            "candidate_w4a16_tp_fp8kv",
        )
    ):
        if arms["reference_w4a16_tp_fp8kv"].get("quality_ok") is not True:
            return {"verdict": "invalid_tp_reference", "complete": True, **details}
        return {"verdict": "expert_parallel_boundary", "complete": True, **details}
    if any(
        arms[name].get("quality_ok") is True
        for name in (
            "candidate_w4a8_ep_autokv",
            "candidate_w4a16_ep_autokv",
        )
    ):
        return {"verdict": "kv_cache_boundary", "complete": True, **details}
    if any(
        arms[name].get("quality_ok") is True
        for name in ("candidate_w4a8_ep_fp8kv", "candidate_w4a16_ep_fp8kv")
    ):
        return {"verdict": "candidate_control_pass", "complete": True, **details}

    first = _first_explosive_boundary(
        arms["reference_w4a16_ep_fp8kv"].get("layer_boundary_records") or [],
        [
            arms["candidate_w4a8_ep_fp8kv"].get("layer_boundary_records") or [],
            arms["candidate_w4a16_ep_fp8kv"].get("layer_boundary_records") or [],
        ],
    )
    details["first_explosive_boundary"] = first
    boundary = first.get("boundary") if first else None
    if boundary == "attention_output":
        verdict = "attention_indexer_boundary"
    elif boundary == "moe_input":
        verdict = "attention_residual_boundary"
    elif boundary == "moe_output":
        verdict = "routed_moe_boundary"
    elif boundary in {"decoder_output_hidden", "decoder_output_residual"}:
        verdict = "residual_collective_boundary"
    else:
        verdict = "unresolved_post_shared_boundary"
    return {"verdict": verdict, "complete": True, **details}


def write_arm_manifest(
    *,
    arm: str,
    run_dir: Path,
    evidence_dir: Path,
    source_checkpoint: Path,
    overlay_checkpoint: Path,
    model_id: str,
    dry_run: bool,
) -> dict[str, Any]:
    if arm not in ARM_SPECS:
        raise ValueError(f"unknown arm: {arm}")
    spec = ARM_SPECS[arm]
    source_checkpoint = source_checkpoint.resolve()
    overlay_checkpoint = overlay_checkpoint.resolve()
    diagnostics = spec.interface == "offline"
    manifest = {
        "schema_version": 1,
        "matrix_id": run_dir.parent.name,
        "arm": arm,
        **asdict(spec),
        "source_checkpoint": str(source_checkpoint),
        "overlay_checkpoint": str(overlay_checkpoint),
        "source_config_sha256": _sha256(source_checkpoint / "config.json"),
        "source_index_sha256": _sha256(source_checkpoint / "model.safetensors.index.json"),
        "overlay_config_sha256": _sha256(overlay_checkpoint / "config.json"),
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
        "scheduler_log": os.environ.get("M3_BOUNDARY_SRUN_LOG"),
        "quality_envelope": {
            "tensor_parallel_size": 8,
            "enable_expert_parallel": spec.enable_ep,
            "kv_cache_dtype": spec.kv_cache_dtype,
            "enforce_eager": True,
            "block_size": 128,
            "max_model_len": 2048,
            "gpu_memory_utilization": 0.85,
            "disable_custom_all_reduce": True,
            "disable_shared_experts_stream": True,
            "max_tokens": 64,
            "temperature": 0.0,
            "thinking_mode": "disabled",
        },
        "diagnostics": {
            "M3_LOAD_AUDIT": "1" if diagnostics else "0",
            "M3_PARAM_FINGERPRINT": "1" if diagnostics else "0",
            "M3_PARAM_FINGERPRINT_LAYERS": "3,4,5,6,7,8,9" if diagnostics else "",
            "M3_LAYER_BOUNDARY": "1" if diagnostics else "0",
            "M3_LAYER_BOUNDARY_LAYERS": "3,4,5,6,7,8,9" if diagnostics else "",
        },
        "config_overlay": {
            "shared_expert_ignore": (
                VLLM_SHARED_EXPERT_IGNORE if spec.checkpoint_role == "candidate" else None
            ),
            "router_ignore": VLLM_ROUTER_IGNORE if spec.router_alias else None,
            "input_activations": None if spec.disable_activations else "unchanged",
        },
        "evidence_dir": str(evidence_dir.resolve()),
        "deviations": [],
        "retries": [],
    }
    _write_json(run_dir / "arm_manifest.json", manifest)
    _write_json(evidence_dir / "arm_manifest.json", manifest)
    return manifest


def bundle_arm(run_dir: Path, evidence_dir: Path) -> dict[str, Any]:
    evidence_dir.mkdir(parents=True, exist_ok=True)
    manifest = _read_json(run_dir / "arm_manifest.json")
    manifest["finished_at"] = datetime.now(timezone.utc).isoformat()
    return_code = run_dir / "return_code.txt"
    if return_code.is_file():
        try:
            manifest["return_code"] = int(return_code.read_text().strip())
        except ValueError:
            manifest["return_code"] = return_code.read_text().strip()
    _write_json(run_dir / "arm_manifest.json", manifest)
    _write_json(evidence_dir / "arm_manifest.json", manifest)

    if manifest.get("interface") == "http":
        responses = [_read_json(run_dir / f"http_response_{index}.json") for index in range(2)]
        report = normalize_http_responses(responses)
    else:
        report = normalize_offline_report(_read_json(run_dir / "serve_report.json"))
    log_path = run_dir / "serve.log"
    log_text = log_path.read_text(encoding="utf-8", errors="replace") if log_path.is_file() else ""
    extracted = extract_log_evidence(log_text)
    arm_report = {
        "arm": manifest.get("arm"),
        "interface": manifest.get("interface"),
        "checkpoint_role": manifest.get("checkpoint_role"),
        "scheme": manifest.get("scheme"),
        "infrastructure_ok": report.get("infrastructure_ok") is True,
        "quality_ok": report.get("quality_ok") is True,
        "quality_cases": report.get("quality_cases", []),
        **extracted,
    }
    _write_json(evidence_dir / "arm_report.json", arm_report)
    _write_json(evidence_dir / "normalized_report.json", report)
    _write_jsonl(evidence_dir / "parameter_fingerprints.jsonl", extracted["fingerprints"])
    _write_jsonl(evidence_dir / "layer_boundary_records.jsonl", extracted["layer_boundary_records"])
    _write_jsonl(evidence_dir / "moe_probe_records.jsonl", extracted["moe_probe_records"])
    (evidence_dir / "loader_audit.txt").write_text(
        "\n".join(extracted["loader_audit_lines"]) + "\n", encoding="utf-8"
    )
    (evidence_dir / "notable_log_excerpt.txt").write_text(
        "\n".join(_notable_log_excerpt(log_text)) + "\n", encoding="utf-8"
    )
    for name in (
        "software_versions.txt", "nvidia_smi.csv", "nvidia_topology.txt",
        "patch_status.txt", "return_code.txt", "serve_report.json", "server_start.txt",
    ):
        source = run_dir / name
        if source.is_file():
            shutil.copy2(source, evidence_dir / name)
    for index in range(2):
        for kind in ("request", "response"):
            source = run_dir / f"http_{kind}_{index}.json"
            if source.is_file():
                shutil.copy2(source, evidence_dir / source.name)
    retention = os.environ.get("RETENTION_UNTIL_UTC") or (
        datetime.now(timezone.utc) + timedelta(days=14)
    ).strftime("%Y-%m-%dT00:00:00Z")
    logs = [log_path]
    if manifest.get("scheduler_log"):
        logs.append(Path(manifest["scheduler_log"]))
    _write_json(
        evidence_dir / "artifact_index.json",
        [
            {"path": str(path.resolve()), "bytes": path.stat().st_size,
             "sha256": _sha256(path), "retention_until_utc": retention}
            for path in logs if path.is_file()
        ],
    )
    return arm_report


def aggregate_matrix(evidence_root: Path) -> dict[str, Any]:
    arms = {
        name: report
        for name in EXPECTED_ARMS
        if (report := _read_json(evidence_root / name / "arm_report.json"))
    }
    comparison = {**classify_matrix(arms), "arms": arms}
    _write_json(evidence_root / "comparison.json", comparison)
    return comparison


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    manifest = commands.add_parser("manifest")
    manifest.add_argument("--arm", choices=EXPECTED_ARMS, required=True)
    manifest.add_argument("--run-dir", type=Path, required=True)
    manifest.add_argument("--evidence-dir", type=Path, required=True)
    manifest.add_argument("--source-checkpoint", type=Path, required=True)
    manifest.add_argument("--overlay-checkpoint", type=Path, required=True)
    manifest.add_argument("--model-id", required=True)
    manifest.add_argument("--dry-run", action="store_true")
    bundle = commands.add_parser("bundle-arm")
    bundle.add_argument("--run-dir", type=Path, required=True)
    bundle.add_argument("--evidence-dir", type=Path, required=True)
    aggregate = commands.add_parser("aggregate")
    aggregate.add_argument("--evidence-root", type=Path, required=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    if args.command == "manifest":
        write_arm_manifest(
            arm=args.arm, run_dir=args.run_dir, evidence_dir=args.evidence_dir,
            source_checkpoint=args.source_checkpoint,
            overlay_checkpoint=args.overlay_checkpoint, model_id=args.model_id,
            dry_run=args.dry_run,
        )
    elif args.command == "bundle-arm":
        bundle_arm(args.run_dir, args.evidence_dir)
    else:
        aggregate_matrix(args.evidence_root)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
