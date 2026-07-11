"""Evidence and orchestration for the MiniMax-M3 shared-expert repair."""

from __future__ import annotations

import argparse
import os
import platform
import re
import shutil
import socket
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

from pipeline.m3_chat_quality import normalize_http_responses, normalize_offline_report
from pipeline.m3_quality_evidence import _notable_log_excerpt, extract_log_evidence
from pipeline.m3_routed_diagnostics import (
    VLLM_SHARED_EXPERT_IGNORE,
    _git_output,
    _read_json,
    _sha256,
    _write_json,
    _write_jsonl,
)

ARM_SPECS = {
    "repaired_w4a8_offline": ("offline", "w4a8"),
    "repaired_w4a16_offline": ("offline", "w4a16_overlay"),
    "repaired_w4a8_http": ("http", "w4a8"),
}
EXPECTED_ARMS = tuple(ARM_SPECS)

_SHARED_COUNTS = re.compile(
    r"unmatched_this_scope=(?P<unmatched>\d+).*shared_seen=(?P<seen>\d+)"
)


def _shared_health(arm: dict[str, Any]) -> dict[str, Any]:
    loader = []
    for line in arm.get("loader_audit_lines") or []:
        match = _SHARED_COUNTS.search(line)
        if match:
            loader.append(
                {
                    "seen": int(match.group("seen")),
                    "unmatched": int(match.group("unmatched")),
                }
            )
    fingerprints = [
        record
        for record in arm.get("fingerprints") or []
        if record.get("category") == "shared_expert"
    ]
    probes = arm.get("moe_probe_records") or []
    fp_ranks = {record.get("rank") for record in fingerprints}
    probe_ranks = {record.get("rank") for record in probes}
    complete = (
        len(loader) >= 8
        and fp_ranks == set(range(8))
        and probe_ranks == set(range(8))
    )
    loader_ok = complete and all(
        item["seen"] == 171 and item["unmatched"] == 0 for item in loader
    )
    fingerprints_ok = complete and bool(fingerprints) and all(
        record.get("name", "").endswith(".weight")
        and record.get("dtype") == "torch.bfloat16"
        and float(record.get("sample_abs_max", 0.0)) > 0.0
        for record in fingerprints
    )
    probes_ok = complete and bool(probes) and all(
        record.get("shared_present") is True
        and float(record.get("shared_norm", 0.0)) > 1e-3
        and record.get("dropped") is not True
        for record in probes
    )
    return {
        "complete": complete,
        "loader_ok": loader_ok,
        "fingerprints_ok": fingerprints_ok,
        "probes_ok": probes_ok,
        "loader_rank_records": len(loader),
        "fingerprint_ranks": sorted(fp_ranks),
        "probe_ranks": sorted(probe_ranks),
    }


def classify_repair(arms: dict[str, dict[str, Any]]) -> dict[str, Any]:
    """Classify the repair without reasoning past missing runtime controls."""

    missing = [arm for arm in EXPECTED_ARMS if arm not in arms]
    if missing:
        return {
            "verdict": "inconclusive_missing_arms",
            "complete": False,
            "missing_arms": missing,
        }
    failed = [
        arm for arm in EXPECTED_ARMS if arms[arm].get("infrastructure_ok") is not True
    ]
    if failed:
        return {
            "verdict": "infrastructure_failure",
            "complete": True,
            "failed_arms": failed,
        }

    health = {
        arm: _shared_health(arms[arm])
        for arm in ("repaired_w4a8_offline", "repaired_w4a16_offline")
    }
    incomplete = [arm for arm, result in health.items() if not result["complete"]]
    if incomplete:
        return {
            "verdict": "inconclusive_missing_diagnostics",
            "complete": True,
            "failed_arms": incomplete,
            "shared_health": health,
        }
    unhealthy = [
        arm
        for arm, result in health.items()
        if not all(
            result[key] for key in ("loader_ok", "fingerprints_ok", "probes_ok")
        )
    ]
    if unhealthy:
        return {
            "verdict": "shared_ignore_repair_failed",
            "complete": True,
            "failed_arms": unhealthy,
            "shared_health": health,
        }

    w4a8_offline = arms["repaired_w4a8_offline"].get("quality_ok") is True
    w4a8_http = arms["repaired_w4a8_http"].get("quality_ok") is True
    w4a16 = arms["repaired_w4a16_offline"].get("quality_ok") is True
    details = {"shared_health": health}
    if w4a8_offline != w4a8_http:
        return {
            "verdict": "candidate_interface_disagreement",
            "complete": True,
            **details,
        }
    if not w4a8_offline and w4a16:
        return {
            "verdict": "activation_boundary_after_shared_repair",
            "complete": True,
            **details,
        }
    if not w4a8_offline and not w4a16:
        return {
            "verdict": "post_shared_routed_boundary",
            "complete": True,
            **details,
        }
    if not w4a16:
        return {
            "verdict": "w4a16_overlay_backend_regression",
            "complete": True,
            **details,
        }
    return {"verdict": "quality_repair_pass", "complete": True, **details}



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
    interface, scheme = ARM_SPECS[arm]
    source_checkpoint = source_checkpoint.resolve()
    overlay_checkpoint = overlay_checkpoint.resolve()
    disable_activations = arm == "repaired_w4a16_offline"
    diagnostics_enabled = interface == "offline"
    manifest = {
        "schema_version": 1,
        "matrix_id": run_dir.parent.name,
        "arm": arm,
        "interface": interface,
        "scheme": scheme,
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
        "scheduler_log": os.environ.get("M3_REPAIR_SRUN_LOG"),
        "quality_envelope": {
            "tensor_parallel_size": 8,
            "enable_expert_parallel": True,
            "enforce_eager": True,
            "block_size": 128,
            "kv_cache_dtype": "fp8",
            "max_model_len": 2048,
            "gpu_memory_utilization": 0.85,
            "disable_custom_all_reduce": True,
            "disable_shared_experts_stream": True,
            "max_tokens": 64,
            "temperature": 0.0,
            "thinking_mode": "disabled",
        },
        "diagnostics": {
            "M3_LOAD_AUDIT": "1" if diagnostics_enabled else "0",
            "M3_MOE_PROBE": "1" if diagnostics_enabled else "0",
            "M3_MOE_PROBE_RECOMPUTE": "1" if diagnostics_enabled else "0",
            "M3_MOE_PROBE_MAX_TOKENS": "256" if diagnostics_enabled else "0",
            "M3_PARAM_FINGERPRINT": "1" if diagnostics_enabled else "0",
            "M3_PARAM_FINGERPRINT_LAYERS": "3,59" if diagnostics_enabled else "",
        },
        "config_overlay": {
            "shared_expert_ignore": VLLM_SHARED_EXPERT_IGNORE,
            "input_activations": None if disable_activations else "unchanged",
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
        raw_return_code = return_code.read_text().strip()
        try:
            manifest["return_code"] = int(raw_return_code)
        except ValueError:
            manifest["return_code"] = raw_return_code
    _write_json(run_dir / "arm_manifest.json", manifest)
    _write_json(evidence_dir / "arm_manifest.json", manifest)

    interface = manifest.get("interface")
    if interface == "http":
        responses = [_read_json(run_dir / f"http_response_{index}.json") for index in range(2)]
        report = normalize_http_responses(responses)
    else:
        report = normalize_offline_report(_read_json(run_dir / "serve_report.json"))

    log_path = run_dir / "serve.log"
    log_text = log_path.read_text(encoding="utf-8", errors="replace") if log_path.is_file() else ""
    extracted = extract_log_evidence(log_text)
    arm_report = {
        "arm": manifest.get("arm"),
        "interface": interface,
        "scheme": manifest.get("scheme"),
        "infrastructure_ok": report.get("infrastructure_ok") is True,
        "quality_ok": report.get("quality_ok") is True,
        "quality_cases": report.get("quality_cases", []),
        **extracted,
    }
    _write_json(evidence_dir / "arm_report.json", arm_report)
    _write_json(evidence_dir / "normalized_report.json", report)
    _write_jsonl(evidence_dir / "parameter_fingerprints.jsonl", extracted["fingerprints"])
    _write_jsonl(evidence_dir / "fingerprint_summaries.jsonl", extracted["fingerprint_summaries"])
    _write_jsonl(evidence_dir / "moe_probe_records.jsonl", extracted["moe_probe_records"])
    (evidence_dir / "loader_audit.txt").write_text("\n".join(extracted["loader_audit_lines"]) + "\n", encoding="utf-8")
    (evidence_dir / "notable_log_excerpt.txt").write_text("\n".join(_notable_log_excerpt(log_text)) + "\n", encoding="utf-8")
    copy_names = [
        "software_versions.txt", "nvidia_smi.csv", "nvidia_topology.txt",
        "patch_status.txt", "return_code.txt", "serve_report.json", "server_start.txt",
    ]
    copy_names.extend(f"http_{kind}_{index}.json" for kind in ("request", "response") for index in range(2))
    for name in copy_names:
        source = run_dir / name
        if source.is_file():
            shutil.copy2(source, evidence_dir / name)
    log_paths = [log_path]
    if manifest.get("scheduler_log"):
        log_paths.append(Path(manifest["scheduler_log"]))
    retention = os.environ.get("RETENTION_UNTIL_UTC") or (
        datetime.now(timezone.utc) + timedelta(days=14)
    ).strftime("%Y-%m-%dT00:00:00Z")
    artifacts = [
        {"path": str(path.resolve()), "bytes": path.stat().st_size,
         "sha256": _sha256(path), "retention_until_utc": retention}
        for path in log_paths if path.is_file()
    ]
    _write_json(evidence_dir / "artifact_index.json", artifacts)
    return arm_report


def aggregate_matrix(evidence_root: Path) -> dict[str, Any]:
    arms = {
        arm: report
        for arm in EXPECTED_ARMS
        if (report := _read_json(evidence_root / arm / "arm_report.json"))
    }
    comparison = {**classify_repair(arms), "arms": arms}
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
