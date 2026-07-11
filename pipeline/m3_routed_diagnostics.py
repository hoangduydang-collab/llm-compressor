"""MiniMax-M3 routed-expert diagnostic evidence and classification."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import shutil
import socket
import subprocess
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

from pipeline.m3_quality_evidence import _notable_log_excerpt, extract_log_evidence

ARM_SPECS = {
    "reference_w4a16": ("reference", "w4a16"),
    "candidate_w4a8": ("candidate", "w4a8"),
    "candidate_w4a16": ("candidate", "w4a16_overlay"),
}
EXPECTED_ARMS = tuple(ARM_SPECS)
# Exact controls shared by both checkpoint schemes. Reference attention is
# W4A16 while candidate attention is BF16, and the vLLM model may fuse the MSA
# indexer into QKV, so neither is a portable byte-equality control.
UNQUANTIZED_CATEGORIES = {"lm_head", "shared_expert"}
VLLM_SHARED_EXPERT_IGNORE = "re:.*block_sparse_moe[.]shared_experts[.].*"
VLLM_ROUTER_IGNORE = "re:.*block_sparse_moe[.]gate$"


def _read_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {}
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return value if isinstance(value, dict) else {}


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _write_jsonl(path: Path, records: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(record, sort_keys=True) + "\n" for record in records),
        encoding="utf-8",
    )


def _sha256(path: Path) -> str | None:
    if not path.is_file():
        return None
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _git_output(*args: str) -> str:
    try:
        return subprocess.check_output(
            ["git", *args], text=True, stderr=subprocess.DEVNULL
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return ""


def _fingerprint_key(record: dict[str, Any]) -> tuple[Any, ...]:
    return (
        record.get("scope"),
        record.get("rank"),
        record.get("name"),
        record.get("category"),
    )


def compare_unquantized_fingerprints(
    reference: list[dict[str, Any]],
    candidate: list[dict[str, Any]],
) -> dict[str, Any]:
    """Compare only components expected to remain unquantized in both arms."""

    ref = {
        _fingerprint_key(record): record
        for record in reference
        if record.get("category") in UNQUANTIZED_CATEGORIES
    }
    cand = {
        _fingerprint_key(record): record
        for record in candidate
        if record.get("category") in UNQUANTIZED_CATEGORIES
    }
    common = sorted(ref.keys() & cand.keys(), key=str)
    mismatched = [
        {
            "key": list(key),
            "reference_sha256": ref[key].get("sample_sha256"),
            "candidate_sha256": cand[key].get("sample_sha256"),
        }
        for key in common
        if ref[key].get("sample_sha256") != cand[key].get("sample_sha256")
    ]
    missing = [list(key) for key in sorted(ref.keys() ^ cand.keys(), key=str)]
    return {
        "compared": len(common),
        "mismatched": mismatched,
        "missing": missing,
    }


def _first_probe_by_rank(records: list[dict[str, Any]]) -> dict[int, dict[str, Any]]:
    selected: dict[int, dict[str, Any]] = {}
    for record in records:
        if record.get("probe_index") != 1:
            continue
        try:
            rank = int(record.get("rank"))
        except (TypeError, ValueError):
            continue
        selected[rank] = record
    return selected


def compare_first_moe_inputs(
    reference: list[dict[str, Any]],
    candidate: list[dict[str, Any]],
) -> dict[str, Any]:
    """Compare the first quantized-MoE input digest rank by rank."""

    ref = _first_probe_by_rank(reference)
    cand = _first_probe_by_rank(candidate)
    common = sorted(ref.keys() & cand.keys())
    mismatched = [
        rank
        for rank in common
        if ref[rank].get("input_sample_sha256")
        != cand[rank].get("input_sample_sha256")
    ]
    return {
        "compared_ranks": common,
        "mismatched_ranks": mismatched,
        "missing_ranks": sorted(ref.keys() ^ cand.keys()),
    }


def _diagnostic_gaps(arm: dict[str, Any]) -> list[str]:
    gaps: list[str] = []
    fingerprints = arm.get("fingerprints") or []
    expected_categories = UNQUANTIZED_CATEGORIES | {"routed_expert"}
    for rank in range(8):
        found = {
            record.get("category")
            for record in fingerprints
            if record.get("rank") == rank
        }
        for category in sorted(expected_categories - found):
            gaps.append(f"rank{rank}.fingerprint.{category}")
    probe_ranks = {
        record.get("rank")
        for record in arm.get("moe_probe_records") or []
        if record.get("probe_index") == 1
    }
    for rank in sorted(set(range(8)) - probe_ranks):
        gaps.append(f"rank{rank}.first_moe_probe")
    if len(arm.get("loader_audit_lines") or []) < 8:
        gaps.append("loader_audit.rank_coverage")
    return gaps


def classify_diagnostics(arms: dict[str, dict[str, Any]]) -> dict[str, Any]:
    """Choose the first candidate-specific boundary supported by all controls."""

    missing = [arm for arm in EXPECTED_ARMS if arm not in arms]
    if missing:
        return {
            "verdict": "inconclusive_missing_arms",
            "complete": False,
            "missing_arms": missing,
        }
    infra = [
        arm for arm in EXPECTED_ARMS if arms[arm].get("infrastructure_ok") is not True
    ]
    if infra:
        return {
            "verdict": "infrastructure_failure",
            "complete": True,
            "failed_arms": infra,
        }
    if arms["reference_w4a16"].get("quality_ok") is not True:
        return {"verdict": "invalid_reference", "complete": True}

    diagnostic_gaps = {
        arm: gaps
        for arm in EXPECTED_ARMS
        if (gaps := _diagnostic_gaps(arms[arm]))
    }
    if diagnostic_gaps:
        return {
            "verdict": "inconclusive_missing_diagnostics",
            "complete": True,
            "failed_arms": list(diagnostic_gaps),
            "diagnostic_gaps": diagnostic_gaps,
        }

    reference = arms["reference_w4a16"]
    fingerprint_comparisons = {
        arm: compare_unquantized_fingerprints(
            reference["fingerprints"], arms[arm]["fingerprints"]
        )
        for arm in ("candidate_w4a8", "candidate_w4a16")
    }
    # The activation-disabled overlay changes only candidate quantization
    # metadata, so these two arms must enter the first routed expert identically.
    # Reference/candidate equality is intentionally not asserted: their attention
    # representations differ (W4A16 versus BF16) before layer-3 MoE.
    first_inputs = {
        "candidate_w4a8_vs_w4a16": compare_first_moe_inputs(
            arms["candidate_w4a8"]["moe_probe_records"],
            arms["candidate_w4a16"]["moe_probe_records"],
        )
    }
    details = {
        "fingerprint_comparisons": fingerprint_comparisons,
        "first_moe_inputs": first_inputs,
    }
    if any(value["mismatched"] or value["missing"] for value in fingerprint_comparisons.values()):
        return {"verdict": "unquantized_load_boundary", "complete": True, **details}
    if any(value["mismatched_ranks"] or value["missing_ranks"] for value in first_inputs.values()):
        return {
            "verdict": "overlay_pre_moe_divergence",
            "complete": True,
            **details,
        }

    w4a8_ok = arms["candidate_w4a8"].get("quality_ok") is True
    w4a16_ok = arms["candidate_w4a16"].get("quality_ok") is True
    if not w4a8_ok and w4a16_ok:
        verdict = "w4a8_activation_boundary"
    elif not w4a8_ok and not w4a16_ok:
        verdict = "routed_weight_or_loader_boundary"
    elif w4a8_ok and not w4a16_ok:
        verdict = "w4a16_overlay_backend_regression"
    else:
        verdict = "candidate_diagnostic_pass"
    return {"verdict": verdict, "complete": True, **details}


def prepare_checkpoint_overlay(
    source: Path,
    destination: Path,
    *,
    disable_activations: bool,
    add_vllm_shared_expert_ignore: bool = False,
    add_vllm_router_ignore: bool = False,
) -> None:
    """Create a metadata-only overlay without mutating the source checkpoint."""

    source = source.resolve()
    if destination.exists():
        raise FileExistsError(f"overlay already exists: {destination}")
    destination.mkdir(parents=True)
    for item in source.iterdir():
        if item.name == "config.json":
            continue
        (destination / item.name).symlink_to(item.resolve())
    config = _read_json(source / "config.json")
    if add_vllm_shared_expert_ignore or add_vllm_router_ignore:
        quantization_config = config.get("quantization_config")
        if not isinstance(quantization_config, dict):
            raise ValueError("candidate config has no quantization_config")
        ignore = quantization_config.setdefault("ignore", [])
        if not isinstance(ignore, list):
            raise ValueError("candidate quantization_config.ignore is not a list")
        if add_vllm_shared_expert_ignore and VLLM_SHARED_EXPERT_IGNORE not in ignore:
            ignore.append(VLLM_SHARED_EXPERT_IGNORE)
        if add_vllm_router_ignore and VLLM_ROUTER_IGNORE not in ignore:
            ignore.append(VLLM_ROUTER_IGNORE)
    if disable_activations:
        groups = config.get("quantization_config", {}).get("config_groups", {})
        if not groups:
            raise ValueError("candidate config has no quantization config groups")
        for group in groups.values():
            group["input_activations"] = None
    _write_json(destination / "config.json", config)


def write_arm_manifest(
    *,
    arm: str,
    run_dir: Path,
    evidence_dir: Path,
    checkpoint: Path,
    model_id: str,
    dry_run: bool,
) -> dict[str, Any]:
    if arm not in ARM_SPECS:
        raise ValueError(f"unknown arm: {arm}")
    role, scheme = ARM_SPECS[arm]
    checkpoint = checkpoint.resolve()
    manifest = {
        "schema_version": 1,
        "matrix_id": run_dir.parent.name,
        "arm": arm,
        "checkpoint_role": role,
        "scheme": scheme,
        "checkpoint": str(checkpoint),
        "checkpoint_config_sha256": _sha256(checkpoint / "config.json"),
        "checkpoint_index_sha256": _sha256(
            checkpoint / "model.safetensors.index.json"
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
        "scheduler_log": os.environ.get("M3_DIAG_SRUN_LOG"),
        "quality_envelope": {
            "tensor_parallel_size": 8,
            "enable_expert_parallel": True,
            "enforce_eager": True,
            "block_size": 128,
            "kv_cache_dtype": "fp8",
            "max_model_len": 2048,
            "gpu_memory_utilization": 0.85,
            "disable_custom_all_reduce": True,
            "prompt_mode": "chat_template",
            "thinking_mode": "disabled",
        },
        "diagnostics": {
            "M3_LOAD_AUDIT": "1",
            "M3_MOE_PROBE": "1",
            "M3_MOE_PROBE_RECOMPUTE": "1",
            "M3_MOE_PROBE_MAX_TOKENS": "256",
            "M3_PARAM_FINGERPRINT": "1",
            "M3_PARAM_FINGERPRINT_LAYERS": "3,59",
        },
        "config_overlay": "input_activations=null" if arm == "candidate_w4a16" else None,
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

    report = _read_json(run_dir / "serve_report.json")
    log_path = run_dir / "serve.log"
    log_text = (
        log_path.read_text(encoding="utf-8", errors="replace")
        if log_path.is_file()
        else ""
    )
    extracted = extract_log_evidence(log_text)
    arm_report = {
        "arm": manifest.get("arm"),
        "scheme": manifest.get("scheme"),
        "infrastructure_ok": report.get("loaded") is True
        and isinstance(report.get("quality_cases"), list),
        "quality_ok": report.get("quality_ok") is True,
        "quality_cases": report.get("quality_cases", []),
        **extracted,
    }
    _write_json(evidence_dir / "arm_report.json", arm_report)
    _write_json(evidence_dir / "serve_report.json", report)
    _write_jsonl(evidence_dir / "parameter_fingerprints.jsonl", extracted["fingerprints"])
    _write_jsonl(evidence_dir / "fingerprint_summaries.jsonl", extracted["fingerprint_summaries"])
    _write_jsonl(evidence_dir / "moe_probe_records.jsonl", extracted["moe_probe_records"])
    (evidence_dir / "loader_audit.txt").write_text(
        "\n".join(extracted["loader_audit_lines"]) + "\n", encoding="utf-8"
    )
    (evidence_dir / "notable_log_excerpt.txt").write_text(
        "\n".join(_notable_log_excerpt(log_text)) + "\n", encoding="utf-8"
    )
    for name in (
        "software_versions.txt",
        "nvidia_smi.csv",
        "nvidia_topology.txt",
        "patch_status.txt",
        "return_code.txt",
    ):
        source = run_dir / name
        if source.is_file():
            shutil.copy2(source, evidence_dir / name)
    log_paths = [log_path]
    scheduler_log = manifest.get("scheduler_log")
    if scheduler_log:
        log_paths.append(Path(scheduler_log))
    retention = os.environ.get("RETENTION_UNTIL_UTC") or (
        datetime.now(timezone.utc) + timedelta(days=14)
    ).strftime("%Y-%m-%dT00:00:00Z")
    artifacts = [
        {
            "path": str(path.resolve()),
            "bytes": path.stat().st_size,
            "sha256": _sha256(path),
            "retention_until_utc": retention,
        }
        for path in log_paths
        if path.is_file()
    ]
    _write_json(evidence_dir / "artifact_index.json", artifacts)
    return arm_report


def aggregate_matrix(evidence_root: Path) -> dict[str, Any]:
    arms = {
        arm: report
        for arm in EXPECTED_ARMS
        if (report := _read_json(evidence_root / arm / "arm_report.json"))
    }
    comparison = {**classify_diagnostics(arms), "arms": arms}
    _write_json(evidence_root / "comparison.json", comparison)
    return comparison


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    manifest = commands.add_parser("manifest")
    manifest.add_argument("--arm", choices=EXPECTED_ARMS, required=True)
    manifest.add_argument("--run-dir", type=Path, required=True)
    manifest.add_argument("--evidence-dir", type=Path, required=True)
    manifest.add_argument("--checkpoint", type=Path, required=True)
    manifest.add_argument("--model-id", required=True)
    manifest.add_argument("--dry-run", action="store_true")
    bundle = commands.add_parser("bundle-arm")
    bundle.add_argument("--run-dir", type=Path, required=True)
    bundle.add_argument("--evidence-dir", type=Path, required=True)
    aggregate = commands.add_parser("aggregate")
    aggregate.add_argument("--evidence-root", type=Path, required=True)
    overlay = commands.add_parser("prepare-overlay")
    overlay.add_argument("--source", type=Path, required=True)
    overlay.add_argument("--destination", type=Path, required=True)
    overlay.add_argument("--disable-activations", action="store_true")
    overlay.add_argument("--add-vllm-shared-expert-ignore", action="store_true")
    overlay.add_argument("--add-vllm-router-ignore", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    if args.command == "manifest":
        write_arm_manifest(
            arm=args.arm,
            run_dir=args.run_dir,
            evidence_dir=args.evidence_dir,
            checkpoint=args.checkpoint,
            model_id=args.model_id,
            dry_run=args.dry_run,
        )
    elif args.command == "bundle-arm":
        bundle_arm(args.run_dir, args.evidence_dir)
    elif args.command == "aggregate":
        aggregate_matrix(args.evidence_root)
    else:
        prepare_checkpoint_overlay(
            args.source,
            args.destination,
            disable_activations=args.disable_activations,
            add_vllm_shared_expert_ignore=args.add_vllm_shared_expert_ignore,
            add_vllm_router_ignore=args.add_vllm_router_ignore,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
