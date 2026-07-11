"""Canonical MiniMax-M3 offline/HTTP chat quality evidence.

Standard-library only so manifests and result aggregation run on login nodes.
"""

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

from pipeline.m3_quality_evidence import (
    M3_QUALITY_CASES,
    _notable_log_excerpt,
    assess_quality_outputs,
)

ARM_SPECS: dict[str, tuple[str, str]] = {
    "reference_offline_chat": ("reference", "offline"),
    "candidate_offline_chat": ("candidate", "offline"),
    "reference_http_chat": ("reference", "http"),
    "candidate_http_chat": ("candidate", "http"),
}
EXPECTED_ARMS = tuple(ARM_SPECS)


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
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


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


def normalize_http_responses(responses: list[dict[str, Any]]) -> dict[str, Any]:
    """Normalize two OpenAI chat responses into the shared quality schema."""

    texts: list[str] = []
    metadata: list[dict[str, Any]] = []
    errors: list[str] = []
    for index, response in enumerate(responses):
        choices = response.get("choices")
        if not isinstance(choices, list) or not choices:
            errors.append(f"response_{index}: missing choices: {response.get('error')!r}")
            texts.append("")
            metadata.append({"finish_reason": None, "raw_response": response})
            continue
        choice = choices[0] if isinstance(choices[0], dict) else {}
        message = choice.get("message") if isinstance(choice.get("message"), dict) else {}
        content = message.get("content")
        text = content if isinstance(content, str) else ""
        if not isinstance(content, str):
            errors.append(f"response_{index}: missing assistant content")
        texts.append(text)
        metadata.append(
            {
                "finish_reason": choice.get("finish_reason"),
                "stop_reason": choice.get("stop_reason"),
                "raw_response": response,
            }
        )
    assessed = assess_quality_outputs(texts)
    assessed["errors"] = [*assessed.get("errors", []), *errors]
    for case, extra in zip(assessed["quality_cases"], metadata):
        case.update(extra)
    infrastructure_ok = len(responses) == len(M3_QUALITY_CASES) and not errors
    return {
        **assessed,
        "interface": "http",
        "prompt_mode": "chat_completions",
        "infrastructure_ok": infrastructure_ok,
        "quality_ok": infrastructure_ok and assessed["quality_ok"],
    }


def normalize_offline_report(report: dict[str, Any]) -> dict[str, Any]:
    """Normalize an offline serve report without discarding raw evidence."""

    cases = report.get("quality_cases")
    infrastructure_ok = report.get("loaded") is True and isinstance(cases, list)
    return {
        **report,
        "interface": "offline",
        "infrastructure_ok": infrastructure_ok,
        "quality_ok": infrastructure_ok and report.get("quality_ok") is True,
    }


def classify_matrix(arms: dict[str, dict[str, Any]]) -> dict[str, Any]:
    """Classify the four-arm matrix without reasoning past missing controls."""

    missing = [arm for arm in EXPECTED_ARMS if arm not in arms]
    if missing:
        return {
            "verdict": "inconclusive_missing_arms",
            "complete": False,
            "missing_arms": missing,
        }
    infrastructure_failed = [
        arm for arm in EXPECTED_ARMS if arms[arm].get("infrastructure_ok") is not True
    ]
    if infrastructure_failed:
        return {
            "verdict": "infrastructure_failure",
            "complete": True,
            "failed_arms": infrastructure_failed,
        }
    reference_failed = [
        arm
        for arm in ("reference_offline_chat", "reference_http_chat")
        if arms[arm].get("quality_ok") is not True
    ]
    if reference_failed:
        return {
            "verdict": "invalid_reference",
            "complete": True,
            "failed_arms": reference_failed,
        }
    candidate_failed = [
        arm
        for arm in ("candidate_offline_chat", "candidate_http_chat")
        if arms[arm].get("quality_ok") is not True
    ]
    if len(candidate_failed) == 1:
        return {
            "verdict": "candidate_interface_disagreement",
            "complete": True,
            "failed_arms": candidate_failed,
        }
    if candidate_failed:
        return {
            "verdict": "candidate_quality_fail",
            "complete": True,
            "failed_arms": candidate_failed,
        }
    return {"verdict": "candidate_quality_pass", "complete": True}


def write_arm_manifest(
    *,
    arm: str,
    run_dir: Path,
    evidence_dir: Path,
    checkpoint: Path,
    model_id: str,
    dry_run: bool,
) -> dict[str, Any]:
    """Record the immutable quality envelope for one independent arm."""

    if arm not in ARM_SPECS:
        raise ValueError(f"unknown arm: {arm}")
    role, interface = ARM_SPECS[arm]
    checkpoint = checkpoint.resolve()
    manifest = {
        "schema_version": 1,
        "matrix_id": run_dir.parent.name,
        "arm": arm,
        "checkpoint_role": role,
        "interface": interface,
        "checkpoint": str(checkpoint),
        "checkpoint_config_sha256": _sha256(checkpoint / "config.json"),
        "checkpoint_index_sha256": _sha256(
            checkpoint / "model.safetensors.index.json"
        ),
        "model_id": model_id,
        "checkpoint_tokenizer_config_sha256": _sha256(
            checkpoint / "tokenizer_config.json"
        ),
        "checkpoint_chat_template_sha256": _sha256(
            checkpoint / "chat_template.jinja"
        ),
        "model_id_tokenizer_config_sha256": _sha256(
            Path(model_id) / "tokenizer_config.json"
        ),
        "model_id_chat_template_sha256": _sha256(
            Path(model_id) / "chat_template.jinja"
        ),
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
        "scheduler_logs": {
            "stdout": os.environ.get("M3_MATRIX_STDOUT"),
            "stderr": os.environ.get("M3_MATRIX_STDERR"),
        },
        "runtime_context": {
            key: os.environ.get(key)
            for key in ("SBATCH_ARGS", "RESULTS_ROOT", "EVIDENCE_ROOT", "PORT")
            if os.environ.get(key)
        },
        "quality_envelope": {
            "tensor_parallel_size": 8,
            "enable_expert_parallel": True,
            "enforce_eager": True,
            "block_size": 128,
            "kv_cache_dtype": "fp8",
            "max_model_len": 2048,
            "gpu_memory_utilization": 0.85,
            "disable_custom_all_reduce": True,
            "max_tokens": 64,
            "temperature": 0.0,
            "thinking_mode": "disabled",
        },
        "command": (
            [
                "python", "-m", "pipeline.run", "--config",
                "pipeline/configs/minimax_m3_full_calib.yaml", "--stage", "serve",
                "--checkpoint", str(run_dir / "checkpoint"), "--set",
                f"model.id={model_id}", "--set", "serve.tensor_parallel_size=8",
                "--set", "serve.enable_expert_parallel=true", "--set",
                "serve.block_size=128", "--set", "serve.kv_cache_dtype=fp8",
                "--set", "serve.max_model_len=2048", "--set",
                "serve.gpu_memory_utilization=0.85", "--set",
                "serve.enforce_eager=true", "--set",
                "serve.disable_custom_all_reduce=true", "--set", "eval.enabled=false",
            ]
            if interface == "offline"
            else [
                "bash", "pipeline/slurm/run_vllm_http_serve_smoke.sh",
                f"CKPT={run_dir / 'checkpoint'}", "MAX_MODEL_LEN=2048", "GPU_UTIL=0.85",
                "ENFORCE_EAGER=1", "ENABLE_EP=1", "DISABLE_CUSTOM_AR=1",
                "LANGUAGE_MODEL_ONLY=1", "DEBUG_CUDAGRAPH=0",
            ]
        ),
        "diagnostics": {
            "M3_LOAD_AUDIT": "0",
            "M3_MOE_PROBE": "0",
            "M3_PARAM_FINGERPRINT": "0",
        },
        "evidence_dir": str(evidence_dir.resolve()),
        "deviations": [],
        "retries": [],
    }
    _write_json(run_dir / "arm_manifest.json", manifest)
    _write_json(evidence_dir / "arm_manifest.json", manifest)
    return manifest


def bundle_arm(run_dir: Path, evidence_dir: Path) -> dict[str, Any]:
    """Build compact evidence for one arm, even after runtime failure."""

    evidence_dir.mkdir(parents=True, exist_ok=True)
    manifest = _read_json(run_dir / "arm_manifest.json")
    manifest["finished_at"] = datetime.now(timezone.utc).isoformat()
    return_code_path = run_dir / "return_code.txt"
    if return_code_path.is_file():
        try:
            manifest["return_code"] = int(return_code_path.read_text().strip())
        except ValueError:
            manifest["return_code"] = return_code_path.read_text().strip()
    _write_json(run_dir / "arm_manifest.json", manifest)
    _write_json(evidence_dir / "arm_manifest.json", manifest)

    interface = manifest.get("interface")
    if interface == "http":
        responses = [
            _read_json(run_dir / f"http_response_{index}.json")
            for index in range(len(M3_QUALITY_CASES))
        ]
        report = normalize_http_responses(responses)
    else:
        report = normalize_offline_report(_read_json(run_dir / "serve_report.json"))
    report.update(
        {
            "arm": manifest.get("arm"),
            "checkpoint_role": manifest.get("checkpoint_role"),
        }
    )
    _write_json(evidence_dir / "arm_report.json", report)

    for name in (
        "software_versions.txt",
        "nvidia_smi.csv",
        "nvidia_topology.txt",
        "server_start.txt",
        "return_code.txt",
    ):
        source = run_dir / name
        if source.is_file():
            shutil.copy2(source, evidence_dir / name)
    for index in range(len(M3_QUALITY_CASES)):
        for prefix in ("http_request", "http_response"):
            source = run_dir / f"{prefix}_{index}.json"
            if source.is_file():
                shutil.copy2(source, evidence_dir / source.name)
    log_paths = [run_dir / "serve.log", run_dir / "operator.log"]
    for value in manifest.get("scheduler_logs", {}).values():
        if value:
            log_paths.append(Path(value))
    log_text = "\n".join(
        path.read_text(encoding="utf-8", errors="replace")
        for path in log_paths
        if path.is_file()
    )
    (evidence_dir / "notable_log_excerpt.txt").write_text(
        "\n".join(_notable_log_excerpt(log_text)) + "\n",
        encoding="utf-8",
    )
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
    return report


def aggregate_matrix(evidence_root: Path) -> dict[str, Any]:
    """Aggregate independently completed arm bundles."""

    arms = {
        arm: report
        for arm in EXPECTED_ARMS
        if (report := _read_json(evidence_root / arm / "arm_report.json"))
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
    manifest.add_argument("--checkpoint", type=Path, required=True)
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
            arm=args.arm,
            run_dir=args.run_dir,
            evidence_dir=args.evidence_dir,
            checkpoint=args.checkpoint,
            model_id=args.model_id,
            dry_run=args.dry_run,
        )
    elif args.command == "bundle-arm":
        bundle_arm(args.run_dir, args.evidence_dir)
    else:
        aggregate_matrix(args.evidence_root)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
