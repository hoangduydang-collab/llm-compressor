"""MiniMax-M3 smoke-quality assessment and paired evidence decisions.

This module intentionally imports only the Python standard library so reports
can be classified on a CPU login node without importing Torch or vLLM.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import re
import shutil
import socket
import subprocess
from collections import Counter
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class QualityCase:
    """One deterministic smoke prompt and its accepted answer fragments."""

    case_id: str
    prompt: str
    expected_any: tuple[str, ...]


M3_QUALITY_CASES = (
    QualityCase("capital_france", "The capital of France is", ("paris",)),
    QualityCase(
        "arithmetic_2_plus_2",
        "What is 2 + 2? Answer with only the number.",
        ("4", "four"),
    ),
)


def _has_consecutive_character_chunk(text: str) -> bool:
    compact = re.sub(r"\s+", "", text.casefold())
    if len(compact) < 6:
        return False
    max_chunk = min(32, len(compact) // 3)
    for size in range(2, max_chunk + 1):
        for start in range(0, len(compact) - (size * 3) + 1):
            chunk = compact[start : start + size]
            repeats = 1
            cursor = start + size
            while compact[cursor : cursor + size] == chunk:
                repeats += 1
                cursor += size
            if repeats >= 3 and repeats * size >= max(12, len(compact) // 2):
                return True
    return False


def _has_dominant_token(text: str) -> bool:
    tokens = re.findall(r"[\w']+", text.casefold(), flags=re.UNICODE)
    if len(tokens) < 4:
        return False
    _, count = Counter(tokens).most_common(1)[0]
    return count >= 4 and count / len(tokens) >= 0.6


def assess_output(text: str, expected_any: tuple[str, ...]) -> dict[str, Any]:
    """Assess one raw generation without discarding evidence."""

    raw = text if isinstance(text, str) else str(text)
    normalized = " ".join(raw.casefold().split())
    reasons: list[str] = []
    if _has_dominant_token(raw):
        reasons.append("dominant_token")
    if _has_consecutive_character_chunk(raw):
        reasons.append("character_chunk")
    expected_match = any(
        re.search(
            rf"(?<!\w){re.escape(expected.casefold())}(?!\w)",
            normalized,
        )
        is not None
        for expected in expected_any
    )
    nonempty = bool(normalized)
    return {
        "text": raw,
        "normalized": normalized,
        "expected_any": list(expected_any),
        "expected_match": expected_match,
        "nonempty": nonempty,
        "repetitive": bool(reasons),
        "repetition_reasons": reasons,
        "passed": nonempty and expected_match and not reasons,
    }


def assess_quality_outputs(outputs: list[str]) -> dict[str, Any]:
    """Assess outputs in the fixed order of :data:`M3_QUALITY_CASES`."""

    errors: list[str] = []
    complete = len(outputs) == len(M3_QUALITY_CASES)
    if not complete:
        errors.append(
            f"output_count={len(outputs)} expected={len(M3_QUALITY_CASES)}"
        )
    cases: list[dict[str, Any]] = []
    for quality_case, output in zip(M3_QUALITY_CASES, outputs):
        assessed = assess_output(output, quality_case.expected_any)
        cases.append({**asdict(quality_case), **assessed})
    quality_ok = complete and all(case["passed"] for case in cases)
    return {
        "complete": complete,
        "quality_ok": quality_ok,
        "quality_cases": cases,
        "errors": errors,
    }


def classify_pair(
    reference: dict[str, Any],
    candidate: dict[str, Any],
    evidence: dict[str, Any],
) -> dict[str, Any]:
    """Select the next quality boundary without guessing past missing evidence."""

    if reference.get("quality_ok") is not True:
        return {
            "verdict": "invalid_reference",
            "next": "repair the reference serving baseline before candidate analysis",
        }
    if candidate.get("quality_ok") is True:
        return {
            "verdict": "candidate_quality_pass",
            "next": "confirm with the broader quality evaluation",
        }
    if evidence.get("required_complete") is not True:
        return {
            "verdict": "inconclusive_missing_evidence",
            "missing": list(evidence.get("missing") or []),
            "next": "collect only the missing paired evidence",
        }
    if evidence.get("lm_head_bad") is True:
        return {
            "verdict": "lm_head_boundary",
            "next": "isolate the MiniMax-M3 lm_head loader mapping",
        }
    if evidence.get("shared_expert_bad") is True:
        return {
            "verdict": "shared_expert_boundary",
            "next": "isolate shared-expert construction, loading, and contribution",
        }
    return {
        "verdict": "attention_indexer_boundary",
        "next": "compare q/k/v and MSA-indexer construction and loading",
    }


def _prefixed_json(line: str, marker: str) -> dict[str, Any] | None:
    if marker not in line:
        return None
    raw = line.split(marker, 1)[1].strip()
    try:
        value = json.loads(raw)
    except json.JSONDecodeError:
        return None
    return value if isinstance(value, dict) else None


def extract_log_evidence(log_text: str) -> dict[str, Any]:
    """Extract structured diagnostic records without dropping raw marker lines."""

    fingerprints: list[dict[str, Any]] = []
    summaries: list[dict[str, Any]] = []
    loader_lines: list[str] = []
    probe_lines: list[str] = []
    shared_bad = False
    for line in log_text.splitlines():
        record = _prefixed_json(line, "M3_PARAM_FINGERPRINT#")
        if record is not None:
            fingerprints.append(record)
            continue
        summary = _prefixed_json(line, "M3_PARAM_FINGERPRINT_SUMMARY#")
        if summary is not None:
            summaries.append(summary)
            continue
        if "M3_LOAD_AUDIT#" in line:
            loader_lines.append(line.strip())
        if "M3_MOE_PROBE#" in line:
            clean = line.strip()
            probe_lines.append(clean)
            shared_match = re.search(r"shared_norm=([-+0-9.eE]+)", clean)
            shared_bad = shared_bad or "shared_present=False" in clean
            shared_bad = shared_bad or "SHARED EXPERT DROPPED" in clean
            if shared_match:
                try:
                    norm = float(shared_match.group(1))
                except ValueError:
                    norm = -1.0
                shared_bad = shared_bad or 0.0 <= norm <= 1e-3
    return {
        "fingerprints": fingerprints,
        "fingerprint_summaries": summaries,
        "loader_audit_lines": loader_lines,
        "moe_probe_lines": probe_lines,
        "shared_expert_bad": shared_bad,
    }


def _sha256(path: Path) -> str | None:
    if not path.is_file():
        return None
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _git_output(*args: str) -> str:
    result = subprocess.run(
        ["git", *args],
        check=False,
        text=True,
        capture_output=True,
    )
    return result.stdout.strip()


def _case_command(
    *,
    case_dir: Path,
    config: str,
    model_id: str,
    max_model_len: int,
    gpu_util: float,
) -> list[str]:
    return [
        "python",
        "-m",
        "pipeline.run",
        "--config",
        config,
        "--stage",
        "serve",
        "--checkpoint",
        str(case_dir / "checkpoint"),
        "--set",
        f"model.id={model_id}",
        "--set",
        "serve.tensor_parallel_size=8",
        "--set",
        "serve.enable_expert_parallel=true",
        "--set",
        "serve.block_size=128",
        "--set",
        "serve.kv_cache_dtype=fp8",
        "--set",
        f"serve.max_model_len={max_model_len}",
        "--set",
        f"serve.gpu_memory_utilization={gpu_util}",
        "--set",
        "serve.enforce_eager=true",
        "--set",
        "serve.disable_custom_all_reduce=true",
        "--set",
        "eval.enabled=false",
    ]


def write_run_manifest(
    *,
    run_dir: Path,
    evidence_dir: Path,
    reference: Path,
    candidate: Path,
    config: str,
    model_id: str,
    dry_run: bool,
    max_model_len: int,
    gpu_util: float,
) -> dict[str, Any]:
    """Write allowlisted provenance and the exact two-case comparison envelope."""

    reference = reference.resolve()
    candidate = candidate.resolve()
    cases = []
    for name, checkpoint in (
        ("cyankiwi_reference", reference),
        ("portable_awq_w4a8", candidate),
    ):
        case_dir = run_dir / name
        cases.append(
            {
                "name": name,
                "checkpoint": str(checkpoint),
                "config_sha256": _sha256(checkpoint / "config.json"),
                "index_sha256": _sha256(
                    checkpoint / "model.safetensors.index.json"
                ),
                "command": _case_command(
                    case_dir=case_dir,
                    config=config,
                    model_id=model_id,
                    max_model_len=max_model_len,
                    gpu_util=gpu_util,
                ),
            }
        )
    case_order = [case["name"] for case in cases]
    manifest = {
        "schema_version": 1,
        "run_id": run_dir.name,
        "evidence_dir": str(evidence_dir.resolve()),
        "dry_run": dry_run,
        "started_at": datetime.now(timezone.utc).isoformat(),
        "finished_at": None,
        "case_order": case_order,
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
        "model_id": model_id,
        "config": config,
        "comparison_envelope": {
            "tensor_parallel_size": 8,
            "enable_expert_parallel": True,
            "enforce_eager": True,
            "block_size": 128,
            "kv_cache_dtype": "fp8",
            "max_model_len": max_model_len,
            "gpu_memory_utilization": gpu_util,
            "disable_custom_all_reduce": True,
        },
        "diagnostics": {
            "M3_LOAD_AUDIT": "1",
            "M3_MOE_PROBE": "1",
            "M3_PARAM_FINGERPRINT": "1",
        },
        "deviations": [],
        "cases": cases,
    }
    run_dir.mkdir(parents=True, exist_ok=True)
    evidence_dir.mkdir(parents=True, exist_ok=True)
    encoded = json.dumps(manifest, indent=2, sort_keys=True) + "\n"
    (run_dir / "run_manifest.json").write_text(encoded, encoding="utf-8")
    (evidence_dir / "run_manifest.json").write_text(encoded, encoding="utf-8")
    return manifest


def _read_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {}
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return value if isinstance(value, dict) else {}


def _write_jsonl(path: Path, records: list[dict[str, Any]]) -> None:
    path.write_text(
        "".join(json.dumps(record, sort_keys=True) + "\n" for record in records),
        encoding="utf-8",
    )


def _fingerprint_bad(records: list[dict[str, Any]], category: str) -> bool:
    selected = [record for record in records if record.get("category") == category]
    return any(
        record.get("finite_fraction", 1.0) < 1.0
        or record.get("sample_abs_max", 1.0) <= 1e-12
        for record in selected
    )


def _combined_evidence(
    reference: dict[str, Any],
    candidate: dict[str, Any],
) -> dict[str, Any]:
    missing: list[str] = []
    required_categories = {"lm_head", "shared_expert"}
    for case_name, case in (("reference", reference), ("candidate", candidate)):
        found = {
            record.get("category")
            for record in case["fingerprints"]
            if record.get("category")
        }
        for category in sorted(required_categories - found):
            missing.append(f"{case_name}.fingerprint.{category}")
        if not case["moe_probe_lines"]:
            missing.append(f"{case_name}.moe_probe")
        if not case["loader_audit_lines"]:
            missing.append(f"{case_name}.loader_audit")
    return {
        "required_complete": not missing,
        "missing": missing,
        "lm_head_bad": _fingerprint_bad(
            candidate["fingerprints"], "lm_head"
        ),
        "shared_expert_bad": (
            candidate["shared_expert_bad"]
            or _fingerprint_bad(candidate["fingerprints"], "shared_expert")
        ),
    }


def bundle_run(run_dir: Path, evidence_dir: Path) -> dict[str, Any]:
    """Build a compact auditable bundle from full per-case logs and reports."""

    evidence_dir.mkdir(parents=True, exist_ok=True)
    manifest = _read_json(run_dir / "run_manifest.json")
    if manifest:
        manifest["finished_at"] = datetime.now(timezone.utc).isoformat()
        for case in manifest.get("cases", []):
            case_dir = run_dir / str(case.get("name"))
            for field, filename in (
                ("started_at", "started_at.txt"),
                ("finished_at", "finished_at.txt"),
                ("return_code", "return_code.txt"),
            ):
                path = case_dir / filename
                if path.is_file():
                    value = path.read_text(encoding="utf-8").strip()
                    case[field] = int(value) if field == "return_code" else value
        encoded = json.dumps(manifest, indent=2, sort_keys=True) + "\n"
        (run_dir / "run_manifest.json").write_text(encoded, encoding="utf-8")
        (evidence_dir / "run_manifest.json").write_text(
            encoded, encoding="utf-8"
        )
    if manifest.get("dry_run") is True:
        comparison = {"verdict": "dry_run", "required_complete": False}
        (evidence_dir / "comparison.json").write_text(
            json.dumps(comparison, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        return comparison

    for name in (
        "software_versions.txt",
        "nvidia_smi.csv",
        "nvidia_topology.txt",
        "patch_status.txt",
    ):
        source = run_dir / name
        if source.is_file():
            shutil.copy2(source, evidence_dir / name)

    case_evidence: dict[str, dict[str, Any]] = {}
    artifact_index: list[dict[str, Any]] = []
    for case_name in ("cyankiwi_reference", "portable_awq_w4a8"):
        source_dir = run_dir / case_name
        output_dir = evidence_dir / case_name
        output_dir.mkdir(parents=True, exist_ok=True)
        report_path = source_dir / "serve_report.json"
        log_path = source_dir / "serve.log"
        report = _read_json(report_path)
        log_text = (
            log_path.read_text(encoding="utf-8", errors="replace")
            if log_path.is_file()
            else ""
        )
        extracted = extract_log_evidence(log_text)
        case_evidence[case_name] = extracted
        (output_dir / "serve_report.json").write_text(
            json.dumps(report, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        return_code = source_dir / "return_code.txt"
        if return_code.is_file():
            shutil.copy2(return_code, output_dir / "return_code.txt")
        _write_jsonl(
            output_dir / "parameter_fingerprints.jsonl",
            extracted["fingerprints"],
        )
        _write_jsonl(
            output_dir / "fingerprint_summaries.jsonl",
            extracted["fingerprint_summaries"],
        )
        (output_dir / "moe_probe.txt").write_text(
            "\n".join(extracted["moe_probe_lines"]) + "\n",
            encoding="utf-8",
        )
        (output_dir / "loader_audit.txt").write_text(
            "\n".join(extracted["loader_audit_lines"]) + "\n",
            encoding="utf-8",
        )
        notable = [
            line
            for line in log_text.splitlines()
            if re.search(r"warning|error|traceback|failed|M3_", line, re.I)
        ][-300:]
        (output_dir / "notable_log_excerpt.txt").write_text(
            "\n".join(notable) + "\n",
            encoding="utf-8",
        )
        if log_path.is_file():
            artifact_index.append(
                {
                    "case": case_name,
                    "path": str(log_path.resolve()),
                    "bytes": log_path.stat().st_size,
                    "sha256": _sha256(log_path),
                }
            )

    reference_report = _read_json(
        run_dir / "cyankiwi_reference/serve_report.json"
    )
    candidate_report = _read_json(
        run_dir / "portable_awq_w4a8/serve_report.json"
    )
    combined = _combined_evidence(
        case_evidence["cyankiwi_reference"],
        case_evidence["portable_awq_w4a8"],
    )
    comparison = {
        **classify_pair(reference_report, candidate_report, combined),
        "evidence": combined,
    }
    (evidence_dir / "comparison.json").write_text(
        json.dumps(comparison, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    (evidence_dir / "artifact_index.json").write_text(
        json.dumps(artifact_index, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return comparison


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    manifest = subparsers.add_parser("manifest")
    manifest.add_argument("--run-dir", type=Path, required=True)
    manifest.add_argument("--evidence-dir", type=Path, required=True)
    manifest.add_argument("--reference", type=Path, required=True)
    manifest.add_argument("--candidate", type=Path, required=True)
    manifest.add_argument("--config", required=True)
    manifest.add_argument("--model-id", required=True)
    manifest.add_argument("--max-model-len", type=int, required=True)
    manifest.add_argument("--gpu-util", type=float, required=True)
    manifest.add_argument("--dry-run", action="store_true")
    bundle = subparsers.add_parser("bundle")
    bundle.add_argument("--run-dir", type=Path, required=True)
    bundle.add_argument("--evidence-dir", type=Path)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    if args.command == "manifest":
        write_run_manifest(
            run_dir=args.run_dir,
            evidence_dir=args.evidence_dir,
            reference=args.reference,
            candidate=args.candidate,
            config=args.config,
            model_id=args.model_id,
            dry_run=args.dry_run,
            max_model_len=args.max_model_len,
            gpu_util=args.gpu_util,
        )
    else:
        evidence_dir = args.evidence_dir
        if evidence_dir is None:
            evidence_value = _read_json(args.run_dir / "run_manifest.json").get(
                "evidence_dir"
            )
            if not isinstance(evidence_value, str) or not evidence_value:
                raise ValueError(
                    "bundle requires --evidence-dir when the run manifest "
                    "does not record one"
                )
            evidence_dir = Path(evidence_value)
        bundle_run(args.run_dir, evidence_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
