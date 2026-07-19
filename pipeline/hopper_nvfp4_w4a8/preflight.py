"""Fail-closed eligibility gate for the Hopper NVFP4 W4A8 specialization."""

from __future__ import annotations

import argparse
import dataclasses
import importlib.metadata
import json
from pathlib import Path
from typing import Mapping, Sequence


@dataclasses.dataclass(frozen=True)
class PreflightInput:
    device_capability: tuple[int, int]
    humming_version: str
    patch_status: str
    source_hashes: Mapping[str, str]
    activation_dtype: str
    weight_dtype: str
    scale_dtype: str
    input_group_size: int
    weight_group_size: int
    weight_scale_type: str
    has_zero_point: bool
    global_scale_count: int
    expected_global_scale_count: int
    global_scale_uniform: bool
    accumulator_dtype: str
    output_dtype: str
    checkpoint_bytes: int
    transformed_bytes: int


@dataclasses.dataclass(frozen=True)
class PreflightReport:
    eligible: bool
    backend: str
    reason_code: str
    runtime_retry: bool
    details: Mapping[str, object]

    def as_dict(self) -> dict[str, object]:
        return dataclasses.asdict(self)


def _details(candidate: PreflightInput) -> dict[str, object]:
    ratio = (
        candidate.transformed_bytes / candidate.checkpoint_bytes
        if candidate.checkpoint_bytes > 0
        else None
    )
    return {
        "device_capability": candidate.device_capability,
        "humming_version": candidate.humming_version,
        "patch_status": candidate.patch_status,
        "source_hashes": dict(candidate.source_hashes),
        "activation_dtype": candidate.activation_dtype,
        "weight_dtype": candidate.weight_dtype,
        "scale_dtype": candidate.scale_dtype,
        "input_group_size": candidate.input_group_size,
        "weight_group_size": candidate.weight_group_size,
        "weight_scale_type": candidate.weight_scale_type,
        "has_zero_point": candidate.has_zero_point,
        "global_scale_count": candidate.global_scale_count,
        "expected_global_scale_count": candidate.expected_global_scale_count,
        "global_scale_uniform": candidate.global_scale_uniform,
        "accumulator_dtype": candidate.accumulator_dtype,
        "output_dtype": candidate.output_dtype,
        "checkpoint_bytes": candidate.checkpoint_bytes,
        "transformed_bytes": candidate.transformed_bytes,
        "persistent_ratio": ratio,
    }


def evaluate_preflight(candidate: PreflightInput) -> PreflightReport:
    """Select the specialization only when every exact policy gate passes."""

    checks = (
        (candidate.device_capability == (9, 0), "SM_NOT_90"),
        (candidate.humming_version == "0.1.10", "HUMMING_VERSION_MISMATCH"),
        (candidate.patch_status == "patched", "PATCH_INCOMPLETE"),
        (candidate.weight_dtype == "float4e2m1", "WEIGHT_DTYPE_NOT_E2M1"),
        (
            candidate.activation_dtype == "float8e4m3",
            "ACTIVATION_DTYPE_NOT_E4M3",
        ),
        (candidate.scale_dtype == "float8e4m3", "SCALE_DTYPE_NOT_E4M3"),
        (candidate.input_group_size == 0, "INPUT_GROUP_SIZE_NOT_TENSOR"),
        (candidate.weight_group_size == 16, "WEIGHT_GROUP_SIZE_NOT_16"),
        (not candidate.has_zero_point, "ZERO_POINT_UNSUPPORTED"),
        (
            candidate.weight_scale_type == "GROUP_TENSOR",
            "SCALE_TYPE_NOT_GROUP_TENSOR",
        ),
        (candidate.global_scale_count > 0, "GLOBAL_SCALE_MISSING"),
        (candidate.global_scale_uniform, "GLOBAL_SCALE_NOT_UNIFORM"),
        (
            candidate.global_scale_count == candidate.expected_global_scale_count,
            "GLOBAL_SCALE_CARDINALITY",
        ),
        (candidate.accumulator_dtype == "float32", "F16_ACCUM_UNSUPPORTED"),
        (candidate.output_dtype == "bfloat16", "OUTPUT_DTYPE_NOT_BF16"),
        (
            candidate.checkpoint_bytes > 0 and candidate.transformed_bytes > 0,
            "INVALID_BYTE_ACCOUNTING",
        ),
        (
            candidate.checkpoint_bytes > 0
            and candidate.transformed_bytes / candidate.checkpoint_bytes <= 1.10,
            "PERSISTENT_RATIO_EXCEEDED",
        ),
    )
    details = _details(candidate)
    for accepted, reason in checks:
        if not accepted:
            return PreflightReport(
                eligible=False,
                backend="marlin_w4a16",
                reason_code=reason,
                runtime_retry=False,
                details=details,
            )
    return PreflightReport(
        eligible=True,
        backend="humming_w4a8_g16",
        reason_code="ELIGIBLE",
        runtime_retry=False,
        details=details,
    )


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Evaluate the Hopper NVFP4 W4A8 fail-closed preflight"
    )
    parser.add_argument("--metadata-json", type=Path, required=True)
    parser.add_argument("--patch-report", type=Path, required=True)
    return parser


def _discover_candidate(metadata_path: Path, patch_path: Path) -> PreflightInput:
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    patch = json.loads(patch_path.read_text(encoding="utf-8"))
    try:
        import torch
    except ImportError as exc:
        raise RuntimeError("torch is required for device discovery") from exc

    version = importlib.metadata.version("humming-kernels")
    capability = tuple(torch.cuda.get_device_capability())
    files = patch.get("files", [])
    source_hashes = {item["relative_path"]: item["after_sha256"] for item in files}
    return PreflightInput(
        device_capability=(int(capability[0]), int(capability[1])),
        humming_version=version,
        patch_status=str(patch.get("status", "unknown")),
        source_hashes=source_hashes,
        **metadata,
    )


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        report = evaluate_preflight(
            _discover_candidate(args.metadata_json, args.patch_report)
        )
    except Exception as exc:
        payload = {
            "eligible": False,
            "backend": "marlin_w4a16",
            "reason_code": "DISCOVERY_ERROR",
            "runtime_retry": False,
            "details": {"error": str(exc)},
        }
        print(json.dumps(payload, sort_keys=True))
        return 2

    print(json.dumps(report.as_dict(), sort_keys=True))
    return 0 if report.backend == "humming_w4a8_g16" else 1


if __name__ == "__main__":
    raise SystemExit(main())
