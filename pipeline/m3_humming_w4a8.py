"""Fail-closed native Humming W4A8 qualification for MiniMax-M3."""

from __future__ import annotations

import argparse
import base64
import dataclasses
import hashlib
import importlib.metadata
import json
import os
import re
from pathlib import Path
from typing import Any, Mapping, Sequence

from pipeline.m3_serve_abi import analyze_checkpoint

EXPECTED_VLLM_VERSION = "0.24.0"
EXPECTED_HUMMING_VERSION = "0.1.10"
EXPECTED_DEVICE_CAPABILITY = (9, 0)
EXPECTED_CACHE_BASENAME = "cache-m3-gptq-w4a8-v1"
NVFP4_OVERLAY_MARKER = b"LLMC_NVFP4_W4A8_G16_V1"

# Third-party Humming files we knowingly modify, pinned to the exact post-patch
# SHA-256. A declared patch is reported, never silently tolerated; any other
# content for these paths, and any modification of an undeclared path, is still
# a hard mismatch. Applied by pipeline/slurm/patch_humming_ct_input_format.py.
DECLARED_PATCH_SHA256: dict[str, str] = {
    "humming/schema/compressed_tensors.py": (
        "8e2ab300b595e98f9b66d76096c6a03272ffe948e11dd29844af701c1f6474c3"
    ),
    # grouped_contiguous last-expert row count: derive it from the loaded
    # expert offsets instead of shape_m (== a.size(0)), which vLLM oversizes to
    # (M * topk, K). Unpatched, the final expert is assigned thousands of
    # phantom rows, inflating m_blocks and corrupting the tail experts' tiles --
    # measured as 100% of experts 13/14/15's rows wrong. See
    # pipeline/slurm/patch_humming_grouped_expert_bounds.py.
    "humming/include/humming/scheduler.cuh": (
        "befa01f9758df24e34be12022c86aec701de81d182e0bec713374d987df1839f"
    ),
}
RECORD_MATCHED = "record-matched"
RECORD_MATCHED_PATCHED = "record-matched-with-declared-patch"
ACCEPTED_INTEGRITY = frozenset({RECORD_MATCHED, RECORD_MATCHED_PATCHED})

_QUANTIZATION_PATTERNS = (
    re.compile(r"""["']quantization["']\s*:\s*["']humming["']"""),
    re.compile(r"\bquantization=humming\b"),
)
_INDEXED_MARKER = "Using indexed gemm for humming moe"
_GROUPED_MARKER = "Using grouped_contiguous gemm for humming moe"

# Humming MoE scheduling strategies we are prepared to attest.
#
# ``indexed`` was qualified first (arm 2 of
# M3_HOPPER_W4A8_KERNEL_INVESTIGATION.md); ``grouped_contiguous`` is arm 3,
# unblocked once arm 2 passed. The two are not interchangeable at the kernel
# level: humming/tune/sm90.py grants TMA + warp specialization to every gemm
# type EXCEPT indexed, so grouped compiles the warp-specialized kernel
# (humming_ws.cuh) while indexed falls back to cp.async.
GEMM_TYPE_INDEXED = "indexed"
GEMM_TYPE_GROUPED = "grouped_contiguous"

# vLLM's get_humming_moe_gemm_type() maps a bare "grouped" onto
# "grouped_contiguous" and silently falls back to "indexed" for anything it does
# not recognise. Mirror that mapping exactly so what we attest is what vLLM
# selected -- an unrecognised value must never be reported as the requested one.
_GEMM_TYPE_ALIASES = {
    "": GEMM_TYPE_INDEXED,
    "indexed": GEMM_TYPE_INDEXED,
    "grouped": GEMM_TYPE_GROUPED,
    "grouped_contiguous": GEMM_TYPE_GROUPED,
}
SUPPORTED_GEMM_TYPES = frozenset(_GEMM_TYPE_ALIASES)


def normalize_gemm_type(raw: str) -> str | None:
    """Resolve a ``VLLM_HUMMING_MOE_GEMM_TYPE`` value, or None if unsupported."""
    return _GEMM_TYPE_ALIASES.get((raw or "").strip().lower())
_CUTLASS_MARKER = "Using CUTLASS W4A8 MoE backend"
_MARLIN_PATTERN = re.compile(r"\bUsing Marlin\b", re.IGNORECASE)
_UNQUANTIZED_MARKER = "UnquantizedFusedMoEMethod"
_COMPILE_CACHE_PATTERN = re.compile(r"compile|cache", re.IGNORECASE)


@dataclasses.dataclass(frozen=True)
class RuntimeFacts:
    vllm_version: str
    humming_version: str
    device_capability: tuple[int, int]
    humming_source_integrity: str
    humming_source_mismatches: tuple[str, ...]
    nvfp4_overlay_detected: bool
    humming_cache_dir: str
    f16_accum: str
    moe_gemm_type: str
    normal_patch_status: str
    humming_patch_status: str
    humming_unhashed_bytecode: int = 0
    humming_declared_patches: tuple[str, ...] = ()


def _mapping(value: object) -> Mapping[str, Any]:
    return value if isinstance(value, Mapping) else {}


def _first_reason(checks: Sequence[tuple[bool, str]]) -> str | None:
    for accepted, reason in checks:
        if not accepted:
            return reason
    return None


def evaluate_preflight(
    config: Mapping[str, object],
    abi_report: Mapping[str, object],
    runtime: RuntimeFacts,
) -> dict[str, object]:
    """Evaluate the exact first-qualification checkpoint and runtime policy."""

    quant = _mapping(config.get("quantization_config"))
    groups = _mapping(quant.get("config_groups"))
    linear_groups = [
        _mapping(group)
        for group in groups.values()
        if "Linear" in list(_mapping(group).get("targets") or [])
    ]
    group = linear_groups[0] if len(linear_groups) == 1 else {}
    weights = _mapping(group.get("weights"))
    activations = _mapping(group.get("input_activations"))
    ignore = quant.get("ignore")
    components = _mapping(abi_report.get("components"))
    routed = _mapping(components.get("routed_experts"))
    cache_basename = (
        Path(runtime.humming_cache_dir).name
        if runtime.humming_cache_dir
        else ""
    )

    checks = (
        (
            runtime.vllm_version == EXPECTED_VLLM_VERSION,
            "VLLM_VERSION_MISMATCH",
        ),
        (
            runtime.humming_version == EXPECTED_HUMMING_VERSION,
            "HUMMING_VERSION_MISMATCH",
        ),
        (
            runtime.device_capability == EXPECTED_DEVICE_CAPABILITY,
            "SM_NOT_90",
        ),
        (not runtime.nvfp4_overlay_detected, "NVFP4_OVERLAY_PRESENT"),
        (
            runtime.humming_source_integrity in ACCEPTED_INTEGRITY
            and not runtime.humming_source_mismatches,
            "HUMMING_SOURCE_INTEGRITY",
        ),
        (
            set(runtime.humming_declared_patches)
            <= set(DECLARED_PATCH_SHA256),
            "HUMMING_UNDECLARED_PATCH",
        ),
        (
            cache_basename == EXPECTED_CACHE_BASENAME,
            "HUMMING_CACHE_NAMESPACE",
        ),
        (
            runtime.f16_accum in {"", "0", "false", "False"},
            "F16_ACCUM_ENABLED",
        ),
        (
            normalize_gemm_type(runtime.moe_gemm_type) is not None,
            "MOE_GEMM_TYPE_UNSUPPORTED",
        ),
        (
            runtime.normal_patch_status == "patched",
            "NORMAL_PATCH_INCOMPLETE",
        ),
        (
            runtime.humming_patch_status == "patched",
            "HUMMING_PATCH_INCOMPLETE",
        ),
        (
            str(quant.get("quant_method") or "").lower()
            == "compressed-tensors",
            "QUANT_METHOD_MISMATCH",
        ),
        (quant.get("format") == "pack-quantized", "FORMAT_MISMATCH"),
        (len(linear_groups) == 1, "LINEAR_GROUP_CARDINALITY"),
        (weights.get("num_bits") == 4, "WEIGHT_BITS_MISMATCH"),
        (
            str(weights.get("type") or "").lower() == "int",
            "WEIGHT_TYPE_MISMATCH",
        ),
        (weights.get("symmetric") is True, "WEIGHT_SYMMETRY_MISMATCH"),
        (weights.get("strategy") == "group", "WEIGHT_STRATEGY_MISMATCH"),
        (weights.get("group_size") == 128, "WEIGHT_GROUP_SIZE_MISMATCH"),
        (weights.get("dynamic") is False, "WEIGHT_DYNAMIC_MISMATCH"),
        (activations.get("num_bits") == 8, "ACTIVATION_BITS_MISMATCH"),
        (
            str(activations.get("type") or "").lower() == "float",
            "ACTIVATION_TYPE_MISMATCH",
        ),
        (
            activations.get("symmetric") is True,
            "ACTIVATION_SYMMETRY_MISMATCH",
        ),
        (
            activations.get("strategy") == "token",
            "ACTIVATION_STRATEGY_MISMATCH",
        ),
        (
            "group_size" in activations
            and activations.get("group_size") is None,
            "ACTIVATION_GROUP_SIZE_MISMATCH",
        ),
        (
            activations.get("dynamic") is True,
            "ACTIVATION_DYNAMIC_MISMATCH",
        ),
        (
            isinstance(ignore, list) and len(ignore) > 0,
            "IGNORE_METADATA_MISSING",
        ),
        (abi_report.get("valid") is True, "ABI_INVALID"),
        (
            isinstance(routed.get("quantized"), int)
            and int(routed["quantized"]) > 0,
            "ROUTED_EXPERTS_MISSING",
        ),
    )
    reason = _first_reason(checks)
    return {
        "schema_version": 1,
        "valid": reason is None,
        "backend": "humming",
        "reason_codes": [] if reason is None else [reason],
        "details": {
            "vllm_version": runtime.vllm_version,
            "humming_version": runtime.humming_version,
            "device_capability": runtime.device_capability,
            "humming_source_integrity": runtime.humming_source_integrity,
            "humming_source_mismatches": runtime.humming_source_mismatches,
            "nvfp4_overlay_detected": runtime.nvfp4_overlay_detected,
            "humming_unhashed_bytecode": runtime.humming_unhashed_bytecode,
            "humming_declared_patches": runtime.humming_declared_patches,
            "humming_cache_dir": runtime.humming_cache_dir,
            "weight_group_size": weights.get("group_size"),
            "activation_strategy": activations.get("strategy"),
            "gemm_type": normalize_gemm_type(runtime.moe_gemm_type),
            "abi_valid": abi_report.get("valid"),
        },
    }


def classify_backend_log(
    text: str,
    preflight: Mapping[str, object],
) -> dict[str, object]:
    """Require positive Humming runtime evidence and reject fallback markers.

    The expected scheduling strategy comes from the preflight's ``gemm_type``
    (itself derived from ``VLLM_HUMMING_MOE_GEMM_TYPE``). Attestation is
    *positive and specific*: the requested strategy's marker must be present and
    the other one's absent. A grouped run that silently fell back to indexed --
    vLLM's behaviour for an unrecognised value -- therefore fails rather than
    passing as "some Humming kernel ran".
    """

    indexed = _INDEXED_MARKER in text
    grouped = _GROUPED_MARKER in text
    expected = preflight.get("details", {}).get("gemm_type")
    if not isinstance(expected, str):
        expected = GEMM_TYPE_INDEXED
    observed = {GEMM_TYPE_INDEXED: indexed, GEMM_TYPE_GROUPED: grouped}
    quantization = any(pattern.search(text) for pattern in _QUANTIZATION_PATTERNS)
    compile_cache_lines = [
        line
        for line in text.splitlines()
        if "humming" in line.lower() and _COMPILE_CACHE_PATTERN.search(line)
    ]
    checks = (
        (
            preflight.get("valid") is True
            and preflight.get("backend") == "humming",
            "PREFLIGHT_INVALID",
        ),
        (_CUTLASS_MARKER not in text, "CUTLASS_FALLBACK_DETECTED"),
        (_MARLIN_PATTERN.search(text) is None, "MARLIN_FALLBACK_DETECTED"),
        (_UNQUANTIZED_MARKER not in text, "UNQUANTIZED_FALLBACK_DETECTED"),
        (quantization, "QUANTIZATION_MARKER_MISSING"),
        (not (indexed and grouped), "CONTRADICTORY_GEMM_MARKERS"),
        (expected in observed, "MOE_GEMM_TYPE_UNSUPPORTED"),
        # Checked before the missing-marker case so that a run which served the
        # *wrong* strategy is named as such, rather than as a generic absence.
        (
            not any(seen for name, seen in observed.items() if name != expected),
            "UNEXPECTED_GEMM_TYPE_SELECTED",
        ),
        (observed.get(expected, False), "HUMMING_GEMM_MARKER_MISSING"),
    )
    reason = _first_reason(checks)
    return {
        "schema_version": 1,
        "valid": reason is None,
        "backend": "humming",
        "gemm_type": expected if reason is None else None,
        "reason_codes": [] if reason is None else [reason],
        "details": {
            "quantization_marker": quantization,
            "expected_gemm_type": expected,
            "indexed_marker": indexed,
            "grouped_marker": grouped,
            "compile_cache_lines": compile_cache_lines,
        },
    }


def _is_derived_bytecode(relative_path: str) -> bool:
    """Cached bytecode is regenerated from source, so RECORD may not hash it."""

    return "__pycache__/" in relative_path and relative_path.endswith(".pyc")


def _distribution_integrity() -> (
    tuple[str, tuple[str, ...], bool, int, tuple[str, ...]]
):
    distribution = importlib.metadata.distribution("humming-kernels")
    files = list(distribution.files or [])
    package_files = [
        file for file in files if file.parts and file.parts[0] == "humming"
    ]
    hashed_files = [file for file in package_files if file.hash is not None]
    if not hashed_files:
        return "unverifiable", ("NO_HASHED_HUMMING_FILES",), False, 0, ()

    mismatches: list[str] = []
    declared_patches: list[str] = []
    overlay_detected = False
    derived_unhashed = 0
    for file in package_files:
        relative_path = str(getattr(file, "path", file)).replace("\\", "/")
        # Some wheels list unhashed __pycache__ entries. They carry no source
        # authority, so they are counted and reported but never verified.
        unverifiable_bytecode = file.hash is None and _is_derived_bytecode(
            relative_path
        )
        path = Path(distribution.locate_file(file))
        try:
            payload = path.read_bytes()
        except OSError:
            if unverifiable_bytecode:
                derived_unhashed += 1
                continue
            mismatches.append(relative_path)
            continue
        if NVFP4_OVERLAY_MARKER in payload:
            overlay_detected = True
        if file.hash is None:
            if unverifiable_bytecode:
                derived_unhashed += 1
                continue
            mismatches.append(relative_path)
            continue
        digest = hashlib.new(file.hash.mode, payload).digest()
        encoded = base64.urlsafe_b64encode(digest).rstrip(b"=").decode("ascii")
        if encoded != file.hash.value:
            declared = DECLARED_PATCH_SHA256.get(relative_path)
            if declared and hashlib.sha256(payload).hexdigest() == declared:
                declared_patches.append(relative_path)
                continue
            mismatches.append(relative_path)
    if mismatches:
        status = "mismatch"
    elif declared_patches:
        status = RECORD_MATCHED_PATCHED
    else:
        status = RECORD_MATCHED
    return (
        status,
        tuple(sorted(mismatches)),
        overlay_detected,
        derived_unhashed,
        tuple(sorted(declared_patches)),
    )


def _patch_statuses() -> tuple[str, str]:
    from pipeline.slurm.patch_vllm_m3_serve import (
        ensure_vllm_m3_humming_patch,
        ensure_vllm_m3_patches,
    )

    try:
        ensure_vllm_m3_patches(apply=False)
    except Exception:
        normal = "missing"
    else:
        normal = "patched"
    try:
        ensure_vllm_m3_humming_patch(apply=False)
    except Exception:
        humming = "missing"
    else:
        humming = "patched"
    return normal, humming


def _discover_runtime() -> RuntimeFacts:
    import torch
    import vllm

    (
        integrity,
        mismatches,
        overlay,
        unhashed_bytecode,
        declared_patches,
    ) = _distribution_integrity()
    normal_patch, humming_patch = _patch_statuses()
    capability = torch.cuda.get_device_capability()
    return RuntimeFacts(
        vllm_version=str(getattr(vllm, "__version__", "")),
        humming_version=importlib.metadata.version("humming-kernels"),
        device_capability=(int(capability[0]), int(capability[1])),
        humming_source_integrity=integrity,
        humming_source_mismatches=mismatches,
        nvfp4_overlay_detected=overlay,
        humming_unhashed_bytecode=unhashed_bytecode,
        humming_declared_patches=declared_patches,
        humming_cache_dir=os.environ.get("HUMMING_CACHE_DIR", ""),
        f16_accum=os.environ.get("VLLM_HUMMING_USE_F16_ACCUM", "0"),
        moe_gemm_type=os.environ.get(
            "VLLM_HUMMING_MOE_GEMM_TYPE",
            "indexed",
        ),
        normal_patch_status=normal_patch,
        humming_patch_status=humming_patch,
    )


def preflight_checkpoint(checkpoint: Path) -> dict[str, object]:
    """Discover and evaluate one checkpoint without modifying it."""

    try:
        config = json.loads(
            (checkpoint / "config.json").read_text(encoding="utf-8")
        )
        abi_report = analyze_checkpoint(checkpoint)
        runtime = _discover_runtime()
    except Exception as exc:
        return {
            "schema_version": 1,
            "valid": False,
            "backend": "humming",
            "reason_codes": ["DISCOVERY_ERROR"],
            "details": {"error": str(exc)},
        }
    return evaluate_preflight(config, abi_report, runtime)


def _write_report(path: Path, report: Mapping[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    preflight = subparsers.add_parser("preflight")
    preflight.add_argument("--checkpoint", type=Path, required=True)
    preflight.add_argument("--out", type=Path, required=True)

    attest = subparsers.add_parser("attest")
    attest.add_argument("--preflight", type=Path, required=True)
    attest.add_argument("--log", type=Path, required=True)
    attest.add_argument("--out", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.command == "preflight":
        report = preflight_checkpoint(args.checkpoint)
    else:
        try:
            preflight = json.loads(args.preflight.read_text(encoding="utf-8"))
            text = args.log.read_text(encoding="utf-8", errors="replace")
            report = classify_backend_log(text, preflight)
        except Exception as exc:
            report = {
                "schema_version": 1,
                "valid": False,
                "backend": "humming",
                "gemm_type": None,
                "reason_codes": ["DISCOVERY_ERROR"],
                "details": {"error": str(exc)},
            }
    _write_report(args.out, report)
    if report["valid"]:
        return 0
    return 2 if "DISCOVERY_ERROR" in report["reason_codes"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
