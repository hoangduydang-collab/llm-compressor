"""Format-tolerant checkpoint diagnostics for MiniMax-M3 evaluation."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
from collections import defaultdict
from pathlib import Path
from typing import Any


_QUANT_WEIGHT_SUFFIXES = (".weight_packed", ".qweight")
_PLAIN_WEIGHT_SUFFIX = ".weight"
_SCALE_SUFFIXES = (".weight_scale", ".scales")
_EXPERT = re.compile(r"(?:^|\.)(?:experts?)\.(\d+)(?:\.|$)")


def _sha256(path: Path) -> str | None:
    if not path.is_file():
        return None
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _json(path: Path) -> dict:
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"expected JSON object in {path}")
    return data


def _weight_map(checkpoint: Path) -> dict[str, str]:
    index = checkpoint / "model.safetensors.index.json"
    if index.is_file():
        mapping = _json(index).get("weight_map")
        if not isinstance(mapping, dict):
            raise ValueError(f"missing weight_map in {index}")
        return {str(key): str(value) for key, value in mapping.items()}
    single = checkpoint / "model.safetensors"
    if not single.is_file():
        raise FileNotFoundError(f"no safetensors checkpoint under {checkpoint}")
    try:
        from safetensors import safe_open
    except ImportError as exc:
        raise RuntimeError("safetensors is required for an unindexed checkpoint") from exc
    with safe_open(str(single), framework="numpy") as handle:
        return {key: single.name for key in handle.keys()}


def _strip_suffix(key: str, suffixes: tuple[str, ...]) -> str | None:
    for suffix in suffixes:
        if key.endswith(suffix):
            return key[: -len(suffix)]
    return None


def classify_module(module_name: str) -> str:
    lower = module_name.lower()
    leaf = lower.rsplit(".", 1)[-1]
    if "vision" in lower or "visual" in lower or "multi_modal_projector" in lower:
        return "vision"
    if lower == "lm_head" or lower.endswith(".lm_head"):
        return "lm_head"
    if "indexer" in lower:
        return "msa_indexer"
    if "shared_expert" in lower:
        return "shared_experts"
    if _EXPERT.search(lower):
        return "routed_experts"
    if leaf in {"gate", "router", "router_gate"}:
        return "routers"
    if "norm" in leaf:
        return "norms"
    if "self_attn" in lower or "attention" in lower:
        return "attention"
    if any(name in lower for name in (".mlp", "feed_forward", "ffn")):
        return "dense_mlp"
    return "other"


def _quantization_summary(config: dict) -> dict[str, Any]:
    quant = config.get("quantization_config")
    if not isinstance(quant, dict):
        return {
            "method": "none",
            "format": None,
            "weight_bits": None,
            "activation_bits": None,
            "group_size": None,
            "ignore": [],
        }

    method = quant.get("quant_method") or quant.get("method") or "unknown"
    weight_bits = quant.get("bits") or quant.get("w_bit")
    activation_bits = quant.get("activation_bits")
    group_size = quant.get("group_size") or quant.get("q_group_size")
    groups = quant.get("config_groups") or {}
    if isinstance(groups, dict):
        for group in groups.values():
            if not isinstance(group, dict):
                continue
            weights = group.get("weights") or {}
            activations = group.get("input_activations") or {}
            weight_bits = weight_bits or weights.get("num_bits")
            group_size = group_size or weights.get("group_size")
            activation_bits = activation_bits or activations.get("num_bits")
            if weight_bits is not None:
                break
    return {
        "method": str(method),
        "format": quant.get("format"),
        "weight_bits": weight_bits,
        "activation_bits": activation_bits,
        "group_size": group_size,
        "ignore": list(quant.get("ignore") or []),
        "raw": quant,
    }



def _packed_code_saturation(values, *, num_bits: int) -> dict[str, Any]:
    if num_bits <= 0 or 32 % num_bits:
        raise ValueError("packed saturation supports bit widths that divide 32")
    mask = (1 << num_bits) - 1
    minimum = maximum = total = 0
    for raw_value in values:
        word = int(raw_value) & 0xFFFFFFFF
        for shift in range(0, 32, num_bits):
            code = (word >> shift) & mask
            minimum += code == 0
            maximum += code == mask
            total += 1
    return {
        "codes": total,
        "minimum_code_fraction": minimum / total if total else None,
        "maximum_code_fraction": maximum / total if total else None,
        "extreme_code_fraction": (minimum + maximum) / total if total else None,
    }


def _sample_packed_saturation(
    checkpoint: Path,
    weight_map: dict[str, str],
    packed_keys: list[str],
    *,
    num_bits: int | None,
    max_tensors: int = 4,
    max_words_per_tensor: int = 32_768,
) -> dict[str, Any]:
    if not packed_keys:
        return {"status": "unavailable", "reason": "no recognized packed weights"}
    if num_bits not in {4, 8}:
        return {
            "status": "unavailable",
            "reason": f"packed decoder does not support weight_bits={num_bits}",
        }
    try:
        from safetensors import safe_open
    except ImportError:
        return {"status": "unavailable", "reason": "safetensors is not installed"}
    words = []
    sampled = []
    try:
        for key in sorted(packed_keys)[:max_tensors]:
            with safe_open(
                str(checkpoint / weight_map[key]), framework="numpy"
            ) as handle:
                tensor = handle.get_tensor(key).reshape(-1)[:max_words_per_tensor]
                words.extend(int(value) for value in tensor)
            sampled.append(key)
    except Exception as exc:
        return {
            "status": "unavailable",
            "reason": f"packed tensor sampling failed: {type(exc).__name__}: {exc}",
        }
    return {
        "status": "available",
        "sampled_tensors": sampled,
        "weight_bits": num_bits,
        **_packed_code_saturation(words, num_bits=num_bits),
    }


def _sample_scale_statistics(
    checkpoint: Path,
    weight_map: dict[str, str],
    scale_keys: list[str],
    *,
    max_tensors: int = 8,
    max_values_per_tensor: int = 65_536,
) -> dict[str, Any]:
    if not scale_keys:
        return {"status": "unavailable", "reason": "no scale tensors detected"}
    try:
        from safetensors import safe_open
    except ImportError:
        return {"status": "unavailable", "reason": "safetensors is not installed"}

    values: list[float] = []
    sampled: list[str] = []
    try:
        for key in sorted(scale_keys)[:max_tensors]:
            shard = checkpoint / weight_map[key]
            with safe_open(str(shard), framework="numpy") as handle:
                tensor = handle.get_tensor(key).reshape(-1)[:max_values_per_tensor]
                values.extend(float(value) for value in tensor)
            sampled.append(key)
    except Exception as exc:
        return {
            "status": "unavailable",
            "reason": f"scale tensor sampling failed: {type(exc).__name__}: {exc}",
        }
    finite = sorted(value for value in values if math.isfinite(value))
    if not finite:
        return {
            "status": "available",
            "sampled_tensors": sampled,
            "values": len(values),
            "finite_values": 0,
            "nonfinite_values": len(values),
            "zero_values": 0,
        }

    def quantile(fraction: float) -> float:
        return finite[round((len(finite) - 1) * fraction)]

    return {
        "status": "available",
        "sampled_tensors": sampled,
        "values": len(values),
        "finite_values": len(finite),
        "nonfinite_values": len(values) - len(finite),
        "zero_values": sum(value == 0.0 for value in finite),
        "min": finite[0],
        "p50": quantile(0.5),
        "p95": quantile(0.95),
        "p99": quantile(0.99),
        "max": finite[-1],
    }


def _calibration_artifacts(checkpoint: Path) -> dict[str, Any]:
    names = ("quant_metrics.jsonl", "quantization_metrics.json", "metrics.json")
    found = []
    for parent in (checkpoint, checkpoint.parent):
        for name in names:
            path = parent / name
            if path.is_file():
                found.append(
                    {"path": str(path.resolve()), "sha256": _sha256(path)}
                )
    if not found:
        return {
            "status": "unavailable",
            "reason": "no GPTQ/AWQ calibration metric artifact found",
        }
    return {"status": "available", "artifacts": found}


def diagnose_checkpoint(
    path: str | Path,
    *,
    baseline_bytes: int | None,
) -> dict[str, Any]:
    checkpoint = Path(path).resolve()
    config_path = checkpoint / "config.json"
    if not config_path.is_file():
        raise FileNotFoundError(f"missing config.json under {checkpoint}")
    config = _json(config_path)
    weight_map = _weight_map(checkpoint)

    packed_keys = [
        key for key in weight_map if key.endswith(_QUANT_WEIGHT_SUFFIXES)
    ]
    quantized = {
        module
        for key in packed_keys
        if (module := _strip_suffix(key, _QUANT_WEIGHT_SUFFIXES)) is not None
    }
    plain = {
        key[: -len(_PLAIN_WEIGHT_SUFFIX)]
        for key in weight_map
        if key.endswith(_PLAIN_WEIGHT_SUFFIX)
        and not key.endswith(_QUANT_WEIGHT_SUFFIXES)
    } - quantized
    scale_keys = [key for key in weight_map if key.endswith(_SCALE_SUFFIXES)]

    coverage: dict[str, dict[str, int]] = defaultdict(
        lambda: {"quantized_modules": 0, "plain_modules": 0}
    )
    for module in quantized:
        coverage[classify_module(module)]["quantized_modules"] += 1
    for module in plain:
        coverage[classify_module(module)]["plain_modules"] += 1
    for component in (
        "attention",
        "dense_mlp",
        "routed_experts",
        "shared_experts",
        "routers",
        "norms",
        "msa_indexer",
        "vision",
        "lm_head",
        "other",
    ):
        coverage[component]

    checkpoint_bytes = sum(
        file.stat().st_size for file in checkpoint.rglob("*") if file.is_file()
    )
    total_modules = len(quantized) + len(plain)
    compression_ratio = (
        baseline_bytes / checkpoint_bytes
        if baseline_bytes is not None and checkpoint_bytes
        else None
    )
    quantization = _quantization_summary(config)
    result = {
        "schema_version": 1,
        "valid": True,
        "checkpoint": str(checkpoint),
        "checkpoint_bytes": checkpoint_bytes,
        "config_sha256": _sha256(config_path),
        "index_sha256": _sha256(checkpoint / "model.safetensors.index.json"),
        "tensor_keys": len(weight_map),
        "quantized_modules": len(quantized),
        "plain_modules": len(plain),
        "quantized_module_coverage": (
            len(quantized) / total_modules if total_modules else None
        ),
        "bf16_fallback_fraction": (
            len(plain) / total_modules if total_modules else None
        ),
        "coverage_by_component": dict(sorted(coverage.items())),
        "quantization": quantization,
        "compression": {
            "baseline_bytes": baseline_bytes,
            "checkpoint_bytes": checkpoint_bytes,
            "ratio_to_baseline": compression_ratio,
            "effective_stored_bits_per_original_parameter": {
                "status": "unavailable",
                "reason": "tensor shapes are not available from the index alone",
            },
        },
        "scale_statistics": _sample_scale_statistics(
            checkpoint, weight_map, scale_keys
        ),
        "packed_saturation": _sample_packed_saturation(
            checkpoint,
            weight_map,
            packed_keys,
            num_bits=quantization["weight_bits"],
        ),
        "calibration_metrics": _calibration_artifacts(checkpoint),
    }
    if quantization["method"] == "none":
        result["scale_statistics"] = {
            "status": "unavailable",
            "reason": "unquantized baseline has no quantization scales",
        }
    return result


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--checkpoint",
        action="append",
        required=True,
        metavar="LABEL=PATH",
    )
    parser.add_argument("--baseline-label", default="bf16")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)

    checkpoints: dict[str, Path] = {}
    for item in args.checkpoint:
        if "=" not in item:
            parser.error(f"--checkpoint must be LABEL=PATH, got {item!r}")
        label, raw_path = item.split("=", 1)
        if not label or label in checkpoints:
            parser.error(f"invalid or duplicate checkpoint label: {label!r}")
        checkpoints[label] = Path(raw_path)
    if args.baseline_label not in checkpoints:
        parser.error(f"baseline label {args.baseline_label!r} is not configured")

    baseline = diagnose_checkpoint(
        checkpoints[args.baseline_label], baseline_bytes=None
    )
    reports = {args.baseline_label: baseline}
    for label, checkpoint in checkpoints.items():
        if label != args.baseline_label:
            reports[label] = diagnose_checkpoint(
                checkpoint,
                baseline_bytes=baseline["checkpoint_bytes"],
            )
    args.out.mkdir(parents=True, exist_ok=True)
    for label, report in reports.items():
        (args.out / f"{label}.json").write_text(
            json.dumps(report, indent=2), encoding="utf-8"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
