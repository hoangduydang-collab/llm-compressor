"""Fail-closed static validation for GLM-5.3 expert-only GPTQ artifacts."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Iterable, Mapping


PROJECTIONS = ("gate_proj", "up_proj", "down_proj")
REQUIRED_SUFFIXES = ("weight_packed", "weight_scale", "weight_shape")
GIB = 1024**3


def _expert_prefix(layer: int, expert: int, projection: str) -> str:
    return (
        f"model.layers.{layer}.mlp.experts.{expert}.{projection}"
    )


def validate_checkpoint_keys(
    keys: Iterable[str], *, layers: Iterable[int], experts: int = 256
) -> list[str]:
    """Require every routed projection and reject packed non-expert weights."""
    actual = set(keys)
    required = {
        f"{_expert_prefix(layer, expert, projection)}.{suffix}"
        for layer in layers
        for expert in range(experts)
        for projection in PROJECTIONS
        for suffix in REQUIRED_SUFFIXES
    }
    errors = [
        f"missing required checkpoint key: {key}"
        for key in sorted(required - actual)
    ]
    allowed_packed = {key for key in required if key.endswith(".weight_packed")}
    errors.extend(
        f"forbidden non-expert packed weight: {key}"
        for key in sorted(
            key
            for key in actual
            if key.endswith(".weight_packed") and key not in allowed_packed
        )
    )
    return errors


def validate_scale_values(values: Iterable[float]) -> list[str]:
    invalid = sum(
        1
        for value in values
        if not math.isfinite(float(value)) or float(value) <= 0
    )
    if invalid:
        return [
            f"weight scales contain {invalid} non-finite-or-non-positive value(s)"
        ]
    return []


def validate_memory_scaling(
    *,
    ep4_peak_bytes: int,
    ep8_peak_bytes: int,
    ceiling_bytes: int = 76 * GIB,
    ratio: float = 0.80,
) -> list[str]:
    errors = []
    for name, value in (("EP4", ep4_peak_bytes), ("EP8", ep8_peak_bytes)):
        if value >= ceiling_bytes:
            errors.append(
                f"{name} peak {value / GIB:.2f} GiB is not below the "
                f"{ceiling_bytes / GIB:.2f} GiB ceiling"
            )
    allowed = ep4_peak_bytes * ratio
    if ep8_peak_bytes > allowed:
        errors.append(
            f"EP8 peak must be <= {ratio:.0%} of EP4 peak "
            f"({ep8_peak_bytes / GIB:.2f} GiB > {allowed / GIB:.2f} GiB)"
        )
    return errors


def validate_phase_summary(
    summary: Mapping[str, object], *, world_size: int
) -> list[str]:
    errors = []
    if not summary.get("available"):
        errors.append("phase evidence is unavailable")
        return errors
    if not summary.get("complete"):
        errors.append("phase evidence is incomplete")
    if summary.get("expected_world_size") != world_size:
        errors.append(
            "phase evidence world size differs: "
            f"{summary.get('expected_world_size')} != {world_size}"
        )
    missing = summary.get("missing_ranks") or []
    if missing:
        errors.append(f"phase evidence is missing ranks: {missing}")
    unexpected = summary.get("unexpected_ranks") or []
    if unexpected:
        errors.append(f"phase evidence has unexpected ranks: {unexpected}")
    return errors


def _metric_paths(run_dir: Path) -> list[Path]:
    return sorted(run_dir.glob("quant_metrics*.jsonl"))


def peak_cuda_bytes(paths: Iterable[Path]) -> int | None:
    peaks: list[int] = []
    for path in paths:
        with Path(path).open("r", encoding="utf-8") as handle:
            for line in handle:
                try:
                    record = json.loads(line).get("record", {})
                    gpu = (
                        record.get("extra", {})
                        .get("snapshot", {})
                        .get("gpu", {})
                    )
                    value = gpu.get("peak_allocated_bytes")
                    if gpu.get("available") and isinstance(value, int):
                        peaks.append(value)
                except (json.JSONDecodeError, AttributeError):
                    continue
    return max(peaks) if peaks else None


def _checkpoint_index(checkpoint: Path) -> dict:
    path = checkpoint / "model.safetensors.index.json"
    return json.loads(path.read_text(encoding="utf-8"))


def _scale_values(checkpoint: Path, keys: Iterable[str]) -> Iterable[float]:
    from safetensors import safe_open

    wanted = {key for key in keys if key.endswith(".weight_scale")}
    index = _checkpoint_index(checkpoint)["weight_map"]
    by_shard: dict[str, list[str]] = {}
    for key in wanted:
        by_shard.setdefault(index[key], []).append(key)
    for shard, shard_keys in sorted(by_shard.items()):
        with safe_open(str(checkpoint / shard), framework="pt", device="cpu") as src:
            for key in sorted(shard_keys):
                tensor = src.get_tensor(key).reshape(-1)
                yield from tensor.tolist()


def validate_checkpoint(
    checkpoint: Path, *, layers: Iterable[int], experts: int = 256
) -> list[str]:
    index = _checkpoint_index(checkpoint)
    keys = set(index["weight_map"])
    errors = validate_checkpoint_keys(keys, layers=layers, experts=experts)
    if not errors:
        errors.extend(validate_scale_values(_scale_values(checkpoint, keys)))
    return errors


def _parse_layers(raw: str) -> tuple[int, ...]:
    layers = tuple(int(item) for item in raw.split(",") if item.strip())
    if not layers:
        raise argparse.ArgumentTypeError("at least one layer is required")
    return layers


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--layers", type=_parse_layers, required=True)
    parser.add_argument("--experts", type=int, default=256)
    parser.add_argument("--run-dir", type=Path)
    parser.add_argument("--world-size", type=int)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)

    errors = validate_checkpoint(
        args.checkpoint, layers=args.layers, experts=args.experts
    )
    report: dict[str, object] = {
        "checkpoint": str(args.checkpoint),
        "layers": list(args.layers),
        "experts": args.experts,
        "errors": errors,
    }
    if args.run_dir is not None:
        if args.world_size is None:
            parser.error("--run-dir requires --world-size")
        from pipeline.metrics import summarize_phases

        paths = _metric_paths(args.run_dir)
        phases = summarize_phases(paths)
        errors.extend(validate_phase_summary(phases, world_size=args.world_size))
        report["phase_summary"] = phases
        report["peak_cuda_bytes"] = peak_cuda_bytes(paths)

    report["ok"] = not errors
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    return 0 if report["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
