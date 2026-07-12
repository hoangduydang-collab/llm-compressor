"""Stream MiniMax-M3 checkpoint tensors and audit AWQ smoothing invariants."""

from __future__ import annotations

import argparse
import json
from functools import lru_cache
from pathlib import Path
from typing import Any

import torch
from safetensors import safe_open


@lru_cache(maxsize=None)
def _index(checkpoint: Path) -> dict[str, str]:
    path = checkpoint / "model.safetensors.index.json"
    data = json.loads(path.read_text(encoding="utf-8"))
    return dict(data["weight_map"])


def resolve_suffix(weight_map: dict[str, str], suffix: str) -> str:
    matches = [name for name in weight_map if name.endswith(suffix)]
    if len(matches) != 1:
        raise ValueError(f"expected one tensor ending {suffix!r}, found {matches}")
    return matches[0]


def load_suffix(checkpoint: Path, suffix: str) -> tuple[str, torch.Tensor]:
    weight_map = _index(checkpoint)
    name = resolve_suffix(weight_map, suffix)
    with safe_open(checkpoint / weight_map[name], framework="pt", device="cpu") as handle:
        tensor = handle.get_tensor(name).float()
    return name, tensor


def tensor_stats(tensor: torch.Tensor) -> dict[str, Any]:
    finite = torch.isfinite(tensor)
    values = tensor[finite]
    return {
        "shape": list(tensor.shape),
        "finite_fraction": float(finite.float().mean()),
        "norm": float(tensor.norm()),
        "abs_max": float(values.abs().max()) if values.numel() else None,
        "mean": float(values.mean()) if values.numel() else None,
    }


def compensation_error(
    base_norm: torch.Tensor,
    candidate_norm: torch.Tensor,
    base_weight: torch.Tensor,
    candidate_weight: torch.Tensor,
) -> dict[str, Any]:
    # MiniMaxM3VLRMSNorm is Gemma-style: smooth the effective 1 + weight.
    scale = (1.0 + base_norm) / (1.0 + candidate_norm)
    predicted = base_weight * scale.reshape(1, -1)
    denom = base_weight.norm().clamp_min(1e-12)
    return {
        "scale": tensor_stats(scale),
        "relative_l2_error": float((candidate_weight - predicted).norm() / denom),
        "candidate": tensor_stats(candidate_weight),
        "base": tensor_stats(base_weight),
    }


def audit_checkpoint(base: Path, candidate: Path, layers: list[int]) -> dict[str, Any]:
    records: dict[str, Any] = {}
    for layer in layers:
        prefix = f"language_model.layers.{layer}."
        _, base_norm = load_suffix(base, prefix + "post_attention_layernorm.weight")
        _, cand_norm = load_suffix(candidate, prefix + "post_attention_layernorm.weight")
        _, base_router = load_suffix(base, prefix + "mlp.gate.weight")
        _, cand_router = load_suffix(candidate, prefix + "mlp.gate.weight")
        _, base_shared = load_suffix(
            base, prefix + "mlp.shared_experts.gate_up_proj.weight"
        )
        _, cand_shared = load_suffix(
            candidate, prefix + "mlp.shared_experts.gate_up_proj.weight"
        )
        records[str(layer)] = {
            "normalization": {
                "base": tensor_stats(base_norm),
                "candidate": tensor_stats(cand_norm),
                "base_effective": tensor_stats(1.0 + base_norm),
                "candidate_effective": tensor_stats(1.0 + cand_norm),
            },
            "router_compensation": compensation_error(
                base_norm, cand_norm, base_router, cand_router
            ),
            "shared_gate_up_compensation": compensation_error(
                base_norm, cand_norm, base_shared, cand_shared
            ),
        }
    return {"checkpoint": str(candidate.resolve()), "layers": records}


def audit_checkpoints(
    base: Path,
    reference: Path,
    awq: Path,
    gptq: Path,
    layers: list[int],
) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "base": str(base.resolve()),
        "reference": audit_checkpoint(base, reference, layers),
        "awq": audit_checkpoint(base, awq, layers),
        "gptq": audit_checkpoint(base, gptq, layers),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", type=Path, required=True)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--awq", type=Path, required=True)
    parser.add_argument("--gptq", type=Path, required=True)
    parser.add_argument("--layers", default="3,4,5,6,7,8,9")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    layers = [int(item) for item in args.layers.split(",")]
    result = audit_checkpoints(args.base, args.reference, args.awq, args.gptq, layers)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
