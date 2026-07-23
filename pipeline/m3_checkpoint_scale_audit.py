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


def resolve_suffix(
    weight_map: dict[str, str], suffix: str | tuple[str, ...]
) -> str:
    matches = [name for name in weight_map if name.endswith(suffix)]
    if len(matches) != 1:
        raise ValueError(f"expected one tensor ending {suffix!r}, found {matches}")
    return matches[0]


def _component_suffixes(layer: int, component: str) -> tuple[str, ...]:
    roots = (
        f"language_model.layers.{layer}",
        f"model.layers.{layer}",
        f"language_model.model.layers.{layer}",
    )
    tails = {
        "norm": ("post_attention_layernorm.weight",),
        "router": ("mlp.gate.weight", "block_sparse_moe.gate.weight"),
        "shared_gate_up": (
            "mlp.shared_experts.gate_up_proj.weight",
            "block_sparse_moe.shared_experts.gate_up_proj.weight",
            # HF-format checkpoints (raw M3, cyankiwi AWQ, our distributed
            # saves) keep gate/up separate; gate_proj sees the same smoothed
            # input, so it is an equivalent witness for the compensation.
            "mlp.shared_experts.gate_proj.weight",
            "block_sparse_moe.shared_experts.gate_proj.weight",
        ),
    }
    if component not in tails:
        raise ValueError(f"unknown component: {component}")
    return tuple(f"{root}.{tail}" for root in roots for tail in tails[component])


def load_suffix(
    checkpoint: Path, suffix: str | tuple[str, ...]
) -> tuple[str, torch.Tensor]:
    weight_map = _index(checkpoint)
    name = resolve_suffix(weight_map, suffix)
    with safe_open(checkpoint / weight_map[name], framework="pt", device="cpu") as handle:
        tensor = handle.get_tensor(name).float()
    return name, tensor


def load_weight_dequant(
    checkpoint: Path, suffix: str | tuple[str, ...]
) -> tuple[str, torch.Tensor, bool]:
    """``load_suffix`` + fp8 dequantization for float-quantized modules.

    r8 mixed checkpoints store shared-expert / attention weights as fp8 codes
    with a per-channel ``weight_scale`` sibling; comparing the raw codes
    against base bf16 is meaningless (rel_l2 ~ 1/scale ~ 1e3, the r8 v3 smoke
    gate failure). Returns (name, dequantized tensor, was_fp8).
    """
    weight_map = _index(checkpoint)
    name = resolve_suffix(weight_map, suffix)
    with safe_open(checkpoint / weight_map[name], framework="pt", device="cpu") as handle:
        tensor = handle.get_tensor(name).float()
    scale_name = name[: -len(".weight")] + ".weight_scale"
    if name.endswith(".weight") and scale_name in weight_map:
        with safe_open(
            checkpoint / weight_map[scale_name], framework="pt", device="cpu"
        ) as handle:
            tensor = tensor * handle.get_tensor(scale_name).float()
        return name, tensor, True
    return name, tensor, False


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
    base_gain = 1.0 + base_norm
    cand_gain = 1.0 + candidate_norm
    scale = base_gain / cand_gain
    # Dead channels (base weight exactly -1 -> gain 0: M3 layers 8/10-13)
    # make the implied scale 0/0. A consistent fold leaves them at gain 0
    # (scale 1); any other combination is inconsistent and must show up in
    # the error, so pin only the both-zero case.
    both_dead = (base_gain == 0) & (cand_gain == 0)
    scale = torch.where(both_dead, torch.ones_like(scale), scale)
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
        norm_suffixes = _component_suffixes(layer, "norm")
        router_suffixes = _component_suffixes(layer, "router")
        shared_suffixes = _component_suffixes(layer, "shared_gate_up")
        _, base_norm = load_suffix(base, norm_suffixes)
        _, cand_norm = load_suffix(candidate, norm_suffixes)
        _, base_router = load_suffix(base, router_suffixes)
        _, cand_router, router_fp8 = load_weight_dequant(candidate, router_suffixes)
        _, base_shared = load_suffix(base, shared_suffixes)
        _, cand_shared, shared_fp8 = load_weight_dequant(candidate, shared_suffixes)
        router_comp = compensation_error(
            base_norm, cand_norm, base_router, cand_router
        )
        router_comp["fp8_dequantized"] = router_fp8
        shared_comp = compensation_error(
            base_norm, cand_norm, base_shared, cand_shared
        )
        shared_comp["fp8_dequantized"] = shared_fp8
        records[str(layer)] = {
            "normalization": {
                "base": tensor_stats(base_norm),
                "candidate": tensor_stats(cand_norm),
                "base_effective": tensor_stats(1.0 + base_norm),
                "candidate_effective": tensor_stats(1.0 + cand_norm),
            },
            "router_compensation": router_comp,
            "shared_gate_up_compensation": shared_comp,
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
