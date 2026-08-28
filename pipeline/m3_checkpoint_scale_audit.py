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
    """Map tensor name -> shard file, for sharded AND single-shard checkpoints.

    THE SINGLE-SHARD FALLBACK IS NOT A CONVENIENCE. Without it this function
    raised FileNotFoundError on any checkpoint small enough for save_pretrained
    to write one ``model.safetensors`` with no index -- and
    ``assert_smooth_fold_consistency`` catches FileNotFoundError and prints
    "smooth-fold gate skipped (names not resolvable)". So the fold gate silently
    NO-OPPED on exactly the checkpoints we use for fast validation: every subset
    probe and every small smoke.

    That is why the GLM-5.2 router-fix validation run (routerfix,
    20260828-150142) never produced a router_compensation number to compare
    against the pre-fix 1.08e-1 to 2.42e-1. The run was launched to audit a
    numerics fix, and the audit was skipped without failing.
    """
    path = checkpoint / "model.safetensors.index.json"
    if path.exists():
        data = json.loads(path.read_text(encoding="utf-8"))
        return dict(data["weight_map"])
    single = checkpoint / "model.safetensors"
    if single.exists():
        with safe_open(single, framework="pt") as handle:
            return {name: single.name for name in handle.keys()}
    raise FileNotFoundError(
        f"no model.safetensors.index.json and no model.safetensors under {checkpoint}"
    )


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
        # ATTENTION-SIDE NORMS AND THEIR CONSUMERS. Added 2026-08-28 to close a
        # gap this audit had since it was written: it looked only at
        # post_attention_layernorm, so the ENTIRE attention block's fold was
        # unchecked. That is what would have hidden the DSA indexer trap --
        # GlmMoeDsaDecoderLayer.forward feeds input_layernorm's output verbatim to
        # the indexer, whose wk and weights_proj consume it, and wq_b consumes
        # q_a_layernorm(q_a_proj(x)); none of the three was an AWQ balance layer.
        # (That trap is latent today, because every attention-side balance layer
        # is FP8 and AWQ skips a mapping with no int-quantized balance layer. The
        # point of auditing is that it stays that way.)
        "attn_input_norm": ("input_layernorm.weight",),
        # q_a_proj on MLA (GLM-5.2/5.3), q_proj on dense QKV (MiniMax-M3).
        "attn_q": ("self_attn.q_a_proj.weight", "self_attn.q_proj.weight"),
        "attn_kv_a": ("self_attn.kv_a_proj_with_mqa.weight",),
        "indexer_wk": ("self_attn.indexer.wk.weight",),
        "indexer_weights_proj": ("self_attn.indexer.weights_proj.weight",),
        "qa_norm": ("self_attn.q_a_layernorm.weight",),
        "attn_q_b": ("self_attn.q_b_proj.weight",),
        "indexer_wq_b": ("self_attn.indexer.wq_b.weight",),
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
    *,
    norm_gain_offset: float,
) -> dict[str, Any]:
    """Relative L2 error of the norm-implied inverse fold.

    ``norm_gain_offset`` is the architecture's norm gain form and there is NO
    default, deliberately. MiniMaxM3VLRMSNorm is Gemma-style, applying
    ``output * (1 + weight)``, so its effective gain is ``1 + w`` and the offset
    is 1.0. GLM-5.2's GlmMoeDsaRMSNorm applies plain ``output * weight``, so its
    offset is 0.0. Getting this wrong does not produce an obviously broken
    number -- it produces a WRONG implied scale, so a perfectly consistent fold
    reports a large error and the post-save gate fails a run that was fine, at
    the very end, after all the calibration has been paid for. The authority for
    which form a class uses is KNOWN_OFFSET_NORM_CLASSES /
    KNOWN_ORDINARY_NORM_CLASSES in llmcompressor/preflight/quantization.py.
    """
    base_gain = norm_gain_offset + base_norm
    cand_gain = norm_gain_offset + candidate_norm
    scale = base_gain / cand_gain
    # Dead channels (offset form: base weight exactly -1 -> gain 0, M3
    # layers 8/10-13; plain form: base weight exactly 0)
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


# Which consumers each attention-side norm feeds. Read off
# transformers/models/glm_moe_dsa/modeling_glm_moe_dsa.py rather than assumed:
#   input_layernorm -> q_a_proj, kv_a_proj_with_mqa, indexer.wk,
#                      indexer.weights_proj
#   q_a_layernorm   -> q_b_proj, indexer.wq_b
# kv_a_layernorm -> kv_b_proj is deliberately absent: kv_a_layernorm normalizes
# kv_a_proj's OUTPUT, which no indexer module consumes, so it adds a component to
# audit without adding a class of failure to catch.
_ATTENTION_FOLD_GROUPS: tuple[tuple[str, tuple[str, ...]], ...] = (
    (
        "attn_input_norm",
        ("attn_q", "attn_kv_a", "indexer_wk", "indexer_weights_proj"),
    ),
    ("qa_norm", ("attn_q_b", "indexer_wq_b")),
)


def _resolves(checkpoint: Path, suffixes: tuple[str, ...]) -> bool:
    try:
        resolve_suffix(_index(checkpoint), suffixes)
    except (ValueError, KeyError):
        return False
    return True


def audit_attention_fold(
    base: Path,
    candidate: Path,
    layer: int,
    *,
    norm_gain_offset: float,
) -> dict[str, Any]:
    """Audit the fold on the attention-side norms of one layer.

    Absence is recorded EXPLICITLY rather than skipped, because silent absence is
    the failure mode this whole function exists to prevent. Four outcomes per
    consumer:

    ``checked``                 both checkpoints have it; a residual is reported.
    ``absent``                  neither has it. Legitimate and common -- only 21
                                of GLM's 78 layers carry a DSA indexer, and M3
                                has no q_a_layernorm at all.
    ``missing_from_candidate``  the base has it and the quantized checkpoint does
                                not. A dropped tensor; the caller must fail.
    ``missing_from_base``       the reverse, which means the base is not the base.
    """
    records: dict[str, Any] = {}
    for norm_component, consumers in _ATTENTION_FOLD_GROUPS:
        norm_suffixes = _component_suffixes(layer, norm_component)
        if not (_resolves(base, norm_suffixes) and _resolves(candidate, norm_suffixes)):
            records[norm_component] = {"status": "norm_absent"}
            continue
        _, base_norm = load_suffix(base, norm_suffixes)
        _, cand_norm = load_suffix(candidate, norm_suffixes)
        group: dict[str, Any] = {
            "status": "checked",
            "normalization": {
                "base": tensor_stats(base_norm),
                "candidate": tensor_stats(cand_norm),
            },
            "consumers": {},
        }
        for consumer in consumers:
            suffixes = _component_suffixes(layer, consumer)
            in_base = _resolves(base, suffixes)
            in_candidate = _resolves(candidate, suffixes)
            if not in_base and not in_candidate:
                group["consumers"][consumer] = {"status": "absent"}
                continue
            if in_base and not in_candidate:
                group["consumers"][consumer] = {"status": "missing_from_candidate"}
                continue
            if in_candidate and not in_base:
                group["consumers"][consumer] = {"status": "missing_from_base"}
                continue
            _, base_weight = load_suffix(base, suffixes)
            _, cand_weight, was_fp8 = load_weight_dequant(candidate, suffixes)
            record = compensation_error(
                base_norm,
                cand_norm,
                base_weight,
                cand_weight,
                norm_gain_offset=norm_gain_offset,
            )
            record["fp8_dequantized"] = was_fp8
            record["status"] = "checked"
            group["consumers"][consumer] = record
        records[norm_component] = group
    return records


def audit_checkpoint(
    base: Path,
    candidate: Path,
    layers: list[int],
    *,
    norm_gain_offset: float,
) -> dict[str, Any]:
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
            base_norm,
            cand_norm,
            base_router,
            cand_router,
            norm_gain_offset=norm_gain_offset,
        )
        router_comp["fp8_dequantized"] = router_fp8
        shared_comp = compensation_error(
            base_norm,
            cand_norm,
            base_shared,
            cand_shared,
            norm_gain_offset=norm_gain_offset,
        )
        shared_comp["fp8_dequantized"] = shared_fp8
        records[str(layer)] = {
            "normalization": {
                "base": tensor_stats(base_norm),
                "candidate": tensor_stats(cand_norm),
                "norm_gain_offset": norm_gain_offset,
                "base_effective": tensor_stats(norm_gain_offset + base_norm),
                "candidate_effective": tensor_stats(norm_gain_offset + cand_norm),
            },
            "router_compensation": router_comp,
            "shared_gate_up_compensation": shared_comp,
            "attention_fold": audit_attention_fold(
                base, candidate, layer, norm_gain_offset=norm_gain_offset
            ),
        }
    return {
        "checkpoint": str(candidate.resolve()),
        "norm_gain_offset": norm_gain_offset,
        "layers": records,
    }


def audit_checkpoints(
    base: Path,
    reference: Path,
    awq: Path,
    gptq: Path,
    layers: list[int],
    *,
    norm_gain_offset: float = 1.0,
) -> dict[str, Any]:
    """Audit three candidates against a base.

    ``norm_gain_offset`` defaults to 1.0 only because every caller of THIS
    function is a MiniMax-M3 path (Gemma-style norm). ``audit_checkpoint`` and
    ``compensation_error`` require it explicitly; new families go through those.
    """
    return {
        "schema_version": 2,
        "base": str(base.resolve()),
        "norm_gain_offset": norm_gain_offset,
        "reference": audit_checkpoint(
            base, reference, layers, norm_gain_offset=norm_gain_offset
        ),
        "awq": audit_checkpoint(base, awq, layers, norm_gain_offset=norm_gain_offset),
        "gptq": audit_checkpoint(base, gptq, layers, norm_gain_offset=norm_gain_offset),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", type=Path, required=True)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--awq", type=Path, required=True)
    parser.add_argument("--gptq", type=Path, required=True)
    parser.add_argument("--layers", default="3,4,5,6,7,8,9")
    parser.add_argument(
        "--norm-gain-offset",
        type=float,
        required=True,
        help="Norm gain form: 1.0 for offset norms applying output*(1+weight) "
        "(Gemma, MiniMaxM3VLRMSNorm), 0.0 for plain norms applying "
        "output*weight (GlmMoeDsaRMSNorm). Required -- a wrong value silently "
        "produces a wrong implied scale and fails a healthy fold. Authority: "
        "KNOWN_OFFSET_NORM_CLASSES / KNOWN_ORDINARY_NORM_CLASSES in "
        "llmcompressor/preflight/quantization.py.",
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    layers = [int(item) for item in args.layers.split(",")]
    result = audit_checkpoints(
        args.base,
        args.reference,
        args.awq,
        args.gptq,
        layers,
        norm_gain_offset=args.norm_gain_offset,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
