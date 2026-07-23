"""Work around MiniMax-M3 checkpoint / transformers config mismatches.

The public ``config.json`` ships ``vision_config.model_type = clip_vision_model``
and nests ``temporal_patch_size`` / ``spatial_merge_size`` under
``img_token_compression_config``. Transformers 5.12+ expects a
``MiniMaxM3VLVisionConfig`` with those fields at the top level; without coercion
``from_pretrained`` builds a generic ``PreTrainedConfig`` and model init fails with::

    AttributeError: 'PreTrainedConfig' object has no attribute 'temporal_patch_size'

``modeling_minimax_m3_vl.get_placeholder_mask`` reads ``config.image_token_id`` /
``config.video_token_id``. The public checkpoint only defines ``*_token_index``;
remote ``AutoConfig`` classes may omit transformers' ``attribute_map`` aliases,
which breaks FX tracing during sequential calibration.

Text-only calibration (no ``pixel_values``) still enters ``get_placeholder_mask`` during FX
trace; absent features appear as non-``None`` proxies, so ``image_features.numel()`` raises.
"""

from __future__ import annotations

import os
from pathlib import Path
from types import MethodType
from typing import Any

import torch


_VISION_KEYS = frozenset(
    {
        "hidden_size",
        "intermediate_size",
        "num_hidden_layers",
        "num_attention_heads",
        "num_channels",
        "image_size",
        "patch_size",
        "temporal_patch_size",
        "spatial_merge_size",
        "hidden_act",
        "layer_norm_eps",
        "attention_dropout",
        "rope_parameters",
        "initializer_range",
    }
)


def _as_dict(obj: Any) -> dict[str, Any]:
    if isinstance(obj, dict):
        return dict(obj)
    if hasattr(obj, "to_dict"):
        return dict(obj.to_dict())
    return {}


def _ensure_token_id_aliases(config: Any) -> None:
    """Expose image_token_id / video_token_id for modeling and FX trace paths."""
    image_idx = getattr(config, "image_token_index", 200025)
    video_idx = getattr(config, "video_token_index", 200026)
    if not hasattr(config, "image_token_index"):
        config.image_token_index = image_idx
    if not hasattr(config, "video_token_index"):
        config.video_token_index = video_idx
    for attr, value in (("image_token_id", image_idx), ("video_token_id", video_idx)):
        if getattr(config, attr, None) == value:
            continue
        try:
            setattr(config, attr, value)
        except (AttributeError, TypeError):
            object.__setattr__(config, attr, value)


def coerce_minimax_m3_vl_config(config: Any) -> Any:
    """Return *config* with M3 sub-configs coerced in place."""
    if getattr(config, "model_type", None) != "minimax_m3_vl":
        return config

    from transformers.models.minimax_m3_vl.configuration_minimax_m3_vl import (
        MiniMaxM3VLConfig,
        MiniMaxM3VLTextConfig,
        MiniMaxM3VLVisionConfig,
    )

    vd = _as_dict(config.vision_config)
    compression = vd.pop("img_token_compression_config", None) or {}
    if isinstance(compression, dict):
        vd.setdefault("spatial_merge_size", compression.get("spatial_merge_size", 2))
        vd.setdefault("temporal_patch_size", compression.get("temporal_patch_size", 2))
    vd.setdefault("spatial_merge_size", 2)
    vd.setdefault("temporal_patch_size", 2)
    if "initializer_factor" in vd and "initializer_range" not in vd:
        vd["initializer_range"] = vd.pop("initializer_factor")
    if "rope_theta" in vd and "rope_parameters" not in vd:
        vd["rope_parameters"] = {"rope_theta": vd.pop("rope_theta")}

    vision_kwargs = {k: vd[k] for k in _VISION_KEYS if k in vd}
    config.vision_config = MiniMaxM3VLVisionConfig(**vision_kwargs)

    tc = config.text_config
    if not isinstance(tc, MiniMaxM3VLTextConfig):
        td = _as_dict(tc)
        td.pop("model_type", None)
        td.pop("architectures", None)
        config.text_config = MiniMaxM3VLTextConfig(**td)

    config.merged_hidden_size = config.text_config.hidden_size * (
        config.vision_config.spatial_merge_size**2
    )

    # M3 only sets torch_dtype at the top level; sub-configs default to float32.
    # linearize_moe reads text_config.dtype when building 2D expert Linears.
    target_dtype = getattr(config, "dtype", None) or torch.bfloat16
    config.text_config.dtype = target_dtype
    config.vision_config.dtype = target_dtype

    _ensure_token_id_aliases(config)

    # Remote AutoConfig may return a generic PreTrainedConfig; rebuild as the
    # official transformers class so sub-config types are consistent.
    if not isinstance(config, MiniMaxM3VLConfig):
        top = _as_dict(config)
        top["vision_config"] = config.vision_config.to_dict()
        top["text_config"] = config.text_config.to_dict()
        config = MiniMaxM3VLConfig.from_dict(top)
        config.text_config.dtype = target_dtype
        config.vision_config.dtype = target_dtype
        _ensure_token_id_aliases(config)

    return config


# vLLM's MiniMaxM3MLP only accepts hidden_act == "swigluoai". Transformers 5.12+
# normalizes the checkpoint's "swigluoai" to "silu" in MiniMaxM3VLTextConfig.__post_init__
# (ACT2FN fallback); quantize save persists that coercion.
_VLLM_TEXT_ACT_KEYS = ("swiglu_alpha", "swiglu_limit", "swiglu_beta")


def ensure_minimax_m3_vllm_serve_config(ckpt: Path, source: str) -> list[str]:
    """Patch a quantized checkpoint's ``config.json`` for vLLM M3 load.

  1. **Vision:** ``coerce_minimax_m3_vl_config()`` hoists
     ``img_token_compression_config`` onto ``MiniMaxM3VLVisionConfig`` and drops the
     nested dict. vLLM's ``MiniMaxVLVisionModel`` still reads
     ``config.img_token_compression_config``.

  2. **Text MLP activation:** transformers rewrites ``text_config.hidden_act`` from
     ``"swigluoai"`` to ``"silu"`` on load/save. vLLM's ``MiniMaxM3MLP`` requires
     ``hidden_act == "swigluoai"`` and wires ``SiluAndMulWithClamp`` from the
     ``swiglu_*`` scalars.

    Vision weights are bf16/unchanged; restoring source ``vision_config`` and
    ``text_config.hidden_act`` is numerically safe. ``quantization_config`` is untouched.
    """
    import json

    cfg_path = ckpt / "config.json"
    if not cfg_path.exists():
        return []

    data = json.loads(cfg_path.read_text(encoding="utf-8"))
    if data.get("model_type") != "minimax_m3_vl":
        return []

    src_path = Path(source)
    if (src_path / "config.json").exists():
        src = json.loads((src_path / "config.json").read_text(encoding="utf-8"))
    else:
        from transformers import AutoConfig

        src = AutoConfig.from_pretrained(source, trust_remote_code=True).to_dict()

    changed: list[str] = []

    src_vc = src.get("vision_config")
    if isinstance(src_vc, dict) and "img_token_compression_config" in src_vc:
        cur_vc = data.get("vision_config") or {}
        if cur_vc.get("img_token_compression_config") != src_vc.get(
            "img_token_compression_config"
        ):
            data["vision_config"] = src_vc
            changed.append(
                "vision_config (restored img_token_compression_config from source)"
            )

    src_tc = src.get("text_config") if isinstance(src.get("text_config"), dict) else {}
    tc = data.setdefault("text_config", {})
    if not isinstance(tc, dict):
        tc = {}
        data["text_config"] = tc

    # M3's language MLP is SwiGLU-OAI by architecture. transformers' config
    # __post_init__ (native 5.x MiniMaxM3VLTextConfig or the remote-code module)
    # rewrites hidden_act "swigluoai" -> "silu" on every load/save, and quantize
    # persists that coercion. Reading the source via AutoConfig re-coerces too, so
    # we must NOT gate on the source string. Force "swigluoai" unconditionally for
    # minimax_m3_vl (model_type was checked above); vLLM's MiniMaxM3MLP requires it.
    if tc.get("hidden_act") != "swigluoai":
        tc["hidden_act"] = "swigluoai"
        changed.append(
            'text_config.hidden_act ("silu" -> "swigluoai" for vLLM SwiGLU-OAI)'
        )

    # SiluAndMulWithClamp needs the swiglu_* scalars. Prefer source values (raw
    # json is not coerced); fall back to whatever the checkpoint already carries.
    for key in _VLLM_TEXT_ACT_KEYS:
        val = src_tc.get(key, tc.get(key))
        if val is not None and tc.get(key) != val:
            tc[key] = val
            changed.append(f"text_config.{key}")

    if not changed:
        return []

    cfg_path.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")
    return changed


def load_minimax_m3_vl_config(model_id: str, *, trust_remote_code: bool = True) -> Any:
    """Load and coerce a MiniMax-M3 VL config from a hub id or local path."""
    from transformers import AutoConfig

    config = AutoConfig.from_pretrained(model_id, trust_remote_code=trust_remote_code)
    return coerce_minimax_m3_vl_config(config)


def apply_minimax_m3_config(
    model_id: str,
    from_pretrained_kwargs: dict[str, Any],
    *,
    trust_remote_code: bool = True,
) -> dict[str, Any]:
    """If *model_id* is MiniMax-M3, inject a coerced ``config`` into kwargs."""
    try:
        config = load_minimax_m3_vl_config(
            model_id, trust_remote_code=trust_remote_code
        )
    except Exception:
        return from_pretrained_kwargs

    if getattr(config, "model_type", None) != "minimax_m3_vl":
        return from_pretrained_kwargs

    out = dict(from_pretrained_kwargs)
    out["config"] = config
    return out


def _find_minimax_m3_vl_backbone(model: Any) -> Any | None:
    backbone = getattr(model, "model", None)
    if backbone is not None and hasattr(backbone, "get_placeholder_mask"):
        return backbone
    return None


def patch_minimax_m3_for_text_calibration(model: Any) -> bool:
    """Patch placeholder-mask validation for text-only AWQ/GPTQ FX tracing.

    Sequential tracing calls the full VL forward without images. FX represents absent
    ``image_features`` / ``video_features`` as non-None proxies, so the stock
    ``get_placeholder_mask`` path can call ``.numel()`` on None.
    """
    if getattr(getattr(model, "config", None), "model_type", None) != "minimax_m3_vl":
        return False

    vl_backbone = _find_minimax_m3_vl_backbone(model)
    if vl_backbone is None:
        return False
    if getattr(vl_backbone, "_llmc_text_calibration_mask_patch", False):
        return True

    original = vl_backbone.get_placeholder_mask

    def get_placeholder_mask(
        self,
        input_ids,
        inputs_embeds,
        image_features=None,
        video_features=None,
    ):
        if not isinstance(image_features, torch.Tensor):
            image_features = None
        if not isinstance(video_features, torch.Tensor):
            video_features = None
        return original(input_ids, inputs_embeds, image_features=image_features, video_features=video_features)

    vl_backbone.get_placeholder_mask = MethodType(get_placeholder_mask, vl_backbone)
    vl_backbone._llmc_text_calibration_mask_patch = True
    return True


# Sparse decoder layers (minimax_m3_sparse attention + MoE MLP); layers 0-2 are dense.
_M3_SPARSE_LAYER = r"(?:[3-9]|[1-5][0-9])"
# Anchor to the language backbone: the vision tower ALSO has `layers.N.self_attn.*`,
# so an unanchored `.*layers[.]` matches vision q/k/v (breaks per-layer grouping since
# vision has no matching `input_layernorm`, collapsing all smooth layers into one set).
_M3_LM = r".*language_model[.]layers[.]"


def get_minimax_m3_awq_mappings(
    disable_mlp_input_smoothing: bool | None = None,
    layer: int | None = None,
) -> list:
    """Return AWQ mappings for all sparse layers or one selected sparse layer.

    Derived from the keep-bf16 strategy of ``cyankiwi/MiniMax-M3-AWQ-INT4``, translated
    to transformers 5.12.1 module names (``self_attn.indexer.*``, ``mlp.experts.N.*``).
    Indexer, router, and shared experts stay bf16 via ``quantization.ignore`` but are
    included as balance layers so smoothed activations stay consistent. Routed-expert
    patterns match only after ``linearize_moe`` splits the fused experts on load.

    NO up_proj -> down_proj mapping (r6, 2026-07-23). A smoothing fold may only pass
    through an activation factor in which it is homogeneous. M3's expert activation is
    ``(clamp(up, ±limit) + 1.0) * glu`` (gpt-oss style, swiglu_beta=1.0): the down
    input is AFFINE and CLAMPED in up's output, so folding ``up_rows /= s`` /
    ``down_cols *= s`` rescales the effective beta to ``s*beta`` and the up-clamp to
    ``±limit*s`` per channel — a function change, not a reparameterization. The shipped
    r5 fold scales (median s≈1.66 at L30) perturbed expert outputs by ~5-33% RMS,
    invisible to the grid-search loss (computed pre-fold) and to the weight-algebra
    gates. See BUGS_AND_FIXES.md "AWQ up->down smoothing fold is not
    function-preserving on MiniMax-M3". This mapping was the M3-specific AWQ damage
    channel; do not re-add it without the gate-side homogeneous fold (see
    docs/superpowers/plans/2026-07-23-m3-awq-gate-alpha-fold.md, "r7").
    The post-attention-norm mapping below is unaffected: it folds across a purely
    linear boundary (norm -> router/shared/expert input projections).
    """
    from llmcompressor.modifiers.transform.awq import AWQMapping

    if layer is not None and layer not in range(3, 60):
        raise ValueError(f"expected sparse MiniMax-M3 layer in [3, 59], got {layer}")
    s = _M3_SPARSE_LAYER if layer is None else str(layer)
    lm = _M3_LM
    if disable_mlp_input_smoothing is None:
        disable_mlp_input_smoothing = os.environ.get(
            "M3_AWQ_DISABLE_MLP_INPUT_SMOOTH", "0"
        ).lower() in {"1", "true", "yes"}

    mappings = [
        AWQMapping(
            rf"re:{lm}{s}[.]input_layernorm$",
            [
                rf"re:{lm}{s}[.]self_attn[.]q_proj$",
                rf"re:{lm}{s}[.]self_attn[.]k_proj$",
                rf"re:{lm}{s}[.]self_attn[.]v_proj$",
                rf"re:{lm}{s}[.]self_attn[.]indexer[.]q_proj$",
                rf"re:{lm}{s}[.]self_attn[.]indexer[.]k_proj$",
            ],
        ),
        AWQMapping(
            rf"re:{lm}{s}[.]self_attn[.]v_proj$",
            [rf"re:{lm}{s}[.]self_attn[.]o_proj$"],
        ),
        # NOTE: deliberately NO ``experts.N.up_proj -> experts.N.down_proj`` mapping —
        # not function-preserving through M3's ``(clamp(up)+1)*glu`` activation (see
        # docstring). Enforced by tests/pipeline/test_minimax_m3_awq_mappings.py.
    ]
    if not disable_mlp_input_smoothing:
        mappings.insert(
            2,
            AWQMapping(
                rf"re:{lm}{s}[.]post_attention_layernorm$",
                [
                    rf"re:{lm}{s}[.]mlp[.]gate$",
                    rf"re:{lm}{s}[.]mlp[.]shared_experts[.]gate_up_proj$",
                    rf"re:{lm}{s}[.]mlp[.]experts[.][0-9]+[.]gate_proj$",
                    rf"re:{lm}{s}[.]mlp[.]experts[.][0-9]+[.]up_proj$",
                ],
            ),
        )
    return mappings


def register_minimax_m3_awq_mappings() -> None:
    """Register M3 AWQ mappings so ``AWQModifier`` auto-discovers them by arch name."""
    from llmcompressor.modifiers.transform.awq import AWQ_MAPPING_REGISTRY

    AWQ_MAPPING_REGISTRY["MiniMaxM3SparseForConditionalGeneration"] = (
        get_minimax_m3_awq_mappings()
    )
