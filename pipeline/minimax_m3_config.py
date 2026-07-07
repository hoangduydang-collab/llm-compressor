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


def get_minimax_m3_awq_mappings() -> list:
    """Return AWQ smooth/balance mappings for MiniMax-M3 sparse layers (3-59).

    Mirrors the keep-bf16 strategy from ``cyankiwi/MiniMax-M3-AWQ-INT4``, translated
    to transformers 5.12.1 module names (``self_attn.indexer.*``, ``mlp.experts.N.*``).
    Indexer, router, and shared experts stay bf16 via ``quantization.ignore`` but are
    included as balance layers so smoothed activations stay consistent. Routed-expert
    patterns match only after ``linearize_moe`` splits the fused experts on load.
    """
    from llmcompressor.modifiers.transform.awq import AWQMapping

    s = _M3_SPARSE_LAYER
    lm = _M3_LM
    return [
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
        AWQMapping(
            rf"re:{lm}{s}[.]post_attention_layernorm$",
            [
                rf"re:{lm}{s}[.]mlp[.]gate$",
                rf"re:{lm}{s}[.]mlp[.]shared_experts[.]gate_up_proj$",
                rf"re:{lm}{s}[.]mlp[.]experts[.][0-9]+[.]gate_proj$",
                rf"re:{lm}{s}[.]mlp[.]experts[.][0-9]+[.]up_proj$",
            ],
        ),
        AWQMapping(
            rf"re:{lm}{s}[.]mlp[.]experts[.][0-9]+[.]up_proj$",
            [rf"re:{lm}{s}[.]mlp[.]experts[.][0-9]+[.]down_proj$"],
        ),
    ]


def register_minimax_m3_awq_mappings() -> None:
    """Register M3 AWQ mappings so ``AWQModifier`` auto-discovers them by arch name."""
    from llmcompressor.modifiers.transform.awq import AWQ_MAPPING_REGISTRY

    AWQ_MAPPING_REGISTRY["MiniMaxM3SparseForConditionalGeneration"] = (
        get_minimax_m3_awq_mappings()
    )
