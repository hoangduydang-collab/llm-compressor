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
"""

from __future__ import annotations

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
