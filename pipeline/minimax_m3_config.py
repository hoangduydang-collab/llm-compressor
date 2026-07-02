"""Work around MiniMax-M3 checkpoint / transformers config mismatches.

The public ``config.json`` ships ``vision_config.model_type = clip_vision_model``
and nests ``temporal_patch_size`` / ``spatial_merge_size`` under
``img_token_compression_config``. Transformers 5.12+ expects a
``MiniMaxM3VLVisionConfig`` with those fields at the top level; without coercion
``from_pretrained`` builds a generic ``PreTrainedConfig`` and model init fails with::

    AttributeError: 'PreTrainedConfig' object has no attribute 'temporal_patch_size'
"""

from __future__ import annotations

from typing import Any


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


def coerce_minimax_m3_vl_config(config: Any) -> Any:
    """Return *config* with M3 sub-configs coerced in place."""
    if getattr(config, "model_type", None) != "minimax_m3_vl":
        return config

    from transformers.models.minimax_m3_vl.configuration_minimax_m3_vl import (
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
