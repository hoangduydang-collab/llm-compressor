"""Tests for MiniMax-M3 config coercion."""

import torch


def test_coerce_propagates_bfloat16_to_sub_configs():
    from transformers import AutoConfig

    from pipeline.minimax_m3_config import coerce_minimax_m3_vl_config

    config = AutoConfig.for_model("minimax_m3_vl")
    config.dtype = torch.bfloat16
    config.text_config = config.text_config.__class__(
        **{k: v for k, v in config.text_config.to_dict().items() if k != "dtype"}
    )
    assert config.text_config.dtype != torch.bfloat16

    coerced = coerce_minimax_m3_vl_config(config)
    assert coerced.text_config.dtype == torch.bfloat16
    assert coerced.vision_config.dtype == torch.bfloat16


def test_coerce_exposes_image_and_video_token_id_aliases():
    from transformers import AutoConfig

    from pipeline.minimax_m3_config import coerce_minimax_m3_vl_config

    config = AutoConfig.for_model("minimax_m3_vl")
    coerced = coerce_minimax_m3_vl_config(config)
    assert coerced.image_token_index == coerced.image_token_id
    assert coerced.video_token_index == coerced.video_token_id


def test_patch_text_calibration_coerces_non_tensor_features():
    from pipeline.minimax_m3_config import patch_minimax_m3_for_text_calibration

    class _VL:
        def __init__(self):
            self.calls = []

        def get_placeholder_mask(
            self, input_ids, inputs_embeds, image_features=None, video_features=None
        ):
            self.calls.append((image_features, video_features))
            return "ok"

    class _Model:
        config = type("Cfg", (), {"model_type": "minimax_m3_vl"})()

        def __init__(self):
            self.model = _VL()

    model = _Model()
    assert patch_minimax_m3_for_text_calibration(model) is True
    assert patch_minimax_m3_for_text_calibration(model) is True  # idempotent

    sentinel = object()
    result = model.model.get_placeholder_mask(None, None, image_features=sentinel, video_features=sentinel)
    assert result == "ok"
    assert model.model.calls[-1] == (None, None)


def test_register_minimax_m3_awq_mappings():
    from llmcompressor.modifiers.transform.awq import AWQ_MAPPING_REGISTRY

    from pipeline.minimax_m3_config import (
        get_minimax_m3_awq_mappings,
        register_minimax_m3_awq_mappings,
    )

    register_minimax_m3_awq_mappings()
    assert "MiniMaxM3SparseForConditionalGeneration" in AWQ_MAPPING_REGISTRY
    assert AWQ_MAPPING_REGISTRY["MiniMaxM3SparseForConditionalGeneration"] == (
        get_minimax_m3_awq_mappings()
    )
    assert len(get_minimax_m3_awq_mappings()) == 4


def test_minimax_m3_awq_mapping_regexes():
    from compressed_tensors.utils.match import match_name

    from pipeline.minimax_m3_config import get_minimax_m3_awq_mappings

    mappings = get_minimax_m3_awq_mappings()
    qkv_smooth, v_to_o, mlp_smooth, up_to_down = mappings

    sparse_attn_q = "model.language_model.layers.3.self_attn.q_proj"
    sparse_indexer_q = "model.language_model.layers.3.self_attn.indexer.q_proj"
    dense_mlp = "model.language_model.layers.0.mlp.gate_up_proj"
    sparse_expert_up = "model.language_model.layers.5.mlp.experts.12.up_proj"

    assert match_name(
        "model.language_model.layers.3.input_layernorm",
        qkv_smooth.smooth_layer,
    )
    assert match_name(sparse_attn_q, qkv_smooth.balance_layers[0])
    assert match_name(sparse_indexer_q, qkv_smooth.balance_layers[3])
    assert not match_name(
        "model.language_model.layers.0.input_layernorm",
        qkv_smooth.smooth_layer,
    )
    assert not match_name(
        "model.language_model.layers.3.self_attn.o_proj",
        qkv_smooth.balance_layers[0],
    )
    # Vision tower also has layers.N.self_attn.q_proj; it must NOT match (bf16, and
    # matching it collapses per-layer grouping -> "single smoothlayer" error).
    assert not match_name(
        "model.vision_tower.layers.3.self_attn.q_proj",
        qkv_smooth.balance_layers[0],
    )
    assert not match_name(
        "model.vision_tower.layers.3.input_layernorm",
        qkv_smooth.smooth_layer,
    )

    assert match_name(
        "model.language_model.layers.10.self_attn.v_proj",
        v_to_o.smooth_layer,
    )
    assert match_name(
        "model.language_model.layers.10.self_attn.o_proj",
        v_to_o.balance_layers[0],
    )

    assert match_name(
        "model.language_model.layers.4.post_attention_layernorm",
        mlp_smooth.smooth_layer,
    )
    assert match_name(
        "model.language_model.layers.4.mlp.gate",
        mlp_smooth.balance_layers[0],
    )
    assert match_name(
        "model.language_model.layers.4.mlp.experts.0.gate_proj",
        mlp_smooth.balance_layers[2],
    )
    assert not match_name(dense_mlp, mlp_smooth.balance_layers[2])

    assert match_name(sparse_expert_up, up_to_down.smooth_layer)
    assert match_name(
        "model.language_model.layers.5.mlp.experts.12.down_proj",
        up_to_down.balance_layers[0],
    )
