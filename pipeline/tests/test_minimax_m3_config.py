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
