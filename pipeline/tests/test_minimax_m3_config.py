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
