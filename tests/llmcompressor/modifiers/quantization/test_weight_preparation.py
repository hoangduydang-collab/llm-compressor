"""FP8 upstream weights must change the inputs used to calibrate INT4 weights."""

from unittest.mock import patch

import pytest
import torch
from compressed_tensors.quantization import quantize

from llmcompressor.core import State
from llmcompressor.modifiers.gptq import GPTQModifier
from llmcompressor.modifiers.quantization import QuantizationModifier
from llmcompressor.modifiers.quantization.quantization.weight_preparation import (
    prepare_subgraph_weights,
    validate_weight_preparation,
)
from llmcompressor.pipelines.registry import CalibrationPipeline
from pipeline.sglang_w4afp8_kernels import dequantize_block_fp8, quantize_block_fp8


def _setup(dtype=torch.float32):
    torch.manual_seed(19)
    model = torch.nn.Sequential(
        torch.nn.Linear(128, 128, bias=False),
        torch.nn.Linear(128, 128, bias=False),
    ).to(dtype)
    gptq = GPTQModifier(targets=["1"], scheme="W4A16")
    fp8 = QuantizationModifier(
        targets=["0"],
        scheme="FP8_BLOCK",
        quantize_weights_before_calibration=True,
    )
    state = State(model=model)
    for modifier in (gptq, fp8):
        modifier.on_initialize(state)
    preparers = validate_weight_preparation(model, [gptq, fp8])
    for modifier in (gptq, fp8):
        modifier.on_calibration_start(state, None)
    return model, gptq, fp8, state, preparers


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_fp8_preparation_changes_gptq_inputs_and_freezes_export_scales(dtype):
    model, gptq, fp8, state, preparers = _setup(dtype)
    original = model[0].weight.detach().clone()
    payload, scale = quantize_block_fp8(original)
    # CT stores qparams in the weight dtype; use the same representable scale
    # for an independent cast-based reference, not the preparation helper.
    scale = scale.to(dtype)
    expanded = scale.repeat_interleave(128, 0).repeat_interleave(128, 1)
    payload = (original / expanded).clamp(-448, 448).to(torch.float8_e4m3fn)
    expected = dequantize_block_fp8(payload, scale.float()).to(dtype)
    prepare_subgraph_weights(preparers, list(model.modules()))
    torch.testing.assert_close(model[0].weight, expected, rtol=0, atol=0)
    assert not torch.equal(original, expected)
    x = torch.randn(2, 5, 128).to(dtype)
    # GPTQ's real input hook observes A's modified output.
    model(x)
    qx = torch.nn.functional.linear(x, expected)
    flat = qx.flatten(0, 1).float().t()
    torch.testing.assert_close(
        gptq._hessians[model[1]], 2 * flat @ flat.t(), rtol=1e-6, atol=1e-6
    )
    original_x = torch.nn.functional.linear(x, original).flatten(0, 1).float().t()
    assert not torch.equal(gptq._hessians[model[1]], 2 * original_x @ original_x.t())
    scales = model[0].weight_scale.detach().clone()
    with patch.object(
        model[0].weight_observer, "forward", side_effect=AssertionError("re-observed")
    ):
        fp8.on_sequential_epoch_end(state, None, list(model.modules()))
        prepare_subgraph_weights(preparers, list(model.modules()))
    torch.testing.assert_close(model[0].weight_scale, scales, rtol=0, atol=0)
    packed = quantize(
        model[0].weight,
        model[0].weight_scale,
        model[0].weight_zero_point,
        model[0].quantization_scheme.weights,
        dtype=torch.float8_e4m3fn,
    )
    assert torch.equal(packed.view(torch.uint8), payload.view(torch.uint8))
    gptq.on_calibration_end(state, None)
    fp8.on_calibration_end(state, None)


def test_preparation_requires_disjoint_target_ownership():
    model, gptq, fp8, state, _ = _setup()
    for group in gptq.resolved_config.config_groups.values():
        group.targets = ["0", "1"]
    with pytest.raises(ValueError, match="disjoint"):
        validate_weight_preparation(model, [gptq, fp8])


def test_preparation_rejects_disabled_propagation():
    model, gptq, fp8, _, _ = _setup()
    with pytest.raises(ValueError, match="propagate_error"):
        validate_weight_preparation(model, [gptq, fp8], False)


@pytest.mark.parametrize("pipeline", ["basic", "independent", "datafree"])
def test_preparation_rejects_nonsequential_pipeline(pipeline):
    _, gptq, fp8, _, _ = _setup()
    with pytest.raises(ValueError, match="sequential"):
        CalibrationPipeline.from_modifiers([gptq, fp8], pipeline)
