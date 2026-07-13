"""Regression coverage for MiniMax-M3 Gemma-style norm calibration."""

import torch

from llmcompressor.modeling.offset_norm import NormCalibrationModule


class MiniMaxM3VLRMSNorm(torch.nn.Module):
    def __init__(self, dim: int = 4):
        super().__init__()
        self.eps = 1e-6
        self.weight = torch.nn.Parameter(torch.zeros(dim))


def test_minimax_m3_offset_norm_uses_effective_one_plus_weight():
    original = MiniMaxM3VLRMSNorm()
    calibration = NormCalibrationModule.load_from_registry(
        "MiniMaxM3VLRMSNorm", original=original, config=None
    )

    assert torch.equal(calibration.weight, torch.ones_like(calibration.weight))
    calibration.weight.data.div_(torch.tensor([0.5, 1.0, 2.0, 4.0]))
    restored = calibration.restore(original)

    expected = torch.tensor([1.0, 0.0, -0.5, -0.75])
    assert torch.equal(restored.weight, expected)
