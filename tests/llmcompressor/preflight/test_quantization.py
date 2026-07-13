from types import SimpleNamespace

import torch
from torch import nn

from llmcompressor.modifiers.gptq import GPTQModifier
from llmcompressor.modifiers.quantization import QuantizationModifier
from llmcompressor.modifiers.transform.awq import AWQModifier
from llmcompressor.modifiers.transform.awq.mappings import AWQMapping
from llmcompressor.preflight.quantization import (
    analyze_quantization_compatibility,
)


class MiniMaxM3VLRMSNorm(nn.Module):
    def __init__(self, width: int):
        super().__init__()
        self.weight = nn.Parameter(torch.zeros(width, device="meta"))
        self.eps = 1e-6

    def forward(self, value):
        return value


class TinyModel(nn.Module):
    def __init__(self, *, width: int = 128, output: int = 16):
        super().__init__()
        self.config = SimpleNamespace(model_type="tiny")
        self.norm = MiniMaxM3VLRMSNorm(width)
        self.linear = nn.Linear(width, output, bias=False, device="meta")


def test_gptq_uses_real_target_planner_and_passes():
    report = analyze_quantization_compatibility(
        TinyModel(),
        [GPTQModifier(targets="Linear", scheme="W4A16")],
    )

    assert report.compatible
    assert report.methods == ("GPTQModifier",)
    assert report.quantized_module_count == 1
    assert report.quantized_modules == ("linear",)
    assert report.planners[0].modifier == "GPTQModifier"
    assert report.planners[0].targets == ("Linear",)
    assert report.planners[0].ignore == ()
    assert report.failures == ()
    assert "calibration_dataset_quality" in report.unverified


def test_gptq_rejects_recipe_with_no_matching_targets():
    report = analyze_quantization_compatibility(
        TinyModel(),
        [GPTQModifier(targets="DoesNotExist", scheme="W4A16")],
    )

    assert not report.compatible
    assert any(finding.code == "no_quantized_modules" for finding in report.failures)


def test_gptq_reports_group_size_divisibility_before_calibration():
    report = analyze_quantization_compatibility(
        TinyModel(width=130),
        [GPTQModifier(targets="Linear", scheme="W4A16")],
    )

    assert not report.compatible
    finding = next(
        item for item in report.failures if item.code == "planner_initialization_failed"
    )
    assert "group_size" in finding.message
    assert "linear" in finding.message


def test_awq_resolves_real_mappings_and_offset_norm_adapter():
    report = analyze_quantization_compatibility(
        TinyModel(),
        [
            AWQModifier(mappings=[AWQMapping("norm", ["linear"])], duo_scaling=False),
            QuantizationModifier(targets="Linear", scheme="W4A16"),
        ],
    )

    assert report.compatible
    assert report.awq_mapping_count == 1
    assert report.awq_mappings[0].smooth_name == "norm"
    assert report.awq_mappings[0].balance_names == ("linear",)
    assert report.norm_adapters[0].module_class == "MiniMaxM3VLRMSNorm"
    assert report.norm_adapters[0].adapter_class == "CalibrationOffsetNorm"
    assert report.norm_adapters[0].status == "supported_offset"


def test_awq_rejects_recipe_when_no_mapping_resolves():
    report = analyze_quantization_compatibility(
        TinyModel(),
        [
            AWQModifier(mappings=[AWQMapping("norm", ["missing"])], duo_scaling=False),
            QuantizationModifier(targets="Linear", scheme="W4A16"),
        ],
    )

    assert not report.compatible
    assert any(finding.code == "no_awq_mappings" for finding in report.failures)


def test_minimax_offset_norm_missing_adapter_is_hard_failure(monkeypatch):
    from llmcompressor.preflight import quantization as preflight

    monkeypatch.setattr(preflight, "_resolve_norm_adapter", lambda _name: None)
    report = analyze_quantization_compatibility(
        TinyModel(),
        [
            AWQModifier(mappings=[AWQMapping("norm", ["linear"])], duo_scaling=False),
            QuantizationModifier(targets="Linear", scheme="W4A16"),
        ],
    )

    assert not report.compatible
    assert any(
        finding.code == "missing_offset_norm_adapter" for finding in report.failures
    )


def test_report_is_json_safe_and_versioned():
    report = analyze_quantization_compatibility(
        TinyModel(),
        [GPTQModifier(targets="Linear", scheme="W4A16")],
    ).to_dict()

    assert report["schema_version"] == 1
    assert report["compatible"] is True
    assert report["quantized_modules"] == ["linear"]
