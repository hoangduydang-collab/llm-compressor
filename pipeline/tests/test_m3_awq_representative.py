import json
import re
from pathlib import Path

import numpy as np
import pytest

from pipeline.m3_awq_representative import (
    BOUNDARIES,
    LAYERS,
    VARIANTS,
    aggregate_matrix,
    build_arm_recipe,
    classify_boundaries,
    fidelity_from_captures,
    prepare_arm_config,
    layer_exclusion_pattern,
    resolved_mapping_snapshot,
    sequential_target_pattern,
    tensor_fidelity,
    unwrap_tensor,
)
from pipeline.config import PipelineConfig


def _regex(pattern: str) -> re.Pattern[str]:
    assert pattern.startswith("re:")
    return re.compile(pattern.removeprefix("re:"))


def test_supported_matrix_is_fixed():
    assert LAYERS == (8, 31, 59)
    assert VARIANTS == ("offsetfix", "nosmooth")
    assert BOUNDARIES == (
        "layer_input",
        "moe_input",
        "moe_output",
        "layer_output",
    )


@pytest.mark.parametrize("layer", [-1, 0, 9, 60])
def test_selection_patterns_reject_unplanned_layers(layer):
    with pytest.raises(ValueError, match="representative layer"):
        layer_exclusion_pattern(layer)
    with pytest.raises(ValueError, match="representative layer"):
        sequential_target_pattern(layer)


def test_layer_exclusion_matches_every_decoder_layer_except_selected():
    pattern = _regex(layer_exclusion_pattern(8))

    assert pattern.fullmatch("language_model.layers.7.mlp.experts.0.up_proj")
    assert pattern.fullmatch("model.language_model.layers.31.self_attn.q_proj")
    assert not pattern.fullmatch("language_model.layers.8.mlp.experts.0.up_proj")
    assert pattern.fullmatch("language_model.layers.80.mlp.experts.0.up_proj")
    assert not pattern.fullmatch("vision_tower.layers.8.mlp")


def test_sequential_target_matches_only_the_selected_decoder_layer():
    pattern = _regex(sequential_target_pattern(31))

    assert pattern.fullmatch("language_model.layers.31")
    assert pattern.fullmatch("model.language_model.layers.31")
    assert not pattern.fullmatch("language_model.layers.3")
    assert not pattern.fullmatch("language_model.layers.31.mlp")
    assert not pattern.fullmatch("vision_tower.layers.31")


def test_tensor_fidelity_identical_tensors():
    tensor = np.array([[1.0, -2.0], [3.0, 4.0]])

    metrics = tensor_fidelity(tensor, tensor.copy())

    assert metrics["finite_fraction"] == 1.0
    assert metrics["reference_l2"] == pytest.approx(np.linalg.norm(tensor))
    assert metrics["candidate_l2"] == pytest.approx(np.linalg.norm(tensor))
    assert metrics["norm_ratio"] == pytest.approx(1.0)
    assert metrics["cosine_similarity"] == pytest.approx(1.0)
    assert metrics["relative_rmse"] == pytest.approx(0.0)
    assert metrics["max_abs_error"] == pytest.approx(0.0)


def test_tensor_fidelity_detects_exploded_tensor():
    reference = np.array([1.0, -2.0, 3.0])

    metrics = tensor_fidelity(reference, reference * 1_000)

    assert metrics["norm_ratio"] == pytest.approx(1_000.0)
    assert metrics["cosine_similarity"] == pytest.approx(1.0)
    assert metrics["relative_rmse"] == pytest.approx(999.0)


def test_tensor_fidelity_reports_non_finite_candidate_without_nan_metrics():
    metrics = tensor_fidelity(
        np.ones(4), np.array([1.0, float("nan"), float("inf"), 1.0])
    )

    assert metrics["finite_fraction"] == 0.5
    assert metrics["norm_ratio"] is None
    assert metrics["cosine_similarity"] is None
    assert metrics["relative_rmse"] is None
    assert metrics["max_abs_error"] is None


def test_tensor_fidelity_identical_zero_tensors_have_zero_relative_error():
    metrics = tensor_fidelity(np.zeros(4), np.zeros(4))
    assert metrics["norm_ratio"] == 1.0
    assert metrics["relative_rmse"] == 0.0


def test_tensor_fidelity_nonfinite_reference_is_strict_json_safe():
    metrics = tensor_fidelity(np.array([1.0, np.inf]), np.ones(2))
    json.dumps(metrics, allow_nan=False)
    assert metrics["reference_l2"] is None


def _passing_metrics(**updates):
    metrics = {
        "finite_fraction": 1.0,
        "norm_ratio": 1.0,
        "cosine_similarity": 0.99,
        "relative_rmse": 0.1,
    }
    metrics.update(updates)
    return metrics


@pytest.mark.parametrize(
    "updates",
    [
        {"finite_fraction": 0.99},
        {"norm_ratio": 0.09},
        {"norm_ratio": 10.01},
        {"cosine_similarity": 0.89},
        {"relative_rmse": 0.51},
        {"norm_ratio": None},
    ],
)
def test_classification_fails_any_boundary_outside_quality_gates(updates):
    boundaries = {name: _passing_metrics() for name in BOUNDARIES}
    boundaries["moe_input"] = _passing_metrics(**updates)

    verdict = classify_boundaries(boundaries)

    assert verdict["verdict"] == "quality_failure"
    assert verdict["boundaries"]["moe_input"]["passed"] is False
    assert verdict["failures"]


def test_classification_passes_threshold_boundaries_inclusively():
    boundaries = {
        name: _passing_metrics(
            norm_ratio=0.1,
            cosine_similarity=0.9,
            relative_rmse=0.5,
        )
        for name in BOUNDARIES
    }

    assert classify_boundaries(boundaries)["verdict"] == "pass"


def test_aggregate_matrix_preserves_pass_failure_and_missing_arms(tmp_path):
    passed = tmp_path / "offsetfix-layer8"
    passed.mkdir()
    (passed / "arm.json").write_text(
        json.dumps({"layer": 8, "variant": "offsetfix", "verdict": "pass"})
    )
    failed = tmp_path / "nosmooth-layer8"
    failed.mkdir()
    (failed / "arm.json").write_text(
        json.dumps(
            {"layer": 8, "variant": "nosmooth", "verdict": "quality_failure"}
        )
    )
    crashed = tmp_path / "offsetfix-layer31"
    crashed.mkdir()
    (crashed / "rc").write_text("137\n")

    matrix = aggregate_matrix(tmp_path)

    assert matrix["arms"]["offsetfix-layer8"]["status"] == "pass"
    assert matrix["arms"]["nosmooth-layer8"]["status"] == "quality_failure"
    assert matrix["arms"]["offsetfix-layer31"] == {
        "status": "infrastructure_failure",
        "return_code": 137,
    }
    assert matrix["arms"]["nosmooth-layer59"]["status"] == "missing"
    assert matrix["summary"] == {
        "pass": 1,
        "quality_failure": 1,
        "infrastructure_failure": 1,
        "missing": 3,
    }
    assert matrix["verdict"] == "incomplete"


@pytest.mark.parametrize("payload", [[], None, {"verdict": "pass"}])
def test_aggregate_matrix_rejects_wrong_schema_or_identity(tmp_path, payload):
    arm = tmp_path / "offsetfix-layer8"
    arm.mkdir()
    (arm / "arm.json").write_text(json.dumps(payload))
    matrix = aggregate_matrix(tmp_path)
    assert matrix["arms"]["offsetfix-layer8"]["status"] == "infrastructure_failure"


def test_aggregate_matrix_rejects_misattributed_evidence(tmp_path):
    arm = tmp_path / "offsetfix-layer8"
    arm.mkdir()
    (arm / "arm.json").write_text(
        json.dumps({"layer": 59, "variant": "nosmooth", "verdict": "pass"})
    )
    matrix = aggregate_matrix(tmp_path)
    assert matrix["arms"]["offsetfix-layer8"]["status"] == "infrastructure_failure"


def test_aggregate_matrix_nonzero_rc_overrides_written_arm_evidence(tmp_path):
    arm = tmp_path / "offsetfix-layer8"
    arm.mkdir()
    (arm / "arm.json").write_text(
        json.dumps({"layer": 8, "variant": "offsetfix", "verdict": "pass"})
    )
    (arm / "rc").write_text("1\n")
    matrix = aggregate_matrix(tmp_path)
    assert matrix["arms"]["offsetfix-layer8"] == {
        "status": "infrastructure_failure",
        "return_code": 1,
    }


def test_prepare_arm_config_isolates_one_layer_without_mutating_source():
    source = PipelineConfig()
    source.quantization.method = "awq"
    source.quantization.scheme = "W4AFP8"
    source.quantization.ignore = ["lm_head", "re:.*self_attn[.].*"]
    source.calibration.sequential_targets = ["MiniMaxM3VLDecoderLayer"]

    prepared = prepare_arm_config(source, layer=31)

    assert prepared is not source
    # Isolation is via the ignore list only; the production class-name sequential
    # target is preserved so tracing/partitioning matches the real quant run
    # (single-instance-path override is opt-in behind M3_AWQ_SINGLE_LAYER_TRACE).
    assert prepared.quantization.ignore[-1] == layer_exclusion_pattern(31)
    assert prepared.calibration.sequential_targets == ["MiniMaxM3VLDecoderLayer"]
    assert source.quantization.ignore == ["lm_head", "re:.*self_attn[.].*"]
    assert source.calibration.sequential_targets == ["MiniMaxM3VLDecoderLayer"]


def test_prepare_arm_config_single_layer_trace_flag_overrides_target(monkeypatch):
    monkeypatch.setenv("M3_AWQ_SINGLE_LAYER_TRACE", "1")
    source = PipelineConfig()
    source.quantization.method = "awq"
    source.quantization.scheme = "W4AFP8"
    source.calibration.sequential_targets = ["MiniMaxM3VLDecoderLayer"]

    prepared = prepare_arm_config(source, layer=31)

    # Opt-in legacy behavior: force the single decoder instance path for A/B tracing.
    assert prepared.calibration.sequential_targets == [sequential_target_pattern(31)]
    assert prepared.quantization.ignore[-1] == layer_exclusion_pattern(31)


def test_prepare_arm_config_rejects_nonproduction_recipe():
    source = PipelineConfig()
    source.quantization.method = "gptq"
    with pytest.raises(ValueError, match="AWQ W4AFP8"):
        prepare_arm_config(source, layer=8)


def test_unwrap_tensor_handles_nested_model_outputs():
    expected = np.array([[1.0, 2.0]])
    assert unwrap_tensor((expected, {"cache": None})) is expected
    assert unwrap_tensor({"last_hidden_state": expected}) is expected
    with pytest.raises(TypeError, match="tensor-like"):
        unwrap_tensor((None, "no tensor"))


def test_resolved_mapping_snapshot_rejects_cross_layer_mapping():
    class Mapping:
        smooth_name = "model.language_model.layers.8.post_attention_layernorm"
        balance_names = [
            "model.language_model.layers.31.mlp.experts.0.up_proj"
        ]

    with pytest.raises(RuntimeError, match="outside selected layer 8"):
        resolved_mapping_snapshot([Mapping()], layer=8, variant="offsetfix")


def test_resolved_mapping_snapshot_distinguishes_variants():
    class Mapping:
        def __init__(self, smooth):
            self.smooth_name = f"model.language_model.layers.8.{smooth}"
            self.balance_names = [
                "model.language_model.layers.8.mlp.experts.0.up_proj"
            ]

    offset = resolved_mapping_snapshot(
        [Mapping("post_attention_layernorm"), Mapping("mlp.experts.0.up_proj")],
        layer=8,
        variant="offsetfix",
    )
    assert any("post_attention_layernorm" in item["smooth_name"] for item in offset)

    with pytest.raises(RuntimeError, match="nosmooth resolved an MLP-input"):
        resolved_mapping_snapshot(
            [Mapping("post_attention_layernorm")], layer=8, variant="nosmooth"
        )


def test_diagnostic_module_has_no_checkpoint_export_call():
    source = Path("pipeline/m3_awq_representative.py").read_text()
    assert ".save_pretrained(" not in source
    assert "reexport_minimax_m3" not in source
    assert "capture_boundaries" not in source


def test_fidelity_from_captures_returns_per_probe_and_aggregate_metrics():
    captures = {
        phase: {
            boundary: [np.ones((1, 2)) * scale for scale in (1.0, 2.0)]
            for boundary in BOUNDARIES
        }
        for phase in ("reference", "candidate")
    }
    per_probe, aggregate = fidelity_from_captures(captures)
    assert len(per_probe) == 2
    assert all(probe["moe_output"]["relative_rmse"] == 0.0 for probe in per_probe)
    assert aggregate["layer_output"]["cosine_similarity"] == pytest.approx(1.0)


def test_fidelity_from_captures_rejects_missing_candidate_probe():
    captures = {
        phase: {
            boundary: [np.ones((1, 2)), np.ones((1, 2))]
            for boundary in BOUNDARIES
        }
        for phase in ("reference", "candidate")
    }
    captures["candidate"]["moe_input"].pop()
    with pytest.raises(RuntimeError, match="candidate moe_input captured 1"):
        fidelity_from_captures(captures)


def test_arm_recipe_constructs_ordered_audited_capture_modifiers():
    pytest.importorskip("torch")
    pytest.importorskip("compressed_tensors")
    source = PipelineConfig()
    source.quantization.method = "awq"
    source.quantization.scheme = "W4AFP8"
    expected = ["model.language_model.layers.8.mlp.experts.0.up_proj"]

    recipe, awq = build_arm_recipe(
        source, layer=8, variant="offsetfix", expected=expected
    )

    assert recipe[0] is awq
    assert recipe[1].capture_owner is awq
    assert recipe[1].expected_targets == expected


def test_audited_awq_retains_completed_and_skipped_lifecycle_metrics():
    pytest.importorskip("torch")
    pytest.importorskip("compressed_tensors")
    source = PipelineConfig()
    source.quantization.method = "awq"
    source.quantization.scheme = "W4AFP8"
    recipe, awq = build_arm_recipe(
        source, layer=8, variant="offsetfix",
        expected=["model.language_model.layers.8.mlp.experts.0.up_proj"],
    )
    awq._error_metrics.append({"layer_name": "completed", "reduction": 0.5})
    awq._skipped_error_metrics.append(
        {"layer_name": "skipped", "reason": "no_parent_outputs"}
    )

    awq._log_error_metrics()

    assert awq.lifecycle_audit["completed_metrics"][0]["layer_name"] == "completed"
    assert awq.lifecycle_audit["skipped_metrics"][0]["reason"] == "no_parent_outputs"
