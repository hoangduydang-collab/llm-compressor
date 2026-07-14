import inspect
import json
import sys

import torch

from pipeline.config import load_config
from pipeline.m3_guarded_full import (
    FullGuardController,
    GuardedRunAbort,
    LayerEvidenceWriter,
    _evenly_spaced_indices,
    _fake_quant_input_descriptor,
    aggregate_runs,
    build_guarded_recipe,
    compare_sketches,
    deterministic_sketch,
    evaluate_layer_record,
    prepare_variant_config,
    tensor_summary,
)


class _FakeScheme:
    def __init__(self, group_size):
        self.num_bits = 4
        self.type = "int"
        self.group_size = group_size
        self.strategy = "group"
        self.symmetric = True


class _FakeQuantModule(torch.nn.Module):
    def __init__(self, weight, scale):
        super().__init__()
        self.weight = torch.nn.Parameter(weight, requires_grad=False)
        self.register_buffer("weight_scale", scale)


def test_fake_quant_descriptor_flags_group_geometry_mismatch():
    # in_features=256, group_size=128 -> 2 groups expected. A scale with 3
    # columns is the out-of-bounds-index shape that device-side asserts W4
    # grouped fake-quant.
    module = _FakeQuantModule(torch.zeros(8, 256), torch.ones(8, 3))
    descriptor = _fake_quant_input_descriptor(module, _FakeScheme(128))
    assert descriptor["expected_scale_groups"] == 2
    assert descriptor["actual_scale_groups"] == 3
    assert descriptor["group_geometry_consistent"] is False
    assert descriptor["weight"]["shape"] == [8, 256]
    assert descriptor["scheme"]["group_size"] == 128


def test_fake_quant_descriptor_accepts_consistent_group_geometry():
    module = _FakeQuantModule(torch.zeros(8, 256), torch.ones(8, 2))
    descriptor = _fake_quant_input_descriptor(module, _FakeScheme(128))
    assert descriptor["expected_scale_groups"] == 2
    assert descriptor["group_geometry_consistent"] is True


def test_deterministic_sketch_is_bounded_and_repeatable():
    value = torch.arange(10000, dtype=torch.float32).reshape(10, 1000)
    first = deterministic_sketch(value, max_values=64)
    second = deterministic_sketch(value, max_values=64)
    assert first.shape == (64,)
    assert torch.equal(first, second)
    assert first[0] == 0 and first[-1] == 9999


def test_sketch_indices_stay_in_bounds_above_float32_integer_limit():
    # torch.linspace defaults to float32, which cannot represent integers above
    # 2**24 exactly, so its rounded endpoint overshoots to `numel` -- one past
    # the end -- and index_select raises a CUDA device-side assert. This is the
    # bug that killed the diagnostic at layer 3 (first MoE layer): dense layers
    # 0-2 have smooth weights <2**24, MoE per-expert weights are 18.9M-37.7M.
    for numel in (18_874_368, 37_748_736, 16_781_312, 30_000_000):
        indices = _evenly_spaced_indices(numel, 4096, torch.device("cpu"))
        assert int(indices.max()) == numel - 1  # exact upper endpoint, no +1
        assert int(indices.min()) == 0
        assert int(indices[0]) == 0 and int(indices[-1]) == numel - 1
        # the naive float32 linspace this replaces DID overshoot here
        naive = torch.linspace(0, numel - 1, steps=4096).round().long()
        assert int(naive.max()) == numel  # documents the bug we fixed


def test_sketch_indices_handle_small_and_degenerate_counts():
    assert torch.equal(
        _evenly_spaced_indices(1, 1, torch.device("cpu")),
        torch.zeros(1, dtype=torch.long),
    )
    idx = _evenly_spaced_indices(5, 5, torch.device("cpu"))
    assert torch.equal(idx, torch.arange(5))  # count==numel covers every element


def test_deterministic_sketch_does_not_assert_above_float32_integer_limit():
    # Regression: index_select must not receive an out-of-bounds index. Uses a
    # sorted input so the sample must be non-decreasing and bracket both ends.
    value = torch.arange(16_781_312, dtype=torch.float64)
    sample = deterministic_sketch(value, max_values=4096)
    assert sample.shape == (4096,)
    assert torch.all(sample[1:] >= sample[:-1])


def test_comparison_metrics_include_sign_flip_and_catastrophic_gates():
    reference = torch.tensor([1.0, -2.0, 3.0, -4.0])
    candidate = torch.tensor([1.0, 2.0, 3.0, -4.0])
    metrics = compare_sketches(reference, candidate)
    assert metrics["sign_flip_ratio"] == 0.25
    assert metrics["finite_fraction"] == 1.0
    assert metrics["cosine_similarity"] < 1.0


def test_tensor_summary_reports_nonfinite_zero_and_percentiles():
    summary = tensor_summary(torch.tensor([0.0, 1.0, 2.0, float("inf")]))
    assert summary["finite_fraction"] == 0.75
    assert summary["zero_fraction"] == 0.25
    assert summary["min"] == 0.0
    assert summary["max"] == 2.0


def test_variant_aware_layer_evaluation_explains_mapping_failure():
    record = {
        "layer": 8,
        "variant": "offsetfix",
        "mapping_lifecycle": {
            "expected_count": 129,
            "resolved_count": 129,
            "completed_count": 0,
            "skipped_count": 0,
            "unprocessed_count": 129,
            "forward_event_count": 0,
        },
        "boundaries": {},
    }
    verdict = evaluate_layer_record(record)
    assert verdict["status"] == "abort"
    assert {item["check"] for item in verdict["violations"]} >= {
        "completed_mappings",
        "mapping_forward_events",
    }


def test_writer_persists_layer_and_abort_before_raising(tmp_path):
    writer = LayerEvidenceWriter(tmp_path, variant="offsetfix")
    record = {
        "layer": 8,
        "variant": "offsetfix",
        "mapping_lifecycle": {
            "expected_count": 1,
            "resolved_count": 1,
            "completed_count": 0,
            "skipped_count": 0,
            "unprocessed_count": 1,
            "forward_event_count": 0,
        },
        "boundaries": {},
    }
    try:
        writer.persist_and_enforce(record)
    except GuardedRunAbort:
        pass
    else:
        raise AssertionError("expected guarded abort")
    assert json.loads((tmp_path / "layers" / "layer-08.json").read_text())["layer"] == 8
    abort = json.loads((tmp_path / "abort.json").read_text())
    assert abort["layer"] == 8
    assert abort["violations"]
    assert json.loads((tmp_path / "heartbeat.json").read_text())["status"] == "aborted"


def test_full_recipe_variants_change_one_transform_and_preserve_contract(tmp_path):
    source = load_config("pipeline/configs/minimax_m3_full_calib.yaml")
    offset = prepare_variant_config(source, "offsetfix")
    nosmooth = prepare_variant_config(source, "nosmooth")
    quant_only = prepare_variant_config(source, "quant_only")
    assert offset.quantization.method == nosmooth.quantization.method == "awq"
    assert quant_only.quantization.method == "quant_only"
    assert source.quantization.method == "awq"

    offset_recipe, _ = build_guarded_recipe(offset, "offsetfix", tmp_path / "offset")
    nosmooth_recipe, _ = build_guarded_recipe(
        nosmooth, "nosmooth", tmp_path / "nosmooth"
    )
    quant_recipe, _ = build_guarded_recipe(quant_only, "quant_only", tmp_path / "quant")
    assert len(offset_recipe) == len(nosmooth_recipe) == 2
    assert len(quant_recipe) == 1
    assert len(offset_recipe[0].mappings) == 4
    assert len(nosmooth_recipe[0].mappings) == 3
    assert (
        offset_recipe[1].ignore == nosmooth_recipe[1].ignore == quant_recipe[0].ignore
    )


def test_aggregate_preserves_aborts_as_diagnostic_outcomes(tmp_path):
    for variant, status in (("offsetfix", "complete"), ("nosmooth", "aborted")):
        arm = tmp_path / variant
        arm.mkdir()
        name = "result.json" if status == "complete" else "failure.json"
        (arm / name).write_text(json.dumps({"status": status, "variant": variant}))
        (arm / "rc").write_text("0\n" if status == "complete" else "1\n")
    report = aggregate_runs(tmp_path)
    assert report["counts"] == {"complete": 1, "aborted": 1, "error": 0, "missing": 1}
    assert report["arms"]["nosmooth"]["status"] == "aborted"
    assert json.loads((tmp_path / "matrix.json").read_text()) == report


def test_quantization_is_enabled_before_candidate_propagation_capture():
    source = inspect.getsource(FullGuardController.note_quant_epoch)
    assert source.index("enable_quantization") < source.index("begin_candidate")


def test_guarded_recipe_would_infer_datafree_so_pipeline_must_be_pinned(tmp_path):
    # The guard wraps the modifiers in renamed subclasses. oneshot's pipeline
    # inference is class-name-based, so it does NOT see a calibration-requiring
    # modifier and silently picks the datafree (whole-model, no-partition)
    # pipeline -- which never invokes the per-layer propagation callback. The
    # runner must therefore pin pipeline="sequential"; this test locks in the
    # reason so the pin cannot be dropped.
    from llmcompressor.pipelines.registry import CalibrationPipeline

    source = load_config("pipeline/configs/minimax_m3_full_calib.yaml")
    offset = prepare_variant_config(source, "offsetfix")
    recipe, _ = build_guarded_recipe(offset, "offsetfix", tmp_path / "offset")

    assert CalibrationPipeline._infer_pipeline(recipe) == "datafree"
    assert type(recipe[0]).__name__ == "GuardedAWQModifier"

    # The runner pins the pipeline explicitly rather than relying on inference.
    runner_src = inspect.getsource(sys.modules[build_guarded_recipe.__module__])
    pin = 'kwargs["pipeline"] = config.calibration.pipeline or "sequential"'
    assert pin in runner_src


def _healthy_boundary():
    return {
        "finite_fraction": 1.0,
        "norm_ratio": 1.0,
        "cosine_similarity": 1.0,
        "relative_rmse": 0.0,
    }


def test_light_mode_does_not_require_intentionally_omitted_quant_evidence():
    record = {
        "layer": 8,
        "variant": "offsetfix",
        "diagnostic_mode": "light",
        "mapping_lifecycle": {
            "expected_count": 1,
            "resolved_count": 1,
            "completed_count": 1,
            "skipped_count": 0,
            "unprocessed_count": 0,
            "forward_event_count": 1,
            "completed_metrics": [
                {
                    "layer_name": "mapping",
                    "initial_error": 1.0,
                    "best_error": 0.5,
                    "reduction": 0.5,
                    "best_ratio": 0.5,
                }
            ],
        },
        "scale_diagnostics": [
            {
                "layer_name": "mapping",
                "scale": {"finite_fraction": 1.0, "zero_fraction": 0.0},
                "inverse_compensation_max_relative_error": 0.0,
            }
        ],
        "boundaries": {
            name: _healthy_boundary()
            for name in ("layer_input", "moe_input", "moe_output", "layer_output")
        },
    }
    assert evaluate_layer_record(record) == {"status": "pass", "violations": []}


def test_diagnostic_modes_persist_synchronization_boundaries(tmp_path, monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)

    light = FullGuardController(
        tmp_path / "light", variant="offsetfix", diagnostic_mode="light"
    )
    light.note_quant_epoch([])
    light_stages = json.loads(
        (tmp_path / "light" / "diagnostic_stages.json").read_text()
    )
    assert [item["stage"] for item in light_stages] == [
        "post_native_quantization",
        "post_enable_quantization",
    ]
    assert all(item["decoder_layers"] == [] for item in light_stages)

    heavy = FullGuardController(
        tmp_path / "heavy", variant="offsetfix", diagnostic_mode="heavy"
    )
    heavy.note_quant_epoch([])
    heavy_stages = json.loads(
        (tmp_path / "heavy" / "diagnostic_stages.json").read_text()
    )
    assert [item["stage"] for item in heavy_stages] == [
        "post_native_quantization",
        "post_enable_quantization",
        "before_qparam_inspection",
        "after_qparam_inspection",
        "before_fake_quantization",
        "after_fake_quantization",
    ]
    assert "weight_scale" not in inspect.getsource(FullGuardController.note_quant_epoch)
    assert "forward_quantize" not in inspect.getsource(
        FullGuardController.note_quant_epoch
    )
