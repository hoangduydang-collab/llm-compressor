import inspect
import json

import torch

from pipeline.config import load_config
from pipeline.m3_guarded_full import (
    FullGuardController,
    GuardedRunAbort,
    LayerEvidenceWriter,
    aggregate_runs,
    build_guarded_recipe,
    compare_sketches,
    deterministic_sketch,
    evaluate_layer_record,
    prepare_variant_config,
    tensor_summary,
)


def test_deterministic_sketch_is_bounded_and_repeatable():
    value = torch.arange(10000, dtype=torch.float32).reshape(10, 1000)
    first = deterministic_sketch(value, max_values=64)
    second = deterministic_sketch(value, max_values=64)
    assert first.shape == (64,)
    assert torch.equal(first, second)
    assert first[0] == 0 and first[-1] == 9999


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
