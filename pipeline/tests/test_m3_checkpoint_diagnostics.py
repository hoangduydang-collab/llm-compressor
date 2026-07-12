"""Tests for format-tolerant MiniMax-M3 checkpoint diagnostics."""

from __future__ import annotations

import json
from pathlib import Path

from pipeline.m3_checkpoint_diagnostics import (
    _packed_code_saturation,
    classify_module,
    diagnose_checkpoint,
)


def _write_checkpoint(
    root: Path,
    *,
    config: dict,
    weight_map: dict[str, str],
) -> None:
    root.mkdir()
    (root / "config.json").write_text(json.dumps(config), encoding="utf-8")
    (root / "model.safetensors.index.json").write_text(
        json.dumps({"weight_map": weight_map}), encoding="utf-8"
    )
    for shard in set(weight_map.values()):
        (root / shard).write_bytes(b"synthetic-shard")


def test_compressed_checkpoint_reports_coverage_and_fallback(tmp_path):
    checkpoint = tmp_path / "quantized"
    _write_checkpoint(
        checkpoint,
        config={
            "quantization_config": {
                "quant_method": "compressed-tensors",
                "format": "pack-quantized",
                "config_groups": {
                    "group_0": {
                        "weights": {"num_bits": 4, "group_size": 128},
                        "input_activations": {"num_bits": 8, "type": "float"},
                    }
                },
                "ignore": ["lm_head"],
            }
        },
        weight_map={
            "language_model.layers.3.mlp.experts.0.gate_proj.weight_packed": "a.safetensors",
            "language_model.layers.3.mlp.experts.0.gate_proj.weight_scale": "a.safetensors",
            "language_model.layers.3.mlp.gate.weight": "a.safetensors",
            "lm_head.weight": "a.safetensors",
        },
    )

    result = diagnose_checkpoint(checkpoint, baseline_bytes=1000)

    assert result["quantization"]["method"] == "compressed-tensors"
    assert result["quantization"]["weight_bits"] == 4
    assert result["quantization"]["activation_bits"] == 8
    assert result["quantization"]["group_size"] == 128
    assert result["coverage_by_component"]["routed_experts"] == {
        "quantized_modules": 1,
        "plain_modules": 0,
    }
    assert result["coverage_by_component"]["routers"]["plain_modules"] == 1
    assert result["coverage_by_component"]["lm_head"]["plain_modules"] == 1
    assert result["compression"]["ratio_to_baseline"] == 1000 / result["checkpoint_bytes"]
    assert result["scale_statistics"]["status"] == "unavailable"


def test_plain_bf16_checkpoint_is_valid_unquantized_baseline(tmp_path):
    checkpoint = tmp_path / "bf16"
    _write_checkpoint(
        checkpoint,
        config={"torch_dtype": "bfloat16"},
        weight_map={
            "language_model.layers.0.self_attn.q_proj.weight": "model.safetensors",
            "language_model.layers.3.mlp.shared_experts.up_proj.weight": "model.safetensors",
        },
    )

    result = diagnose_checkpoint(checkpoint, baseline_bytes=None)

    assert result["valid"] is True
    assert result["quantization"]["method"] == "none"
    assert result["quantized_modules"] == 0
    assert result["plain_modules"] == 2
    assert result["compression"]["ratio_to_baseline"] is None


def test_gptq_and_autoround_metadata_variants_are_tolerated(tmp_path):
    checkpoint = tmp_path / "autoround"
    _write_checkpoint(
        checkpoint,
        config={
            "quantization_config": {
                "quant_method": "auto-round",
                "bits": 3,
                "group_size": 128,
            }
        },
        weight_map={
            "model.layers.3.block_sparse_moe.experts.0.up_proj.qweight": "a.safetensors",
            "model.layers.3.block_sparse_moe.experts.0.up_proj.scales": "a.safetensors",
        },
    )

    result = diagnose_checkpoint(checkpoint, baseline_bytes=None)

    assert result["quantization"]["method"] == "auto-round"
    assert result["quantization"]["weight_bits"] == 3
    assert result["quantized_modules"] == 1
    assert result["coverage_by_component"]["routed_experts"]["quantized_modules"] == 1


def test_component_classifier_handles_minimax_naming_aliases():
    assert classify_module("language_model.layers.8.self_attn.indexer") == "msa_indexer"
    assert (
        classify_module("model.layers.8.block_sparse_moe.shared_experts.up_proj")
        == "shared_experts"
    )
    assert classify_module("model.layers.8.block_sparse_moe.gate") == "routers"
    assert (
        classify_module("model.layers.8.block_sparse_moe.experts.12.down_proj")
        == "routed_experts"
    )
    assert classify_module("vision_tower.encoder.layers.0.mlp") == "vision"
    assert classify_module("language_model.norm") == "norms"



def test_packed_code_saturation_decodes_int4_nibbles():
    result = _packed_code_saturation([0x00000000, 0xFFFFFFFF], num_bits=4)

    assert result["codes"] == 16
    assert result["minimum_code_fraction"] == 0.5
    assert result["maximum_code_fraction"] == 0.5
    assert result["extreme_code_fraction"] == 1.0
