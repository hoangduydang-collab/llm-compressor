from __future__ import annotations

import json
import math
from pathlib import Path

import torch
from safetensors.torch import save_file

try:
    from pipeline import validate_glm53_ep_gptq as validator
except ImportError:
    class _MissingValidator:
        def __getattr__(self, name):
            raise AssertionError(f"validator API is missing: {name}")

    validator = _MissingValidator()


def _complete_keys(layers=(3, 4), experts=2):
    keys = set()
    for layer in layers:
        for expert in range(experts):
            stem = f"model.layers.{layer}.mlp.experts.{expert}"
            for projection in ("gate_proj", "up_proj", "down_proj"):
                for suffix in ("weight_packed", "weight_scale", "weight_shape"):
                    keys.add(f"{stem}.{projection}.{suffix}")
    return keys


def _complete_tensors(layers=(3,), experts=1):
    tensors = {}
    for key in _complete_keys(layers=layers, experts=experts):
        if key.endswith(".weight_packed"):
            tensors[key] = torch.zeros((1, 1), dtype=torch.int32)
        elif key.endswith(".weight_scale"):
            tensors[key] = torch.ones((1,), dtype=torch.float32)
        else:
            tensors[key] = torch.tensor([1, 1], dtype=torch.int64)
    return tensors


def test_validate_checkpoint_accepts_single_file_emitted_layout(tmp_path: Path):
    save_file(_complete_tensors(), tmp_path / "model.safetensors")

    assert validator.validate_checkpoint(tmp_path, layers=(3,), experts=1) == []


def test_validate_checkpoint_accepts_sharded_emitted_layout(tmp_path: Path):
    tensors = _complete_tensors()
    keys = sorted(tensors)
    first = {key: tensors[key] for key in keys[::2]}
    second = {key: tensors[key] for key in keys[1::2]}
    save_file(first, tmp_path / "model-00001-of-00002.safetensors")
    save_file(second, tmp_path / "model-00002-of-00002.safetensors")
    weight_map = {
        key: (
            "model-00001-of-00002.safetensors"
            if key in first
            else "model-00002-of-00002.safetensors"
        )
        for key in keys
    }
    (tmp_path / "model.safetensors.index.json").write_text(
        json.dumps({"metadata": {}, "weight_map": weight_map}),
        encoding="utf-8",
    )

    assert validator.validate_checkpoint(tmp_path, layers=(3,), experts=1) == []


def test_validate_checkpoint_reads_bad_scale_from_single_file(tmp_path: Path):
    tensors = _complete_tensors()
    scale = next(key for key in tensors if key.endswith(".weight_scale"))
    tensors[scale] = torch.tensor([0.0], dtype=torch.float32)
    save_file(tensors, tmp_path / "model.safetensors")

    assert validator.validate_checkpoint(tmp_path, layers=(3,), experts=1) == [
        "weight scales contain 1 non-finite-or-non-positive value(s)"
    ]


def test_complete_expert_only_key_set_passes():
    errors = validator.validate_checkpoint_keys(
        _complete_keys(), layers=(3, 4), experts=2
    )
    assert errors == []


def test_missing_expert_qparameter_fails_with_exact_key():
    keys = _complete_keys()
    missing = "model.layers.4.mlp.experts.1.down_proj.weight_scale"
    keys.remove(missing)

    errors = validator.validate_checkpoint_keys(keys, layers=(3, 4), experts=2)

    assert errors == [f"missing required checkpoint key: {missing}"]


def test_packed_attention_is_rejected():
    keys = _complete_keys()
    forbidden = "model.layers.3.self_attn.o_proj.weight_packed"
    keys.add(forbidden)

    errors = validator.validate_checkpoint_keys(keys, layers=(3, 4), experts=2)

    assert errors == [f"forbidden non-expert packed weight: {forbidden}"]


def test_scales_must_be_finite_and_positive():
    assert validator.validate_scale_values([0.5, 1.0]) == []
    assert validator.validate_scale_values([0.5, 0.0, -1.0, math.nan]) == [
        "weight scales contain 3 non-finite-or-non-positive value(s)"
    ]


def test_memory_gate_requires_headroom_and_ep8_reduction():
    assert validator.validate_memory_scaling(
        ep4_peak_bytes=50 * 1024**3,
        ep8_peak_bytes=39 * 1024**3,
    ) == []
    assert validator.validate_memory_scaling(
        ep4_peak_bytes=50 * 1024**3,
        ep8_peak_bytes=41 * 1024**3,
    ) == ["EP8 peak must be <= 80% of EP4 peak (41.00 GiB > 40.00 GiB)"]
    assert validator.validate_memory_scaling(
        ep4_peak_bytes=77 * 1024**3,
        ep8_peak_bytes=30 * 1024**3,
    ) == ["EP4 peak 77.00 GiB is not below the 76.00 GiB ceiling"]


def test_phase_summary_requires_complete_expected_rank_coverage():
    summary = {
        "available": True,
        "complete": True,
        "expected_world_size": 8,
        "missing_ranks": [],
        "unexpected_ranks": [],
    }
    assert validator.validate_phase_summary(summary, world_size=8) == []
    summary["missing_ranks"] = [7]
    summary["complete"] = False
    assert validator.validate_phase_summary(summary, world_size=8) == [
        "phase evidence is incomplete",
        "phase evidence is missing ranks: [7]",
    ]
