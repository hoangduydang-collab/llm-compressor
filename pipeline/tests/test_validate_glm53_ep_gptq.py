from __future__ import annotations

import math

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
