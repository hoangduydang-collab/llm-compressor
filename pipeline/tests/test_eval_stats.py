"""Tests for paired quantization-fidelity statistics."""

from __future__ import annotations

import pytest

from pipeline.evalsuite.stats import exact_mcnemar, paired_bootstrap


def _delta_mean(a: list[float], b: list[float]) -> float:
    return sum(y - x for x, y in zip(a, b, strict=True)) / len(a)


def test_exact_mcnemar_all_one_sided_flips():
    assert exact_mcnemar(6, 0) == {
        "method": "exact_binomial",
        "discordant": 6,
        "p_value": 0.03125,
    }


def test_exact_mcnemar_no_discordant_pairs():
    assert exact_mcnemar(0, 0)["p_value"] == 1.0


def test_paired_bootstrap_is_deterministic_and_contains_estimate():
    first = paired_bootstrap(
        [1.0, 0.0, 1.0, 0.0],
        [0.0, 0.0, 1.0, 1.0],
        statistic=_delta_mean,
        seed=42,
        iterations=1000,
    )
    second = paired_bootstrap(
        [1.0, 0.0, 1.0, 0.0],
        [0.0, 0.0, 1.0, 1.0],
        statistic=_delta_mean,
        seed=42,
        iterations=1000,
    )

    assert first == second
    assert first["estimate"] == pytest.approx(0.0)
    assert first["ci95_low"] <= first["estimate"] <= first["ci95_high"]
    assert first["seed"] == 42
    assert first["iterations"] == 1000


@pytest.mark.parametrize(
    "a,b,iterations,message",
    [
        ([], [], 10, "non-empty"),
        ([1.0], [1.0, 2.0], 10, "same length"),
        ([1.0], [1.0], 0, "iterations"),
    ],
)
def test_paired_bootstrap_rejects_invalid_inputs(a, b, iterations, message):
    with pytest.raises(ValueError, match=message):
        paired_bootstrap(a, b, statistic=_delta_mean, seed=42, iterations=iterations)
