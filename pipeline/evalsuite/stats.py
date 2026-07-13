"""Dependency-free paired statistics for quantization comparisons."""

from __future__ import annotations

import math
import random
from collections.abc import Callable, Sequence


def exact_mcnemar(regressions: int, recoveries: int) -> dict:
    if regressions < 0 or recoveries < 0:
        raise ValueError("McNemar counts must be non-negative")
    discordant = regressions + recoveries
    if discordant == 0:
        p_value = 1.0
    else:
        tail = min(regressions, recoveries)
        probability = sum(
            math.comb(discordant, index) for index in range(tail + 1)
        ) / (2**discordant)
        p_value = min(1.0, 2.0 * probability)
    return {
        "method": "exact_binomial",
        "discordant": discordant,
        "p_value": p_value,
    }


def _percentile(sorted_values: list[float], quantile: float) -> float:
    if not 0.0 <= quantile <= 1.0:
        raise ValueError("quantile must be between zero and one")
    position = (len(sorted_values) - 1) * quantile
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return sorted_values[lower]
    fraction = position - lower
    return sorted_values[lower] * (1.0 - fraction) + sorted_values[upper] * fraction


def paired_bootstrap(
    values_a: Sequence[float],
    values_b: Sequence[float],
    *,
    statistic: Callable[[list[float], list[float]], float],
    seed: int,
    iterations: int,
) -> dict:
    if not values_a or not values_b:
        raise ValueError("paired bootstrap inputs must be non-empty")
    if len(values_a) != len(values_b):
        raise ValueError("paired bootstrap inputs must have the same length")
    if iterations <= 0:
        raise ValueError("paired bootstrap iterations must be positive")

    a = [float(value) for value in values_a]
    b = [float(value) for value in values_b]
    estimate = float(statistic(a, b))
    rng = random.Random(seed)
    sampled: list[float] = []
    for _ in range(iterations):
        indices = [rng.randrange(len(a)) for _ in a]
        sampled.append(
            float(
                statistic(
                    [a[index] for index in indices],
                    [b[index] for index in indices],
                )
            )
        )
    sampled.sort()
    return {
        "estimate": estimate,
        "ci95_low": _percentile(sampled, 0.025),
        "ci95_high": _percentile(sampled, 0.975),
        "seed": seed,
        "iterations": iterations,
    }
