"""Regression tests for the AWQ smoothing-scale degeneracy on dead channels.

MiniMax-M3 layers 8/10/11/12/13 carry a post-attention norm channel whose
base weight is exactly -1: the Gemma-style gain (1 + w) is 0, so that
channel's activation is always zero. With the unprotected formula, its
x_mean of 0 clamps to the 1e-4 floor and poisons the geometric
normalization ``scales / sqrt(max * min)``, inflating every live channel
~x100 and destroying the checkpoint (full-run r4, 2026-07-19).
"""

import torch

from llmcompressor.modifiers.transform.awq.base import (
    _SCALE_CLAMP,
    _grid_search_scales,
)


def _healthy_x_mean(n: int = 64) -> torch.Tensor:
    torch.manual_seed(0)
    return torch.rand(n) * 2.0 + 0.5  # bounded away from 0


def test_dead_channel_pinned_to_one():
    x_mean = _healthy_x_mean()
    x_mean[7] = 0.0
    scales = _grid_search_scales(x_mean, None, ratio=0.5)
    assert scales[7] == 1.0


def test_dead_channel_does_not_inflate_live_scales():
    x_mean = _healthy_x_mean()
    reference = _grid_search_scales(x_mean, None, ratio=0.5)

    poisoned = x_mean.clone()
    poisoned[7] = 0.0
    scales = _grid_search_scales(poisoned, None, ratio=0.5)

    live = torch.ones_like(x_mean, dtype=torch.bool)
    live[7] = False
    # Live channels keep the same normalization as the all-live case:
    # dropping one channel from the geometric mean moves it only slightly.
    ratio = scales[live] / reference[live]
    assert ratio.max() / ratio.min() < 1.01  # uniform shift only
    assert scales[live].max() < 10 * reference[live].max()


def test_old_formula_reproduces_r4_inflation():
    """Documents the failure mode the fix removes: with the unprotected
    formula a single floored channel inflates every scale by ~1/sqrt(floor/max)."""
    x_mean = _healthy_x_mean()
    x_mean[7] = 0.0
    ratio = 0.5
    raw = x_mean.pow(ratio).clamp(min=1e-4)
    old = raw / (raw.max() * raw.min()).sqrt()
    new = _grid_search_scales(x_mean, None, ratio=ratio)
    live = torch.ones_like(x_mean, dtype=torch.bool)
    live[7] = False
    assert old[live].min() > 30  # every live channel blown up
    assert new[live].max() < 10  # fixed path stays bounded


def test_duo_scaling_dead_channel():
    x_mean = _healthy_x_mean()
    x_mean[3] = 0.0
    w_mean = torch.rand_like(x_mean) + 0.1
    scales = _grid_search_scales(x_mean, w_mean, ratio=0.5)
    assert scales[3] == 1.0
    live = torch.ones_like(x_mean, dtype=torch.bool)
    live[3] = False
    assert scales[live].max() < 10
    assert scales[live].min() > 0.1


def test_all_dead_returns_ones():
    scales = _grid_search_scales(torch.zeros(16), None, ratio=0.7)
    assert torch.equal(scales, torch.ones(16))


def test_all_live_matches_original_formula():
    x_mean = _healthy_x_mean()
    ratio = 0.4
    raw = x_mean.pow(ratio).clamp(min=1e-4)
    expected = raw / (raw.max() * raw.min()).sqrt()
    got = _grid_search_scales(x_mean, None, ratio=ratio)
    assert torch.allclose(got, expected)


def test_hard_clamp_band():
    # Extreme dynamic range without any dead channel still stays in band.
    x_mean = torch.tensor([1e-8, 1e8], dtype=torch.float32)
    scales = _grid_search_scales(x_mean, None, ratio=1.0)
    assert scales.max() <= _SCALE_CLAMP
    assert scales.min() >= 1.0 / _SCALE_CLAMP
