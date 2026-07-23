"""Tests for the AWQ landscape telemetry (log_landscape_stats toggle)."""

import torch

from llmcompressor.modifiers.transform.awq.base import (
    AWQModifier,
    _channel_landscape_stats,
    _scale_landscape_stats,
)


def test_flat_vs_peaked_discrimination():
    torch.manual_seed(0)
    flat = torch.rand(4096) + 0.5
    peaked = flat.clone()
    peaked[:16] *= 50.0

    s_flat = _channel_landscape_stats(flat)
    s_peaked = _channel_landscape_stats(peaked)

    # uniform fair-share for the top 1% is 0.01; flat sits near it
    assert s_flat["top1pct_energy"] < 0.05
    assert s_peaked["top1pct_energy"] > 0.5
    assert s_peaked["max_over_median"] > 10 * s_flat["max_over_median"]
    assert s_flat["n_dead"] == 0


def test_dead_channels_excluded():
    v = torch.ones(1000)
    v[:10] = 0.0  # exactly dead (offset-norm gain 0 case)
    stats = _channel_landscape_stats(v)
    assert stats["n_dead"] == 10
    # live channels are uniform -> ratios exactly 1
    assert abs(stats["max_over_median"] - 1.0) < 1e-6
    assert abs(stats["top1pct_share"] - 0.01) < 5e-3


def test_scale_stats_identity_and_fold():
    ident = _scale_landscape_stats(torch.ones(128))
    assert ident["abs_dev_max"] == 0.0
    folded = _scale_landscape_stats(torch.full((128,), 1.66))
    assert abs(folded["abs_dev_median"] - 0.66) < 1e-5


def test_toggle_env_override(monkeypatch):
    modifier = AWQModifier(log_landscape_stats=False)
    assert modifier.log_landscape_stats is False
    monkeypatch.setenv("AWQ_LOG_LANDSCAPE_STATS", "1")
    monkeypatch.setenv("AWQ_LOG_LANDSCAPE_VECTORS_DIR", "/tmp/somewhere")

    # on_initialize applies env overrides before anything model-specific;
    # exercise just that block via a minimal fake state
    class _Model(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.linear = torch.nn.Linear(2, 2)

    class _State:
        model = _Model()

    try:
        modifier.on_initialize(_State())
    except Exception:
        # mapping inference may fail on the toy model; env overrides run first
        pass
    assert modifier.log_landscape_stats is True
    assert modifier.log_landscape_vectors_dir == "/tmp/somewhere"
