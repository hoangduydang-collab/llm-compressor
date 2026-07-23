"""Fold-identity tests for the r7 gate-alpha fold (pipeline/m3_gate_alpha_fold.py).

The r7 fold must be an EXACT reparameterization of the M3 expert:
  gate rows /= s_r ; alpha_r = a*s_r ; limit_r = L/s_r ; down cols *= s_r
composed with AWQModifier's generic weight algebra. These tests replicate the
modifier's `_smooth` weight updates bit-for-bit and assert the expert's
input->output function is unchanged in fp64 — including in the clamp-active
regime, which is precisely where the removed up->down fold broke.
"""

import torch

from pipeline.m3_gate_alpha_fold import (
    GATE_SMOOTH_SCALE_NAME,
    attach_minimax_m3_gate_alpha_fold,
    make_m3_vector_apply_gate,
)

ALPHA, LIMIT = 1.702, 7.0
HIDDEN, INTER = 32, 24


def _reference_m3_activation(gate_up: torch.Tensor) -> torch.Tensor:
    """Verbatim MiniMaxM3VLExperts._apply_gate semantics (scalar constants)."""
    gate, up = gate_up.chunk(2, dim=-1)
    gate = gate.clamp(max=LIMIT)
    up = up.clamp(min=-LIMIT, max=LIMIT)
    glu = gate * torch.sigmoid(gate * ALPHA)
    return (up + 1.0) * glu


class _Expert(torch.nn.Module):
    def __init__(self, dtype=torch.float64):
        super().__init__()
        self.gate_proj = torch.nn.Linear(HIDDEN, INTER, bias=False, dtype=dtype)
        self.up_proj = torch.nn.Linear(HIDDEN, INTER, bias=False, dtype=dtype)
        self.down_proj = torch.nn.Linear(INTER, HIDDEN, bias=False, dtype=dtype)
        self._apply_gate = make_m3_vector_apply_gate(self, ALPHA, LIMIT)

    def forward(self, x):
        return self.down_proj(
            self._apply_gate(
                torch.cat([self.gate_proj(x), self.up_proj(x)], dim=-1)
            )
        )


class _Container(torch.nn.ModuleList):
    """Stands in for the linearized M3 experts container."""

    def __init__(self, n=2):
        super().__init__([_Expert() for _ in range(n)])
        self.swiglu_alpha = ALPHA
        self.swiglu_limit = LIMIT


def _apply_awq_fold(expert: _Expert, scales: torch.Tensor) -> None:
    """Replicate AWQModifier._smooth exactly + the fold consumer call."""
    with torch.no_grad():
        # balance layer: down cols *= s   (weight * scales.view(1, -1))
        expert.down_proj.weight.mul_(scales.view(1, -1))
        # smooth layer (ndim==2): last len(s) rows /= s
        expert.gate_proj.weight[-scales.size(0):].div_(scales.view(-1, 1))
    expert.gate_proj.awq_fold_scale_consumer(scales)


def _prepared_expert(buffer_dtype=torch.float64) -> _Expert:
    container = _Container(n=1)
    prepared = attach_minimax_m3_gate_alpha_fold(container)
    assert prepared == 1
    expert = container[0]
    # production keeps the buffer fp32; tests use fp64 so exactness assertions
    # isolate the fold ALGEBRA from buffer-precision rounding (bounded
    # separately in test_fp32_buffer_precision_bound)
    expert._buffers[GATE_SMOOTH_SCALE_NAME] = torch.ones(
        INTER, dtype=buffer_dtype
    )
    return expert


def test_scalar_path_matches_reference():
    torch.manual_seed(0)
    expert = _Expert()
    x = torch.randn(64, HIDDEN, dtype=torch.float64) * 3
    gate_up = torch.cat([expert.gate_proj(x), expert.up_proj(x)], dim=-1)
    assert torch.equal(expert._apply_gate(gate_up), _reference_m3_activation(gate_up))


def test_fold_is_exact_identity_including_clamp_regime():
    torch.manual_seed(1)
    expert = _prepared_expert()
    # inputs large enough that both clamps fire for some channels/tokens
    x = torch.randn(256, HIDDEN, dtype=torch.float64) * 4
    before = expert(x)

    scales = torch.exp(torch.randn(INTER, dtype=torch.float64) * 0.4).clamp(0.3, 3.0)
    _apply_awq_fold(expert, scales)
    after = expert(x)

    torch.testing.assert_close(after, before, rtol=1e-12, atol=1e-12)
    # sanity: the fold really happened (weights differ), per-expert per-channel
    assert getattr(expert, GATE_SMOOTH_SCALE_NAME).shape == (INTER,)
    assert not torch.allclose(
        getattr(expert, GATE_SMOOTH_SCALE_NAME).double(), torch.ones(INTER).double()
    )


def test_up_down_fold_would_NOT_be_identity():
    """Control: the removed r5 fold on the same expert is provably not exact."""
    torch.manual_seed(2)
    expert = _Expert()
    x = torch.randn(256, HIDDEN, dtype=torch.float64)
    before = expert(x)
    scales = torch.full((INTER,), 1.66, dtype=torch.float64)
    with torch.no_grad():
        expert.down_proj.weight.mul_(scales.view(1, -1))
        expert.up_proj.weight[-scales.size(0):].div_(scales.view(-1, 1))
    after = expert(x)
    assert not torch.allclose(after, before, rtol=1e-3, atol=1e-3)


def test_fold_composes_multiplicatively():
    torch.manual_seed(3)
    expert = _prepared_expert()
    x = torch.randn(128, HIDDEN, dtype=torch.float64) * 4
    before = expert(x)
    s1 = torch.exp(torch.randn(INTER, dtype=torch.float64) * 0.3)
    s2 = torch.exp(torch.randn(INTER, dtype=torch.float64) * 0.3)
    _apply_awq_fold(expert, s1)
    _apply_awq_fold(expert, s2)
    torch.testing.assert_close(expert(x), before, rtol=1e-12, atol=1e-12)
    torch.testing.assert_close(
        getattr(expert, GATE_SMOOTH_SCALE_NAME).double(), s1 * s2, rtol=1e-6, atol=1e-6
    )


def test_derived_vectors_satisfy_invariant():
    """alpha_r * limit_r == alpha * limit identically (the self-check gate)."""
    torch.manual_seed(4)
    expert = _prepared_expert()
    _apply_awq_fold(expert, torch.exp(torch.randn(INTER, dtype=torch.float64) * 0.4))
    inv = (expert.swiglu_alpha_vec * expert.swiglu_limit_vec).double()
    torch.testing.assert_close(
        inv, torch.full((INTER,), ALPHA * LIMIT, dtype=torch.float64),
        rtol=1e-5, atol=1e-5,
    )


def test_fp32_buffer_precision_bound():
    """Production path (fp32 gate_smooth_scale): fold residual bounded by fp32
    rounding — orders of magnitude below bf16 weight storage error."""
    torch.manual_seed(5)
    expert = _prepared_expert(buffer_dtype=torch.float32)
    x = torch.randn(256, HIDDEN, dtype=torch.float64) * 4
    before = expert(x)
    scales = torch.exp(torch.randn(INTER, dtype=torch.float64) * 0.4).clamp(0.3, 3.0)
    _apply_awq_fold(expert, scales)
    after = expert(x)
    torch.testing.assert_close(after, before, rtol=1e-5, atol=1e-6)


def test_consumer_rejects_bad_scales():
    expert = _prepared_expert()
    import pytest

    with pytest.raises(ValueError):
        expert.gate_proj.awq_fold_scale_consumer(torch.ones(INTER + 1))
    with pytest.raises(ValueError):
        expert.gate_proj.awq_fold_scale_consumer(torch.full((INTER,), -1.0))
