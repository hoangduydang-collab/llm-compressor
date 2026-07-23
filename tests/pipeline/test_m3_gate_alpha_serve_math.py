"""Offline tests for the r7 serve-side per-channel activation math
(pipeline/m3_gate_alpha_serve_patch.py — the vLLM-independent core)."""

import torch

from pipeline.m3_gate_alpha_fold import make_m3_vector_apply_gate
from pipeline.m3_gate_alpha_serve_patch import (
    expand_tables_for_rows,
    rows_to_experts_batched,
    rows_to_experts_nonbatched,
    swigluoai_per_channel,
)

ALPHA, LIMIT = 1.702, 7.0
D = 16


def test_row_expert_mapping_nonbatched():
    # experts with 3, 0, 2 rows -> offsets [0, 3, 3, 5]
    off = torch.tensor([0, 3, 3, 5])
    mapped = rows_to_experts_nonbatched(off, 5)
    assert mapped.tolist() == [0, 0, 0, 2, 2]


def test_row_expert_mapping_batched():
    assert rows_to_experts_batched(3, 2).tolist() == [0, 0, 1, 1, 2, 2]


def test_serve_activation_matches_calibration_apply_gate():
    """The serve-side per-channel swiglu must equal the calibration-side
    vector-aware _apply_gate given the same scale table — this is the
    calibration/serve consistency contract for r7."""
    torch.manual_seed(0)
    n_experts, rows_per = 4, 8
    table = torch.exp(torch.randn(n_experts, D) * 0.4).clamp(0.3, 3.0)

    gate_up = torch.randn(n_experts * rows_per, 2 * D, dtype=torch.float64) * 4
    row_expert = rows_to_experts_batched(n_experts, rows_per)
    alpha_rows, limit_rows = expand_tables_for_rows(table, row_expert, ALPHA, LIMIT)
    served = swigluoai_per_channel(gate_up, alpha_rows, limit_rows, LIMIT)

    # calibration reference: per-expert module with the same vectors
    for e in range(n_experts):
        holder = torch.nn.Module()
        holder.swiglu_alpha_vec = ALPHA * table[e]
        holder.swiglu_limit_vec = LIMIT / table[e]
        apply_gate = make_m3_vector_apply_gate(holder, ALPHA, LIMIT)
        rows = slice(e * rows_per, (e + 1) * rows_per)
        torch.testing.assert_close(
            served[rows], apply_gate(gate_up[rows]), rtol=1e-6, atol=1e-8
        )


def test_identity_table_matches_scalar_activation():
    """A unit scale table must reproduce the stock scalar swiglu exactly."""
    torch.manual_seed(1)
    gate_up = torch.randn(32, 2 * D, dtype=torch.float64) * 4
    ones = torch.ones(1, D, dtype=torch.float64)  # fp64: isolate algebra from
    # table-storage precision (production fp32 bound covered in the fold tests)
    alpha_rows, limit_rows = expand_tables_for_rows(
        ones, torch.zeros(32, dtype=torch.long), ALPHA, LIMIT
    )
    served = swigluoai_per_channel(gate_up, alpha_rows, limit_rows, LIMIT)

    gate, up = gate_up.chunk(2, dim=-1)
    gate = gate.clamp(max=LIMIT)
    up = up.clamp(min=-LIMIT, max=LIMIT)
    reference = (up + 1.0) * (gate * torch.sigmoid(gate * ALPHA))
    torch.testing.assert_close(served, reference, rtol=0.0, atol=0.0)
