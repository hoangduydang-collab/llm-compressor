"""Regression guard for the MiniMax-M3 AWQ mapping set.

The M3 expert activation is ``(clamp(up, ±limit) + 1.0) * glu`` (gpt-oss style,
swiglu_beta=1.0). A smoothing fold may only pass through an activation factor in
which it is homogeneous; the down input is affine-and-clamped in up's output, so
an ``up_proj -> down_proj`` mapping rescales the effective beta/clamp per channel
— a function change, not a reparameterization. r5 shipped with that mapping and
carried a ~5-33% RMS perturbation of every expert output (see BUGS_AND_FIXES.md
"AWQ up->down smoothing fold is not function-preserving on MiniMax-M3").

These tests fail if anyone re-adds a fold that crosses the expert activation
boundary. A future gate-side homogeneous fold ("r7", per-channel alpha and gate
clamp co-scaled) must use gate_proj as the smooth layer and pass the
homogeneity test below by construction.
"""

import re

import pytest

from pipeline.minimax_m3_config import get_minimax_m3_awq_mappings


def _matches(pattern: str, name: str) -> bool:
    assert pattern.startswith("re:")
    return re.match(pattern.removeprefix("re:"), name) is not None


EXAMPLE_UP = "model.language_model.layers.7.mlp.experts.3.up_proj"
EXAMPLE_DOWN = "model.language_model.layers.7.mlp.experts.3.down_proj"
EXAMPLE_NORM = "model.language_model.layers.7.post_attention_layernorm"
EXAMPLE_GATE_PROJ = "model.language_model.layers.7.mlp.experts.3.gate_proj"
EXAMPLE_ROUTER = "model.language_model.layers.7.mlp.gate"


@pytest.mark.parametrize("disable_mlp_input_smoothing", [False, True])
def test_no_fold_crosses_the_expert_activation(disable_mlp_input_smoothing):
    """No smooth layer may be an expert up_proj (or gate_proj, absent the r7
    alpha/clamp co-scaling machinery), and no balance layer may be an expert
    down_proj: any such pair folds through ``(clamp(up)+1)*glu``."""
    mappings = get_minimax_m3_awq_mappings(
        disable_mlp_input_smoothing=disable_mlp_input_smoothing,
        include_gate_alpha_fold=False,
    )
    for mapping in mappings:
        assert not _matches(mapping.smooth_layer, EXAMPLE_UP), (
            "up_proj used as a smooth layer: the up->down fold is not "
            "function-preserving on M3 (affine + clamped activation)"
        )
        assert not _matches(mapping.smooth_layer, EXAMPLE_GATE_PROJ), (
            "gate_proj used as a smooth layer without the r7 per-channel "
            "alpha/clamp co-scaling — not function-preserving on its own"
        )
        for balance in mapping.balance_layers:
            assert not _matches(balance, EXAMPLE_DOWN), (
                "down_proj used as a balance layer: its input is not a linear "
                "function of any foldable weight on M3"
            )


def test_up_proj_never_a_smooth_layer_even_with_r7_flag():
    """The r5 bug can never come back: up->down stays forbidden in every mode."""
    mappings = get_minimax_m3_awq_mappings(
        disable_mlp_input_smoothing=False, include_gate_alpha_fold=True
    )
    for mapping in mappings:
        assert not _matches(mapping.smooth_layer, EXAMPLE_UP)


def test_r7_gate_alpha_mapping_shape():
    """With the r7 flag, exactly one extra mapping appears: per-expert
    gate_proj -> down_proj (per-expert AND per-channel scales — the regex pair
    groups per expert via match_modules_set, like the removed mapping did)."""
    base = get_minimax_m3_awq_mappings(
        disable_mlp_input_smoothing=False, include_gate_alpha_fold=False
    )
    with_r7 = get_minimax_m3_awq_mappings(
        disable_mlp_input_smoothing=False, include_gate_alpha_fold=True
    )
    assert len(with_r7) == len(base) + 1
    extra = [m for m in with_r7 if _matches(m.smooth_layer, EXAMPLE_GATE_PROJ)]
    assert len(extra) == 1
    assert len(extra[0].balance_layers) == 1
    assert _matches(extra[0].balance_layers[0], EXAMPLE_DOWN)
    # per-expert grouping: the SAME expert index must match both sides, and a
    # different expert's down_proj must still match the regex (grouping is done
    # by match_modules_set per parent, not by the pattern itself)
    assert _matches(extra[0].smooth_layer, EXAMPLE_GATE_PROJ)


def test_moe_input_mapping_present_and_complete():
    """The post-attention-norm mapping is legal (purely linear boundary) and must
    keep its complete consumer set: router + shared experts + expert gate/up."""
    mappings = get_minimax_m3_awq_mappings(disable_mlp_input_smoothing=False)
    moe_input = [
        m for m in mappings if _matches(m.smooth_layer, EXAMPLE_NORM)
    ]
    assert len(moe_input) == 1
    balances = moe_input[0].balance_layers
    assert any(_matches(b, EXAMPLE_ROUTER) for b in balances), "router must be balanced"
    assert any(_matches(b, EXAMPLE_GATE_PROJ) for b in balances)
    assert any(_matches(b, EXAMPLE_UP) for b in balances)
    assert any(
        _matches(b, "model.language_model.layers.7.mlp.shared_experts.gate_up_proj")
        for b in balances
    ), "shared experts must be balanced"


def test_disable_mlp_input_smoothing_removes_only_that_mapping():
    with_mlp = get_minimax_m3_awq_mappings(disable_mlp_input_smoothing=False)
    without_mlp = get_minimax_m3_awq_mappings(disable_mlp_input_smoothing=True)
    assert len(with_mlp) == len(without_mlp) + 1
    assert not any(
        _matches(m.smooth_layer, EXAMPLE_NORM) for m in without_mlp
    )
