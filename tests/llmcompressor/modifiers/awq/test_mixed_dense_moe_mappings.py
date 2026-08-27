"""AWQ mapping guards for MLA models with a mixed dense/MoE stack.

A model whose first `first_k_dense_replace` layers are dense has no `mlp.gate`
router in those layers. `match_modules_set` closes a mapping per layer only when
every balance pattern matches inside that layer, so a balance pattern absent from
the dense layers stops any set from closing; the resolver then accumulates smooth
layers across the whole stack and raises

    ValueError: AWQ needs to match a single smoothlayer for each mapping
        but got ['model.layers.0.post_attention_layernorm', ...]

GLM-5.2 (`GlmMoeDsaForCausalLM`, first_k_dense_replace=3) shipped pointed at
`_deepseek_mappings`, which includes `mlp.gate`, and failed exactly that way in
the pre-quantization gate with **zero** mappings resolved — the same failure class
as MiniMax-M3's "AWQ fails on default smooth-layer mappings". These tests pin the
fix so a future registry edit cannot silently reintroduce it.
"""

import pytest

from llmcompressor.modifiers.transform.awq import (
    AWQ_MAPPING_REGISTRY,
    AWQMapping,
)

# Architectures known to have a mixed dense/MoE stack AND MLA attention, so their
# early layers carry no router. Keep this list in sync with the registry.
MIXED_DENSE_MOE_ARCHES = [
    "GlmMoeDsaForCausalLM",     # GLM-5.2,        first_k_dense_replace=3
    "Glm4MoeLiteForCausalLM",   # GLM-4.7-Flash,  first_k_dense_replace=1
]


def _balance_patterns(mappings: list[AWQMapping]) -> list[str]:
    return [p for m in mappings for p in m.balance_layers]


@pytest.mark.parametrize("arch", MIXED_DENSE_MOE_ARCHES)
def test_mixed_dense_moe_arch_is_registered(arch):
    """Falling back to default_mappings would silently mis-smooth these models."""
    assert arch in AWQ_MAPPING_REGISTRY, (
        f"{arch} is not in AWQ_MAPPING_REGISTRY, so AWQModifier would fall back "
        "to default_mappings, which assume dense q/k/v projections this model "
        "does not have."
    )


@pytest.mark.parametrize("arch", MIXED_DENSE_MOE_ARCHES)
def test_router_is_never_a_balance_layer(arch):
    """The regression this file exists for.

    `mlp.gate` must not appear as a balance layer: it is absent from the dense
    layers, which breaks per-layer grouping. It is also never quantized, so it
    was never a legitimate balance layer.
    """
    offenders = [p for p in _balance_patterns(AWQ_MAPPING_REGISTRY[arch]) if "gate$" in p and "gate_proj" not in p]
    assert not offenders, (
        f"{arch} lists router pattern(s) {offenders} as AWQ balance layers. "
        "Layers below first_k_dense_replace have no mlp.gate, so match_modules_set "
        "cannot close a per-layer set and AWQ resolves zero mappings. Use "
        "_mla_mixed_dense_moe_mappings."
    )


@pytest.mark.parametrize("arch", MIXED_DENSE_MOE_ARCHES)
def test_mla_projections_are_smoothed(arch):
    """Dropping mlp.gate must not have dropped the MLA coverage too."""
    patterns = _balance_patterns(AWQ_MAPPING_REGISTRY[arch])
    for required in ("re:.*kv_a_proj_with_mqa$", "re:.*q_b_proj$", "re:.*kv_b_proj$"):
        assert required in patterns, f"{arch} lost MLA balance layer {required}"


@pytest.mark.parametrize("arch", MIXED_DENSE_MOE_ARCHES)
def test_mlp_boundary_still_covered(arch):
    """gate_proj/up_proj match both the dense and the per-expert MLPs."""
    patterns = _balance_patterns(AWQ_MAPPING_REGISTRY[arch])
    assert "re:.*gate_proj$" in patterns
    assert "re:.*up_proj$" in patterns
    assert "re:.*down_proj$" in patterns


@pytest.mark.parametrize("arch", MIXED_DENSE_MOE_ARCHES)
def test_up_to_down_fold_is_legal_for_plain_swiglu(arch):
    """Document why the up->down fold is kept here but was removed for M3.

    The fold `up_rows /= s`, `down_cols *= s` is a pure reparameterization only
    when down's input is LINEAR in up. GLM computes `act_fn(gate) * up`, so it is.
    MiniMax-M3 computed `(clamp(up, +-7) + 1.0) * glu` — affine and clamped — so
    the same fold changed the function per channel and cost 24 pts on GPQA.
    See GLM52_FROM_M3_CARRYOVER.md.
    """
    folds = [
        m
        for m in AWQ_MAPPING_REGISTRY[arch]
        if m.smooth_layer == "re:.*up_proj$"
    ]
    assert len(folds) == 1, f"{arch} should fold up->down exactly once"
    assert folds[0].balance_layers == ["re:.*down_proj$"]
