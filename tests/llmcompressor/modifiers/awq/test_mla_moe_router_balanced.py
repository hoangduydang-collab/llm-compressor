"""The MoE router must be an AWQ balance layer on MLA mixed dense/MoE stacks.

This is the test whose absence let a real defect ship. Commit 7d08e0fa
("router must not be a balance layer") dropped `mlp.gate` from GLM-5.2's balance
set to fix a resolution error, a month after MiniMax-M3 had established the
opposite as a tested invariant. The GLM-5.2 AWQ smoke then saved a checkpoint
whose router was uncompensated: the norm was divided by s, the shared experts were
multiplied by s (audit residual 2.15e-3), and the router was left at base
(residual 1.08e-1 to 2.42e-1), so top-k expert selection differs from the base
model. See BUGS_AND_FIXES.md.

The resolution error that motivated the drop is real, so this file also pins the
property that made it go away: the router pattern must be LAYER-SCOPED, present
only in the mapping for layers that actually have a router.
"""

import re

import pytest
import torch

from llmcompressor.modifiers.transform.awq.dynamic_mappings import (
    build_mla_mixed_dense_moe_mappings,
)


def _glm_like_model(num_dense: int = 3, num_moe: int = 4, num_experts: int = 2):
    """Module tree shaped like GLM-5.2: dense prefix, then MoE layers with a
    router, shared experts and per-expert gate/up/down projections."""
    model = torch.nn.Module()
    model.layers = torch.nn.Module()
    for index in range(num_dense + num_moe):
        layer = torch.nn.Module()
        layer.input_layernorm = torch.nn.Module()
        layer.post_attention_layernorm = torch.nn.Module()
        mlp = torch.nn.Module()
        if index < num_dense:
            mlp.gate_proj = torch.nn.Linear(4, 4)
            mlp.up_proj = torch.nn.Linear(4, 4)
            mlp.down_proj = torch.nn.Linear(4, 4)
        else:
            mlp.gate = torch.nn.Linear(4, num_experts)  # the router
            shared = torch.nn.Module()
            shared.gate_proj = torch.nn.Linear(4, 4)
            shared.up_proj = torch.nn.Linear(4, 4)
            mlp.shared_experts = shared
            experts = torch.nn.Module()
            for e in range(num_experts):
                expert = torch.nn.Module()
                expert.gate_proj = torch.nn.Linear(4, 4)
                expert.up_proj = torch.nn.Linear(4, 4)
                expert.down_proj = torch.nn.Linear(4, 4)
                setattr(experts, str(e), expert)
            mlp.experts = experts
        layer.mlp = mlp
        setattr(model.layers, str(index), layer)
    return model


def _moe_input_mapping(mappings):
    """The mapping whose smooth layer is a post_attention_layernorm carrying a
    router in its balance set."""
    found = [
        m
        for m in mappings
        if "post_attention_layernorm" in m.smooth_layer
        and any(b.endswith("mlp[.]gate$") for b in m.balance_layers)
    ]
    assert len(found) == 1, f"expected exactly one MoE-input mapping, got {found}"
    return found[0]


def test_router_is_balanced():
    """The regression this file exists for."""
    mappings = build_mla_mixed_dense_moe_mappings(_glm_like_model())
    assert mappings is not None
    balance = _moe_input_mapping(mappings).balance_layers
    assert any(b.endswith("mlp[.]gate$") for b in balance), "router must be balanced"


def test_router_pattern_is_layer_scoped_to_moe_layers_only():
    """Why the router can be included now: the smooth pattern names only the
    layers that have one. An unscoped router pattern is what made
    match_modules_set resolve zero mappings."""
    mappings = build_mla_mixed_dense_moe_mappings(_glm_like_model(num_dense=3, num_moe=4))
    smooth = _moe_input_mapping(mappings).smooth_layer
    indices = re.search(r"layers\[\.\]\(([0-9|]+)\)", smooth)
    assert indices, f"router mapping must be layer-scoped, got {smooth!r}"
    assert sorted(int(i) for i in indices.group(1).split("|")) == [3, 4, 5, 6]


def test_dense_mapping_excludes_the_router():
    """The dense prefix has no router, so its mapping must not ask for one --
    that is the condition match_modules_set needs to close a per-layer set."""
    mappings = build_mla_mixed_dense_moe_mappings(_glm_like_model())
    dense = [
        m
        for m in mappings
        if "post_attention_layernorm" in m.smooth_layer
        and not any(b.endswith("mlp[.]gate$") for b in m.balance_layers)
    ]
    assert len(dense) == 1
    assert not any("mlp[.]gate$" in b for b in dense[0].balance_layers)
    assert re.search(r"layers\[\.\]\(0\|1\|2\)", dense[0].smooth_layer)


def test_experts_and_shared_experts_still_balanced():
    """Adding the router must not have dropped the consumers that already worked
    (the shared experts audited clean at 2.15e-3)."""
    balance = _moe_input_mapping(
        build_mla_mixed_dense_moe_mappings(_glm_like_model())
    ).balance_layers
    assert any("experts[.][0-9]+[.]gate_proj$" in b for b in balance)
    assert any("experts[.][0-9]+[.]up_proj$" in b for b in balance)
    assert any("shared_experts[.]gate_proj$" in b for b in balance)


def test_mla_projections_still_smoothed():
    patterns = [
        b for m in build_mla_mixed_dense_moe_mappings(_glm_like_model())
        for b in m.balance_layers
    ]
    for required in ("re:.*kv_a_proj_with_mqa$", "re:.*q_b_proj$", "re:.*kv_b_proj$"):
        assert required in patterns, f"lost MLA balance layer {required}"


def test_absent_shared_expert_form_is_not_emitted():
    """Only patterns present in EVERY MoE layer are emitted -- the same rule
    match_modules_set applies. A gate_up_proj pattern must not appear for a model
    that spells its shared experts gate_proj/up_proj."""
    balance = _moe_input_mapping(
        build_mla_mixed_dense_moe_mappings(_glm_like_model())
    ).balance_layers
    assert not any("gate_up_proj" in b for b in balance)


def test_dense_only_model_falls_back():
    """No MoE layers -> return None so the static registry is used unchanged."""
    assert build_mla_mixed_dense_moe_mappings(_glm_like_model(num_dense=3, num_moe=0)) is None


def test_no_dense_prefix_is_fine():
    """An all-MoE stack needs no dense mapping."""
    mappings = build_mla_mixed_dense_moe_mappings(_glm_like_model(num_dense=0, num_moe=3))
    assert mappings is not None
    assert _moe_input_mapping(mappings)


@pytest.mark.parametrize("arch", ["GlmMoeDsaForCausalLM", "Glm4MoeLiteForCausalLM"])
def test_registered_in_the_dynamic_registry(arch):
    from llmcompressor.modifiers.transform.awq.dynamic_mappings import (
        AWQ_DYNAMIC_MAPPING_REGISTRY,
    )

    assert AWQ_DYNAMIC_MAPPING_REGISTRY[arch] is build_mla_mixed_dense_moe_mappings
