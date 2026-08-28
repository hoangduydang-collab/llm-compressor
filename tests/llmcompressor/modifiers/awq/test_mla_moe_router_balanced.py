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


def _glm_like_model(
    num_dense: int = 3,
    num_moe: int = 4,
    num_experts: int = 2,
    indexer_layers: tuple[int, ...] | None = (0, 4),
):
    """Module tree shaped like GLM-5.2: dense prefix, then MoE layers with a
    router, shared experts and per-expert gate/up/down projections.

    `indexer_layers` mirrors `indexer_types`: only some layers carry a DSA
    indexer (21 of 78 on the real model), which is what makes an unscoped indexer
    balance pattern the same trap the router hit.
    """
    if indexer_layers is None:
        indexer_layers = ()
    model = torch.nn.Module()
    model.layers = torch.nn.Module()
    for index in range(num_dense + num_moe):
        layer = torch.nn.Module()
        layer.input_layernorm = torch.nn.Module()
        layer.post_attention_layernorm = torch.nn.Module()
        attn = torch.nn.Module()
        attn.q_a_proj = torch.nn.Linear(4, 4)
        attn.q_a_layernorm = torch.nn.Module()
        attn.q_b_proj = torch.nn.Linear(4, 4)
        attn.kv_a_proj_with_mqa = torch.nn.Linear(4, 4)
        attn.kv_a_layernorm = torch.nn.Module()
        attn.kv_b_proj = torch.nn.Linear(4, 4)
        attn.o_proj = torch.nn.Linear(4, 4)
        if index in indexer_layers:
            indexer = torch.nn.Module()
            indexer.wq_b = torch.nn.Linear(4, 4)
            indexer.wk = torch.nn.Linear(4, 4)
            indexer.weights_proj = torch.nn.Linear(4, 2)
            indexer.k_norm = torch.nn.LayerNorm(4)
            attn.indexer = indexer
        layer.self_attn = attn
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


# --- the DSA indexer: the same trap one block earlier -----------------------
#
# Found 2026-08-28 while answering "does it matter if the indexer is fp8 or
# bf16?". It does not, much -- but asking sent me to
# transformers/models/glm_moe_dsa/modeling_glm_moe_dsa.py, where
# GlmMoeDsaDecoderLayer.forward computes hidden_states = input_layernorm(x) and
# self_attn passes that tensor VERBATIM into self.indexer(...), whose wk and
# weights_proj consume it, while wq_b consumes q_a_layernorm(q_a_proj(x)). None of
# the three was a balance layer, so if those norms are ever divided by s the
# indexer sees x/s -- changing the DSA index scores and therefore WHICH TOKENS ARE
# ATTENDED. The router defect, one block earlier.
#
# LATENT, NOT SHIPPED, and this was checked rather than assumed. awq/base.py
# `continue`s a mapping when `any_targeted` is False, and _is_grid_search_targeted
# excludes float-schemed modules on purpose. Every attention-side balance layer on
# today's GLM recipes is FP8_DYNAMIC, so all three attention mappings are SKIPPED
# and those norms are never divided by s. No checkpoint we produced has an
# uncompensated indexer -- unlike the MoE-input mapping, whose routed experts are
# int4, which is why the router defect was real and measured.
#
# It goes live the moment anything in the attention block is int4: deliberately, or
# via a typo dropping an entry from fp8_dynamic_targets, which sends those
# projections THROUGH to the int4 modifier rather than to BF16. Nothing would catch
# it -- m3_checkpoint_scale_audit.py audits only post_attention_layernorm, mlp.gate
# and the shared experts, never input_layernorm or q_a_layernorm. Hence these
# tests: the mapping is correct now, so it cannot become wrong later.


def _mapping_for(mappings, smooth_substring, must_contain=None):
    found = [
        m
        for m in mappings
        if smooth_substring in m.smooth_layer
        and (must_contain is None or any(must_contain in b for b in m.balance_layers))
    ]
    assert len(found) == 1, f"expected one mapping, got {[m.smooth_layer for m in found]}"
    return found[0]


def test_indexer_wk_and_weights_proj_balance_the_attention_input():
    mappings = build_mla_mixed_dense_moe_mappings(_glm_like_model())
    balance = _mapping_for(mappings, "input_layernorm", "indexer[.]wk").balance_layers
    assert "re:.*self_attn[.]indexer[.]wk$" in balance
    assert "re:.*self_attn[.]indexer[.]weights_proj$" in balance
    # the projections that already worked must not have been dropped
    assert "re:.*(q|q_a)_proj$" in balance
    assert "re:.*kv_a_proj_with_mqa$" in balance


def test_indexer_wq_b_balances_q_a_layernorm():
    mappings = build_mla_mixed_dense_moe_mappings(_glm_like_model())
    balance = _mapping_for(mappings, "q_a_layernorm", "indexer[.]wq_b").balance_layers
    assert "re:.*self_attn[.]indexer[.]wq_b$" in balance
    assert "re:.*q_b_proj$" in balance


def test_indexer_patterns_are_layer_scoped_to_indexer_layers_only():
    """Why they can be included now. Only some layers carry an indexer, so an
    unscoped pattern would stop match_modules_set from closing a set on the
    others -- the exact failure that made commit 7d08e0fa drop the router."""
    mappings = build_mla_mixed_dense_moe_mappings(
        _glm_like_model(num_dense=3, num_moe=4, indexer_layers=(0, 4))
    )
    smooth = _mapping_for(mappings, "input_layernorm", "indexer[.]wk").smooth_layer
    indices = re.search(r"layers\[\.\]\(([0-9|]+)\)", smooth)
    assert indices, f"indexer mapping must be layer-scoped, got {smooth!r}"
    assert sorted(int(i) for i in indices.group(1).split("|")) == [0, 4]


def test_layers_without_an_indexer_get_their_own_unpolluted_mapping():
    """The complement mapping must exist and must NOT ask for indexer modules,
    otherwise those 57-of-78 layers resolve nothing."""
    mappings = build_mla_mixed_dense_moe_mappings(
        _glm_like_model(num_dense=3, num_moe=4, indexer_layers=(0, 4))
    )
    plain = [
        m
        for m in mappings
        if "input_layernorm" in m.smooth_layer
        and not any("indexer" in b for b in m.balance_layers)
    ]
    assert len(plain) == 1
    indices = re.search(r"layers\[\.\]\(([0-9|]+)\)", plain[0].smooth_layer)
    assert sorted(int(i) for i in indices.group(1).split("|")) == [1, 2, 3, 5, 6]
    assert "re:.*(q|q_a)_proj$" in plain[0].balance_layers


def test_every_layer_is_covered_by_exactly_one_attention_input_mapping():
    """No layer may be left without a mapping, and none may be claimed twice --
    a double claim would apply s twice."""
    model = _glm_like_model(num_dense=3, num_moe=4, indexer_layers=(0, 4))
    mappings = build_mla_mixed_dense_moe_mappings(model)
    claimed = []
    for m in mappings:
        if "input_layernorm" not in m.smooth_layer:
            continue
        found = re.search(r"layers\[\.\]\(([0-9|]+)\)", m.smooth_layer)
        claimed += [int(i) for i in found.group(1).split("|")] if found else list(range(7))
    assert sorted(claimed) == list(range(7)), claimed


def test_uniform_indexer_stack_stays_unscoped():
    """If every layer has an indexer there is no conflict to scope around, so the
    mapping stays simple -- and must still carry the indexer patterns."""
    mappings = build_mla_mixed_dense_moe_mappings(
        _glm_like_model(num_dense=1, num_moe=3, indexer_layers=(0, 1, 2, 3))
    )
    mapping = _mapping_for(mappings, "input_layernorm")
    assert mapping.smooth_layer == "re:.*input_layernorm$"
    assert "re:.*self_attn[.]indexer[.]wk$" in mapping.balance_layers


def test_no_indexer_anywhere_reproduces_the_previous_mapping():
    """A model with no DSA indexer must be unaffected by this change."""
    mappings = build_mla_mixed_dense_moe_mappings(
        _glm_like_model(indexer_layers=())
    )
    mapping = _mapping_for(mappings, "input_layernorm")
    assert mapping.smooth_layer == "re:.*input_layernorm$"
    assert mapping.balance_layers == ["re:.*(q|q_a)_proj$", "re:.*kv_a_proj_with_mqa$"]
    assert not any("indexer" in b for m in mappings for b in m.balance_layers)


def test_partial_indexer_module_set_does_not_emit_an_absent_pattern():
    """Same presence rule as the MoE set: a form some layers lack is not emitted.
    Here one indexer layer is missing weights_proj, so that pattern must drop out
    while wk and wq_b survive."""
    model = _glm_like_model(indexer_layers=(0, 4))
    del model.layers._modules["4"].self_attn.indexer.weights_proj
    mappings = build_mla_mixed_dense_moe_mappings(model)
    balance = _mapping_for(mappings, "input_layernorm", "indexer[.]wk").balance_layers
    assert "re:.*self_attn[.]indexer[.]wk$" in balance
    assert not any("weights_proj" in b for b in balance)
