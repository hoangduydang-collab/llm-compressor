"""Tests for the upstream-vs-recipe quantization scope gate.

The gate's whole value is that it fails on an unintended scope change, so most of
these tests are about what it must NOT wave through: a component silently left in
BF16 by an over-broad ignore pattern, a component quantized that upstream keeps
at source precision, and the difference between those and the divergences we
chose. The synthetic model here is GLM-shaped (dense prefix, MoE layers with
router / shared experts / routed experts, a DSA indexer, an MTP layer) but tiny.
"""

import json

import pytest

from pipeline.compare_upstream_quant_scope import (
    ACCEPTED_DIVERGENCES,
    classify_recipe,
    classify_upstream,
    compare,
    component_of,
    layer_index,
    match_name,
)

DEPTH = 6          # layers 0..5 are the body
DENSE = 2          # layers 0,1 are dense
EXPERTS = 2
MTP = DEPTH        # layer 6 is the MTP head
INDEXER_LAYERS = {0, 4, MTP}


def _weight_map(quantize_indexer=True, quantize_router=False):
    """A weight index shaped like a released block-FP8 GLM checkpoint.

    Modules that upstream quantized carry `.weight_scale_inv` next to `.weight`;
    modules it left alone carry only `.weight`. That is the only signal the gate
    reads, exactly as the real index provides it.
    """
    fp8, src = [], ["lm_head.weight", "model.embed_tokens.weight", "model.norm.weight"]
    for layer in list(range(DEPTH)) + [MTP]:
        p = f"model.layers.{layer}"
        src += [f"{p}.input_layernorm.weight", f"{p}.post_attention_layernorm.weight"]
        src += [f"{p}.self_attn.q_a_layernorm.weight", f"{p}.self_attn.kv_a_layernorm.weight"]
        fp8 += [
            f"{p}.self_attn.{proj}.weight"
            for proj in ("q_a_proj", "q_b_proj", "kv_a_proj_with_mqa", "kv_b_proj", "o_proj")
        ]
        if layer in INDEXER_LAYERS:
            fp8 += [f"{p}.self_attn.indexer.wq_b.weight", f"{p}.self_attn.indexer.wk.weight"]
            # weights_proj is [32, H]; a [128,128] block grid does not tile it,
            # so the vendor quantizer skips it despite not listing it in
            # modules_to_not_convert.
            src += [f"{p}.self_attn.indexer.weights_proj.weight"]
            src += [f"{p}.self_attn.indexer.k_norm.weight"]
        if layer < DENSE:
            fp8 += [f"{p}.mlp.{proj}.weight" for proj in ("gate_proj", "up_proj", "down_proj")]
            continue
        (fp8 if quantize_router else src).append(f"{p}.mlp.gate.weight")
        fp8 += [
            f"{p}.mlp.shared_experts.{proj}.weight"
            for proj in ("gate_proj", "up_proj", "down_proj")
        ]
        for e in range(EXPERTS):
            fp8 += [
                f"{p}.mlp.experts.{e}.{proj}.weight"
                for proj in ("gate_proj", "up_proj", "down_proj")
            ]
    if not quantize_indexer:
        moved = [k for k in fp8 if ".indexer." in k]
        fp8 = [k for k in fp8 if k not in moved]
        src += moved
    out = {k: "shard.safetensors" for k in src}
    for k in fp8:
        out[k] = "shard.safetensors"
        out[k[: -len(".weight")] + ".weight_scale_inv"] = "shard.safetensors"
    return out


def _recipe(ignore=None, fp8_targets=None):
    return {
        "quantization": {
            "ignore": ignore
            if ignore is not None
            else [
                "lm_head",
                "re:.*mlp[.]gate$",
                "re:.*mlp[.]shared_experts[.].*",
                "re:.*self_attn[.].*",
                f"re:.*layers[.][0-{DENSE - 1}][.].*",
                f"re:.*layers[.]{MTP}[.].*",
            ],
            "fp8_targets_placeholder": None,
            "fp8_dynamic_targets": fp8_targets
            if fp8_targets is not None
            else [
                "re:.*model[.]layers[.][0-9]+[.]self_attn[.]"
                "(q_a_proj|q_b_proj|kv_a_proj_with_mqa|kv_b_proj|o_proj)$",
                "re:.*model[.]layers[.][0-9]+[.]mlp[.]shared_experts[.]"
                "(gate_proj|up_proj|down_proj)$",
                f"re:.*model[.]layers[.][0-{DENSE - 1}][.]mlp[.]"
                "(gate_proj|up_proj|down_proj)$",
            ],
        }
    }


# --- the matcher must be the same one compressed_tensors uses -----------------


def test_match_name_is_prefix_regex_not_fullmatch():
    """compressed_tensors uses re.match. A gate using re.fullmatch would judge
    the recipe by different rules than the pipeline applies, which is worse than
    no gate."""
    assert match_name("model.layers.3.mlp.experts.0.gate_proj", "re:.*mlp[.]experts")
    assert not match_name("model.layers.3.mlp.gate", "re:.*mlp[.]experts")


def test_match_name_plain_string_is_exact():
    assert match_name("lm_head", "lm_head")
    assert not match_name("model.lm_head_extra", "lm_head")


# --- structural labelling -----------------------------------------------------


@pytest.mark.parametrize(
    "module,expected",
    [
        ("model.layers.3.mlp.experts.0.gate_proj", "routed expert gate_proj"),
        ("model.layers.3.mlp.shared_experts.up_proj", "shared expert up_proj"),
        ("model.layers.0.mlp.down_proj", "dense mlp down_proj"),
        ("model.layers.3.mlp.gate", "router (mlp.gate)"),
        ("model.layers.3.self_attn.o_proj", "MLA o_proj"),
        ("model.layers.0.self_attn.indexer.wk", "indexer.wk"),
        ("model.layers.0.self_attn.indexer.k_norm", "norm"),
        ("model.layers.0.self_attn.indexers_proj", "indexers_proj"),
        ("model.layers.0.input_layernorm", "norm"),
        ("model.layers.6.eh_proj", "MTP eh_proj"),
        ("lm_head", "lm_head"),
    ],
)
def test_component_of(module, expected):
    assert component_of(module) == expected


def test_layer_index():
    assert layer_index("model.layers.42.mlp.gate") == 42
    assert layer_index("lm_head") is None


# --- upstream classification comes from the artifact, not the config ----------


def test_upstream_classification_reads_block_scales():
    up = classify_upstream(_weight_map())
    assert up["model.layers.2.mlp.experts.0.gate_proj"] == "fp8"
    assert up["model.layers.2.mlp.gate"] == "src"
    assert up["model.layers.0.self_attn.indexer.weights_proj"] == "src"
    assert up["model.layers.0.self_attn.indexer.wk"] == "fp8"
    # a scale-only key must not invent a module
    assert not any(k.endswith("weight_scale_inv") for k in up)


# --- fp8 targets outranking ignore, as recipe.py builds it --------------------


def test_fp8_target_beats_the_main_modifiers_ignore():
    """recipe.py constructs QuantizationModifier(targets=fp8_dynamic_targets)
    with NO ignore list, so a module can be both ignored by the int4 modifier and
    quantized by the FP8 one. That is the whole design of the two-modifier
    recipe, and a gate that applied `ignore` first would report the shared
    experts as BF16."""
    r = _recipe()["quantization"]
    module = "model.layers.2.mlp.shared_experts.gate_proj"
    assert any(match_name(module, t) for t in r["ignore"])
    assert classify_recipe(module, r["ignore"], r["fp8_dynamic_targets"]) == "fp8"


def test_router_is_never_a_quantization_target():
    r = _recipe()["quantization"]
    assert classify_recipe("model.layers.2.mlp.gate", r["ignore"], r["fp8_dynamic_targets"]) == "src"


def test_routed_experts_are_int4():
    r = _recipe()["quantization"]
    assert (
        classify_recipe(
            "model.layers.2.mlp.experts.0.gate_proj", r["ignore"], r["fp8_dynamic_targets"]
        )
        == "int4"
    )


# --- the gate's verdicts ------------------------------------------------------


def test_matching_recipe_passes():
    report = compare(_weight_map(), _recipe(), DEPTH)
    assert report.ok, [d.__dict__ for d in report.real]
    # the indexer divergence is accepted, not silently dropped
    assert {d.component for d in report.accepted} == {"indexer.wq_b", "indexer.wk"}


def test_mtp_layer_is_reported_separately_and_does_not_fail_the_gate():
    report = compare(_weight_map(), _recipe(), DEPTH)
    assert report.ok
    mtp = {d.component for d in report.mtp}
    assert "routed expert gate_proj" in mtp
    assert all(layer_index(d.example) == MTP for d in report.mtp)
    # body rows and MTP rows must not double-count
    assert report.rows["MLA o_proj"][("fp8", "fp8")] == DEPTH
    assert report.mtp_rows["MLA o_proj"][("fp8", "fp8")] == 1


def test_over_broad_ignore_leaves_a_component_bf16_and_fails():
    """The r8 failure class: a pattern wide enough to swallow a whole component.
    Dropping the shared-expert fp8 target leaves 4 layers x 3 projections in BF16
    while every int4 gate still passes."""
    recipe = _recipe(
        fp8_targets=[
            "re:.*model[.]layers[.][0-9]+[.]self_attn[.]"
            "(q_a_proj|q_b_proj|kv_a_proj_with_mqa|kv_b_proj|o_proj)$",
            f"re:.*model[.]layers[.][0-{DENSE - 1}][.]mlp[.](gate_proj|up_proj|down_proj)$",
        ]
    )
    report = compare(_weight_map(), recipe, DEPTH)
    assert not report.ok
    assert {d.component for d in report.real} == {
        "shared expert gate_proj",
        "shared expert up_proj",
        "shared expert down_proj",
    }
    assert all(d.ours == "src" and d.upstream == "fp8" for d in report.real)


def test_quantizing_something_upstream_keeps_at_source_fails():
    """The opposite error, and the one that would break the router. Upstream
    keeps mlp.gate at source precision; if it ever became an nn.Linear and the
    ignore line were dropped, the gate must say so."""
    weight_map = _weight_map(quantize_router=False)
    recipe = _recipe(ignore=["lm_head", f"re:.*layers[.]{MTP}[.].*"])
    report = compare(weight_map, recipe, DEPTH)
    assert not report.ok
    bad = {(d.component, d.upstream, d.ours) for d in report.real}
    assert ("norm", "src", "src") not in bad  # norms are not Linear
    assert any(c.startswith("indexer") for c, _, _ in bad)


def test_partial_scope_is_not_reported_as_a_defect():
    """A smoke recipe that targets one MoE layer diverges everywhere else by
    construction. Those are attributed to the restricted scope, not to a pattern
    bug -- otherwise the gate is useless on every config we actually run first."""
    recipe = _recipe()
    recipe["quantization"]["ignore"].append(
        "re:.*model[.]layers[.](?!2(?:[.]|$))[0-9]+(?:[.]|$).*"
    )
    recipe["quantization"]["fp8_dynamic_targets"] = [
        "re:.*model[.]layers[.]2[.]self_attn[.]"
        "(q_a_proj|q_b_proj|kv_a_proj_with_mqa|kv_b_proj|o_proj)$",
        "re:.*model[.]layers[.]2[.]mlp[.]shared_experts[.](gate_proj|up_proj|down_proj)$",
    ]
    report = compare(_weight_map(), recipe, DEPTH)
    assert report.covered_layers == {2}
    assert report.ok, [d.__dict__ for d in report.real]
    assert report.layer_restricted


def test_int4_outside_the_routed_experts_fails():
    """A typo dropping the MLA entry from fp8_dynamic_targets does not send those
    projections to BF16 -- it sends them to the int4 modifier, because `ignore`
    is what keeps them out of it. An earlier revision of the gate treated any
    `fp8 -> int4` transition as the intended one and would have passed this."""
    recipe = _recipe(
        ignore=[
            "lm_head",
            "re:.*mlp[.]gate$",
            "re:.*mlp[.]shared_experts[.].*",
            f"re:.*layers[.][0-{DENSE - 1}][.].*",
            f"re:.*layers[.]{MTP}[.].*",
        ],
        fp8_targets=[
            "re:.*model[.]layers[.][0-9]+[.]mlp[.]shared_experts[.]"
            "(gate_proj|up_proj|down_proj)$",
            f"re:.*model[.]layers[.][0-{DENSE - 1}][.]mlp[.](gate_proj|up_proj|down_proj)$",
        ],
    )
    report = compare(_weight_map(), recipe, DEPTH)
    assert not report.ok
    assert {d.component for d in report.real} >= {
        "MLA q_a_proj",
        "MLA q_b_proj",
        "MLA kv_a_proj_with_mqa",
        "MLA kv_b_proj",
        "MLA o_proj",
    }
    # Two distinct failures show up, and both are real: the MoE layers' MLA
    # projections fall through to int4, while layers 0-1 are additionally covered
    # by the dense-prefix ignore and land in BF16 instead.
    mla = {(d.component, d.ours) for d in report.real if d.component.startswith("MLA")}
    assert ("MLA o_proj", "int4") in mla
    assert ("MLA o_proj", "src") in mla


def test_partial_is_decided_by_layer_coverage_not_by_a_flag():
    """A recipe is judged partial because it targets fewer than num_hidden_layers
    layers, not because someone remembered to pass a flag. The smoke configs all
    restrict scope with a regex, and a flag would be exactly the thing that goes
    stale when one is copied into a production config."""
    full = compare(_weight_map(), _recipe(), DEPTH)
    assert not full.partial
    assert full.covered_layers == set(range(DEPTH))

    recipe = _recipe()
    recipe["quantization"]["ignore"].append(
        "re:.*model[.]layers[.](?!2(?:[.]|$))[0-9]+(?:[.]|$).*"
    )
    recipe["quantization"]["fp8_dynamic_targets"] = []
    partial = compare(_weight_map(), recipe, DEPTH)
    assert partial.partial
    # divergences inside the one sampled layer are true of that checkpoint, so
    # they are still listed -- they just do not fail a recipe that never claimed
    # full scope.
    assert partial.ok
    assert partial.real


def test_accepted_divergences_are_declared_not_inferred():
    """The allowlist is policy and must stay small and explicit; a divergence
    that is merely common must not become accepted by accident."""
    assert set(ACCEPTED_DIVERGENCES) == {
        ("indexer.wq_b", "fp8", "src"),
        ("indexer.wk", "fp8", "src"),
    }


def test_upstream_leaving_the_indexer_alone_removes_the_divergence():
    """If a future release stops quantizing the indexer, our RED LINE agrees with
    it and the accepted-divergence list should go quiet rather than stay stale."""
    report = compare(_weight_map(quantize_indexer=False), _recipe(), DEPTH)
    assert report.ok
    assert not report.accepted


# --- the real configs ---------------------------------------------------------


def test_real_glm_configs_share_one_scope():
    """The GPTQ full, AWQ full and GLM-5.3 configs must be scope-identical; the
    only intended difference between the arms is the algorithm."""
    import yaml
    from pathlib import Path

    root = Path(__file__).resolve().parents[1] / "configs"
    scopes = []
    for name in (
        "glm52_distributed_w4afp8_full.yaml",
        "glm52_distributed_w4afp8_awq_full.yaml",
        "glm53_distributed_w4afp8_awq_full.yaml",
    ):
        q = yaml.safe_load((root / name).read_text())["quantization"]
        scopes.append((sorted(q["ignore"]), sorted(q["fp8_dynamic_targets"])))
    assert scopes[0] == scopes[1] == scopes[2]
