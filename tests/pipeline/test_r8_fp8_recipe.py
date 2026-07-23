"""r8 recipe tests: mixed int4-experts + FP8_DYNAMIC-rest recipe building and
target-regex hygiene against the real M3 module naming (verified 2026-07-23 on
a meta-device build of MiniMaxM3VLDecoderLayer)."""

import re

import yaml

from pipeline.config import QuantizationConfig
from pipeline.recipe import build_recipe, describe_recipe

R8_FULL_CONFIG = "pipeline/configs/minimax_m3_distributed_r8_full.yaml"
R8A_FULL_CONFIG = "pipeline/configs/minimax_m3_distributed_r8a_awq_full.yaml"

# Representative in-memory module names (post linearize_moe, quant pipeline).
M3_NAMES = {
    "attn_q": "model.language_model.layers.30.self_attn.q_proj",
    "attn_o": "model.language_model.layers.30.self_attn.o_proj",
    "indexer_q": "model.language_model.layers.30.self_attn.indexer.q_proj",
    "indexer_k": "model.language_model.layers.30.self_attn.indexer.k_proj",
    "routed_gate": "model.language_model.layers.30.mlp.experts.7.gate_proj",
    "routed_down": "model.language_model.layers.30.mlp.experts.7.down_proj",
    "shared_gateup": "model.language_model.layers.30.mlp.shared_experts.gate_up_proj",
    "shared_down": "model.language_model.layers.30.mlp.shared_experts.down_proj",
    "dense0_gateup": "model.language_model.layers.0.mlp.gate_up_proj",
    "dense2_down": "model.language_model.layers.2.mlp.down_proj",
    "router": "model.language_model.layers.30.mlp.gate",
    "vision_q": "model.vision_tower.vision_model.encoder.layers.5.self_attn.q_proj",
    "lm_head": "lm_head",
    # layer-20 has no direct mlp.down_proj (MoE layer), but guard the
    # [0-2] pattern against multi-digit layer indices anyway:
    "layer20_trap": "model.language_model.layers.20.mlp.down_proj",
}

FP8_EXPECTED = {"attn_q", "attn_o", "shared_gateup", "shared_down",
                "dense0_gateup", "dense2_down"}


def _load_fp8_targets(config: str = R8_FULL_CONFIG) -> list[str]:
    cfg = yaml.safe_load(open(config))
    return cfg["quantization"]["fp8_dynamic_targets"]


def _matches(regex_target: str, name: str) -> bool:
    assert regex_target.startswith("re:")
    return re.match(regex_target[3:], name) is not None


def test_r8_full_fp8_targets_match_exactly_the_intended_modules():
    targets = _load_fp8_targets()
    for key, name in M3_NAMES.items():
        hit = any(_matches(t, name) for t in targets)
        assert hit == (key in FP8_EXPECTED), (key, name, hit)


def test_r8_full_gptq_ignore_still_covers_all_fp8_targets():
    """The two modifiers' scopes must be disjoint: everything FP8 targets
    must be ignored by GPTQ (else it would be int4-quantized first)."""
    cfg = yaml.safe_load(open(R8_FULL_CONFIG))["quantization"]
    ignore = [p for p in cfg["ignore"] if p.startswith("re:")]
    for key in FP8_EXPECTED:
        name = M3_NAMES[key]
        assert any(re.search(p[3:], name) for p in ignore), (key, name)


def test_build_recipe_appends_fp8_modifier():
    quant = QuantizationConfig(
        method="gptq",
        scheme="W4AFP8",
        ignore=["lm_head"],
        fp8_dynamic_targets=["re:.*self_attn[.](q|k|v|o)_proj$"],
    )
    recipe = build_recipe(quant)
    assert len(recipe) == 2
    gptq, fp8 = recipe
    assert type(gptq).__name__ == "GPTQModifier"
    assert type(fp8).__name__ == "QuantizationModifier"
    assert fp8.targets == ["re:.*self_attn[.](q|k|v|o)_proj$"]
    assert fp8.scheme == "FP8_DYNAMIC"
    assert describe_recipe(quant)["fp8_dynamic_targets"] == list(
        quant.fp8_dynamic_targets
    )


def test_r8a_awq_full_config_matches_r8_scoping():
    """The AWQ variant must target/ignore the same module sets as r8."""
    targets = _load_fp8_targets(R8A_FULL_CONFIG)
    for key, name in M3_NAMES.items():
        hit = any(_matches(t, name) for t in targets)
        assert hit == (key in FP8_EXPECTED), (key, name, hit)
    cfg = yaml.safe_load(open(R8A_FULL_CONFIG))["quantization"]
    assert cfg["method"] == "awq"
    ignore = [p for p in cfg["ignore"] if p.startswith("re:")]
    for key in FP8_EXPECTED:
        name = M3_NAMES[key]
        assert any(re.search(p[3:], name) for p in ignore), (key, name)


def test_build_recipe_awq_appends_fp8_modifier_last():
    """AWQ + fp8_dynamic_targets -> [AWQ, int4 quant, FP8 quant]. The FP8
    modifier must come AFTER the AWQ modifier so its weight qparams are
    observed on the post-fold (compensated) shared-expert weights."""
    quant = QuantizationConfig(
        method="awq",
        scheme="W4AFP8",
        ignore=["lm_head"],
        fp8_dynamic_targets=["re:.*self_attn[.](q|k|v|o)_proj$"],
    )
    recipe = build_recipe(quant)
    assert [type(m).__name__ for m in recipe] == [
        "AWQModifier",
        "QuantizationModifier",
        "QuantizationModifier",
    ]
    fp8 = recipe[-1]
    assert fp8.scheme == "FP8_DYNAMIC"
    assert fp8.targets == ["re:.*self_attn[.](q|k|v|o)_proj$"]


def test_build_recipe_without_fp8_targets_is_unchanged():
    quant = QuantizationConfig(method="gptq", scheme="W4AFP8", ignore=["lm_head"])
    recipe = build_recipe(quant)
    assert len(recipe) == 1
    assert type(recipe[0]).__name__ == "GPTQModifier"
