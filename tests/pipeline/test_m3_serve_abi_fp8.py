"""fp8-aware static serving-ABI validation (mixed int4+fp8 r8/r8a layouts)."""

from pipeline.m3_serve_abi import analyze_serving_abi


def _mixed_config(ignore=None, float_targets=None):
    return {
        "quantization_config": {
            "quant_method": "compressed-tensors",
            "format": "mixed-precision",
            "ignore": ignore
            if ignore is not None
            else ["re:.*self_attn[.](q_proj|k_proj|v_proj)$", "lm_head"],
            "config_groups": {
                "group_0": {
                    "targets": float_targets
                    if float_targets is not None
                    else ["re:.*self_attn[.]o_proj$"],
                    "weights": {"type": "float", "num_bits": 8},
                },
                "group_1": {
                    "targets": ["Linear"],
                    "weights": {"type": "int", "num_bits": 4},
                },
            },
        }
    }


_KEYS = [
    # int4 packed expert
    "language_model.model.layers.3.block_sparse_moe.experts.0.w1.weight_packed",
    "language_model.model.layers.3.block_sparse_moe.experts.0.w1.weight_scale",
    # fp8 module: plain weight + scale, matched by the float group's targets
    "language_model.model.layers.3.self_attn.o_proj.weight",
    "language_model.model.layers.3.self_attn.o_proj.weight_scale",
    # bf16 modules covered by ignore
    "language_model.model.layers.3.self_attn.q_proj.weight",
    "lm_head.weight",
]


def test_mixed_fp8_layout_is_valid():
    report = analyze_serving_abi(_mixed_config(), _KEYS)
    assert report["valid"], report["errors"]
    assert report["inventory"]["fp8_targeted_modules"] == 1


def test_fp8_module_matched_by_ignore_is_rejected():
    # vLLM checks ignore before targets: an fp8 module that also matches an
    # ignore pattern serves raw fp8 bits cast to bf16 (the r8 v1 bug class).
    config = _mixed_config(
        ignore=[
            "re:.*self_attn[.].*",
            "lm_head",
        ]
    )
    report = analyze_serving_abi(config, _KEYS)
    codes = {error["code"] for error in report["errors"]}
    assert not report["valid"]
    assert "fp8_module_is_ignored" in codes


def test_fp8_targeted_module_without_scale_is_rejected():
    keys = [k for k in _KEYS if k != "language_model.model.layers.3.self_attn.o_proj.weight_scale"]
    report = analyze_serving_abi(_mixed_config(), keys)
    codes = {error["code"] for error in report["errors"]}
    assert not report["valid"]
    assert "fp8_targeted_module_missing_scale" in codes


def test_float_group_matching_nothing_is_rejected():
    # Quant-layout names (e.g. missing ".model.") never match serve/disk names;
    # a float group that hits no runtime module is a dead serving contract.
    config = _mixed_config(
        # every fp8 module must then be ignored or it fails as plain-unignored
        ignore=["re:.*self_attn[.].*", "lm_head"],
        float_targets=["re:.*language_model[.]layers[.].*self_attn[.]o_proj$"],
    )
    report = analyze_serving_abi(config, _KEYS)
    codes = {error["code"] for error in report["errors"]}
    assert not report["valid"]
    assert "quantization_group_does_not_target_linear" in codes


def test_pure_int4_layout_still_valid():
    config = {
        "quantization_config": {
            "quant_method": "compressed-tensors",
            "format": "pack-quantized",
            "ignore": ["re:.*self_attn[.].*", "re:.*mlp[.].*", "lm_head"],
            "config_groups": {
                "group_0": {
                    "targets": ["Linear"],
                    "weights": {"type": "int", "num_bits": 4},
                }
            },
        }
    }
    keys = [
        "language_model.model.layers.3.block_sparse_moe.experts.0.w1.weight_packed",
        "language_model.model.layers.3.block_sparse_moe.experts.0.w1.weight_scale",
        "language_model.model.layers.3.self_attn.q_proj.weight",
        "language_model.model.layers.3.mlp.gate_proj.weight",
        "lm_head.weight",
    ]
    report = analyze_serving_abi(config, keys)
    assert report["valid"], report["errors"]
