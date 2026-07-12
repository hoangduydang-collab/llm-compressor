"""CPU-only tests for the MiniMax-M3 Transformers-to-vLLM ABI gate."""

from pipeline.m3_serve_abi import analyze_serving_abi


def _keys():
    prefix = "language_model.model.layers.3.block_sparse_moe"
    return [
        f"{prefix}.experts.0.w1.weight_packed",
        f"{prefix}.experts.0.w1.weight_scale",
        f"{prefix}.shared_experts.gate_up_proj.weight",
        f"{prefix}.gate.weight",
        "language_model.lm_head.weight",
    ]


def _config(*ignore):
    return {
        "quantization_config": {
            "format": "pack-quantized",
            "config_groups": {"group_0": {"targets": ["Linear"]}},
            "ignore": list(ignore),
        }
    }


def test_source_only_shared_expert_rule_fails_runtime_namespace():
    report = analyze_serving_abi(
        _config(
            "re:.*mlp[.]shared_experts[.].*",
            "re:.*block_sparse_moe[.]gate$",
            "lm_head",
        ),
        _keys(),
    )

    assert report["valid"] is False
    assert report["patterns"][0]["source_matches"] == 1
    assert report["patterns"][0]["runtime_matches"] == 0
    assert any(error["code"] == "plain_runtime_module_not_ignored" for error in report["errors"])


def test_dual_namespace_rules_match_plain_runtime_modules():
    report = analyze_serving_abi(
        _config(
            "re:.*mlp[.]shared_experts[.].*",
            "re:.*block_sparse_moe[.]shared_experts[.].*",
            "re:.*block_sparse_moe[.]gate$",
            "lm_head",
        ),
        _keys(),
    )

    assert report["valid"] is True
    assert report["inventory"]["quantized_modules"] == 1
    assert report["inventory"]["plain_quantizable_modules"] == 3


def test_ignore_rule_matching_packed_runtime_module_is_fatal():
    report = analyze_serving_abi(
        _config(
            "re:.*block_sparse_moe[.]experts[.].*",
            "re:.*block_sparse_moe[.]shared_experts[.].*",
            "re:.*block_sparse_moe[.]gate$",
            "lm_head",
        ),
        _keys(),
    )

    assert report["valid"] is False
    assert any(error["code"] == "packed_module_is_ignored" for error in report["errors"])


def test_packed_tensor_contract_rejects_missing_scale_and_plain_collision():
    keys = _keys()
    packed = keys[0].removesuffix(".weight_packed")
    keys.remove(f"{packed}.weight_scale")
    keys.append(f"{packed}.weight")
    report = analyze_serving_abi(
        _config(
            "re:.*block_sparse_moe[.]shared_experts[.].*",
            "re:.*block_sparse_moe[.]gate$",
            "lm_head",
        ),
        keys,
    )

    codes = {error["code"] for error in report["errors"]}
    assert "packed_module_missing_scale" in codes
    assert "module_has_packed_and_plain_weight" in codes


def test_invalid_ignore_regex_and_missing_linear_target_are_fatal():
    config = _config("re:[", "re:.*block_sparse_moe[.]shared_experts[.].*", "lm_head")
    config["quantization_config"]["config_groups"]["group_0"]["targets"] = ["Conv2d"]
    report = analyze_serving_abi(config, _keys())

    codes = {error["code"] for error in report["errors"]}
    assert "invalid_ignore_regex" in codes
    assert "quantization_group_does_not_target_linear" in codes
