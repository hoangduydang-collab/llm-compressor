"""CPU-only tests for the MiniMax-M3 Transformers-to-vLLM ABI gate."""

from pipeline.m3_serve_abi import _detect_format, analyze_serving_abi


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


# --- ModelOpt block-scale (mxfp8/nvfp4) path -------------------------------

def _modelopt_keys():
    """Mirror MiniMaxAI/MiniMax-M3-MXFP8: quantized Linear = .weight +
    .weight_scale_inv; router gate / lm_head / vision stay plain (.weight only).
    """
    d = "language_model.model.layers.3"
    return [
        # quantized routed expert + dense + attention (weight + block scale)
        f"{d}.block_sparse_moe.experts.0.w1.weight",
        f"{d}.block_sparse_moe.experts.0.w1.weight_scale_inv",
        f"{d}.self_attn.q_proj.weight",
        f"{d}.self_attn.q_proj.weight_scale_inv",
        # intentionally-plain quantizable modules (must be in ignored_layers)
        f"{d}.block_sparse_moe.gate.weight",
        "language_model.lm_head.weight",
        "vision_tower.vision_model.encoder.layers.0.mlp.fc1.weight",
        # genuinely-unquantizable (norm) — never flagged
        f"{d}.input_layernorm.weight",
    ]


def _modelopt_config(*ignore):
    return {"quantization_config": {"quant_method": "mxfp8", "ignored_layers": list(ignore)}}


def test_detect_format_routes_modelopt_and_compressed_tensors():
    assert _detect_format({"quant_method": "mxfp8"}, ["x.weight_scale_inv"]) == "modelopt-scale"
    assert _detect_format({"ignored_layers": ["lm_head"]}, ["x.weight"]) == "modelopt-scale"
    assert (
        _detect_format({"config_groups": {"g": {}}}, ["x.weight_packed"])
        == "compressed-tensors"
    )


def test_modelopt_valid_when_all_quantizable_covered():
    # bare-leaf suffix (lm_head), exact router path, and subtree prefix (vision_tower)
    report = analyze_serving_abi(
        _modelopt_config(
            "lm_head",
            "language_model.model.layers.3.block_sparse_moe.gate",
            "vision_tower",
        ),
        _modelopt_keys(),
    )
    assert report["valid"] is True
    assert report["format"] == "mxfp8"
    assert report["inventory"]["quantized_modules"] == 2
    # gate + lm_head + vision fc1 are plain-but-quantizable, all ignored; norm excluded
    assert report["inventory"]["plain_quantizable_modules"] == 3


def test_modelopt_flags_uncovered_plain_quantizable_module():
    # drop the vision_tower ignore -> vision fc1 is a plain quantizable gap
    report = analyze_serving_abi(
        _modelopt_config(
            "lm_head",
            "language_model.model.layers.3.block_sparse_moe.gate",
        ),
        _modelopt_keys(),
    )
    assert report["valid"] is False
    flagged = {e["module"] for e in report["errors"] if e["code"] == "plain_runtime_module_not_ignored"}
    assert "vision_tower.vision_model.encoder.layers.0.mlp.fc1" in flagged


def test_modelopt_flags_quantized_module_that_is_ignored():
    # ignoring a module that actually carries a block scale is a contradiction
    report = analyze_serving_abi(
        _modelopt_config(
            "lm_head",
            "language_model.model.layers.3.block_sparse_moe.gate",
            "vision_tower",
            "re:.*self_attn[.]q_proj$",
        ),
        _modelopt_keys(),
    )
    assert report["valid"] is False
    assert any(e["code"] == "quantized_module_is_ignored" for e in report["errors"])
