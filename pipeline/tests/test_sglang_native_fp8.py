"""Native CT bytes must survive SGLang conversion, including AWQ-folded weights."""

import json
from pathlib import Path

import pytest
import torch
from compressed_tensors.compressors.naive_quantized import FloatQuantizationCompressor
from compressed_tensors.compressors.pack_quantized import PackedQuantizationCompressor
from compressed_tensors.quantization import preset_name_to_scheme
from safetensors import safe_open
from safetensors.torch import load_file, save_file

from llmcompressor.observers import Observer
from pipeline.config import load_config
from pipeline.recipe import build_recipe
from pipeline.serve_ignore import match_name
from pipeline.to_sglang_w4afp8 import convert, source_fp8_layout
from pipeline.verify_sglang_w4afp8 import verify


@pytest.fixture
def native(tmp_path):
    torch.manual_seed(53)
    src = tmp_path / "checkpoint"
    src.mkdir()
    targets = [
        "model.layers.3.self_attn.o_proj",
        "model.layers.3.self_attn.indexer.wk",
        "model.layers.3.self_attn.indexer.wq_b",
        "model.layers.3.mlp.shared_experts.gate_proj",
    ]
    fp8 = preset_name_to_scheme("FP8_BLOCK", targets)
    expert = "model.layers.3.mlp.experts.0.gate_proj"
    int4 = preset_name_to_scheme("W4AFP8", [expert])
    tensors = {}
    for name, shape, scheme, compressor in [
        *[(n, (128 if not n.endswith('.wk') else 64, 256), fp8,
           FloatQuantizationCompressor) for n in targets],
        (expert, (128, 256), int4, PackedQuantizationCompressor),
    ]:
        # Include a nontrivial per-input-channel AWQ-style fold. Conversion
        # has no access to the original weight or the fold factor.
        weight = (torch.randn(*shape) * torch.linspace(0.5, 1.5, shape[1])).bfloat16()
        observer = Observer.load_from_registry(
            "minmax", base_name="weight", args=scheme.weights
        )
        qparams = observer(weight).get_qparams()
        compressed = compressor.compress(
            {"weight": weight, "weight_scale": qparams["scale"],
             "weight_zero_point": qparams["zero_point"]}, scheme
        )
        tensors.update({f"{name}.{k}": v.contiguous() for k, v in compressed.items()})
    unquantized = "model.layers.3.self_attn.indexer.weights_proj"
    tensors[f"{unquantized}.weight"] = torch.randn(4, 256).bfloat16()
    save_file(tensors, src / "model.safetensors")
    config = {"architectures": ["GlmMoeDsaForCausalLM"],
              "num_hidden_layers": 78, "num_nextn_predict_layers": 1,
              "quantization_config": {
                  "quant_method": "compressed-tensors", "ignore": [unquantized],
                  "config_groups": {"fp8": fp8.model_dump(mode="json"),
                                    "int4": int4.model_dump(mode="json")}}}
    (src / "config.json").write_text(json.dumps(config))
    return src, tmp_path / "converted", tensors, config, targets


def test_real_compressor_bytes_survive_without_base_or_requantization(
    native, monkeypatch
):
    src, dst, tensors, _, targets = native
    import pipeline.to_sglang_w4afp8 as converter

    def forbidden(*args, **kwargs):
        raise AssertionError("native block FP8 must not be requantized")

    monkeypatch.setattr(converter, "quantize_block_fp8", forbidden)
    assert convert(src, None, dst, shard_bytes=50000) == 0
    index = json.loads((dst / "model.safetensors.index.json").read_text())["weight_map"]
    for name in targets:
        for source_suffix, output_suffix in [
            ("weight", "weight"), ("weight_scale", "weight_scale_inv")
        ]:
            key = f"{name}.{output_suffix}"
            with safe_open(dst / index[key], framework="pt") as handle:
                got = handle.get_tensor(key)
            want = tensors[f"{name}.{source_suffix}"]
            if source_suffix == "weight":
                assert got.dtype == want.dtype == torch.float8_e4m3fn
                assert torch.equal(got.view(torch.uint8), want.view(torch.uint8))
            else:
                assert got.dtype == torch.float32
                assert torch.equal(got, want.float())
    manifest = json.loads((dst / "conversion_manifest.json").read_text())
    assert manifest["fp8_paths"] == {"preserved_block": 4, "rebuilt_from_base": 0}
    assert verify(src, dst, samples=10, all_experts=True) == 0
    # MTP absence stays explicit; this change cannot manufacture the draft head.
    output_config = json.loads((dst / "config.json").read_text())
    assert output_config["num_nextn_predict_layers"] == 0


@pytest.mark.parametrize("damage", ["weight", "scale"])
def test_verifier_rejects_even_one_changed_native_fp8_value(native, damage):
    src, dst, _, _, targets = native
    assert convert(src, None, dst) == 0
    shard = dst / "model-00001-of-00001.safetensors"
    tensors = load_file(shard)
    suffix = "weight" if damage == "weight" else "weight_scale_inv"
    tensor = tensors[f"{targets[0]}.{suffix}"]
    if damage == "weight":
        tensor.view(torch.uint8)[0, 0] ^= 1
    else:
        tensor[0, 0] = torch.nextafter(tensor[0, 0], torch.tensor(float("inf")))
    save_file(tensors, shard)
    assert verify(src, dst, samples=10, all_experts=True) == 1


@pytest.mark.parametrize("damage", ["block_size", "shape", "nan", "zero", "asymmetric",
                                    "dtype", "missing_metadata"])
def test_invalid_native_block_metadata_is_rejected(native, damage):
    src, dst, tensors, config, targets = native
    key = f"{targets[0]}.weight_scale"
    weights = config["quantization_config"]["config_groups"]["fp8"]["weights"]
    if damage == "block_size":
        weights["block_structure"] = [64, 128]
    elif damage == "shape":
        tensors[key] = torch.ones(128, 1)
    elif damage == "nan":
        tensors[key][0, 0] = float("nan")
    elif damage == "zero":
        tensors[key][0, 0] = 0
    elif damage == "asymmetric":
        weights["symmetric"] = False
    elif damage == "dtype":
        tensors[f"{targets[0]}.weight"] = tensors[f"{targets[0]}.weight"].bfloat16()
    else:
        # Retain recipe discovery but remove the actual scheme declaration.
        (src.parent / "recipe.json").write_text(
            json.dumps({"fp8_dynamic_targets": targets})
        )
        config["quantization_config"].pop("config_groups")
    save_file(tensors, src / "model.safetensors")
    (src / "config.json").write_text(json.dumps(config))
    assert convert(src, None, dst) == 2
    assert not (dst / "conversion_manifest.json").exists()


def test_shape_alone_does_not_select_block_fast_path():
    weight = torch.ones(1, 128).to(torch.float8_e4m3fn)
    scale = torch.ones(1, 1)
    assert source_fp8_layout("projection", weight, scale, {}) == "channel"


def test_glm53_recipe_assigns_indexer_fp8_without_quantizing_router_or_weights_proj():
    path = Path(__file__).parents[1] / "configs/glm53_distributed_w4afp8_awq_full.yaml"
    quant = load_config(path).quantization
    recipe = build_recipe(quant)
    assert recipe[-1].scheme == "FP8_BLOCK"
    for layer in (0, 2, 6, 30, 74, 77):
        for suffix, is_fp8 in [("self_attn.indexer.wk", True),
                               ("self_attn.indexer.wq_b", True),
                               ("self_attn.indexer.weights_proj", False),
                               ("self_attn.indexer.k_norm", False),
                               ("mlp.gate", False),
                               ("self_attn.q_a_proj", True),
                               ("mlp.shared_experts.gate_proj", True)]:
            name = f"model.layers.{layer}.{suffix}"
            assert any(match_name(name, t) for t in quant.fp8_dynamic_targets) == is_fp8
            assert any(match_name(name, t) for t in quant.ignore)
