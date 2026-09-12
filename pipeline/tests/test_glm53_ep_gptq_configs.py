from pathlib import Path

import pytest

from pipeline.config import load_config

ROOT = Path(__file__).resolve().parents[2]
REPRESENTATIVE = ROOT / "pipeline/configs/glm53_ep_gptq_representative.yaml"
FULL = ROOT / "pipeline/configs/glm53_ep_gptq_full.yaml"


def _assert_expert_only(cfg):
    assert cfg.quantization.method == "gptq"
    assert cfg.quantization.scheme == "W4A16"
    assert cfg.quantization.gptq_expert_parallel is True
    assert cfg.quantization.gptq_offload_hessians is False
    assert cfg.quantization.fp8_dynamic_targets == []
    assert cfg.quantization.sample_generation is False
    ignore = "\n".join(cfg.quantization.ignore)
    for protected in (
        "lm_head",
        "mlp[.]gate",
        "mlp[.]shared_experts",
        "self_attn",
        "layers[.][0-2]",
        "layers[.]78",
    ):
        assert protected in ignore
    assert cfg.calibration.moe_calibrate_all_experts is True
    assert cfg.calibration.sequential_targets == ["GlmMoeDsaDecoderLayer"]
    assert cfg.calibration.pipeline == "sequential"
    assert cfg.serve.enabled is False
    assert cfg.eval.enabled is False


def test_representative_config_is_fixed_real_width_gate():
    cfg = load_config(REPRESENTATIVE)
    _assert_expert_only(cfg)
    assert cfg.model.id == "/scratch/glm53-bf16-layers-0-4"
    assert cfg.calibration.dataset_id == "json"
    assert (
        cfg.calibration.dataset_data_files
        == "pipeline/fixtures/glm53_ep_gptq_representative.jsonl"
    )
    assert cfg.calibration.dataset_split == "train"
    assert cfg.calibration.num_samples == 8
    assert cfg.calibration.max_seq_length == 512


def test_full_config_pins_source_and_production_calibration():
    cfg = load_config(FULL)
    _assert_expert_only(cfg)
    assert cfg.model.id.endswith(
        "models--zai-org--GLM-5.3-BF16/snapshots/"
        "304b8051cfb2b260b61ce0cbe330e02a98e73639"
    )
    assert cfg.calibration.dataset_id == "HuggingFaceH4/ultrachat_200k"
    assert cfg.calibration.dataset_data_files is None
    assert cfg.calibration.dataset_split == "train_sft"
    assert cfg.calibration.num_samples == 256
    assert cfg.calibration.max_seq_length == 2048
    assert cfg.calibration.seed == 42


def test_w4afp8_recipes_prepare_weights_before_gptq_and_export_natively():
    from pipeline.recipe import build_recipe, describe_recipe

    for lane in ("full", "representative"):
        cfg = load_config(ROOT / f"pipeline/configs/glm53_ep_gptq_w4afp8_{lane}.yaml")
        assert cfg.quantization.scheme == "W4AFP8"
        assert cfg.quantization.checkpoint_format == "sglang-w4afp8"
        assert cfg.quantization.fp8_weights_before_gptq
        assert cfg.quantization.gptq_expert_parallel
        assert cfg.quantization.fp8_scheme == "FP8_BLOCK"
        assert any(
            "indexer" in target for target in cfg.quantization.fp8_dynamic_targets
        )
        main, rest = build_recipe(cfg.quantization)
        assert main.expert_parallel
        assert rest.quantize_weights_before_calibration
        assert describe_recipe(cfg.quantization)["fp8_weights_before_gptq"] is True


def test_preparation_configuration_rejects_unsupported_composition():
    from pipeline.config import ModelConfig, PipelineConfig, QuantizationConfig

    for overrides in (
        {"method": "awq"},
        {"fp8_scheme": "FP8_DYNAMIC"},
        {"fp8_dynamic_targets": []},
    ):
        kwargs = dict(
            method="gptq",
            fp8_scheme="FP8_BLOCK",
            fp8_dynamic_targets=["attention"],
            fp8_weights_before_gptq=True,
        )
        kwargs.update(overrides)
        cfg = PipelineConfig(
            model=ModelConfig(id="local"), quantization=QuantizationConfig(**kwargs)
        )
        with pytest.raises(ValueError, match="fp8_weights_before_gptq"):
            cfg.validate()


def test_full_awq_w4afp8_recipe_exports_natively_without_gptq_preparation():
    from pipeline.recipe import build_recipe

    cfg = load_config(
        ROOT / "pipeline/configs/glm53_distributed_w4afp8_awq_full.yaml"
    )
    assert cfg.quantization.method == "awq"
    assert cfg.quantization.scheme == "W4AFP8"
    assert cfg.quantization.fp8_scheme == "FP8_BLOCK"
    assert cfg.quantization.fp8_dynamic_targets
    assert cfg.quantization.fp8_weights_before_gptq is False
    assert cfg.quantization.checkpoint_format == "sglang-w4afp8"
    cfg.validate()
    awq, quant, fp8 = build_recipe(cfg.quantization)
    assert type(awq).__name__ == "AWQModifier"
    assert type(quant).__name__ == "QuantizationModifier"
    assert type(fp8).__name__ == "QuantizationModifier"


@pytest.mark.parametrize(
    "overrides",
    [
        {"method": "quant_only"},
        {"method": "smoothquant+awq"},
        {"method": "gptq", "fp8_weights_before_gptq": False},
        {"scheme": "W4A16"},
        {"fp8_scheme": "FP8_DYNAMIC"},
        {"fp8_dynamic_targets": []},
    ],
)
def test_native_export_rejects_unsupported_composition(overrides):
    from pipeline.config import ModelConfig, PipelineConfig, QuantizationConfig

    kwargs = dict(
        method="awq",
        scheme="W4AFP8",
        fp8_scheme="FP8_BLOCK",
        fp8_dynamic_targets=["attention"],
        checkpoint_format="sglang-w4afp8",
    )
    kwargs.update(overrides)
    cfg = PipelineConfig(
        model=ModelConfig(id="local"), quantization=QuantizationConfig(**kwargs)
    )
    with pytest.raises(ValueError, match="plain AWQ or GPTQ"):
        cfg.validate()
