from pathlib import Path

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
