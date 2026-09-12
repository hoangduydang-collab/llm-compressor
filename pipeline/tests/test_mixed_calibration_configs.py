"""Guard the native recipes while offering a different calibration source."""

from dataclasses import asdict
from pathlib import Path

import pytest
import yaml

from pipeline.config import load_config

CONFIGS = Path(__file__).resolve().parents[1] / "configs"


@pytest.mark.parametrize(
    "base_name",
    ["glm53_ep_gptq_w4afp8_full", "glm53_distributed_w4afp8_awq_full"],
)
def test_mixed_configs_preserve_quantization_and_isolate_artifacts(base_name):
    base = load_config(CONFIGS / f"{base_name}.yaml")
    mixed = load_config(CONFIGS / f"{base_name}_mixed.yaml")
    assert asdict(mixed.quantization) == asdict(base.quantization)
    assert mixed.quantization.checkpoint_format == "sglang-w4afp8"
    assert mixed.quantization.mtp_policy == "source-rtn"
    assert mixed.name != base.name
    assert mixed.output_dir != base.output_dir
    assert mixed.model.offload_folder != base.model.offload_folder
    expected_calibration = asdict(base.calibration)
    expected_calibration["prepared_dataset"] = mixed.calibration.prepared_dataset
    assert asdict(mixed.calibration) == expected_calibration
    assert mixed.calibration.prepared_dataset
    assert base.calibration.prepared_dataset is None


def test_both_methods_consume_same_budget_and_bundle():
    configs = [
        load_config(CONFIGS / f"{name}_mixed.yaml")
        for name in (
            "glm53_ep_gptq_w4afp8_full",
            "glm53_distributed_w4afp8_awq_full",
        )
    ]
    preparation = yaml.safe_load(
        (CONFIGS / "calibration/glm53_generic_agentic_mix.yaml").read_text()
    )
    assert (
        configs[0].calibration.prepared_dataset
        == configs[1].calibration.prepared_dataset
    )
    for cfg in configs:
        assert cfg.calibration.num_samples == preparation["num_samples"]
        assert cfg.calibration.max_seq_length == preparation["max_seq_length"]
        assert cfg.calibration.seed == preparation["seed"]
    assert len(preparation["sources"]) == 2
    assert all(
        len(source["dataset_revision"]) == 40 for source in preparation["sources"]
    )
