import os
from pathlib import Path

import pytest
import torch
from compressed_tensors.utils import patch_attr
from safetensors import safe_open
from transformers import AutoModelForCausalLM
from transformers import initialization as init
from transformers.core_model_loading import WeightConverter, WeightRenaming
from transformers.models.afmoe.configuration_afmoe import AfmoeConfig
from transformers.models.afmoe.modeling_afmoe import AfmoeExperts
from transformers.models.deepseek_v3.configuration_deepseek_v3 import DeepseekV3Config


def _optional_import(module: str, name: str):
    """Fixture classes restructured by transformers 5.14 (*NaiveMoe,
    GraniteMoeParallelExperts); None marks the arch's params as skipped until
    its linearize fixture is ported to the new layout."""
    try:
        return getattr(__import__(module, fromlist=[name]), name)
    except (ImportError, AttributeError):
        return None


DeepseekV3NaiveMoe = _optional_import(
    "transformers.models.deepseek_v3.modeling_deepseek_v3", "DeepseekV3NaiveMoe"
)
from transformers.models.deepseek_v4.configuration_deepseek_v4 import DeepseekV4Config
from transformers.models.deepseek_v4.modeling_deepseek_v4 import (
    DeepseekV4Experts,
    DeepseekV4PreTrainedModel,
)
from transformers.models.gemma4.configuration_gemma4 import Gemma4TextConfig
from transformers.models.gemma4.modeling_gemma4 import Gemma4TextExperts
from transformers.models.glm4_moe.configuration_glm4_moe import Glm4MoeConfig
Glm4MoeNaiveMoe = _optional_import(
    "transformers.models.glm4_moe.modeling_glm4_moe", "Glm4MoeNaiveMoe"
)
from transformers.models.glm4_moe_lite.configuration_glm4_moe_lite import (
    Glm4MoeLiteConfig,
)
Glm4MoeLiteNaiveMoe = _optional_import(
    "transformers.models.glm4_moe_lite.modeling_glm4_moe_lite", "Glm4MoeLiteNaiveMoe"
)
from transformers.models.glm_moe_dsa.configuration_glm_moe_dsa import GlmMoeDsaConfig
GlmMoeDsaNaiveMoe = _optional_import(
    "transformers.models.glm_moe_dsa.modeling_glm_moe_dsa", "GlmMoeDsaNaiveMoe"
)
from transformers.models.gpt_oss.configuration_gpt_oss import GptOssConfig
from transformers.models.gpt_oss.modeling_gpt_oss import GptOssExperts
from transformers.models.granitemoe.configuration_granitemoe import GraniteMoeConfig
GraniteMoeParallelExperts = _optional_import(
    "transformers.models.granitemoe.modeling_granitemoe", "GraniteMoeParallelExperts"
)
from transformers.models.hy_v3.configuration_hy_v3 import HYV3Config
from transformers.models.hy_v3.modeling_hy_v3 import HYV3Experts
from transformers.models.llama4.configuration_llama4 import (
    Llama4Config,
    Llama4TextConfig,
)
from transformers.models.llama4.modeling_llama4 import Llama4TextExperts
from transformers.models.nemotron_h.configuration_nemotron_h import NemotronHConfig
from transformers.models.nemotron_h.modeling_nemotron_h import NemotronHExperts
from transformers.models.qwen3_5_moe.configuration_qwen3_5_moe import (
    Qwen3_5MoeTextConfig,
)
from transformers.models.qwen3_5_moe.modeling_qwen3_5_moe import Qwen3_5MoeExperts
from transformers.models.qwen3_moe.configuration_qwen3_moe import Qwen3MoeConfig
from transformers.models.qwen3_moe.modeling_qwen3_moe import Qwen3MoeExperts
from transformers.models.qwen3_next.configuration_qwen3_next import Qwen3NextConfig
from transformers.models.qwen3_next.modeling_qwen3_next import Qwen3NextExperts
from transformers.models.qwen3_vl_moe.configuration_qwen3_vl_moe import (
    Qwen3VLMoeTextConfig,
)
from transformers.models.qwen3_vl_moe.modeling_qwen3_vl_moe import Qwen3VLMoeTextExperts

from llmcompressor.modeling.moe.context import (
    moe_calibration_context,
)
from llmcompressor.modeling.moe.conversion_mappings import (
    ARCH_TO_2D_MAPPINGS,
    ARCH_TO_EXPERTS_MODULE_CLS,
    get_linearize_load_mappings,
    has_linearize_load_mappings,
)
from llmcompressor.modeling.moe.helpers import (
    FusedExpertsProtocol,
    MoEConfig,
    _getattr_fallbacks,
)
from llmcompressor.modeling.moe.linearize import linearize_moe, load_quantizable_moe
from tests.testing_utils import requires_gpu

NUM_TEST_TOKENS = 64
MODEL_MSE = 1e-2
MODULE_MSE = 1e-10


def test_minimax_m3_load_mapping_keeps_experts_two_dimensional():
    experts_cls, load_mappings, save_mappings = get_linearize_load_mappings(
        "minimax_m3_vl"
    )

    assert experts_cls.__name__ == "MiniMaxM3VLExperts"
    for mappings in (load_mappings, save_mappings):
        assert not any(
            isinstance(mapping, WeightConverter)
            and any(
                target in {"mlp.experts.gate_up_proj", "mlp.experts.down_proj"}
                for target in mapping.target_patterns
            )
            for mapping in mappings
        )

    expert_renames = [
        mapping
        for mapping in load_mappings
        if isinstance(mapping, WeightRenaming)
        and any("mlp.experts" in target for target in mapping.target_patterns)
    ]
    expected_targets = {
        r"mlp.experts.\1.gate_proj.",
        r"mlp.experts.\1.down_proj.",
        r"mlp.experts.\1.up_proj.",
    }
    assert expected_targets <= {
        target for mapping in expert_renames for target in mapping.target_patterns
    }


@pytest.mark.parametrize(
    "model_type",
    sorted(
        set(ARCH_TO_EXPERTS_MODULE_CLS)
        | set(ARCH_TO_2D_MAPPINGS)
        # architectures that reach a registered 2D mapping only by alias — these
        # are the ones the guard used to wave through into a KeyError
        | {"glm_moe_dsa", "glm4_moe", "glm4_moe_lite", "glm4v_moe", "qwen3_moe"}
    ),
)
def test_linearize_guard_never_promises_what_the_getter_cannot_deliver(model_type):
    """
    `has_linearize_load_mappings` gates the fast path in `load_quantizable_moe`.
    If it returns True, `get_linearize_load_mappings` must succeed — otherwise the
    post-load fallback is skipped and `from_pretrained` raises `KeyError` before
    any calibration work. `ARCH_TO_EXPERTS_MODULE_CLS` is keyed by raw model_type
    and `ARCH_TO_2D_MAPPINGS` by conversion pattern, so checking only one of them
    breaks every architecture that reaches a mapping by alias.
    """
    if not has_linearize_load_mappings(model_type):
        pytest.skip(f"{model_type} correctly routes to the post-load fallback")

    experts_cls, load_mappings, save_mappings = get_linearize_load_mappings(model_type)
    assert experts_cls is not None
    assert isinstance(load_mappings, list)
    assert isinstance(save_mappings, list)


def test_glm_moe_dsa_load_mapping_keeps_experts_two_dimensional():
    """
    GLM-5.x DSA presents the same expert contract as MiniMax-M3, so it must
    linearize through the same fast path. Its 2D mappings are inherited from
    `qwen2_moe` by alias, which is why the guard has to agree with the getter.
    """
    assert has_linearize_load_mappings("glm_moe_dsa")
    experts_cls, load_mappings, save_mappings = get_linearize_load_mappings(
        "glm_moe_dsa"
    )

    assert experts_cls.__name__ == "GlmMoeDsaNaiveMoe"
    for mappings in (load_mappings, save_mappings):
        assert not any(
            isinstance(mapping, WeightConverter)
            and any(
                target in {"mlp.experts.gate_up_proj", "mlp.experts.down_proj"}
                for target in mapping.target_patterns
            )
            for mapping in mappings
        )

    expert_renames = [
        mapping
        for mapping in load_mappings
        if isinstance(mapping, WeightRenaming)
        and any("mlp.experts" in target for target in mapping.target_patterns)
    ]
    # qwen2_moe's patterns capture both the layer index and the expert index
    expected_targets = {
        r"layers.\1.mlp.experts.\2.gate_proj.",
        r"layers.\1.mlp.experts.\2.down_proj.",
        r"layers.\1.mlp.experts.\2.up_proj.",
    }
    assert expected_targets <= {
        target for mapping in expert_renames for target in mapping.target_patterns
    }


@pytest.mark.skipif(
    GlmMoeDsaNaiveMoe is None, reason="GlmMoeDsaNaiveMoe not in this transformers"
)
def test_glm_moe_dsa_matches_minimax_m3_expert_contract():
    """
    The registration above is only valid because GLM's expert module is
    contract-identical to M3's. If transformers ever changes either, this fails
    here rather than in a quantization run.
    """
    from llmcompressor.modeling.moe.helpers import (
        get_use_experts_implementation_args,
    )

    glm_args = get_use_experts_implementation_args(GlmMoeDsaNaiveMoe)
    m3_args = get_use_experts_implementation_args(
        ARCH_TO_EXPERTS_MODULE_CLS["minimax_m3_vl"]
    )
    assert glm_args == m3_args, f"GLM {glm_args} != M3 {m3_args}"
    assert glm_args == {
        "is_concatenated": True,
        "is_transposed": False,
        "has_bias": False,
        "has_gate": True,
    }


@pytest.fixture
def patch_deepseek_fp32_modules():
    """
    Monkey patch to force DeepseekV4 models to load in bfloat16.

    BUG: norms should be loaded in float32, but usually aren't due to the base
    model having a quant_config which overrides this. Loading in float32 actually
    breaks the model definition (it expects bfloat16). Let's force load in bfloat16.
    """
    with patch_attr(DeepseekV4PreTrainedModel, "_keep_in_fp32_modules_strict", set()):
        yield


@torch.no_grad()
@requires_gpu
@pytest.mark.parametrize(
    "model_stub,exp_keys",
    [
        (
            "inference-optimization/DSV4-tiny-empty",
            [
                "model.layers.0.mlp.experts.2.up_proj.weight",
                "model.layers.1.mlp.experts.0.gate_proj.weight",
                "model.layers.2.mlp.experts.1.down_proj.weight",
            ],
        ),
        (
            "inference-optimization/Qwen3-1.6B-A0.9B",
            [
                "model.layers.0.mlp.experts.2.up_proj.weight",
                "model.layers.1.mlp.experts.0.gate_proj.weight",
                "model.layers.2.mlp.experts.1.down_proj.weight",
            ],
        ),
    ],
)
def test_load_quantizable_moe(
    model_stub, exp_keys, tmp_path, patch_deepseek_fp32_modules
):
    input_ids = torch.randint(1024, size=(1, NUM_TEST_TOKENS), device="cuda")
    model = AutoModelForCausalLM.from_pretrained(model_stub, device_map="cuda")
    true_outputs = model(input_ids=input_ids).logits
    del model

    with load_quantizable_moe():
        model2 = AutoModelForCausalLM.from_pretrained(model_stub, device_map="cuda")

    select_exp_outputs = model2(input_ids=input_ids).logits

    with moe_calibration_context():
        all_exp_outputs = model2(input_ids=input_ids).logits

    assert torch.any(true_outputs != 0), "Bad test setup, output is all zeros"
    assert torch.nn.functional.mse_loss(true_outputs, select_exp_outputs) < MODEL_MSE
    assert torch.nn.functional.mse_loss(true_outputs, all_exp_outputs) < MODEL_MSE

    save_dir = tmp_path / "save_path"
    os.mkdir(save_dir)
    model2.save_pretrained(save_dir)
    assert keys_exist(save_dir, exp_keys)


def keys_exist(model_path: Path, keys: list[str]) -> bool:
    """
    Utility to check that expected expert keys exist in a saved model.

    Args:
        model_path: Path to the saved model directory
        expected_patterns: List of key patterns to check for

    Returns:
        True if all expected patterns are found in the model checkpoint
    """
    safetensor_files = list(model_path.glob("*.safetensors"))
    all_keys = set()
    keys = set(keys)

    for st_file in safetensor_files:
        with safe_open(st_file, framework="pt", device="cpu") as f:
            all_keys.update(f.keys())

    return keys <= all_keys


class DummyModel(torch.nn.Module):
    def __init__(self, module, config):
        super().__init__()
        self.config = config
        self.module = module

    def forward(self, *args, **kwargs):
        return self.module(*args, **kwargs)


def _param_if_available(config_cls, experts_cls, kwargs):
    """Skip params whose transformers fixture class was restructured away
    (5.14 *NaiveMoe removals) until the linearize fixture is ported."""
    return pytest.param(
        config_cls,
        experts_cls,
        kwargs,
        marks=pytest.mark.skipif(
            experts_cls is None,
            reason="fixture class removed by installed transformers (>=5.14)",
        ),
    )


@torch.no_grad()
@requires_gpu
@pytest.mark.parametrize(
    "config_cls,experts_cls,kwargs",
    [
        (AfmoeConfig, AfmoeExperts, {}),
        _param_if_available(
            DeepseekV3Config,
            DeepseekV3NaiveMoe,
            {"hidden_size": 512, "moe_intermediate_size": 1024},
        ),
        (DeepseekV4Config, DeepseekV4Experts, {}),
        (
            Gemma4TextConfig,
            Gemma4TextExperts,
            {"num_experts": 16, "top_k_experts": 4, "moe_intermediate_size": 2304},
        ),
        _param_if_available(Glm4MoeConfig, Glm4MoeNaiveMoe, {}),
        _param_if_available(Glm4MoeLiteConfig, Glm4MoeLiteNaiveMoe, {}),
        _param_if_available(GlmMoeDsaConfig, GlmMoeDsaNaiveMoe, {"hidden_size": 512}),
        (Qwen3_5MoeTextConfig, Qwen3_5MoeExperts, {}),
        (Qwen3MoeConfig, Qwen3MoeExperts, {}),
        (Qwen3NextConfig, Qwen3NextExperts, {}),
        (Qwen3VLMoeTextConfig, Qwen3VLMoeTextExperts, {}),
        (GptOssConfig, GptOssExperts, {}),
        (HYV3Config, HYV3Experts, {}),
        (
            NemotronHConfig,
            NemotronHExperts,
            {"hidden_size": 32, "moe_intermediate_size": 64},
        ),
    ],
)
def test_linearize_moe(config_cls, experts_cls, kwargs):
    with torch.device("cuda"):
        config = config_cls(**kwargs)
        experts = experts_cls(config)
        assert isinstance(experts, FusedExpertsProtocol)
        up_proj = _getattr_fallbacks(experts, ["gate_up_proj", "up_proj"])
        init.normal_(up_proj, mean=0.0, std=config.initializer_range)
        init.normal_(experts.down_proj, mean=0.0, std=config.initializer_range)

        mock_model = DummyModel(experts, config)
        linearize_moe(mock_model)
        assert mock_model.module is not experts

        moe_config = MoEConfig.from_config(config)
        hidden_states = torch.randn(
            NUM_TEST_TOKENS, moe_config.hidden_dim, dtype=moe_config.dtype
        )
        top_k_index = torch.randint(
            0,
            moe_config.num_experts,
            size=(NUM_TEST_TOKENS, moe_config.num_experts_per_tok),
        )
        top_k_weights = torch.randn(
            NUM_TEST_TOKENS, moe_config.num_experts_per_tok, dtype=moe_config.dtype
        )
        true_outputs = experts(hidden_states, top_k_index, top_k_weights)
        outputs = mock_model(hidden_states, top_k_index, top_k_weights)
        with moe_calibration_context():
            calib_outputs = mock_model(hidden_states, top_k_index, top_k_weights)

        assert torch.any(true_outputs != 0), "Bad test setup, output is all zeros"
        assert torch.nn.functional.mse_loss(outputs, true_outputs) < MODULE_MSE
        assert torch.nn.functional.mse_loss(calib_outputs, true_outputs) < MODULE_MSE


@pytest.mark.skipif(
    GraniteMoeParallelExperts is None,
    reason="transformers>=5.14 fused GraniteMoeExperts layout not ported yet",
)
def test_linearize_moe_granite():
    config = GraniteMoeConfig(hidden_size=512, intermediate_size=1024)
    experts = GraniteMoeParallelExperts(
        config.num_local_experts, config.hidden_size, config.intermediate_size
    )
    init.normal_(experts.weight, mean=0.0, std=config.initializer_range)

    mock_model = DummyModel(experts, config)
    linearize_moe(mock_model)
    assert mock_model.module is not experts

    hidden_states = torch.randn(NUM_TEST_TOKENS, config.hidden_size, dtype=config.dtype)
    expert_size = [
        (NUM_TEST_TOKENS // config.num_local_experts)
        for _ in range(config.num_local_experts)
    ]
    expert_size[-1] += NUM_TEST_TOKENS % config.num_local_experts
    true_outputs = experts(hidden_states, expert_size)
    outputs = mock_model(hidden_states, expert_size)
    with moe_calibration_context():
        calib_outputs = mock_model(hidden_states, expert_size)

    assert torch.any(true_outputs != 0), "Bad test setup, output is all zeros"
    assert torch.nn.functional.mse_loss(outputs, true_outputs) < MODULE_MSE
    assert torch.nn.functional.mse_loss(calib_outputs, true_outputs) < MODULE_MSE


def test_linearize_moe_llama4():
    text_config = Llama4TextConfig(hidden_size=512, intermediate_size=1024)
    config = Llama4Config(text_config=text_config)
    experts = Llama4TextExperts(config.text_config)
    init.normal_(experts.gate_up_proj, mean=0.0, std=text_config.initializer_range)
    init.normal_(experts.down_proj, mean=0.0, std=text_config.initializer_range)

    mock_model = DummyModel(experts, config)
    linearize_moe(mock_model)
    assert mock_model.module is not experts

    moe_config = MoEConfig.from_config(text_config)
    hidden_states = torch.randn(
        NUM_TEST_TOKENS, moe_config.hidden_dim, dtype=moe_config.dtype
    )
    true_outputs = experts(hidden_states)
    outputs = mock_model(hidden_states)
    with moe_calibration_context():
        calib_outputs = mock_model(hidden_states)

    assert torch.any(true_outputs != 0), "Bad test setup, output is all zeros"
    assert torch.nn.functional.mse_loss(outputs, true_outputs) < MODULE_MSE
    assert torch.nn.functional.mse_loss(calib_outputs, true_outputs) < MODULE_MSE
