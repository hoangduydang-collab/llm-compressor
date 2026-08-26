import torch
from loguru import logger
from transformers import PreTrainedModel
from transformers.conversion_mapping import (
    _MODEL_TO_CONVERSION_PATTERN,
    get_checkpoint_conversion_mapping,
)
from transformers.core_model_loading import (
    WeightConverter,
    WeightRenaming,
    WeightTransform,
)
from transformers.models.deepseek_v4.modeling_deepseek_v4 import DeepseekV4Experts
from transformers.models.minimax_m3_vl.modeling_minimax_m3_vl import (
    MiniMaxM3VLExperts,
)
from transformers.models.qwen2_moe.modeling_qwen2_moe import Qwen2MoeExperts
from transformers.models.qwen3_moe.modeling_qwen3_moe import Qwen3MoeExperts

__all__ = [
    "has_linearize_load_mappings",
    "get_linearize_load_mappings",
    "set_save_conversion_mapping",
]


def _first_class(module_path: str, *candidates: str) -> type | None:
    """
    Resolve the first of `candidates` that exists in `module_path`, or None.

    transformers renames experts classes between minor versions — GLM's became
    `GlmMoeDsaExperts` in 5.14 having been `GlmMoeDsaNaiveMoe` in 5.12 — and this
    module is imported on every quantization run. A hard `from ... import` of the
    wrong spelling is therefore not a missing-GLM-support problem, it is an
    ImportError that takes the whole MoE linearize path down, MiniMax-M3
    included. Resolve leniently and let the registry simply omit the entry if no
    spelling matches; `has_linearize_load_mappings` then routes that
    architecture to the post-load fallback, which is the correct degradation.
    """
    import importlib

    try:
        module = importlib.import_module(module_path)
    except ImportError:  # architecture absent from this transformers
        return None
    for name in candidates:
        cls = getattr(module, name, None)
        if cls is not None:
            return cls
    return None


# GLM-5.x DSA. `GlmMoeDsaNaiveMoe` on transformers 5.12, `GlmMoeDsaExperts` from
# 5.14 (which is what the quant venv pins) — accept either spelling.
_GlmMoeDsaExperts = _first_class(
    "transformers.models.glm_moe_dsa.modeling_glm_moe_dsa",
    "GlmMoeDsaExperts",
    "GlmMoeDsaNaiveMoe",
)

# Keyed by the model's own `config.model_type`, *not* by conversion pattern —
# unlike `ARCH_TO_2D_MAPPINGS` below, which is keyed by conversion pattern. Keep
# `has_linearize_load_mappings` in agreement with both or the fast path is entered
# for architectures it cannot serve (see the note on that function).
# TODO: in the future, we can potentially grep the source code for this
ARCH_TO_EXPERTS_MODULE_CLS = {
    "deepseek_v4": DeepseekV4Experts,
    "minimax_m3_vl": MiniMaxM3VLExperts,
    "qwen2_moe": Qwen2MoeExperts,
    "qwen3_moe": Qwen3MoeExperts,
}

# GLM-5.x DSA (`GlmMoeDsaForCausalLM`). Its expert module presents exactly the
# same contract as MiniMax-M3's — same `@use_experts_implementation` arguments
# (is_concatenated/has_gate True, is_transposed/has_bias False) and the same
# `gate_up_proj[E, 2I, H]` / `down_proj[E, H, I]` layout — so it linearizes
# through the identical path. The 2D mappings are inherited from `qwen2_moe` via
# `_MODEL_TO_CONVERSION_PATTERN`, which matches the per-expert
# `gate_proj`/`up_proj`/`down_proj` layout GLM ships in its checkpoints.
# Registered conditionally so an unknown future spelling degrades to the
# post-load fallback rather than breaking module import.
if _GlmMoeDsaExperts is not None:
    ARCH_TO_EXPERTS_MODULE_CLS["glm_moe_dsa"] = _GlmMoeDsaExperts

ARCH_TO_2D_MAPPINGS = {
    "deepseek_v4": (
        ["mlp.experts.gate_up_proj", "mlp.experts.down_proj"],
        [
            WeightRenaming(
                source_patterns=r"^layers\.(\d+)\.mlp\.experts\.(\d+)\.w1\.",
                target_patterns=r"layers.\1.mlp.experts.\2.gate_proj.",
            ),
            WeightRenaming(
                source_patterns=r"^layers\.(\d+)\.mlp\.experts\.(\d+)\.w2\.",
                target_patterns=r"layers.\1.mlp.experts.\2.down_proj.",
            ),
            WeightRenaming(
                source_patterns=r"^layers\.(\d+)\.mlp\.experts\.(\d+)\.w3\.",
                target_patterns=r"layers.\1.mlp.experts.\2.up_proj.",
            ),
        ],
    ),
    "minimax_m3_vl": (
        ["mlp.experts.gate_up_proj", "mlp.experts.down_proj"],
        [
            WeightRenaming(
                source_patterns=r"mlp\.experts\.(\d+)\.w1\.",
                target_patterns=r"mlp.experts.\1.gate_proj.",
            ),
            WeightRenaming(
                source_patterns=r"mlp\.experts\.(\d+)\.w2\.",
                target_patterns=r"mlp.experts.\1.down_proj.",
            ),
            WeightRenaming(
                source_patterns=r"mlp\.experts\.(\d+)\.w3\.",
                target_patterns=r"mlp.experts.\1.up_proj.",
            ),
        ],
    ),
    "qwen2_moe": (
        ["mlp.experts.gate_up_proj", "mlp.experts.down_proj"],
        [
            WeightRenaming(
                source_patterns=r"^layers\.(\d+)\.mlp\.experts\.(\d+)\.gate_proj\.",
                target_patterns=r"layers.\1.mlp.experts.\2.gate_proj.",
            ),
            WeightRenaming(
                source_patterns=r"^layers\.(\d+)\.mlp\.experts\.(\d+)\.up_proj\.",
                target_patterns=r"layers.\1.mlp.experts.\2.up_proj.",
            ),
            WeightRenaming(
                source_patterns=r"^layers\.(\d+)\.mlp\.experts\.(\d+)\.down_proj\.",
                target_patterns=r"layers.\1.mlp.experts.\2.down_proj.",
            ),
        ],
    ),
}


def has_linearize_load_mappings(model_type: str) -> bool:
    """
    Whether `get_linearize_load_mappings` can serve this model type, i.e. whether
    the direct 2D-load fast path is available.

    Both lookups that `get_linearize_load_mappings` performs must be checked, and
    they are keyed differently: `ARCH_TO_EXPERTS_MODULE_CLS` by the raw
    `model_type`, `ARCH_TO_2D_MAPPINGS` by its conversion pattern. Testing only
    the latter reports True for any architecture that reaches a registered
    pattern by alias while having no experts class of its own — every GLM MoE
    variant aliases to `qwen2_moe` this way. `load_quantizable_moe` would then
    skip its post-load fallback and raise `KeyError` from inside
    `from_pretrained`, before any calibration work.

    :param model_type: the model's `config.model_type`
    :return: True if the fast path is fully registered for this model type
    """
    if model_type not in ARCH_TO_EXPERTS_MODULE_CLS:
        return False
    return _MODEL_TO_CONVERSION_PATTERN.get(model_type, model_type) in (
        ARCH_TO_2D_MAPPINGS
    )


def get_linearize_load_mappings(
    model_type: str,
) -> tuple[type[torch.nn.Module], list[WeightTransform], list[WeightTransform]]:
    """ """
    experts_cls = ARCH_TO_EXPERTS_MODULE_CLS[model_type]
    model_type = _MODEL_TO_CONVERSION_PATTERN.get(model_type, model_type)

    mapping: list[WeightTransform] = get_checkpoint_conversion_mapping(model_type)
    remove_targets, new_mappings = ARCH_TO_2D_MAPPINGS[model_type]

    # forwards has conversion mappings
    # backwards has no mappings (stay 2d)
    save_mappings = [
        converter
        for converter in mapping
        if not any(target in remove_targets for target in converter.target_patterns)
    ]
    load_mappings = save_mappings + new_mappings

    # validate that no transforms occur during loading/saving
    for converter in load_mappings:
        if isinstance(converter, WeightConverter):
            logger.warning(
                "Linearized model performs a weight conversion during loading. This "
                f"may lead to longer load times\n{converter}"
            )
    for converter in save_mappings:
        if isinstance(converter, WeightConverter):
            logger.warning(
                "Linearized model performs a weight conversion during saving. This "
                f"may lead to longer save times\n{converter}"
            )

    return experts_cls, load_mappings, save_mappings


def set_save_conversion_mapping(
    model: PreTrainedModel, save_mappings: list[WeightTransform]
):
    """
    Set the conversion mappings used when saving the model. The inverse of these
    mappings will be applied to the model during saving via
    `transformers.core_model_loading.py::revert_weight_conversion`.

    :param model: model to override conversion mapping of
    :param save_mappings: mappings to override with
    """
    model._weight_conversions = save_mappings
