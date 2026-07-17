import torch
from transformers import initialization as init
from transformers.models.qwen3_moe.configuration_qwen3_moe import Qwen3MoeConfig
from transformers.models.qwen3_moe.modeling_qwen3_moe import Qwen3MoeExperts

from llmcompressor.modeling.moe.context import moe_calibration_context
from llmcompressor.modeling.moe.helpers import MoEConfig
from llmcompressor.modeling.moe.linear_experts import (
    LinearExperts2D,
    _carry_over_gate_scalars,
)


def test_carry_over_gate_scalars():
    """Custom `_apply_gate` scalars (e.g. MiniMax-M3 swiglu params) survive linearize.

    The generic linearized experts reuse the source module's `_apply_gate`, which may
    read config-derived scalars off `self`. `from_experts_module` must copy those scalars
    so calibration does not hit `AttributeError`, while leaving structural fields and
    parameters/buffers/submodules untouched.
    """

    class _Source(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.num_experts = 8  # structural: must NOT overwrite target's value
            self.swiglu_limit = 7.0  # custom gate scalar: must be copied
            self.swiglu_alpha = 1.702  # custom gate scalar: must be copied
            self.use_bias = False  # bool scalar: must be copied
            self.name = "src"  # non-scalar: must be ignored
            self.weight = torch.nn.Parameter(torch.zeros(2))  # param: ignored

    class _Target(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.num_experts = 4  # pre-existing: must be preserved

    source, target = _Source(), _Target()
    _carry_over_gate_scalars(target, source)

    assert target.swiglu_limit == 7.0
    assert target.swiglu_alpha == 1.702
    assert target.use_bias is False
    assert target.num_experts == 4  # not clobbered by source's 8
    assert not hasattr(target, "name")  # non-scalar skipped
    assert "weight" not in target._parameters  # parameter not copied


@torch.no_grad()
def test_linear_experts_2d_with_hooks():
    """
    Test LinearExperts2D with forward hooks to verify calibration context behavior.

    This test verifies that:
    1. Outside moe_calibration_context: only selected tokens are sent to each expert
    2. Inside moe_calibration_context: all tokens are sent to all experts
    """
    # Create a Qwen3MoeConfig
    config = Qwen3MoeConfig(
        hidden_size=16,
        intermediate_size=32,
        num_experts=4,
        num_experts_per_tok=2,
    )

    # Get the LinearExperts2D class for Qwen3MoeExperts
    linear_experts_cls = LinearExperts2D.get_linear_experts_cls(Qwen3MoeExperts)

    # Create LinearExperts2D instance
    linear_experts = linear_experts_cls(config)

    # Initialize weights to avoid NaN/Inf
    moe_config = MoEConfig.from_config(config)
    for expert_idx in range(linear_experts.num_experts):
        expert = linear_experts[expert_idx]
        init.normal_(expert.up_proj.weight, mean=0.0, std=config.initializer_range)
        init.normal_(expert.gate_proj.weight, mean=0.0, std=config.initializer_range)
        init.normal_(expert.down_proj.weight, mean=0.0, std=config.initializer_range)

    # Create hook counters to track input shapes for each expert
    expert_num_tokens = dict()

    def make_hook(expert_idx):
        def hook(module, input, output):
            num_tokens = input[0].size(0)
            expert_num_tokens[expert_idx] = num_tokens

        return hook

    # Register hooks on each expert
    hooks = []
    for expert_idx in range(linear_experts.num_experts):
        expert = linear_experts[expert_idx]
        hook_handle = expert.register_forward_hook(make_hook(expert_idx))
        hooks.append(hook_handle)

    # Create test inputs
    num_tokens = 16
    hidden_states = torch.randn(
        num_tokens, moe_config.hidden_dim, dtype=moe_config.dtype
    )

    # Create routing: each token goes to 2 experts (top_k=2)
    # Make sure not all tokens go to all experts
    top_k_index = torch.tensor([[0], [1], [2], [3]])
    top_k_weights = torch.randn(
        num_tokens, moe_config.num_experts_per_tok, dtype=moe_config.dtype
    )

    # Test 1: Forward pass WITHOUT calibration context
    expert_num_tokens = dict()
    output_normal = linear_experts(hidden_states, top_k_index, top_k_weights)

    # Verify that not all tokens went to all experts (1 token, see top_k_index)
    for expert_idx in range(moe_config.num_experts):
        input_size = expert_num_tokens[expert_idx]
        assert input_size == 1, (
            f"Without calibration context, expert {expert_idx} should receive "
            f"exactly 1 token, but received {input_size}"
        )

    # Test 2: Forward pass WITH calibration context
    expert_num_tokens = dict()
    with moe_calibration_context():
        output_calib = linear_experts(hidden_states, top_k_index, top_k_weights)

    # Verify that all tokens went to all experts
    for expert_idx in range(moe_config.num_experts):
        input_size = expert_num_tokens[expert_idx]
        assert input_size == num_tokens, (
            f"With calibration context, expert {expert_idx} should receive "
            f"all {num_tokens} tokens, but received {input_size}"
        )

    # Test 3: Verify outputs are valid tensors (not checking exact values since
    # calibration mode changes computation by passing all tokens through experts)
    assert output_normal.shape == hidden_states.shape
    assert output_calib.shape == hidden_states.shape
    assert not torch.isnan(output_normal).any()
    assert not torch.isnan(output_calib).any()

    # Clean up hooks
    for hook in hooks:
        hook.remove()


@torch.no_grad()
def test_linear_experts_2d_init_carries_source_gate_scalars():
    """Load-time-path regression (distributed r6): when Transformers constructs
    the dynamic `LinearExperts2D` directly via `register_patch_mapping` during
    `from_pretrained`, only `__init__` runs — `from_experts_module` (and its
    `_carry_over_gate_scalars`) is never called. The reused MiniMax-M3
    `_apply_gate` reads `swiglu_limit` / `swiglu_alpha` off `self`, so `__init__`
    must harvest those scalars itself (from a weightless meta instantiation of
    the source experts class) or calibration dies with an AttributeError.
    """
    from transformers.models.minimax_m3_vl.configuration_minimax_m3_vl import (
        MiniMaxM3VLTextConfig,
    )
    from transformers.models.minimax_m3_vl.modeling_minimax_m3_vl import (
        MiniMaxM3VLExperts,
    )

    config = MiniMaxM3VLTextConfig(
        hidden_size=16,
        intermediate_size=32,
        num_local_experts=4,
        num_experts_per_tok=2,
    )
    linear_experts_cls = LinearExperts2D.get_linear_experts_cls(MiniMaxM3VLExperts)

    # construct exactly like the load-time patch mapping does: __init__ only
    linear_experts = linear_experts_cls(config)

    assert linear_experts.swiglu_limit == config.swiglu_limit
    assert linear_experts.swiglu_alpha == config.swiglu_alpha

    for expert in linear_experts:
        if isinstance(expert, torch.nn.Module) and hasattr(expert, "up_proj"):
            init.normal_(expert.up_proj.weight, mean=0.0, std=0.02)
            init.normal_(expert.gate_proj.weight, mean=0.0, std=0.02)
            init.normal_(expert.down_proj.weight, mean=0.0, std=0.02)

    num_tokens = 8
    hidden_states = torch.randn(num_tokens, config.hidden_size)
    top_k_index = torch.randint(
        0, config.num_local_experts, size=(num_tokens, config.num_experts_per_tok)
    )
    top_k_weights = torch.randn(num_tokens, config.num_experts_per_tok)

    # r6 raised AttributeError('swiglu_limit') inside the reused `_apply_gate`,
    # both with and without the calibrate-all-experts context
    output = linear_experts(hidden_states, top_k_index, top_k_weights)
    with moe_calibration_context():
        calib_output = linear_experts(hidden_states, top_k_index, top_k_weights)

    assert output.shape == hidden_states.shape
    assert calib_output.shape == hidden_states.shape
    assert not torch.isnan(output).any()
    assert not torch.isnan(calib_output).any()
