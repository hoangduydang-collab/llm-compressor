"""Meta-device MoE linearization regression.

The pre-quantization compatibility gate builds a disposable model under
``accelerate.init_empty_weights`` (meta tensors) and then calls ``linearize_moe`` so
the planner sees the same per-expert Linear representation used by calibration.
``compressed-tensors`` intentionally has no ``meta`` offload backend, so
``LinearExperts2D.from_experts_module`` must skip runtime offload initialization when
the source experts are on ``meta`` and leave the linearized modules as meta-only. This
CPU-only test covers that boundary; the GPU offload integration tests in
``test_linearize_offload.py`` never exercise it.
"""

import torch
from accelerate import init_empty_weights
from compressed_tensors.offload.cache.base import OffloadCache
from transformers.models.qwen3_moe.configuration_qwen3_moe import Qwen3MoeConfig
from transformers.models.qwen3_moe.modeling_qwen3_moe import Qwen3MoeExperts

from llmcompressor.modeling.moe.linear_experts import LinearExperts2D


@torch.no_grad()
def test_from_experts_module_meta_skips_offload():
    """A meta source must linearize without touching the (absent) meta offload backend.

    Regression for ``NotImplementedError: Offload of type meta and distributed=False
    has not been implemented`` raised while the pre-quantization gate constructs its
    meta MiniMax-M3 model. The gate builds the fused experts and linearizes them inside
    ``accelerate.init_empty_weights`` (which patches meta ``copy_`` into a no-op), so
    the meta offload initialization is the sole remaining failure boundary.
    """
    config = Qwen3MoeConfig(
        hidden_size=16,
        intermediate_size=32,
        num_experts=4,
        num_experts_per_tok=2,
    )

    linear_experts_cls = LinearExperts2D.get_linear_experts_cls(Qwen3MoeExperts)

    # Mirror the gate: fused experts built and linearized under init_empty_weights.
    with init_empty_weights():
        experts = Qwen3MoeExperts(config)
        assert all(p.is_meta for p in experts.parameters())

        # Must not raise NotImplementedError from the meta offload backend.
        linear_experts = linear_experts_cls.from_experts_module(experts, config)

    # Offload was skipped, not applied: modules stay plain meta modules.
    for module in linear_experts.modules():
        assert not isinstance(module._parameters, OffloadCache)
    assert all(p.is_meta for p in linear_experts.parameters())
