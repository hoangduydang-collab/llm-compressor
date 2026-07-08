"""Runtime vLLM patches for serving MiniMax-M3 W4AFP8 (W4A8 MoE) checkpoints.

Why this exists
---------------
M3's routed experts store ``w13`` as ``[all gates; all ups]`` (the natural
``MergedColumnParallelLinear`` layout produced by ``llm-compressor``), so the M3
model requests ``MoEActivation.SWIGLUOAI_UNINTERLEAVE``. The W4A8 CUTLASS MoE
path (``CutlassExpertsW4A8Fp8``) applies the activation *generically* via
``apply_moe_activation`` after GEMM1 (it is NOT fused into the CUTLASS epilogue),
and ``apply_moe_activation`` already fully implements
``SWIGLUOAI_UNINTERLEAVE``. But two gaps block it in every available build
(stock, this NVIDIA build, and toncao's ``minimax-m3-compressed-tensors``):

1. ``CutlassExpertsW4A8Fp8._supports_activation`` omits
   ``SWIGLUOAI_UNINTERLEAVE`` -> kernel selection raises
   ``NotImplementedError: ... kernel does not support
   MoEActivation.SWIGLUOAI_UNINTERLEAVE activation``.
2. The W4A8 call site ``apply_moe_activation(activation, act_out, mm1_out)``
   passes no ``clamp_limit``/``alpha``/``beta``, but ``SWIGLUOAI_UNINTERLEAVE``
   asserts ``clamp_limit is not None``.

Both are pure Python plumbing that the code was clearly *designed* to support
(the ``MoEActivation`` docstring explicitly names MiniMax-M3). This module
patches them in-process before ``vllm.LLM(...)`` is constructed, sourcing
``swiglu_limit``/``alpha``/``beta`` from the checkpoint config so the MoE path
is numerically identical to the dense ``MiniMaxM3MLP`` (``SiluAndMulWithClamp``).

Long-term fix: upstream this into vLLM (add the enum to
``CutlassExpertsW4A8Fp8._supports_activation`` and thread the swiglu scalars into
the W4A8 ``apply_moe_activation`` call). Remove this shim once a vLLM release
serves M3 W4A8 out of the box.
"""

from __future__ import annotations

from pathlib import Path

# gpt-oss / SwiGLU-OAI defaults (MiniMax-M3 uses these exact constants). Used
# only as a last-resort fallback if the checkpoint config omits a scalar.
_DEFAULT_SWIGLU_LIMIT = 7.0
_DEFAULT_SWIGLU_ALPHA = 1.702
_DEFAULT_SWIGLU_BETA = 1.0  # OAI "up + 1" bias

_PATCH_FLAG = "_llmc_w4a8_swigluoai_uninterleave_patched"


def read_swiglu_params(ckpt: Path, source: str | None = None) -> tuple[float, float, float]:
    """Return ``(swiglu_limit, swiglu_alpha, swiglu_beta)`` for an M3 checkpoint.

    Prefers the transformers-resolved config (identical to what the model uses at
    runtime, where ``swiglu_beta=null`` in raw json becomes a concrete float);
    falls back to raw json, then to gpt-oss/M3 defaults. ``beta`` is never left
    ``None`` (``SiluAndMulWithClamp`` requires a float).
    """
    limit = alpha = beta = None

    try:
        from transformers import AutoConfig

        cfg = AutoConfig.from_pretrained(str(ckpt), trust_remote_code=True)
        tc = getattr(cfg, "text_config", None)
        if tc is not None:
            limit = getattr(tc, "swiglu_limit", None)
            alpha = getattr(tc, "swiglu_alpha", None)
            beta = getattr(tc, "swiglu_beta", None)
    except Exception:
        pass

    if limit is None or alpha is None or beta is None:
        import json

        for base in (ckpt, Path(source) if source else None):
            if base is None:
                continue
            cfg_path = base / "config.json"
            if not cfg_path.exists():
                continue
            try:
                tc = json.loads(cfg_path.read_text(encoding="utf-8")).get(
                    "text_config", {}
                )
            except Exception:
                continue
            limit = limit if limit is not None else tc.get("swiglu_limit")
            alpha = alpha if alpha is not None else tc.get("swiglu_alpha")
            beta = beta if beta is not None else tc.get("swiglu_beta")

    limit = float(limit) if limit is not None else _DEFAULT_SWIGLU_LIMIT
    alpha = float(alpha) if alpha is not None else _DEFAULT_SWIGLU_ALPHA
    beta = float(beta) if beta is not None else _DEFAULT_SWIGLU_BETA
    return limit, alpha, beta


def patch_vllm_w4a8_swigluoai_uninterleave(
    limit: float, alpha: float, beta: float
) -> list[str]:
    """Enable W4A8 CUTLASS MoE + ``SWIGLUOAI_UNINTERLEAVE`` in the loaded vLLM.

    Idempotent. Returns a list of applied changes (empty if already patched or if
    the target vLLM modules are absent, e.g. a build without W4A8 MoE support).
    """
    changes: list[str] = []

    try:
        from vllm.model_executor.layers.fused_moe.activation import MoEActivation
        from vllm.model_executor.layers.fused_moe.experts import cutlass_moe
    except Exception:
        return changes

    uninterleave = getattr(MoEActivation, "SWIGLUOAI_UNINTERLEAVE", None)
    if uninterleave is None:
        # This vLLM predates the uninterleaved SwiGLU-OAI activation; nothing to do.
        return changes

    kernel_cls = getattr(cutlass_moe, "CutlassExpertsW4A8Fp8", None)
    if kernel_cls is None:
        return changes

    if getattr(kernel_cls, _PATCH_FLAG, False):
        return changes

    # 1) Widen the kernel's supported-activation gate to include the uninterleaved
    #    SwiGLU-OAI variant (the generic apply_moe_activation() already handles it).
    original_supports = kernel_cls._supports_activation

    def _supports_activation(activation: "MoEActivation") -> bool:
        if activation is uninterleave:
            return True
        return original_supports(activation)

    kernel_cls._supports_activation = staticmethod(_supports_activation)
    changes.append("CutlassExpertsW4A8Fp8._supports_activation += SWIGLUOAI_UNINTERLEAVE")

    # 2) The W4A8 run path calls apply_moe_activation(activation, out, in) with no
    #    clamp scalars, but SWIGLUOAI_UNINTERLEAVE requires clamp_limit/alpha/beta.
    #    Wrap the symbol imported into cutlass_moe's namespace to inject them.
    original_apply = cutlass_moe.apply_moe_activation
    resolved_limit, resolved_alpha, resolved_beta = limit, alpha, beta

    def apply_moe_activation(
        activation, output, input, *, clamp_limit=None, alpha=1.0, beta=0.0
    ):
        if activation is uninterleave and clamp_limit is None:
            clamp_limit, alpha, beta = resolved_limit, resolved_alpha, resolved_beta
        return original_apply(
            activation, output, input, clamp_limit=clamp_limit, alpha=alpha, beta=beta
        )

    cutlass_moe.apply_moe_activation = apply_moe_activation
    changes.append(
        f"cutlass_moe.apply_moe_activation injects clamp for SWIGLUOAI_UNINTERLEAVE "
        f"(limit={limit}, alpha={alpha}, beta={beta})"
    )

    setattr(kernel_cls, _PATCH_FLAG, True)
    return changes


_CG_AR_PATCH_FLAG = "_llmc_m3_cudagraph_fused_ar_patched"


def patch_vllm_m3_fused_ar_for_cudagraph() -> list[str]:
    """Use NCCL all-reduce + GemmaRMSNorm when CUDA graphs are enabled.

    FlashInfer's fused AR+RMSNorm is invoked directly from M3's forward
    (``fused_allreduce_gemma_rms_norm``). ``breakable_cudagraph`` captures it
    whole; on TP8 H100 this triggers ``cudaErrorIllegalAddress`` at
    ``capture_end`` (vLLM #46253). The module already has a numerically identical
    NCCL fallback — we force that path whenever ``cudagraph_mode != NONE``.
    """
    changes: list[str] = []
    try:
        import vllm.model_executor.layers.fused_allreduce_gemma_rms_norm as far
    except ImportError:
        return changes

    if getattr(far, _CG_AR_PATCH_FLAG, False):
        return changes

    original = far._can_use_flashinfer

    def _can_use_flashinfer(hidden_states, tp_size: int) -> tuple[bool, int]:
        try:
            from vllm.config import get_current_vllm_config
            from vllm.config.compilation import CUDAGraphMode

            vc = get_current_vllm_config()
            if vc is not None and not vc.enforce_eager:
                mode = vc.compilation_config.cudagraph_mode
                if mode is not None and mode != CUDAGraphMode.NONE:
                    return False, 0
        except Exception:
            pass
        return original(hidden_states, tp_size)

    far._can_use_flashinfer = _can_use_flashinfer
    setattr(far, _CG_AR_PATCH_FLAG, True)
    changes.append(
        "fused_allreduce_gemma_rms_norm: NCCL fallback when CUDA graphs enabled"
    )
    return changes
