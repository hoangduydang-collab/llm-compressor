"""MiniMax-M3 r7 "gate-alpha fold": function-preserving down-side AWQ smoothing.

Background (BUGS_AND_FIXES.md "AWQ up->down smoothing fold is not
function-preserving on MiniMax-M3"): M3's expert activation is
``h = (clamp(up, ±L) + 1.0) * glu`` with ``glu = clamp(g, max=L) * sigmoid(a*g)``
(gpt-oss style, a=1.702, L=7.0). The removed r5 mapping folded ``1/s`` into
``up`` rows, which rescales beta/clamp per channel — a function change. r7
carries ``1/s`` through the gate path's *homogeneous* factor instead
(design: docs/superpowers/plans/2026-07-23-m3-awq-gate-alpha-fold.md):

    gate rows   /= s_r          (AWQModifier's generic smooth-layer fold)
    alpha_r      = a * s_r      (per channel, per expert — this module)
    limit_r      = L / s_r      (per channel, per expert — this module)
    down cols   *= s_r          (AWQModifier's generic balance fold)

which gives ``glu' = glu / s`` exactly for every input, so the composition is
the identity. Scales are **per-expert AND per-channel**, same granularity as
the removed up->down mapping: the AWQ mapping is
``experts.N.gate_proj -> experts.N.down_proj``, grouped per expert, and each
expert carries its own fp32 ``gate_smooth_scale`` buffer of size
[intermediate_size] (persisted into the checkpoint for serve-side use).

Wiring:
- ``get_minimax_m3_awq_mappings(include_gate_alpha_fold=True)`` emits the
  per-expert gate->down mapping (env: ``M3_AWQ_GATE_ALPHA_FOLD=1``).
- ``attach_minimax_m3_gate_alpha_fold(model)`` (this module) must run after
  ``linearize_moe`` and before calibration: it registers the per-expert
  buffer, swaps in a vector-aware ``_apply_gate``, and attaches an
  ``awq_fold_scale_consumer`` to every expert's ``gate_proj`` that
  AWQModifier._apply_smoothing calls synchronously at fold time.
- Fail closed: enabling the mapping without attaching consumers must abort
  (checked in pipeline.quantize), because the generic fold alone would change
  the function through the un-co-scaled sigmoid.
"""

from __future__ import annotations

from typing import Any

import torch

GATE_SMOOTH_SCALE_NAME = "gate_smooth_scale"
# Backstop band, mirroring the r4 dead-channel lesson: bound every fold.
_SCALE_MIN, _SCALE_MAX = 0.125, 8.0


def _is_m3_gated_expert(module: torch.nn.Module) -> bool:
    return (
        hasattr(module, "gate_proj")
        and hasattr(module, "up_proj")
        and hasattr(module, "down_proj")
        and hasattr(module, "_apply_gate")
        and isinstance(getattr(module, "gate_proj", None), torch.nn.Linear)
    )


def make_m3_vector_apply_gate(expert: torch.nn.Module, alpha: float, limit: float):
    """M3's ``_apply_gate`` with optional per-channel alpha / gate-limit vectors.

    Identical to ``MiniMaxM3VLExperts._apply_gate`` when no vectors are set
    (scalar path), which the fold-identity unit test asserts. The up-clamp and
    the ``+ 1.0`` are NOT vectorized — they are exactly the terms the r7 fold
    is designed never to touch.
    """

    def _apply_gate(gate_up: torch.Tensor) -> torch.Tensor:
        gate, up = gate_up.chunk(2, dim=-1)
        limit_vec = getattr(expert, "swiglu_limit_vec", None)
        if limit_vec is None:
            gate = gate.clamp(max=limit)
        else:
            gate = torch.minimum(gate, limit_vec.to(gate.device, gate.dtype))
        up = up.clamp(min=-limit, max=limit)
        alpha_vec = getattr(expert, "swiglu_alpha_vec", None)
        a: Any = (
            alpha_vec.to(gate.device, gate.dtype) if alpha_vec is not None else alpha
        )
        glu = gate * torch.sigmoid(gate * a)
        return (up + 1.0) * glu

    return _apply_gate


def _make_fold_consumer(expert: torch.nn.Module, alpha: float, limit: float):
    """Consumer called by AWQModifier at fold time with the chosen scales.

    Composes multiplicatively (a second fold multiplies into the same buffer),
    clamps to the backstop band, and refreshes the derived per-channel
    alpha/limit vectors in fp32 so serve-side derivation from the stored
    ``gate_smooth_scale`` reproduces them exactly.
    """

    def _consume(scales: torch.Tensor) -> None:
        buf = getattr(expert, GATE_SMOOTH_SCALE_NAME)
        # buf.dtype (fp32 in production) governs the derived-vector precision;
        # tests swap in fp64 to assert the algebra is exactly the identity.
        s = scales.detach().to(device=buf.device, dtype=buf.dtype).flatten()
        if s.numel() != buf.numel():
            raise ValueError(
                f"gate-alpha fold scale length {s.numel()} != intermediate size "
                f"{buf.numel()}"
            )
        if not torch.isfinite(s).all() or (s <= 0).any():
            raise ValueError("gate-alpha fold received non-finite or non-positive scales")
        buf.mul_(s)
        buf.clamp_(min=_SCALE_MIN, max=_SCALE_MAX)
        expert.swiglu_alpha_vec = alpha * buf
        expert.swiglu_limit_vec = limit / buf

    return _consume


def attach_minimax_m3_gate_alpha_fold(model: torch.nn.Module) -> int:
    """Prepare every linearized M3 expert for the gate-alpha fold.

    Returns the number of experts prepared. Must run AFTER ``linearize_moe``
    (the per-expert ``gate_proj`` Linears only exist then). Idempotent.
    """
    prepared = 0
    for name, container in model.named_modules():
        alpha = getattr(container, "swiglu_alpha", None)
        limit = getattr(container, "swiglu_limit", None)
        if not isinstance(alpha, float) or not isinstance(limit, float):
            continue
        for expert in container.children() if hasattr(container, "children") else []:
            if not _is_m3_gated_expert(expert):
                continue
            if not hasattr(expert, GATE_SMOOTH_SCALE_NAME):
                inter = expert.gate_proj.weight.shape[0]
                # Small fp32 CPU buffer; persistent => lands in the saved
                # checkpoint as `...experts.N.gate_smooth_scale`.
                expert.register_buffer(
                    GATE_SMOOTH_SCALE_NAME,
                    torch.ones(inter, dtype=torch.float32),
                    persistent=True,
                )
            expert._apply_gate = make_m3_vector_apply_gate(expert, alpha, limit)
            expert.gate_proj.awq_fold_scale_consumer = _make_fold_consumer(
                expert, alpha, limit
            )
            prepared += 1
    return prepared


def assert_gate_alpha_fold_ready(model: torch.nn.Module, prepared: int) -> None:
    """Fail-closed guard: the gate->down mapping is only safe with consumers."""
    if prepared <= 0:
        raise RuntimeError(
            "M3_AWQ_GATE_ALPHA_FOLD is enabled but no experts were prepared for "
            "the gate-alpha fold (linearize_moe not run, or model is not M3). "
            "Running the gate->down mapping without alpha/limit co-scaling would "
            "change the model function — aborting."
        )
