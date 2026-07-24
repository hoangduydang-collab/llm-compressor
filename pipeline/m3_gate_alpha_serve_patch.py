"""Serve-side support for the r7 gate-alpha fold (per-channel swiglu alpha/limit).

The r7 checkpoint's experts compute ``glu = clamp(g, max=limit_r) * sigmoid(alpha_r*g)``
with per-expert, per-channel ``alpha_r = 1.702*s_r`` and ``limit_r = 7.0/s_r``
derived from the stored fp32 ``gate_smooth_scale`` (see
pipeline/m3_gate_alpha_fold.py). vLLM's W4A8 CUTLASS path applies the swiglu
via a scalar-parameter fused op, so serving an r7 checkpoint needs this patch.

Design (three narrow shims; the pure math below is unit-tested offline):

1. ``run_cutlass_moe_w4a8_fp8`` wrapper: reads per-layer tables off the ``w1``
   weight tensor (attached at bind time as ``w1._m3_gate_alpha``), stores them
   in a contextvar for the duration of the call together with ``expert_map``.
2. ``ops.get_cutlass_moe_mm_problem_sizes_from_expert_offsets`` shim: records
   ``expert_first_token_offset`` (its first argument) into the same context —
   this is the only place the non-batched row->expert grouping is visible.
3. The existing ``apply_moe_activation`` wrapper (see
   ``patch_vllm_w4a8_swigluoai_uninterleave``) is extended: when context tables
   are present, compute the activation with per-channel vectors in plain torch
   (elementwise, broadcasting) instead of the scalar fused op.

Binding: ``bind_gate_alpha_tables(model, sidecar)`` walks the loaded vLLM
model, matches FusedMoE layers by decoder-layer index, and attaches each
layer's [n_experts, intermediate] fp32 scale table to its ``w13``/``w1``
weight tensor. Build the sidecar from the (pre-reexport) r7 checkpoint with
``build_sidecar_from_checkpoint``. OPEN SMOKE QUESTION: whether the vLLM M3
loader tolerates the ``gate_smooth_scale`` keys present in the export — if it
rejects unknown tensors, extend ``reexport_minimax_m3_vllm`` to drop them
(they are fully captured by the sidecar).

STATUS: pure-math core is final and unit-tested
(tests/pipeline/test_m3_gate_alpha_serve_math.py); the vLLM integration shims
target vLLM 0.24.0 and REQUIRE an executor-side smoke (single node, r7 smoke
checkpoint) before any eval — see the r7 design note's validation ladder.
"""

from __future__ import annotations

import contextvars
import json
import os
import re
from pathlib import Path

import torch

SIDECAR_NAME = "gate_smooth_scale_sidecar.pt"
_LAYER_RE = re.compile(r"\.layers\.(\d+)\.")

# ---------------------------------------------------------------------------
# Pure math (no vLLM imports) — unit-tested offline
# ---------------------------------------------------------------------------


def rows_to_experts_nonbatched(
    expert_first_token_offset: torch.Tensor, num_rows: int
) -> torch.Tensor:
    """Row-index -> local-expert-index for the grouped (permuted) layout.

    ``expert_first_token_offset`` has length n_local_experts+1; rows in
    ``[off[e], off[e+1])`` belong to local expert e. Rows past the last offset
    (padding) map to expert 0 — their outputs are never unpermuted back.
    """
    off = expert_first_token_offset.to(torch.int64)
    idx = torch.arange(num_rows, device=off.device)
    row_expert = torch.bucketize(idx, off[1:-1], right=True)
    return row_expert


def rows_to_experts_batched(num_local_experts: int, padded_m: int) -> torch.Tensor:
    """Row-index -> local-expert-index for the batched [E*padded_M, ...] layout."""
    return torch.arange(num_local_experts).repeat_interleave(padded_m)


def swigluoai_per_channel(
    gate_up: torch.Tensor,
    alpha_rows: torch.Tensor,
    limit_rows: torch.Tensor,
    up_limit: float,
) -> torch.Tensor:
    """M3 clamped swiglu with per-ROW per-channel gate alpha/limit vectors.

    ``gate_up`` is the packed (uninterleaved) [rows, 2d] GEMM1 output;
    ``alpha_rows``/``limit_rows`` are [rows, d] (already expanded per row's
    expert). The up-clamp and the ``+1.0`` stay scalar by design — the r7 fold
    never touches them.
    """
    d = gate_up.shape[-1] // 2
    gate = gate_up[..., :d]
    up = gate_up[..., d:]
    gate = torch.minimum(gate, limit_rows.to(gate.dtype))
    up = up.clamp(min=-up_limit, max=up_limit)
    glu = gate * torch.sigmoid(gate * alpha_rows.to(gate.dtype))
    return (up + 1.0) * glu


def expand_tables_for_rows(
    scale_table: torch.Tensor,  # [n_experts_local, d] fp32 gate_smooth_scale
    row_expert: torch.Tensor,  # [rows] local expert index per row
    alpha: float,
    limit: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Per-row alpha/limit vectors from the per-expert scale table."""
    s = scale_table.index_select(0, row_expert.to(scale_table.device))
    return alpha * s, limit / s


# ---------------------------------------------------------------------------
# Sidecar I/O
# ---------------------------------------------------------------------------


def load_sidecar(path: str | Path) -> dict[int, torch.Tensor]:
    """{decoder_layer_index: fp32 [n_experts, intermediate] gate_smooth_scale}."""
    payload = torch.load(Path(path), map_location="cpu", weights_only=True)
    return {int(k): v.to(torch.float32) for k, v in payload["layers"].items()}


def build_sidecar_from_checkpoint(ckpt: str | Path, out: str | Path) -> int:
    """Collect per-expert ``gate_smooth_scale`` tensors from a (pre-reexport)
    r7 checkpoint into one sidecar file. Returns number of layers collected."""
    from safetensors import safe_open

    ckpt = Path(ckpt)
    index = json.loads((ckpt / "model.safetensors.index.json").read_text())
    weight_map = index["weight_map"]
    per_layer: dict[int, dict[int, torch.Tensor]] = {}
    pat = re.compile(r"\.layers\.(\d+)\..*experts\.(\d+)\.gate_smooth_scale$")
    by_shard: dict[str, list[tuple[str, int, int]]] = {}
    for name, shard in weight_map.items():
        m = pat.search(name)
        if not m:
            continue
        by_shard.setdefault(shard, []).append(
            (name, int(m.group(1)), int(m.group(2)))
        )
    # one mmap per shard — re-opening a ~50 GB shard per tensor exhausts
    # the process's mmap budget (ENOMEM) long before all 7296 scales load
    for shard, entries in by_shard.items():
        with safe_open(ckpt / shard, framework="pt") as fh:
            for name, layer, expert in entries:
                per_layer.setdefault(layer, {})[expert] = fh.get_tensor(name).float()
    layers = {
        layer: torch.stack([experts[i] for i in sorted(experts)])
        for layer, experts in per_layer.items()
    }
    torch.save({"schema": 1, "layers": layers}, Path(out))
    return len(layers)


# ---------------------------------------------------------------------------
# vLLM integration (0.24.0) — executor smoke required
# ---------------------------------------------------------------------------

_ctx_tables: contextvars.ContextVar = contextvars.ContextVar(
    "m3_gate_alpha_ctx", default=None
)
_PATCH_FLAG = "_m3_gate_alpha_patched"


def bind_gate_alpha_tables(model: torch.nn.Module, sidecar: str | Path) -> int:
    """Attach per-layer scale tables to each FusedMoE layer's w13 weight tensor.

    Under expert parallelism each rank's w13 holds only local experts; the
    runner's ``expert_map`` (global->local) is used at call time, so the FULL
    global table is attached here. Returns number of layers bound.
    """
    tables = load_sidecar(sidecar)
    bound = 0
    for name, module in model.named_modules():
        match = _LAYER_RE.search(name)
        if match is None:
            continue
        layer_idx = int(match.group(1))
        if layer_idx not in tables:
            continue
        weight = getattr(module, "w13_weight_packed", None)
        if weight is None:
            weight = getattr(module, "w13_weight", None)
        if weight is None:
            continue
        weight._m3_gate_alpha = tables[layer_idx]
        bound += 1
    return bound


def patch_vllm_w4a8_gate_alpha(alpha: float, limit: float) -> list[str]:
    """Install the three shims. Idempotent; returns applied-change strings.

    Must be applied ON TOP of ``patch_vllm_w4a8_swigluoai_uninterleave`` (which
    routes SWIGLUOAI_UNINTERLEAVE through ``cutlass_moe.apply_moe_activation``,
    the symbol wrapped here).
    """
    changes: list[str] = []
    try:
        from vllm.model_executor.layers.fused_moe.activation import MoEActivation
        from vllm.model_executor.layers.fused_moe.experts import cutlass_moe
    except Exception:
        return changes

    if getattr(cutlass_moe, _PATCH_FLAG, False):
        return changes
    uninterleave = getattr(MoEActivation, "SWIGLUOAI_UNINTERLEAVE", None)
    runner = getattr(cutlass_moe, "run_cutlass_moe_w4a8_fp8", None)
    if uninterleave is None or runner is None:
        return changes

    # (1) runner wrapper: expose this layer's table + expert_map to the context
    def run_w4a8_with_tables(output, hidden_states, w1, w2, topk_ids, activation,
                             global_num_experts, expert_map, *args, **kwargs):
        table = getattr(w1, "_m3_gate_alpha", None)
        if table is None:
            return runner(output, hidden_states, w1, w2, topk_ids, activation,
                          global_num_experts, expert_map, *args, **kwargs)
        token = _ctx_tables.set(
            {"table": table, "expert_map": expert_map, "offsets": None}
        )
        try:
            return runner(output, hidden_states, w1, w2, topk_ids, activation,
                          global_num_experts, expert_map, *args, **kwargs)
        finally:
            _ctx_tables.reset(token)

    cutlass_moe.run_cutlass_moe_w4a8_fp8 = run_w4a8_with_tables
    changes.append("cutlass_moe.run_cutlass_moe_w4a8_fp8 wrapped (table context)")

    # (2) offsets capture: the runner passes expert_first_token_offset as the
    # first argument right before GEMM1/activation.
    from vllm import _custom_ops as ops

    original_sizes = ops.get_cutlass_moe_mm_problem_sizes_from_expert_offsets

    def sizes_shim(expert_first_token_offset, *args, **kwargs):
        ctx = _ctx_tables.get()
        if ctx is not None:
            ctx["offsets"] = expert_first_token_offset
        return original_sizes(expert_first_token_offset, *args, **kwargs)

    ops.get_cutlass_moe_mm_problem_sizes_from_expert_offsets = sizes_shim
    changes.append("ops.get_cutlass_moe_mm_problem_sizes shim (offsets capture)")

    # (3) per-channel activation: extend the (already-wrapped) module-level
    # apply_moe_activation.
    prior_apply = cutlass_moe.apply_moe_activation

    def apply_with_tables(activation, output, input, **kwargs):
        ctx = _ctx_tables.get()
        if ctx is None or activation is not uninterleave:
            return prior_apply(activation, output, input, **kwargs)
        offsets = ctx["offsets"]
        table = ctx["table"].to(input.device, non_blocking=True)
        expert_map = ctx["expert_map"]
        if expert_map is not None:
            # global table -> local rows: local expert i is the i-th global
            # expert with expert_map[g] == i
            local_to_global = torch.argsort(expert_map[expert_map >= 0])
            inverse = torch.nonzero(expert_map >= 0).flatten()[local_to_global]
            table = table.index_select(0, inverse.to(table.device))
        if offsets is not None:
            row_expert = rows_to_experts_nonbatched(offsets, input.shape[0])
        else:
            # batched layout: rows = n_local_experts * padded_M
            n_local = table.shape[0]
            assert input.shape[0] % n_local == 0
            row_expert = rows_to_experts_batched(n_local, input.shape[0] // n_local)
        alpha_rows, limit_rows = expand_tables_for_rows(
            table, row_expert.to(input.device), _ALPHA, _LIMIT
        )
        output.copy_(
            swigluoai_per_channel(input, alpha_rows, limit_rows, _LIMIT).to(
                output.dtype
            )
        )
        return output

    global _ALPHA, _LIMIT
    _ALPHA, _LIMIT = alpha, limit
    cutlass_moe.apply_moe_activation = apply_with_tables
    changes.append(
        f"cutlass_moe.apply_moe_activation supports per-channel gate alpha/limit "
        f"(base alpha={alpha}, limit={limit})"
    )

    setattr(cutlass_moe, _PATCH_FLAG, True)
    return changes


_ALPHA = 1.702
_LIMIT = 7.0
