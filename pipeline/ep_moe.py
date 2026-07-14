"""Expert-parallel (EP) all-to-all MoE forward: the dispatch/combine core.

Phase 2 of `M3_QUANT_SPEEDUP_PLAN`, EP path. To shard MiniMax-M3's experts across
`world_size` ranks for calibration, each rank must run the MoE block forward with
only its 1/world_size experts resident, yet produce the SAME output as the full
dense forward (so per-expert Hessians are complete and activations propagate
correctly). That requires an all-to-all: route each token to the rank owning its
top-k expert, compute locally, route results back, combine.

The part that is subtle and gets forwards wrong is NOT the collective or the gate
-- it is the token permutation / split-size / scatter-back bookkeeping. That part
is semantics-agnostic and is what this module owns and tests: `plan_dispatch`
computes the per-rank split sizes + permutation that `torch.distributed.
all_to_all_single` consumes, and `ep_moe_reference` runs the whole dispatch ->
local-expert -> combine pipeline by SIMULATING the ranks in one process, so it can
be checked bit-for-bit against a plain dense MoE on CPU.

NOT owned here (executor, GPU): swapping the simulated per-rank loop for real
`all_to_all_single` under `torchrun`/nccl, and using MiniMax-M3's exact router
gate + fused expert op in place of the generic ones below. The gate is passed in,
so reference and EP paths share it and parity holds regardless of which gate.
"""
from __future__ import annotations

from dataclasses import dataclass

import torch

# MiniMax-M3 MoE facts (from HF config.json text_config, verified 2026-07-14):
#   num_local_experts=128, num_experts_per_tok=4, scoring_func="sigmoid",
#   use_routing_bias=True (per-layer `e_score_correction_bias`),
#   routed_scaling_factor=2.0, n_shared_experts=1 (always-on),
#   hidden_act="swigluoai" (swiglu_alpha=1.702, swiglu_limit=7.0),
#   dense layers 0-2, MoE layers 3-59 (moe_layer_freq).
# The checkpoint stores experts PER-EXPERT as
#   language_model.model.layers.N.block_sparse_moe.experts.{i}.w{1,2,3}.weight
# (w1=gate, w3=up, w2=down) -- NOT fused on disk -- so the loader shards by expert
# index exactly like MoE-Quant. (The fused gate_up_proj[128,...] is a runtime form.)
TOP_K = 4
ROUTED_SCALING_FACTOR = 2.0
SWIGLU_ALPHA = 1.702
SWIGLU_LIMIT = 7.0


def route(
    logits: torch.Tensor, top_k: int = TOP_K,
    bias: torch.Tensor | None = None, scaling: float = ROUTED_SCALING_FACTOR,
) -> tuple[torch.Tensor, torch.Tensor]:
    """MiniMax-M3 sigmoid router (DeepSeek-V3-style, aux-loss-free bias selection).

    scores = sigmoid(logits); select top-k by (scores + e_score_correction_bias);
    combine weight = the *unbiased* scores at the selected experts, normalized over
    the k picks, then * routed_scaling_factor. `bias` is the per-layer correction
    (shape [E]); None => plain top-k by score.

    NOTE: the top-k-weight NORMALIZATION is the one detail to reconcile against the
    on-cluster modeling forward; sigmoid + bias-selection + scaling are from config.
    Both reference and EP paths call this same function, so dispatch parity holds
    regardless of which normalization is correct.
    """
    scores = torch.sigmoid(logits)
    select_on = scores if bias is None else scores + bias
    _, expert_ids = select_on.topk(top_k, dim=-1)
    weights = scores.gather(-1, expert_ids)
    weights = weights / weights.sum(dim=-1, keepdim=True).clamp_min(1e-20) * scaling
    return expert_ids, weights


def _swigluoai(gate: torch.Tensor, up: torch.Tensor) -> torch.Tensor:
    """OpenAI-style clamped SwiGLU (hidden_act=swigluoai). Matches the vLLM runtime
    clamp patch (alpha=1.702, limit=7.0) recorded in the GPTQ serving evidence."""
    gate = gate.clamp(max=SWIGLU_LIMIT)
    up = up.clamp(min=-SWIGLU_LIMIT, max=SWIGLU_LIMIT)
    return (gate * torch.sigmoid(SWIGLU_ALPHA * gate)) * (up + 1)


def expert_forward(x: torch.Tensor, gate_up: torch.Tensor, down: torch.Tensor) -> torch.Tensor:
    """One expert on a batch of its tokens. gate_up:[2I,H] (w1|w3 stacked),
    down:[H,I] (w2), x:[n,H]. swigluoai-gated. Executor may swap the fused op; only
    matters that reference and EP share it (parity is about dispatch, not the gate)."""
    gu = x @ gate_up.t()            # [n, 2I]
    gate, up = gu.chunk(2, dim=-1)  # [n, I], [n, I]
    return _swigluoai(gate, up) @ down.t()  # [n, H]


def _shared_expert(x, shared_gate_up, shared_down):
    """Always-on shared expert (n_shared_experts=1). Replicated on every rank in
    EP -- computed locally, never routed -- so it is just a dense add here."""
    if shared_gate_up is None:
        return 0.0
    return expert_forward(x, shared_gate_up, shared_down)


def ep_moe_reference(
    x: torch.Tensor, router_w: torch.Tensor, gate_up: torch.Tensor,
    down: torch.Tensor, top_k: int = TOP_K, bias: torch.Tensor | None = None,
    shared_gate_up: torch.Tensor | None = None, shared_down: torch.Tensor | None = None,
) -> torch.Tensor:
    """Ground-truth dense MoE: all E experts present, no sharding. x:[T,H],
    router_w:[E,H], gate_up:[E,2I,H], down:[E,H,I]."""
    logits = x @ router_w.t()
    expert_ids, weights = route(logits, top_k, bias)
    out = torch.zeros_like(x)
    for slot in range(top_k):
        for e in range(gate_up.shape[0]):
            mask = expert_ids[:, slot] == e
            if mask.any():
                y = expert_forward(x[mask], gate_up[e], down[e])
                out[mask] += weights[mask, slot].unsqueeze(1) * y
    return out + _shared_expert(x, shared_gate_up, shared_down)


@dataclass
class Dispatch:
    """Everything all_to_all_single needs, plus what combine needs to scatter back."""

    perm: torch.Tensor          # [N] row order grouped by destination rank
    send_counts: torch.Tensor   # [world_size] rows going to each rank
    dest_rank: torch.Tensor     # [N] destination rank per (token,slot) assignment
    src_token: torch.Tensor     # [N] originating token index per assignment
    local_expert: torch.Tensor  # [N] expert index LOCAL to its destination rank
    weight: torch.Tensor        # [N] router weight per assignment


def plan_dispatch(expert_ids: torch.Tensor, weights: torch.Tensor,
                  world_size: int, experts_per_rank: int) -> Dispatch:
    """Flatten [T,k] assignments to N=T*k rows and group them by owning rank.
    The permutation + split sizes are exactly what all_to_all_single expects."""
    top_k = expert_ids.shape[1]
    src_token = torch.arange(expert_ids.shape[0]).repeat_interleave(top_k)
    flat_e = expert_ids.reshape(-1)
    dest_rank = torch.div(flat_e, experts_per_rank, rounding_mode="floor")
    local_expert = flat_e - dest_rank * experts_per_rank
    perm = torch.argsort(dest_rank, stable=True)  # contiguous per-rank runs
    send_counts = torch.bincount(dest_rank, minlength=world_size)
    return Dispatch(perm, send_counts, dest_rank, src_token, local_expert, weights.reshape(-1))


def ep_moe_simulated(
    x: torch.Tensor, router_w: torch.Tensor, gate_up: torch.Tensor,
    down: torch.Tensor, top_k: int = TOP_K, world_size: int = 8,
    bias: torch.Tensor | None = None,
    shared_gate_up: torch.Tensor | None = None, shared_down: torch.Tensor | None = None,
) -> torch.Tensor:
    """The EP pipeline with ranks simulated in one process: route -> dispatch ->
    each rank computes ONLY its experts -> combine. Structurally identical to the
    real all_to_all version; only the transport is faked (direct indexing instead
    of a collective), so this validates the permutation/combine bookkeeping."""
    E = gate_up.shape[0]
    assert E % world_size == 0, "experts must divide evenly across ranks"
    epr = E // world_size
    logits = x @ router_w.t()
    expert_ids, weights = route(logits, top_k, bias)
    d = plan_dispatch(expert_ids, weights, world_size, epr)

    out = torch.zeros_like(x)
    for r in range(world_size):
        # rows destined to rank r (real path: what rank r receives via all-to-all)
        rows = d.perm[d.dest_rank[d.perm] == r]
        if rows.numel() == 0:
            continue
        xr = x[d.src_token[rows]]           # tokens sent to rank r
        # rank r holds experts [r*epr : (r+1)*epr] only
        gate_up_r, down_r = gate_up[r * epr:(r + 1) * epr], down[r * epr:(r + 1) * epr]
        yr = torch.zeros_like(xr)
        for le in range(epr):               # local experts on this rank
            m = d.local_expert[rows] == le
            if m.any():
                yr[m] = expert_forward(xr[m], gate_up_r[le], down_r[le])
        # combine (real path: all-to-all back, then weighted scatter-add on origin)
        out.index_add_(0, d.src_token[rows], d.weight[rows].unsqueeze(1) * yr)
    # shared expert is replicated on every rank -> local dense add (no all-to-all)
    return out + _shared_expert(x, shared_gate_up, shared_down)
