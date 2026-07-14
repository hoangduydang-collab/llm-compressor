"""Parity tests for the EP all-to-all MoE dispatch/combine core.

The EP path (experts sharded across ranks, tokens routed by all-to-all) must
produce the SAME output as the dense forward (all experts present). These tests
simulate the ranks in one process and assert bit-level agreement, so the
permutation/split/scatter bookkeeping is validated on CPU before it ever runs
under torchrun/nccl on GPU.
"""
import torch

from pipeline.ep_moe import (
    ep_moe_reference,
    ep_moe_simulated,
    plan_dispatch,
    route,
)


def _rand_moe(T, H, I, E, seed=0, dtype=torch.float64):
    g = torch.Generator().manual_seed(seed)
    x = torch.randn(T, H, generator=g, dtype=dtype)
    router_w = torch.randn(E, H, generator=g, dtype=dtype)
    gate_up = torch.randn(E, 2 * I, H, generator=g, dtype=dtype) * 0.1
    down = torch.randn(E, H, I, generator=g, dtype=dtype) * 0.1
    return x, router_w, gate_up, down


def test_ep_matches_dense_across_world_sizes():
    # float64 so agreement is exact up to reduction order (index_add is stable).
    x, router_w, gate_up, down = _rand_moe(T=40, H=16, I=12, E=8)
    for top_k in (1, 2, 4):
        ref = ep_moe_reference(x, router_w, gate_up, down, top_k)
        for world_size in (1, 2, 4, 8):
            got = ep_moe_simulated(x, router_w, gate_up, down, top_k, world_size)
            assert torch.allclose(got, ref, atol=1e-9, rtol=1e-6), (top_k, world_size)


def test_ep_matches_dense_with_routing_bias_and_shared_expert():
    # MiniMax-M3 semantics: top-4, per-expert routing bias, one always-on shared expert.
    g = torch.Generator().manual_seed(5)
    x, router_w, gate_up, down = _rand_moe(T=48, H=16, I=12, E=8, seed=5)
    bias = torch.randn(8, generator=g, dtype=torch.float64) * 0.5
    sh_gate_up = torch.randn(2 * 12, 16, generator=g, dtype=torch.float64) * 0.1
    sh_down = torch.randn(16, 12, generator=g, dtype=torch.float64) * 0.1
    ref = ep_moe_reference(x, router_w, gate_up, down, 4, bias, sh_gate_up, sh_down)
    for world_size in (1, 2, 4, 8):
        got = ep_moe_simulated(x, router_w, gate_up, down, 4, world_size, bias, sh_gate_up, sh_down)
        assert torch.allclose(got, ref, atol=1e-9, rtol=1e-6), world_size
    # shared expert actually contributes (guards against it being silently dropped)
    no_shared = ep_moe_simulated(x, router_w, gate_up, down, 4, 4, bias)
    assert not torch.allclose(got, no_shared)


def test_dispatch_metadata_is_consistent():
    x, router_w, gate_up, down = _rand_moe(T=25, H=8, I=8, E=8, seed=1)
    top_k, world_size, epr = 3, 4, 2
    logits = x @ router_w.t()
    expert_ids, weights = route(logits, top_k)
    d = plan_dispatch(expert_ids, weights, world_size, epr)
    n = x.shape[0] * top_k
    # every assignment is accounted for exactly once
    assert d.perm.shape[0] == n
    assert torch.equal(torch.sort(d.perm).values, torch.arange(n))
    # split sizes partition N and match the destination grouping
    assert int(d.send_counts.sum()) == n
    assert torch.equal(d.dest_rank[d.perm].sort().values, d.dest_rank[d.perm])  # grouped
    # local expert index is always within [0, epr) and maps back to the global id
    assert (d.local_expert >= 0).all() and (d.local_expert < epr).all()
    recovered = d.dest_rank * epr + d.local_expert
    assert torch.equal(recovered, expert_ids.reshape(-1))


def test_all_tokens_receive_output_when_covered():
    # with top_k>=1 every token has at least one expert, so no row is dropped
    x, router_w, gate_up, down = _rand_moe(T=30, H=8, I=8, E=4, seed=2)
    got = ep_moe_simulated(x, router_w, gate_up, down, top_k=2, world_size=4)
    ref = ep_moe_reference(x, router_w, gate_up, down, top_k=2)
    assert torch.allclose(got, ref, atol=1e-9, rtol=1e-6)
    assert (got.abs().sum(dim=1) > 0).all()  # no all-zero (dropped) token rows
