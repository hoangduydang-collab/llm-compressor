"""Ownership and dispatch bookkeeping for expert-parallel calibration.

These are the invariants that separate "sharded" from "silently quantized a
fraction of the model". An expert dropped by ``shard_experts``, or routed by
``routed_dispatch`` to a rank that does not own it, produces a checkpoint that
loads, serves, and is wrong -- no exception anywhere. So coverage and agreement
between the two are asserted directly, including for the world sizes that do not
divide GLM-5.2's 256 experts.
"""

import pytest
import torch

from llmcompressor.modeling.moe.expert_parallel import (
    ExpertParallelContext,
    expert_parallel_context,
    get_expert_parallel_context,
    is_expert_parallel_enabled,
    routed_dispatch,
    shard_experts,
)

# GLM-5.2 has 256 routed experts. 6 is a world size we have actually had to run
# (the cluster's largest free node was 6 GPUs), and it does NOT divide 256 --
# which is exactly where an `expert // experts_per_rank` owner calculation
# breaks.
GLM_EXPERTS = 256
WORLD_SIZES = [1, 2, 3, 4, 5, 6, 7, 8, 16]


# --------------------------------------------------------------------------
# shard_experts
# --------------------------------------------------------------------------


@pytest.mark.parametrize("world_size", WORLD_SIZES)
def test_shards_cover_every_expert_exactly_once(world_size):
    shards = [shard_experts(GLM_EXPERTS, r, world_size) for r in range(world_size)]
    flat = [e for s in shards for e in s]
    assert sorted(flat) == list(range(GLM_EXPERTS))
    assert len(flat) == len(set(flat)), "an expert is owned by two ranks"


@pytest.mark.parametrize("world_size", WORLD_SIZES)
def test_shards_are_balanced_within_one(world_size):
    sizes = [
        len(shard_experts(GLM_EXPERTS, r, world_size)) for r in range(world_size)
    ]
    assert max(sizes) - min(sizes) <= 1, sizes


@pytest.mark.parametrize("world_size", WORLD_SIZES)
def test_shards_are_contiguous_and_ascending(world_size):
    for rank in range(world_size):
        shard = shard_experts(GLM_EXPERTS, rank, world_size)
        assert shard == sorted(shard)
        if shard:
            assert shard == list(range(shard[0], shard[-1] + 1))


def test_non_divisible_world_size_distributes_the_remainder():
    """256 over 6: four ranks get 43, two get 42 -- not 42 with 4 dropped."""
    sizes = [len(shard_experts(256, r, 6)) for r in range(6)]
    assert sizes == [43, 43, 43, 43, 42, 42]
    assert sum(sizes) == 256


def test_more_ranks_than_experts_leaves_some_ranks_empty():
    shards = [shard_experts(3, r, 8) for r in range(8)]
    assert [len(s) for s in shards] == [1, 1, 1, 0, 0, 0, 0, 0]
    assert sorted(e for s in shards for e in s) == [0, 1, 2]


def test_single_rank_owns_everything():
    assert shard_experts(GLM_EXPERTS, 0, 1) == list(range(GLM_EXPERTS))


def test_zero_experts_is_empty_not_an_error():
    assert shard_experts(0, 0, 4) == []


@pytest.mark.parametrize(
    "num_experts, rank, world_size",
    [(8, 4, 4), (8, -1, 4), (8, 0, 0), (-1, 0, 4)],
)
def test_shard_experts_rejects_invalid_arguments(num_experts, rank, world_size):
    with pytest.raises(ValueError):
        shard_experts(num_experts, rank, world_size)


# --------------------------------------------------------------------------
# routed_dispatch
# --------------------------------------------------------------------------


def owner_of(expert, num_experts, world_size):
    for rank in range(world_size):
        if expert in shard_experts(num_experts, rank, world_size):
            return rank
    raise AssertionError(f"expert {expert} unowned")


@pytest.mark.parametrize("world_size", [2, 3, 4, 6, 8])
def test_dispatch_routes_each_assignment_to_the_owning_rank(world_size):
    """The trap: `expert // experts_per_rank` is wrong when world_size does not
    divide num_experts, and routes tokens to a rank holding no such expert."""
    torch.manual_seed(0)
    top_k_index = torch.randint(0, GLM_EXPERTS, (64, 8))
    weights = torch.rand(64, 8)

    d = routed_dispatch(top_k_index, weights, GLM_EXPERTS, world_size)

    flat = top_k_index.reshape(-1)
    for i in range(flat.numel()):
        expert = int(flat[i])
        expected = owner_of(expert, GLM_EXPERTS, world_size)
        assert int(d.dest_rank[i]) == expected, (
            f"expert {expert} sent to rank {int(d.dest_rank[i])}, "
            f"owned by {expected}"
        )


@pytest.mark.parametrize("world_size", [2, 3, 4, 6])
def test_local_expert_index_is_valid_on_its_destination_rank(world_size):
    torch.manual_seed(1)
    top_k_index = torch.randint(0, GLM_EXPERTS, (32, 4))
    d = routed_dispatch(
        top_k_index, torch.rand(32, 4), GLM_EXPERTS, world_size
    )
    flat = top_k_index.reshape(-1)
    for i in range(flat.numel()):
        rank = int(d.dest_rank[i])
        shard = shard_experts(GLM_EXPERTS, rank, world_size)
        local = int(d.local_expert[i])
        assert 0 <= local < len(shard)
        # the local index must name the same expert globally
        assert shard[local] == int(flat[i])


def test_permutation_groups_rows_into_contiguous_per_rank_runs():
    """all_to_all_single requires each rank's rows to be contiguous."""
    torch.manual_seed(2)
    top_k_index = torch.randint(0, 64, (40, 4))
    d = routed_dispatch(top_k_index, torch.rand(40, 4), 64, 4)

    ordered = d.dest_rank[d.perm].tolist()
    assert ordered == sorted(ordered), "destination ranks not grouped"
    # and the run lengths must equal the declared split sizes
    runs = torch.bincount(d.dest_rank[d.perm], minlength=4)
    assert runs.tolist() == d.send_counts.tolist()


def test_send_counts_sum_to_every_assignment():
    torch.manual_seed(3)
    top_k_index = torch.randint(0, 64, (25, 6))
    d = routed_dispatch(top_k_index, torch.rand(25, 6), 64, 4)
    assert int(d.send_counts.sum()) == 25 * 6
    assert d.perm.numel() == 25 * 6


def test_permutation_is_a_true_permutation():
    torch.manual_seed(4)
    d = routed_dispatch(
        torch.randint(0, 32, (17, 3)), torch.rand(17, 3), 32, 4
    )
    assert sorted(d.perm.tolist()) == list(range(17 * 3))


def test_src_token_maps_each_assignment_back_to_its_token():
    d = routed_dispatch(
        torch.tensor([[0, 1], [2, 3], [4, 5]]), torch.rand(3, 2), 8, 2
    )
    assert d.src_token.tolist() == [0, 0, 1, 1, 2, 2]


def test_weights_are_carried_through_in_assignment_order():
    weights = torch.tensor([[0.1, 0.2], [0.3, 0.4]])
    d = routed_dispatch(torch.tensor([[0, 1], [2, 3]]), weights, 8, 2)
    assert torch.allclose(d.weight, torch.tensor([0.1, 0.2, 0.3, 0.4]))


def test_dispatch_rejects_mismatched_index_and_weight_shapes():
    with pytest.raises(ValueError, match="shape mismatch"):
        routed_dispatch(torch.zeros(4, 2, dtype=torch.long), torch.rand(4, 3), 8, 2)


def test_every_expert_is_reachable_by_dispatch():
    """A dispatch that can never route to expert j leaves it uncalibrated."""
    world_size = 6
    top_k_index = torch.arange(GLM_EXPERTS).reshape(-1, 1)  # each expert once
    d = routed_dispatch(
        top_k_index, torch.ones(GLM_EXPERTS, 1), GLM_EXPERTS, world_size
    )
    reached = set()
    for i in range(GLM_EXPERTS):
        rank = int(d.dest_rank[i])
        reached.add(shard_experts(GLM_EXPERTS, rank, world_size)[int(d.local_expert[i])])
    assert reached == set(range(GLM_EXPERTS))


# --------------------------------------------------------------------------
# context
# --------------------------------------------------------------------------


def test_context_is_inactive_by_default():
    assert get_expert_parallel_context() is None
    assert not is_expert_parallel_enabled()


def test_context_activates_and_restores():
    with expert_parallel_context(rank=1, world_size=4) as ctx:
        assert ctx.rank == 1 and ctx.world_size == 4
        assert get_expert_parallel_context() is ctx
        assert is_expert_parallel_enabled()
    assert get_expert_parallel_context() is None


def test_context_restores_after_an_exception():
    """A failed calibration must not leave the module globally sharded."""
    with pytest.raises(RuntimeError, match="boom"):
        with expert_parallel_context(rank=0, world_size=2):
            raise RuntimeError("boom")
    assert get_expert_parallel_context() is None
    assert not is_expert_parallel_enabled()


def test_world_size_one_reports_disabled():
    """The sharded path would be a no-op wrapped in collectives; prefer the
    path that has validation history."""
    with expert_parallel_context(rank=0, world_size=1) as ctx:
        assert get_expert_parallel_context() is ctx
        assert not is_expert_parallel_enabled()


def test_nested_contexts_restore_the_outer_one():
    with expert_parallel_context(rank=0, world_size=2) as outer:
        with expert_parallel_context(rank=1, world_size=8) as inner:
            assert get_expert_parallel_context() is inner
        assert get_expert_parallel_context() is outer


@pytest.mark.parametrize("rank, world_size", [(4, 4), (-1, 2), (0, 0)])
def test_context_rejects_invalid_rank_or_world_size(rank, world_size):
    with pytest.raises(ValueError):
        ExpertParallelContext(rank=rank, world_size=world_size)


def test_context_without_dist_and_without_explicit_args_raises():
    if torch.distributed.is_available() and torch.distributed.is_initialized():
        pytest.skip("a process group is initialized in this environment")
    with pytest.raises(RuntimeError, match="initialized process group"):
        with expert_parallel_context():
            pass
