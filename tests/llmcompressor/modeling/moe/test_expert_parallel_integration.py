"""Wiring: does LinearExperts2D.forward actually take the sharded path, and is
anything it must not touch left alone?

The arithmetic is covered in test_expert_parallel_equivalence.py, which calls
expert_parallel_forward directly. What is left to check is the dispatch decision
and its blast radius:

  * the default path must be untouched when no context is active, and when the
    world size is 1 (where sharding would be a no-op wrapped in collectives)
  * the shared expert must stay REPLICATED. It is always-on -- every token needs
    it -- so a sharded shared expert would drop most of its contribution.
    LinearExperts2D holds only routed experts, so this is true by construction;
    asserted here rather than assumed, because the structure could change.
  * DistCollectives must work against a real process group, including its
    padding path for uneven shards.
"""

import threading

import pytest
import torch
from torch import distributed as dist

import llmcompressor.modeling.moe.linear_experts as le
from llmcompressor.modeling.moe.context import moe_calibration_context
from llmcompressor.modeling.moe.expert_parallel import (
    DistCollectives,
    expert_parallel_context,
)
from llmcompressor.modeling.moe.linear_experts import LinearExperts2D

# Plain module name, not a relative import: this test directory is not a
# package (no __init__.py), and pytest puts the file's directory on sys.path.
from test_expert_parallel_equivalence import (  # noqa: E402
    N_EXPERTS,
    ThreadCollectives,
    ThreadWorld,
    build_experts,
    routing,
)


# --------------------------------------------------------------------------
# dispatch decision
# --------------------------------------------------------------------------


def test_default_path_when_no_context_is_active():
    experts = build_experts()
    hidden, index, weights = routing()
    calls = []
    original = le.expert_parallel_forward

    def spy(*args, **kwargs):
        calls.append(1)
        return original(*args, **kwargs)

    le.expert_parallel_forward = spy
    try:
        with moe_calibration_context(), torch.no_grad():
            experts(hidden, index, weights)
    finally:
        le.expert_parallel_forward = original

    assert calls == [], "sharded path taken without an active context"


def test_world_size_one_takes_the_default_path():
    """Sharding at W=1 would be a no-op wrapped in collectives; the unsharded
    path is the one with validation history."""
    experts = build_experts()
    hidden, index, weights = routing()
    calls = []
    original = le.expert_parallel_forward

    def spy(*args, **kwargs):
        calls.append(1)
        return original(*args, **kwargs)

    le.expert_parallel_forward = spy
    try:
        with expert_parallel_context(rank=0, world_size=1):
            with moe_calibration_context(), torch.no_grad():
                experts(hidden, index, weights)
    finally:
        le.expert_parallel_forward = original

    assert calls == []


def test_forward_dispatches_to_the_sharded_path_when_enabled():
    """Substitutes the transport only -- the dispatch decision is the real one."""
    experts = build_experts()
    hidden, index, weights = routing()
    world_size = 4

    shards = list(torch.chunk(hidden, world_size, dim=0))
    idx = list(torch.chunk(index, world_size, dim=0))
    wts = list(torch.chunk(weights, world_size, dim=0))

    world = ThreadWorld(world_size)
    outputs = [None] * world_size
    errors = [None] * world_size
    dispatched = []

    def body(rank):
        try:
            # Transport injected through the context, so each simulated rank
            # gets its own without racing on a module global. The dispatch
            # decision inside forward() is the real one.
            with expert_parallel_context(
                rank=rank, world_size=world_size,
                collectives=ThreadCollectives(world, rank),
            ):
                dispatched.append(rank)
                with torch.no_grad():
                    outputs[rank] = experts(shards[rank], idx[rank], wts[rank])
        except Exception as exc:
            errors[rank] = exc
            world.barrier.abort()

    # moe_calibration_context is entered ONCE, in this thread, around the whole
    # threaded section. Its flag is a plain module global, so entering it inside
    # each worker would interleave the save/restore values and leave it stuck
    # on -- polluting later tests. A real run is one process per rank and never
    # nests it across threads, so this is a constraint on the harness, not a bug.
    threads = [threading.Thread(target=body, args=(r,)) for r in range(world_size)]
    with moe_calibration_context():
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=60)

    for rank, err in enumerate(errors):
        assert err is None, f"rank {rank}: {err!r}"
    assert sorted(dispatched) == list(range(world_size))

    got = torch.cat(outputs, dim=0)
    with moe_calibration_context(), torch.no_grad():
        reference = experts(hidden, index, weights)
    torch.testing.assert_close(got, reference, rtol=1e-5, atol=1e-6)


# --------------------------------------------------------------------------
# blast radius: the shared expert
# --------------------------------------------------------------------------


def test_shared_expert_is_not_inside_the_sharded_container():
    """The shared expert is always-on, so sharding it would drop most of its
    contribution. It must live outside LinearExperts2D."""
    from transformers.models.glm_moe_dsa.configuration_glm_moe_dsa import (
        GlmMoeDsaConfig,
    )
    from transformers.models.glm_moe_dsa.modeling_glm_moe_dsa import (
        GlmMoeDsaForCausalLM,
    )
    from llmcompressor.modeling.moe.linearize import linearize_moe

    cfg = GlmMoeDsaConfig(
        hidden_size=32, num_hidden_layers=4, n_routed_experts=N_EXPERTS,
        num_experts_per_tok=2, moe_intermediate_size=16,
        first_k_dense_replace=3, n_shared_experts=1, num_attention_heads=4,
        num_key_value_heads=4, intermediate_size=64, vocab_size=128,
        kv_lora_rank=16, q_lora_rank=16, qk_rope_head_dim=8, v_head_dim=8,
        qk_nope_head_dim=8, index_topk=8, max_position_embeddings=32,
        tie_word_embeddings=False,
    )
    torch.manual_seed(0)
    model = GlmMoeDsaForCausalLM(cfg)
    with moe_calibration_context():
        linearize_moe(model)

    moe = model.model.layers[3].mlp
    assert isinstance(moe.experts, LinearExperts2D)
    assert hasattr(moe, "shared_experts"), "GLM-5.2 has an always-on shared expert"

    sharded_names = {name for name, _ in moe.experts.named_modules()}
    shared_names = {name for name, _ in moe.shared_experts.named_modules()}
    sharded_params = {id(p) for p in moe.experts.parameters()}
    shared_params = {id(p) for p in moe.shared_experts.parameters()}

    assert shared_params, "shared expert has no parameters?"
    assert not (sharded_params & shared_params), (
        "shared expert parameters are reachable from the sharded container"
    )
    assert not any("shared" in n for n in sharded_names), sharded_names
    assert shared_names, shared_names


def test_whole_moe_block_output_is_unchanged_under_sharding():
    """End-to-end through the MoE block, so the always-on shared expert and the
    router are both in the comparison, not just the routed experts."""
    from transformers.models.glm_moe_dsa.configuration_glm_moe_dsa import (
        GlmMoeDsaConfig,
    )
    from transformers.models.glm_moe_dsa.modeling_glm_moe_dsa import (
        GlmMoeDsaForCausalLM,
    )
    from llmcompressor.modeling.moe.linearize import linearize_moe

    cfg = GlmMoeDsaConfig(
        hidden_size=32, num_hidden_layers=4, n_routed_experts=N_EXPERTS,
        num_experts_per_tok=2, moe_intermediate_size=16,
        first_k_dense_replace=3, n_shared_experts=1, num_attention_heads=4,
        num_key_value_heads=4, intermediate_size=64, vocab_size=128,
        kv_lora_rank=16, q_lora_rank=16, qk_rope_head_dim=8, v_head_dim=8,
        qk_nope_head_dim=8, index_topk=8, max_position_embeddings=32,
        tie_word_embeddings=False,
    )
    torch.manual_seed(0)
    model = GlmMoeDsaForCausalLM(cfg)
    with moe_calibration_context():
        linearize_moe(model)
    moe = model.model.layers[3].mlp

    torch.manual_seed(5)
    hidden = torch.randn(1, 16, 32)
    world_size = 4

    with moe_calibration_context(), torch.no_grad():
        reference = moe(hidden)

    # Shared-expert weights must be untouched by an EP forward.
    before = [p.detach().clone() for p in moe.shared_experts.parameters()]

    # Split along the TOKEN dim; the block reshapes internally, so shard the
    # sequence rather than the batch.
    shards = list(torch.chunk(hidden, world_size, dim=1))
    world = ThreadWorld(world_size)
    outputs = [None] * world_size
    errors = [None] * world_size

    def body(rank):
        try:
            with expert_parallel_context(
                rank=rank, world_size=world_size,
                collectives=ThreadCollectives(world, rank),
            ):
                with torch.no_grad():
                    outputs[rank] = moe(shards[rank])
        except Exception as exc:
            errors[rank] = exc
            world.barrier.abort()

    # moe_calibration_context is entered ONCE, in this thread, around the whole
    # threaded section. Its flag is a plain module global, so entering it inside
    # each worker would interleave the save/restore values and leave it stuck
    # on -- polluting later tests. A real run is one process per rank and never
    # nests it across threads, so this is a constraint on the harness, not a bug.
    threads = [threading.Thread(target=body, args=(r,)) for r in range(world_size)]
    with moe_calibration_context():
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=60)

    for rank, err in enumerate(errors):
        assert err is None, f"rank {rank}: {err!r}"

    got = torch.cat(outputs, dim=1)
    assert got.shape == reference.shape
    torch.testing.assert_close(got, reference, rtol=1e-5, atol=1e-6)

    after = [p.detach() for p in moe.shared_experts.parameters()]
    for b, a in zip(before, after):
        torch.testing.assert_close(a, b, rtol=0, atol=0)


# --------------------------------------------------------------------------
# DistCollectives against a real process group
# --------------------------------------------------------------------------


@pytest.fixture
def gloo_group(tmp_path):
    """A real single-rank gloo group, so DistCollectives is exercised for real.

    world_size=1 means is_expert_parallel_enabled() is False, so this cannot be
    reached through forward(); DistCollectives is called directly. It still
    covers the collectives themselves, including the size exchange that the
    padding path depends on.
    """
    if not dist.is_available():
        pytest.skip("torch.distributed unavailable")
    if dist.is_initialized():
        yield None
        return
    try:
        # FileStore rather than HashStore: HashStore is absent in torch 2.11,
        # and FileStore works on both POSIX and Windows.
        store = dist.FileStore(str(tmp_path / "gloo_store"), 1)
        dist.init_process_group(backend="gloo", store=store, rank=0, world_size=1)
    except Exception as exc:  # noqa: BLE001
        pytest.skip(f"cannot init gloo: {exc}")
    try:
        yield None
    finally:
        dist.destroy_process_group()


def test_dist_collectives_all_gather_round_trips(gloo_group):
    c = DistCollectives()
    x = torch.randn(5, 3)
    torch.testing.assert_close(c.all_gather(x), x)


def test_dist_collectives_all_reduce_is_identity_at_world_size_one(gloo_group):
    c = DistCollectives()
    x = torch.randn(4, 2)
    expected = x.clone()
    out = c.all_reduce_sum(x)
    torch.testing.assert_close(out, expected)
    assert out is x, "all_reduce_sum must be in-place"


def test_dist_collectives_all_gather_preserves_dtype_and_shape(gloo_group):
    c = DistCollectives()
    for tensor in (
        torch.randint(0, 7, (6, 2)),
        torch.randn(6, 4),
        torch.tensor([3]),
    ):
        out = c.all_gather(tensor)
        assert out.dtype == tensor.dtype
        assert out.shape == tensor.shape
