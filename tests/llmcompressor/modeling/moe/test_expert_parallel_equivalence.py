"""Expert-parallel calibration preserves exact coverage and tolerant numerical parity.

Two properties, and the second is the one that bites. Sharding experts across
ranks is easy to get subtly wrong: calibration is ALREADY data-parallel, so a
rank that runs only its own experts sees only its own data shard, and expert j's
Hessian ends up covering 1/W of the calibration set. Nothing raises -- a Hessian
over a quarter of the data is a well-formed matrix -- and the checkpoint loads
and serves. So this file asserts, per expert, that the rows it was fed under
expert parallelism are exactly the rows it was fed unsharded.

The reference is the REAL ``LinearExperts2D.forward`` on a real linearized GLM
MoE layer, not a transcription of it, so the comparison cannot drift away from
the implementation it is supposed to match.

Thread simulations allow direct inspection of every rank's input records.
They complement the real multiprocess integration tests; collective ordering,
offload persistence and save/reload require separate distributed validation.
"""

import threading

import pytest
import torch

from llmcompressor.modeling.moe.context import (
    get_calibrate_all_experts_flag,
    moe_calibration_context,
)
from llmcompressor.modeling.moe.expert_parallel import (
    Collectives,
    ExpertParallelContext,
    expert_parallel_forward,
    get_expert_hessian_contributions,
    shard_experts,
)
from llmcompressor.modeling.moe.linear_experts import LinearExperts2D
from llmcompressor.modeling.moe.linearize import linearize_moe
from llmcompressor.modifiers.gptq import GPTQModifier

N_EXPERTS = 8
TOP_K = 2
HIDDEN = 32
TOKENS = 24


# --------------------------------------------------------------------------
# lockstep in-process world
# --------------------------------------------------------------------------


class ThreadWorld:
    """Shared slots + barrier, giving real collective ordering semantics."""

    def __init__(self, world_size):
        self.world_size = world_size
        self.barrier = threading.Barrier(world_size)
        self.slots = [None] * world_size


class ThreadCollectives(Collectives):
    def __init__(self, world: ThreadWorld, rank: int):
        self.world = world
        self.rank = rank

    def all_gather(self, tensor):
        self.world.slots[self.rank] = tensor
        self.world.barrier.wait()
        out = torch.cat(list(self.world.slots), dim=0)
        # second barrier: nobody may overwrite a slot until all have read
        self.world.barrier.wait()
        return out

    def all_reduce_sum(self, tensor):
        self.world.slots[self.rank] = tensor.clone()
        self.world.barrier.wait()
        total = self.world.slots[0].clone()
        for other in self.world.slots[1:]:
            total = total + other
        self.world.barrier.wait()
        tensor.copy_(total)
        return tensor


# --------------------------------------------------------------------------
# fixtures
# --------------------------------------------------------------------------


def build_model(
    num_hidden_layers=4, first_k_dense_replace=3, n_experts=None, config_overrides=None
):
    """Build the shared real GLM fixture, retaining its complete model lifecycle."""
    if n_experts is None:
        n_experts = N_EXPERTS
    from transformers.models.glm_moe_dsa.configuration_glm_moe_dsa import (
        GlmMoeDsaConfig,
    )
    from transformers.models.glm_moe_dsa.modeling_glm_moe_dsa import (
        GlmMoeDsaForCausalLM,
    )

    config_kwargs = dict(
        hidden_size=HIDDEN, num_hidden_layers=num_hidden_layers,
        n_routed_experts=n_experts,
        num_experts_per_tok=TOP_K, moe_intermediate_size=16,
        first_k_dense_replace=first_k_dense_replace, n_shared_experts=1,
        num_attention_heads=4,
        num_key_value_heads=4, intermediate_size=64, vocab_size=128,
        kv_lora_rank=16, q_lora_rank=16, qk_rope_head_dim=8, v_head_dim=8,
        qk_nope_head_dim=8, index_topk=8, max_position_embeddings=32,
        tie_word_embeddings=False,
    )
    cfg = GlmMoeDsaConfig(**(config_kwargs | (config_overrides or {})))
    torch.manual_seed(0)
    model = GlmMoeDsaForCausalLM(cfg)
    model.eval()
    with moe_calibration_context():
        linearize_moe(model)
    return model


def build_experts():
    """A real linearized LinearExperts2D, taken from a tiny GLM MoE layer."""
    model = build_model()
    experts = model.model.layers[3].mlp.experts
    assert isinstance(experts, LinearExperts2D), type(experts)
    assert experts.num_experts == N_EXPERTS
    return experts


def routing(tokens=TOKENS, seed=7):
    g = torch.Generator().manual_seed(seed)
    hidden = torch.randn(tokens, HIDDEN, generator=g)
    logits = torch.randn(tokens, N_EXPERTS, generator=g)
    weights, index = torch.softmax(logits, dim=-1).topk(TOP_K, dim=-1)
    return hidden, index, weights


class InputRecorder:
    """Records, per expert, the rows fed to its gate_proj.

    gate_proj's input is the expert's input, which is exactly what GPTQ
    accumulates its Hessian from -- so these records are a faithful stand-in
    for "what data did this expert's Hessian cover".
    """

    def __init__(self, experts):
        self.seen = {i: [] for i in range(experts.num_experts)}
        self.handles = []
        for i in range(experts.num_experts):
            self.handles.append(
                experts[i].gate_proj.register_forward_pre_hook(self._make(i))
            )

    def _make(self, index):
        def hook(_module, args):
            self.seen[index].append(args[0].detach().clone())
        return hook

    def rows(self, index):
        if not self.seen[index]:
            return torch.empty(0, HIDDEN)
        return torch.cat(self.seen[index], dim=0)

    def remove(self):
        for h in self.handles:
            h.remove()


def run_expert_parallel(experts, hidden, index, weights, world_size,
                        calibrate_all_experts=True):
    """Shard rows across ranks, run all ranks in lockstep, reassemble."""
    shards = list(torch.chunk(hidden, world_size, dim=0))
    idx_shards = list(torch.chunk(index, world_size, dim=0))
    w_shards = list(torch.chunk(weights, world_size, dim=0))
    assert len(shards) == world_size, "test needs one shard per rank"

    world = ThreadWorld(world_size)
    outputs = [None] * world_size
    errors = [None] * world_size

    def body(rank):
        try:
            outputs[rank] = expert_parallel_forward(
                experts,
                shards[rank],
                idx_shards[rank],
                w_shards[rank],
                ExpertParallelContext(rank=rank, world_size=world_size),
                ThreadCollectives(world, rank),
                calibrate_all_experts,
            )
            assert get_expert_hessian_contributions() is None
        except Exception as exc:  # surfaced by the caller, not swallowed
            errors[rank] = exc
            world.barrier.abort()

    threads = [threading.Thread(target=body, args=(r,)) for r in range(world_size)]
    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout=60)
    for rank, err in enumerate(errors):
        if err is not None:
            raise AssertionError(f"rank {rank} failed: {err!r}") from err
    assert all(o is not None for o in outputs), "a rank produced no output"
    return torch.cat(outputs, dim=0)


# --------------------------------------------------------------------------
# numerical equivalence
# --------------------------------------------------------------------------


@pytest.mark.parametrize("world_size", [2, 4, 8])
def test_output_matches_the_unsharded_forward(world_size):
    experts = build_experts()
    hidden, index, weights = routing()

    with moe_calibration_context(), torch.no_grad():
        reference = experts(hidden, index, weights)
        got = run_expert_parallel(experts, hidden, index, weights, world_size)

    assert got.shape == reference.shape
    torch.testing.assert_close(got, reference, rtol=1e-5, atol=1e-6)


@pytest.mark.parametrize("world_size", [2, 3, 4])
def test_output_matches_with_routed_calibration(world_size):
    """calibrate_all_experts=False: only routed tokens reach each expert.

    Run OUTSIDE moe_calibration_context, which takes no argument and
    unconditionally sets the flag; False is the module default.
    """
    experts = build_experts()
    hidden, index, weights = routing()
    assert not get_calibrate_all_experts_flag(), "expected routed mode here"

    with torch.no_grad():
        reference = experts(hidden, index, weights)
        got = run_expert_parallel(
            experts, hidden, index, weights, world_size,
            calibrate_all_experts=False,
        )

    torch.testing.assert_close(got, reference, rtol=1e-5, atol=1e-6)


def test_uneven_shards_still_reassemble_correctly():
    """TOKENS need not divide world_size; the short shard must not shift rows."""
    experts = build_experts()
    hidden, index, weights = routing(tokens=25)   # 25 over 4 ranks -> 7,7,7,4

    with moe_calibration_context(), torch.no_grad():
        reference = experts(hidden, index, weights)
        got = run_expert_parallel(experts, hidden, index, weights, 4)

    assert got.shape[0] == 25
    torch.testing.assert_close(got, reference, rtol=1e-5, atol=1e-6)


# --------------------------------------------------------------------------
# Hessian completeness -- the silent failure
# --------------------------------------------------------------------------


@pytest.mark.parametrize("world_size", [2, 4, 8])
def test_every_expert_sees_the_full_calibration_set(world_size):
    """The property whose violation is silent.

    Under expert parallelism each expert runs on exactly one rank. If that rank
    fed it only its own data shard, the expert's Hessian would cover 1/W of the
    calibration set with no error anywhere. Assert the rows match the unsharded
    run, per expert.
    """
    experts = build_experts()
    hidden, index, weights = routing()

    with moe_calibration_context(), torch.no_grad():
        ref_rec = InputRecorder(experts)
        experts(hidden, index, weights)
        reference_rows = {i: ref_rec.rows(i) for i in range(N_EXPERTS)}
        ref_rec.remove()

        ep_rec = InputRecorder(experts)
        run_expert_parallel(experts, hidden, index, weights, world_size)
        ep_rows = {i: ep_rec.rows(i) for i in range(N_EXPERTS)}
        ep_rec.remove()

    for expert in range(N_EXPERTS):
        ref, got = reference_rows[expert], ep_rows[expert]
        assert got.shape[0] == ref.shape[0], (
            f"expert {expert} saw {got.shape[0]} rows under EP but "
            f"{ref.shape[0]} unsharded -- its Hessian would cover "
            f"{got.shape[0] / max(ref.shape[0], 1):.0%} of the data"
        )
        torch.testing.assert_close(got, ref, rtol=0, atol=0)


@pytest.mark.parametrize("world_size", [2, 4, 8])
def test_each_expert_is_evaluated_on_exactly_one_rank(world_size):
    """Double evaluation would double-count that expert in the all-reduce."""
    experts = build_experts()
    hidden, index, weights = routing()

    with moe_calibration_context(), torch.no_grad():
        rec = InputRecorder(experts)
        run_expert_parallel(experts, hidden, index, weights, world_size)
        calls = {i: len(rec.seen[i]) for i in range(N_EXPERTS)}
        rec.remove()

    assert all(n == 1 for n in calls.values()), calls


@pytest.mark.parametrize("world_size", [2, 4, 8])
def test_no_rank_evaluates_an_expert_it_does_not_own(world_size):
    """Where the memory saving comes from: a non-owned expert must never run,
    or GPTQ allocates its Hessian on that rank anyway."""
    experts = build_experts()
    hidden, index, weights = routing()

    owner_of = {}
    for rank in range(world_size):
        for expert in shard_experts(N_EXPERTS, rank, world_size):
            owner_of[expert] = rank

    seen_by = {}

    def make_hook(expert_index):
        def hook(_module, _args):
            seen_by.setdefault(expert_index, []).append(
                threading.current_thread().name
            )
        return hook

    handles = [
        experts[i].gate_proj.register_forward_pre_hook(make_hook(i))
        for i in range(N_EXPERTS)
    ]
    try:
        with moe_calibration_context(), torch.no_grad():
            run_expert_parallel(experts, hidden, index, weights, world_size)
    finally:
        for h in handles:
            h.remove()

    # Every expert ran, and each on a single thread (i.e. a single rank).
    assert set(seen_by) == set(range(N_EXPERTS))
    for expert, threads_used in seen_by.items():
        assert len(set(threads_used)) == 1, (
            f"expert {expert} evaluated on {len(set(threads_used))} ranks"
        )
    # Distinct owners must map to distinct threads.
    thread_of = {e: t[0] for e, t in seen_by.items()}
    for a in range(N_EXPERTS):
        for b in range(N_EXPERTS):
            if owner_of[a] == owner_of[b]:
                assert thread_of[a] == thread_of[b]
            else:
                assert thread_of[a] != thread_of[b]


@pytest.mark.parametrize("world_size", [2, 4, 8])
@torch.no_grad()
def test_gptq_hooks_preserve_gathered_hessians_and_counts(world_size):
    """Real GPTQ hooks see all gate/up/down inputs and rank contributions."""
    experts = build_experts()
    hidden, index, weights = routing(tokens=31)
    modules = {
        name: module for name, module in experts.named_modules()
        if isinstance(module, torch.nn.Linear)
    }
    assert len(modules) == N_EXPERTS * 3

    def collect(run):
        modifier = GPTQModifier()
        inputs = {name: [] for name in modules}
        contributions = {name: [] for name in modules}
        handles = []

        def record(name):
            def hook(module, args, output):
                inputs[name].append(args[0].detach().clone())
                contributions[name].append(get_expert_hessian_contributions())
                modifier.calibrate_module(module, args, output)
            return hook

        for name, module in modules.items():
            handles.append(module.register_forward_hook(record(name)))
        try:
            run()
        finally:
            for handle in handles:
                handle.remove()
        return modifier, inputs, contributions

    def reference_forward():
        for h, i, w in zip(
            torch.chunk(hidden, world_size),
            torch.chunk(index, world_size),
            torch.chunk(weights, world_size),
        ):
            experts(h, i, w)

    with moe_calibration_context():
        reference, reference_inputs, reference_counts = collect(reference_forward)
        gathered, gathered_inputs, gathered_counts = collect(
            lambda: run_expert_parallel(experts, hidden, index, weights, world_size)
        )

    for name, module in modules.items():
        assert reference_counts[name] == [None] * world_size
        assert gathered_counts[name] == [world_size]
        # Gate/up receive exact original rows; down receives computed activations.
        tolerance = dict(rtol=1e-5, atol=1e-6) if "down_proj" in name else dict(
            rtol=0, atol=0
        )
        torch.testing.assert_close(
            torch.cat(gathered_inputs[name]), torch.cat(reference_inputs[name]),
            **tolerance,
        )
        ref_n = reference._num_samples[module]
        ep_n = gathered._num_samples[module]
        assert ref_n.item() == ep_n.item() == world_size
        ref_h = reference._hessians[module]
        ep_h = gathered._hessians[module]
        torch.testing.assert_close(ep_h, ref_h, rtol=1e-5, atol=1e-6)
        torch.testing.assert_close(ep_h / ep_n, ref_h / ref_n, rtol=1e-5, atol=1e-6)
