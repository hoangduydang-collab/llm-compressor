"""Expert-parallel calibration for linearized MoE experts.

WHY. GPTQ accumulates one Hessian per targeted Linear, and every rank
accumulates a Hessian for EVERY targeted module before the reduce frees
non-owners (``modifiers/gptq/base.py``). Peak residency is therefore a property
of the layer, not of the world size: for GLM-5.2 (hidden 6144,
moe_intermediate 2048, 256 routed experts) that is

    256 experts x (gate 144 + up 144 + down 16 MiB) = 76.0 GiB per rank

which does not fit next to the layer's own weights on an 80 GiB card. Adding
GPUs does not help. ``offload_hessians`` does, but it round-trips every Hessian
over PCIe once per module per sample: measured 3:46 for one MoE layer at 8
samples, i.e. hours per layer at 512.

WHAT THIS DOES. Shard the experts across ranks so rank r only ever instantiates
Hessians for the experts it owns: 76 GiB -> ~19 GiB at world size 4.

The subtlety that makes it correct. Calibration is already data-parallel: rank r
holds data shard D_r. If rank r also only ran experts E_r, then expert j's
Hessian would cover only D_r -- silently incomplete, and it would look fine,
because a Hessian accumulated over a quarter of the data is a perfectly
well-formed matrix. So the activations are all-gathered first: every rank sees
ALL tokens, and runs only its own experts over them. Expert j's Hessian is then
complete on its owning rank and needs no cross-rank reduction at all.

Total work is unchanged, only redistributed: at W=4, 256 experts x T/4 tokens
per rank becomes 64 experts x T tokens per rank.

The MoE output is then all-reduced so every rank can propagate the true layer
output, and each rank returns its own data slice -- so everything downstream of
this module sees exactly the data-parallel semantics it saw before.

``calibrate_all_experts`` is the reason the collectives are this simple. With it
set (the default, and what MiniMax-M3 was validated with) every expert sees
every token, so a plain all-gather of activations suffices. With it unset only
routed tokens reach each expert, which needs a token permutation and an
all-to-all; ``routed_dispatch`` below covers that case.
"""

import contextlib
import threading
from dataclasses import dataclass

import torch
from torch import distributed as dist

__all__ = [
    "Collectives",
    "DistCollectives",
    "ExpertParallelContext",
    "expert_parallel_context",
    "expert_parallel_forward",
    "get_expert_parallel_context",
    "is_expert_parallel_enabled",
    "shard_experts",
    "routed_dispatch",
    "RoutedDispatch",
]


@dataclass(frozen=True)
class ExpertParallelContext:
    """How experts are split, and over what process group.

    :param rank: this process's index within ``world_size``.
    :param world_size: number of ranks sharing the experts.
    :param group: process group for the collectives; ``None`` means default.
    """

    rank: int
    world_size: int
    group: object = None
    # Transport override. Production leaves this None and gets DistCollectives;
    # carrying it here rather than patching a module global is what lets several
    # simulated ranks coexist in one process, each with its own transport.
    collectives: "Collectives | None" = None

    def __post_init__(self):
        if self.world_size < 1:
            raise ValueError(f"world_size must be >= 1, got {self.world_size}")
        if not 0 <= self.rank < self.world_size:
            raise ValueError(
                f"rank {self.rank} outside [0, {self.world_size})"
            )


# Thread-local, not a plain global. A production run is one process per rank, so
# this costs nothing there -- but it means several ranks can be simulated as
# threads without their contexts clobbering each other, and it prevents an
# unrelated worker thread from inheriting a calibration context it never entered.
_STATE = threading.local()


def get_expert_parallel_context() -> ExpertParallelContext | None:
    return getattr(_STATE, "context", None)


def is_expert_parallel_enabled() -> bool:
    """True only when sharding would actually change anything.

    A world size of 1 is deliberately reported as disabled: the sharded path
    would be a no-op wrapped in collectives, and the unsharded path is the one
    with validation history behind it.
    """
    context = get_expert_parallel_context()
    return context is not None and context.world_size > 1


@contextlib.contextmanager
def expert_parallel_context(
    rank: int | None = None,
    world_size: int | None = None,
    group=None,
    collectives: "Collectives | None" = None,
):
    """Enable expert-parallel calibration for the duration of the block.

    Rank and world size default to the initialized default process group, so
    the caller does not have to thread them through. Restores the previous
    context on exit, including on exception, so a failed calibration cannot
    leave the module globally sharded.

    :param rank: this process's rank; inferred from ``dist`` when omitted.
    :param world_size: total ranks; inferred from ``dist`` when omitted.
    :param group: process group for collectives.
    :param collectives: transport override; ``None`` uses ``DistCollectives``.
    """
    if rank is None or world_size is None:
        if not (dist.is_available() and dist.is_initialized()):
            raise RuntimeError(
                "expert_parallel_context needs an initialized process group, "
                "or explicit rank/world_size"
            )
        rank = dist.get_rank(group) if rank is None else rank
        world_size = dist.get_world_size(group) if world_size is None else world_size

    previous = get_expert_parallel_context()
    _STATE.context = ExpertParallelContext(
        rank=rank, world_size=world_size, group=group, collectives=collectives
    )
    try:
        yield _STATE.context
    finally:
        _STATE.context = previous


def shard_experts(num_experts: int, rank: int, world_size: int) -> list[int]:
    """Assign expert indices to ``rank``, balanced and covering.

    Contiguous blocks, with the first ``num_experts % world_size`` ranks taking
    one extra. Contiguity is not cosmetic: it keeps a rank's experts adjacent in
    the checkpoint, so a future loader can shard by index without a gather.

    Every expert is owned by exactly one rank and none is dropped -- the
    property that separates "sharded" from "silently quantizing 3/4 of the
    model". Guaranteed by construction here and asserted in the tests.

    :return: expert indices owned by ``rank``, ascending.
    """
    if num_experts < 0:
        raise ValueError(f"num_experts must be >= 0, got {num_experts}")
    if world_size < 1:
        raise ValueError(f"world_size must be >= 1, got {world_size}")
    if not 0 <= rank < world_size:
        raise ValueError(f"rank {rank} outside [0, {world_size})")

    base, extra = divmod(num_experts, world_size)
    start = rank * base + min(rank, extra)
    count = base + (1 if rank < extra else 0)
    return list(range(start, start + count))


@dataclass
class RoutedDispatch:
    """Bookkeeping for the routed (``calibrate_all_experts=False``) path.

    Flattens ``[T, top_k]`` assignments into ``N = T * top_k`` rows grouped by
    owning rank, which is exactly the layout ``all_to_all_single`` wants, and
    keeps what the combine step needs to scatter results back to their tokens.

    :param perm: row order grouped by destination rank.
    :param send_counts: rows going to each rank (all_to_all split sizes).
    :param dest_rank: destination rank per (token, slot) assignment.
    :param src_token: originating token index per assignment.
    :param local_expert: expert index local to its destination rank.
    :param weight: router weight per assignment.
    """

    perm: torch.Tensor
    send_counts: torch.Tensor
    dest_rank: torch.Tensor
    src_token: torch.Tensor
    local_expert: torch.Tensor
    weight: torch.Tensor


def routed_dispatch(
    top_k_index: torch.Tensor,
    top_k_weights: torch.Tensor,
    num_experts: int,
    world_size: int,
) -> RoutedDispatch:
    """Plan the token exchange for routed (non-all-experts) calibration.

    Uses the same ``shard_experts`` ownership as the all-gather path, so a
    model can switch between them without experts changing hands. Owner rank is
    derived by searching the shard boundaries rather than by ``expert //
    experts_per_rank``, because that division is only correct when
    ``world_size`` divides ``num_experts`` -- GLM-5.2's 256 over 6 ranks does
    not, and the naive form would silently route to a rank that does not own
    the expert.

    :param top_k_index: ``[T, top_k]`` selected expert per token per slot.
    :param top_k_weights: ``[T, top_k]`` router weight, same layout.
    :param num_experts: total routed experts in the layer.
    :param world_size: ranks sharing them.
    """
    if top_k_index.shape != top_k_weights.shape:
        raise ValueError(
            f"index/weight shape mismatch: {tuple(top_k_index.shape)} vs "
            f"{tuple(top_k_weights.shape)}"
        )

    device = top_k_index.device
    # boundaries[r] = first expert owned by rank r; boundaries[-1] = num_experts
    counts = [
        len(shard_experts(num_experts, r, world_size)) for r in range(world_size)
    ]
    boundaries = torch.tensor(
        [0] + list(torch.tensor(counts).cumsum(0)), device=device
    )

    top_k = top_k_index.shape[1]
    src_token = torch.arange(top_k_index.shape[0], device=device).repeat_interleave(
        top_k
    )
    flat_expert = top_k_index.reshape(-1)
    # right=True then -1 maps expert e to the rank whose block contains it.
    dest_rank = torch.searchsorted(boundaries, flat_expert, right=True) - 1
    local_expert = flat_expert - boundaries[dest_rank]

    perm = torch.argsort(dest_rank, stable=True)  # contiguous per-rank runs
    send_counts = torch.bincount(dest_rank, minlength=world_size)

    return RoutedDispatch(
        perm=perm,
        send_counts=send_counts,
        dest_rank=dest_rank,
        src_token=src_token,
        local_expert=local_expert,
        weight=top_k_weights.reshape(-1),
    )


class Collectives:
    """The two collectives the sharded forward needs.

    Injectable so the equivalence tests can simulate a whole world inside one
    CPU process. That is not a convenience: the property under test is that
    every expert's Hessian covers ALL ranks' data, and checking it requires
    holding every rank's state at once and comparing against a single-rank
    reference. A real multi-process gloo test cannot make that comparison
    without shipping Hessians between processes, which is the very thing this
    design removes.
    """

    def all_gather(self, tensor: torch.Tensor) -> torch.Tensor:
        """Concatenate ``tensor`` from every rank along dim 0, in rank order."""
        raise NotImplementedError

    def all_reduce_sum(self, tensor: torch.Tensor) -> torch.Tensor:
        """Sum ``tensor`` across ranks; every rank receives the total."""
        raise NotImplementedError


class DistCollectives(Collectives):
    """``torch.distributed`` implementation."""

    def __init__(self, group=None):
        self.group = group

    def all_gather(self, tensor: torch.Tensor) -> torch.Tensor:
        world_size = dist.get_world_size(self.group)
        # all_gather requires equally-shaped buffers. Calibration shards are
        # produced by partition_bounds' floor division, so the last rank can
        # hold fewer samples; pad to the max and trim after concatenating.
        local = torch.tensor([tensor.shape[0]], device=tensor.device)
        sizes = [torch.zeros_like(local) for _ in range(world_size)]
        dist.all_gather(sizes, local, group=self.group)
        counts = [int(s.item()) for s in sizes]
        largest = max(counts)

        padded = tensor
        if tensor.shape[0] < largest:
            pad = torch.zeros(
                (largest - tensor.shape[0], *tensor.shape[1:]),
                dtype=tensor.dtype,
                device=tensor.device,
            )
            padded = torch.cat([tensor, pad], dim=0)

        buffers = [torch.empty_like(padded) for _ in range(world_size)]
        dist.all_gather(buffers, padded, group=self.group)
        return torch.cat([b[:n] for b, n in zip(buffers, counts)], dim=0)

    def all_reduce_sum(self, tensor: torch.Tensor) -> torch.Tensor:
        dist.all_reduce(tensor, op=dist.ReduceOp.SUM, group=self.group)
        return tensor


def expert_parallel_forward(
    experts,
    hidden_states: torch.Tensor,
    top_k_index: torch.Tensor,
    top_k_weights: torch.Tensor,
    context: ExpertParallelContext,
    collectives: Collectives,
    calibrate_all_experts: bool,
) -> torch.Tensor:
    """``LinearExperts2D.forward``, with the expert loop sharded across ranks.

    Mirrors the unsharded forward's arithmetic exactly -- one-hot mask,
    ``torch.where`` token selection, router weighting, ``index_add_``
    accumulation -- so the only difference is WHICH experts this rank evaluates
    and over WHOSE tokens.

    Order of operations, and why:

    1. All-gather activations and routing, so this rank sees every rank's
       tokens. Without this step rank r would accumulate expert j's Hessian
       over its own data shard only. Nothing would raise -- a Hessian over
       1/W of the data is a well-formed matrix -- and the resulting checkpoint
       would load and serve. This is the failure mode the design exists to
       prevent, so the gather is not an optimization detail.
    2. Evaluate ONLY this rank's experts. Non-owned experts are never called,
       so GPTQ never allocates their Hessians: that is where the memory is won.
    3. All-reduce the accumulator. Each rank has only its own experts'
       contributions; the true layer output is the sum.
    4. Return this rank's slice, so everything downstream sees the same
       data-parallel semantics as before.

    The gather is correct for BOTH calibration modes -- routed selection just
    happens over the gathered token set. ``routed_dispatch`` exists to cut
    communication when only routed tokens are needed, not to fix a correctness
    gap here.

    :param experts: the ``LinearExperts2D`` (indexable, ``num_experts``).
    :param hidden_states: ``[T, H]`` this rank's tokens.
    :param top_k_index: ``[T, top_k]`` selected experts.
    :param top_k_weights: ``[T, top_k]`` router weights.
    :param context: expert ownership for this rank.
    :param collectives: transport.
    :param calibrate_all_experts: route every token through every expert.
    :return: ``[T, H]`` this rank's slice of the layer output.
    """
    local_rows = hidden_states.shape[0]

    gathered_states = collectives.all_gather(hidden_states)
    gathered_index = collectives.all_gather(top_k_index)
    gathered_weights = collectives.all_gather(top_k_weights)

    # Where this rank's rows sit inside the gathered tensor. Recovered from the
    # gathered size rather than assumed to be rank * local_rows, because uneven
    # shards make that product wrong for every rank after the short one.
    offsets = collectives.all_gather(
        torch.tensor([local_rows], device=hidden_states.device)
    )
    start = int(offsets[: context.rank].sum())

    final_hidden_states = torch.zeros_like(gathered_states)

    with torch.no_grad():
        expert_mask = torch.nn.functional.one_hot(
            gathered_index, experts.num_experts
        )
        expert_mask = expert_mask.permute(2, 1, 0)

    for expert_index in shard_experts(
        experts.num_experts, context.rank, context.world_size
    ):
        top_k_pos, token_indices = torch.where(expert_mask[expert_index])

        expert = experts[expert_index]
        if calibrate_all_experts:
            expert_output = expert(gathered_states)[token_indices]
        else:
            expert_output = expert(gathered_states[token_indices])

        expert_weights = gathered_weights[token_indices, top_k_pos, None]
        weighted_output = expert_output * expert_weights

        final_hidden_states.index_add_(
            0, token_indices, weighted_output.to(final_hidden_states.dtype)
        )

    # Sum the per-rank partial accumulators into the true layer output.
    collectives.all_reduce_sum(final_hidden_states)

    return final_hidden_states[start : start + local_rows]
