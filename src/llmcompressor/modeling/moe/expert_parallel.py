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
from dataclasses import dataclass

import torch
from torch import distributed as dist

__all__ = [
    "ExpertParallelContext",
    "expert_parallel_context",
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

    def __post_init__(self):
        if self.world_size < 1:
            raise ValueError(f"world_size must be >= 1, got {self.world_size}")
        if not 0 <= self.rank < self.world_size:
            raise ValueError(
                f"rank {self.rank} outside [0, {self.world_size})"
            )


_CONTEXT: ExpertParallelContext | None = None


def get_expert_parallel_context() -> ExpertParallelContext | None:
    return _CONTEXT


def is_expert_parallel_enabled() -> bool:
    """True only when sharding would actually change anything.

    A world size of 1 is deliberately reported as disabled: the sharded path
    would be a no-op wrapped in collectives, and the unsharded path is the one
    with validation history behind it.
    """
    return _CONTEXT is not None and _CONTEXT.world_size > 1


@contextlib.contextmanager
def expert_parallel_context(
    rank: int | None = None,
    world_size: int | None = None,
    group=None,
):
    """Enable expert-parallel calibration for the duration of the block.

    Rank and world size default to the initialized default process group, so
    the caller does not have to thread them through. Restores the previous
    context on exit, including on exception, so a failed calibration cannot
    leave the module globally sharded.

    :param rank: this process's rank; inferred from ``dist`` when omitted.
    :param world_size: total ranks; inferred from ``dist`` when omitted.
    :param group: process group for collectives.
    """
    global _CONTEXT

    if rank is None or world_size is None:
        if not (dist.is_available() and dist.is_initialized()):
            raise RuntimeError(
                "expert_parallel_context needs an initialized process group, "
                "or explicit rank/world_size"
            )
        rank = dist.get_rank(group) if rank is None else rank
        world_size = dist.get_world_size(group) if world_size is None else world_size

    previous = _CONTEXT
    _CONTEXT = ExpertParallelContext(rank=rank, world_size=world_size, group=group)
    try:
        yield _CONTEXT
    finally:
        _CONTEXT = previous


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
