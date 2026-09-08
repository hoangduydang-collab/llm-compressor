"""GPTQ ownership and publication adapters for all-expert calibration.

Reuse MoE-Quant's distinction between expert-local and replicated Hessians:
https://github.com/IST-DASLab/MoE-Quant/blob/master/quant.py
Transport, solving and persistent storage remain owned by PyTorch, GPTQ and
compressed-tensors respectively. No checkpoint loader or cache is implemented here.
"""

from itertools import zip_longest

import torch
from compressed_tensors.offload.cache.base import OffloadCache
from compressed_tensors.offload.cache.cpu import CPUCache
from compressed_tensors.offload.cache.dist_disk import DistributedDiskCache
from compressed_tensors.offload.dist_utils import as_broadcastable
from compressed_tensors.utils import get_execution_device, update_offload_parameter
from torch import distributed as dist

from llmcompressor.modeling.moe.expert_parallel import shard_experts
from llmcompressor.modeling.moe.linear_experts import LinearExperts2D


def routed_expert_owners(model, world_size):
    """Assign actual routed Linear objects, excluding routers/shared experts."""
    owners = {}
    for experts in model.modules():
        if not isinstance(experts, LinearExperts2D):
            continue
        for rank in range(world_size):
            for index in shard_experts(experts.num_experts, rank, world_size):
                for module in experts[index].modules():
                    if isinstance(module, torch.nn.Linear):
                        previous = owners.setdefault(module, rank)
                        if previous != rank:
                            raise ValueError(
                                "A routed module has conflicting EP owners"
                            )
    return owners


def gather_ep_records(record):
    records = [None] * dist.get_world_size()
    dist.all_gather_object(records, record)
    return records


def agree_ep_manifest(manifest):
    """Every rank either agrees or raises before entering tensor collectives."""
    records = gather_ep_records(manifest)
    if any(record != records[0] for record in records[1:]):
        raise ValueError(f"EP manifests differ across ranks: {records!r}")
    errors = manifest.get("errors", [])
    if errors:
        raise ValueError("EP preflight failed: " + "; ".join(errors))


def parameter_storage(module, name):
    """Read metadata without onloading a weight just to inspect its shape."""
    for cache in (module._parameters, module._buffers):
        values = cache.offloaded_values if isinstance(cache, OffloadCache) else cache
        if name in values:
            return cache, values[name]
    return None, None


def solve_rounds(rank_to_modules):
    """Bound retained solver outputs to one module per rank, even uneven queues."""
    for row in zip_longest(*rank_to_modules, fillvalue=None):
        yield [
            (owner, module) for owner, module in enumerate(row) if module is not None
        ]


def validate_hessian_coverage(modules, names, owners, hessians, counts):
    expected = {names[m]: owners.get(m) for m in modules}
    agree_ep_manifest({"targets": sorted(expected.items())})
    local = {names[m]: float(counts[m]) for m in hessians if m in names and m in counts}
    records = gather_ep_records(local)
    errors = []
    for rank, record in enumerate(records):
        wanted = {name for name, owner in expected.items() if owner in (None, rank)}
        actual = set(record)
        if actual != wanted:
            errors.append(
                f"rank {rank}: missing={sorted(wanted - actual)}, "
                f"extra={sorted(actual - wanted)}"
            )
        for name, count in record.items():
            if not 0 < count < float("inf"):
                errors.append(
                    f"rank {rank}: invalid contribution count for {name}: {count}"
                )
    if errors:
        raise ValueError("EP Hessian coverage failed: " + "; ".join(errors))


def _persist_result(module, name, tensor):
    cache, old = parameter_storage(module, name)
    if isinstance(cache, DistributedDiskCache) and "update_offload" not in vars(
        type(cache)
    ):
        raise ValueError(
            "EP disk publication requires distributed-safe update_offload; "
            "use pipeline.run's existing distributed disk update patch"
        )
    if cache is None:
        # All ranks register the same optional GPTQ attribute. Distributed caches
        # perform their existing allocation/metadata collectives in this call.
        module.register_parameter(name, torch.nn.Parameter(tensor, requires_grad=False))
    elif old is None or old.shape != tensor.shape or old.dtype != tensor.dtype:
        if isinstance(cache, OffloadCache):
            # OffloadCache handles allocation on shape changes; remove a same-
            # shape different-dtype entry so it cannot silently cast on update.
            if old is not None and old.shape == tensor.shape:
                del cache[name]
            cache[name] = tensor
        elif name in module._parameters:
            module.register_parameter(
                name, torch.nn.Parameter(tensor, requires_grad=False)
            )
        else:
            module.register_buffer(name, tensor)
    elif isinstance(cache, CPUCache):
        # Existing CPU entries can be private or shared. Serial helper calls
        # refresh both storage and each rank's cached onload, without races or
        # another shared-memory abstraction. Creation above remains collective.
        for writer in range(dist.get_world_size()):
            if dist.get_rank() == writer:
                update_offload_parameter(module, name, tensor)
            dist.barrier()
    else:
        # For distributed disk, EVERY rank enters the source-write/barrier patch.
        # Device-local/plain parameters simply update locally after broadcast.
        update_offload_parameter(module, name, tensor)


def publish_gptq_result(module, owner, local_result, name=None):
    """Publish one solved module without reading/pinning non-owner old weights."""
    rank = dist.get_rank()
    metadata = [
        [
            (key, tuple(value.shape), value.dtype)
            for key, value in sorted(local_result.items())
        ]
        if rank == owner
        else None
    ]
    dist.broadcast_object_list(metadata, src=owner)
    for attribute, shape, dtype in metadata[0]:
        tensor = (
            local_result[attribute].contiguous()
            if rank == owner
            else torch.empty(shape, dtype=dtype, device=get_execution_device(module))
        )
        dist.broadcast(as_broadcastable(tensor), src=owner)
        _persist_result(module, attribute, tensor)
