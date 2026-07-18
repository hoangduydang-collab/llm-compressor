"""Small pipeline-level wrapper around compressed-tensors distributed setup.

The quantization algorithms and model offload remain owned by llm-compressor and
compressed-tensors.  This module only coordinates the outer ``pipeline.run``
lifecycle and rank-safe artifact names.
"""

from __future__ import annotations

import os
from dataclasses import asdict, dataclass
from datetime import timedelta
from pathlib import Path
from typing import Any

# 8h: the disk-offload checkpoint save (r11, 2026-07-18) parks ranks 1..7 in a
# single collective while rank 0 serially gathers ~900 GB from NFS and writes
# ~766 GB of shards (transformers modeling_utils has a TODO acknowledging the
# save loop cannot be parallelized: safetensors holds the GIL). The old 3h
# timeout would watchdog-kill an otherwise-successful full-model save.
_DISTRIBUTED_TIMEOUT = timedelta(hours=8)


@dataclass
class DistributedContext:
    """Distributed process metadata with single-process no-op behavior."""

    enabled: bool = False
    rank: int = 0
    world_size: int = 1
    local_rank: int = 0
    _owns_process_group: bool = False

    @classmethod
    def from_environment(cls) -> "DistributedContext":
        """Initialize CT distributed state when launched by ``torchrun``."""
        raw_world_size = os.environ.get("WORLD_SIZE", "1")
        try:
            world_size = int(raw_world_size)
        except ValueError as exc:
            raise RuntimeError(
                f"WORLD_SIZE must be an integer, got {raw_world_size!r}"
            ) from exc

        if world_size <= 0:
            raise RuntimeError(f"WORLD_SIZE must be positive, got {world_size}")
        if world_size == 1:
            return cls()

        missing = [name for name in ("RANK", "LOCAL_RANK") if name not in os.environ]
        if missing:
            raise RuntimeError(
                "WORLD_SIZE > 1 requires torchrun rank variables RANK and "
                f"LOCAL_RANK; missing {', '.join(missing)}"
            )

        # Keep imports lazy so ordinary config/launcher commands do not require
        # the GPU quantization environment.
        import torch
        import torch.distributed as dist

        owns_process_group = not dist.is_initialized()
        if owns_process_group:
            rank = int(os.environ["RANK"])
            local_rank = int(os.environ["LOCAL_RANK"])
            device = torch.device(f"cuda:{local_rank}")
            torch.cuda.set_device(device)
            dist.init_process_group(
                backend="nccl",
                init_method="env://",
                rank=rank,
                world_size=world_size,
                device_id=device,
                timeout=_DISTRIBUTED_TIMEOUT,
            )
            dist.barrier()

        if not dist.is_initialized():
            raise RuntimeError("distributed process group initialization failed")

        return cls(
            enabled=True,
            rank=dist.get_rank(),
            world_size=dist.get_world_size(),
            local_rank=int(os.environ["LOCAL_RANK"]),
            _owns_process_group=owns_process_group,
        )

    @property
    def is_source(self) -> bool:
        return self.rank == 0

    def rank_path(self, path: Path) -> Path:
        """Return a per-rank filename while preserving the final suffix."""
        path = Path(path)
        if not self.enabled:
            return path
        return path.with_name(f"{path.stem}.rank-{self.rank}{path.suffix}")

    def broadcast_path(self, path: Path | None) -> Path:
        """Broadcast a rank-zero path to every process."""
        if not self.enabled:
            if path is None:
                raise ValueError("single-process path cannot be None")
            return Path(path)

        import torch.distributed as dist

        payload: list[Any] = [
            str(path) if self.is_source and path is not None else None
        ]
        if self.is_source and payload[0] is None:
            raise ValueError("source rank must provide a path to broadcast")
        dist.broadcast_object_list(payload, src=0)
        if payload[0] is None:
            raise RuntimeError("source rank broadcast an empty run path")
        return Path(payload[0])

    def barrier(self) -> None:
        if self.enabled:
            import torch.distributed as dist

            dist.barrier()

    def close(self) -> None:
        """Destroy only a process group initialized by this context."""
        if self.enabled and self._owns_process_group:
            import torch.distributed as dist

            if dist.is_initialized():
                dist.destroy_process_group()
            self._owns_process_group = False

    def snapshot(self) -> dict[str, object]:
        data = asdict(self)
        data.pop("_owns_process_group")
        data.update(
            {
                "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
                "slurm_step_id": os.environ.get("SLURM_STEP_ID"),
                "node": os.environ.get("COMPUTERNAME") or os.environ.get("HOSTNAME"),
            }
        )
        if self.enabled:
            import torch

            current_device = torch.cuda.current_device()
            data.update(
                {
                    "cuda_current_device": current_device,
                    "cuda_device_name": torch.cuda.get_device_name(current_device),
                }
            )
        return data
