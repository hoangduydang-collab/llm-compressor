"""
Utility functions for metrics logging and GPU memory monitoring.

This module provides functions for tracking device memory usage, loss, and runtime
during module compression (optimization). Supports both NVIDIA and AMD GPU monitoring
"""

import os
import socket
import time
import uuid
from collections.abc import Iterable
from contextlib import contextmanager
from contextvars import ContextVar
from pathlib import Path

import torch
from compressed_tensors.offload import is_distributed
from loguru import logger

__all__ = ["CompressionLogger", "compression_phase"]


_PARENT_SPAN = ContextVar("compression_parent_span", default=None)


def _read_counter_file(path: Path) -> dict:
    """Preserve raw platform counters and their source, including unavailable reads."""
    try:
        return {"available": True, "source": str(path), "raw": path.read_text()}
    except (OSError, UnicodeError) as exc:
        return {"available": False, "source": str(path), "reason": type(exc).__name__}


def _phase_rank() -> int:
    if torch.distributed.is_available() and torch.distributed.is_initialized():
        return torch.distributed.get_rank()
    return int(os.environ.get("RANK", "0"))


def _phase_snapshot(rank: int) -> dict:
    # /proc reports process I/O, not unique filesystem or CephFS traffic. Raw
    # counters retain units/semantics and make later diagnostic parsing possible.
    memory = _read_counter_file(Path("/proc/self/status"))
    if memory["available"]:
        values = dict(
            line.split(":", 1) for line in memory.pop("raw").splitlines() if ":" in line
        )
        for source_key, key in (("VmRSS", "rss_bytes"), ("VmHWM", "peak_rss_bytes")):
            value = values.get(source_key)
            memory[key] = int(value.split()[0]) * 1024 if value else None
        memory["peak_scope"] = "process lifetime high water mark"
    io = _read_counter_file(Path("/proc/self/io"))
    if io["available"]:
        io["counters"] = {
            key: int(value)
            for key, value in (
                line.split(":", 1) for line in io.pop("raw").splitlines()
            )
        }
    process = {
        "scope": "process; I/O is not unique filesystem traffic",
        "memory": memory,
        "io": io,
    }
    gpu = {
        "available": False,
        "reason": "CUDA allocator not initialized",
        "peak_scope": "rank process allocator since initialization or last peak reset",
    }
    # Never initialize CUDA, reset peaks, or synchronize just for a measurement.
    if torch.cuda.is_initialized():
        try:
            device = torch.cuda.current_device()
            gpu.update(
                available=True,
                reason=None,
                device=device,
                allocated_bytes=torch.cuda.memory_allocated(device),
                reserved_bytes=torch.cuda.memory_reserved(device),
                peak_allocated_bytes=torch.cuda.max_memory_allocated(device),
                peak_reserved_bytes=torch.cuda.max_memory_reserved(device),
            )
        except (RuntimeError, AssertionError) as exc:
            gpu["reason"] = type(exc).__name__

    # One collector per node at each boundary. The supported EP topology is one
    # node; torchrun LOCAL_RANK extends the same policy to multi-node callers.
    local_rank = int(os.environ.get("LOCAL_RANK", str(rank)))
    node = {"available": False, "reason": "collected by node local rank zero"}
    if local_rank == 0:
        membership = _read_counter_file(Path("/proc/self/cgroup"))
        cgroup = None
        for line in membership.get("raw", "").splitlines():
            if line.startswith("0::"):
                relative = line[3:].lstrip("/")
                # A cgroup namespace may expose the host-relative membership
                # with '..'. Never accidentally inspect outside the mount.
                if ".." not in Path(relative).parts:
                    cgroup = Path("/sys/fs/cgroup") / relative
                break
        node = {
            "available": True,
            "scope": "node collector; cgroup counters cover current cgroup only",
            "membership": membership,
            "host_memory": _read_counter_file(Path("/proc/meminfo")),
            "host_memory_pressure": _read_counter_file(Path("/proc/pressure/memory")),
            "host_io_pressure": _read_counter_file(Path("/proc/pressure/io")),
            "cgroup": {"available": False, "reason": "cgroup v2 path unavailable"},
        }
        if cgroup is not None:
            node["cgroup"] = {
                "source": str(cgroup),
                "counters": {
                    name: _read_counter_file(cgroup / name)
                    for name in (
                        "memory.current",
                        "memory.peak",
                        "memory.stat",
                        "memory.events",
                        "memory.pressure",
                        "io.stat",
                        "io.pressure",
                    )
                },
            }
            node["cgroup"]["available"] = any(
                counter["available"] for counter in node["cgroup"]["counters"].values()
            )
    return {"process": process, "gpu": gpu, "node": node}


def _safe_phase_snapshot(rank: int) -> dict:
    try:
        return _phase_snapshot(rank)
    except Exception as exc:
        return {"available": False, "reason": type(exc).__name__}


@contextmanager
def compression_phase(name: str, *, collect_snapshot: bool = True, **identity):
    """Emit paired structured loguru spans without changing work semantics.

    Durations use the local monotonic clock and include waits. Snapshots are
    boundary observations, not phase-specific peaks or pure disk/compute time.
    External sinks installed by callers preserve these records across resets.
    An interrupted process can leave an unmatched start; consumers must retain
    it as incomplete rather than inventing a zero duration.
    """
    rank = _phase_rank()
    span_id = uuid.uuid4().hex
    parent = _PARENT_SPAN.get()
    world_size = (
        torch.distributed.get_world_size()
        if torch.distributed.is_available() and torch.distributed.is_initialized()
        else int(os.environ.get("WORLD_SIZE", "1"))
    )
    bound = logger.bind(
        phase=name,
        span_id=span_id,
        parent_span_id=parent,
        rank=rank,
        world_size=world_size,
        node=socket.gethostname(),
        identity=identity,
    )
    start = time.perf_counter_ns()
    bound.bind(
        event="phase_start",
        timestamp_ns=start,
        duration_ns=None,
        status="running",
        snapshot=_safe_phase_snapshot(rank) if collect_snapshot else {},
    ).debug("phase_start {}", name)
    token = _PARENT_SPAN.set(span_id)
    status = "ok"
    error_type = None
    try:
        yield span_id
    except BaseException as exc:
        status = "error"
        error_type = type(exc).__name__
        raise
    finally:
        stop = time.perf_counter_ns()
        _PARENT_SPAN.reset(token)
        bound.bind(
            event="phase_end",
            timestamp_ns=stop,
            duration_ns=stop - start,
            status=status,
            error_type=error_type,
            snapshot=_safe_phase_snapshot(rank) if collect_snapshot else {},
        ).debug("phase_end {}", name)


class CompressionLogger:
    """
    Log metrics related to compression algorithms
    """

    def __init__(self, module: torch.nn.Module):
        self.module = module
        self.start_tick = None

        self._name = None
        self._loss = None

    def set_results(
        self,
        name: str | None = None,
        loss: float | None = None,
    ):
        self._name = name
        self._loss = loss

    def __enter__(self) -> "CompressionLogger":
        self.start_tick = time.time()
        return self

    def __exit__(self, _exc_type, _exc_val, _exc_tb):
        stop_tick = time.time()

        patch = logger.patch(lambda r: r.update(function=(self._name or "compress")))

        patch.log("METRIC", f"time {(stop_tick - self.start_tick):.2f}s")
        if self._loss is not None:
            patch.log("METRIC", f"error {self._loss:.2f}")

        if not torch.accelerator.is_available():
            return

        for device_id in _get_visible_devices():
            used_memory = torch.accelerator.max_memory_allocated(device_id) / 1e9
            max_memory = torch.accelerator.get_memory_info(device_id)[1] / 1e9
            perc_used = 100 * used_memory / max_memory
            patch.log(
                "METRIC",
                (
                    f"Accelerator {device_id} | usage: {perc_used:.2f}%"
                    f" | total memory: {max_memory:.1f} Gb"
                ),
            )


def _get_visible_devices() -> Iterable:
    if is_distributed():
        return [torch.accelerator.current_device_index()]

    else:
        return range(torch.accelerator.device_count())
