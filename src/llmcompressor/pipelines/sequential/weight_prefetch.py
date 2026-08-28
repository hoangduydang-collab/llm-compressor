"""Overlap a sequential-pipeline layer's weight read with the previous layer's
compute, and release page cache once a layer is done with.

Both halves matter, and the second is not an optimization -- it is what keeps the
first from making things worse.

WHY PREFETCH (and why the earlier verdict was the opposite)
-----------------------------------------------------------
The sequential pipeline reads one decoder layer's weights, then calibrates it,
then moves on. On the Rancher infermesh-test cluster GLM-5.2's MoE layers are 18.4
GiB each and cephfs delivers 125-153 MB/s to 8 concurrent ranks, so the read costs
~120 s/layer while the GPUs sit idle.

Prefetching was initially rejected: during the 32-sample smoke the GPUs were
measured idle in 9 of 10 samples, meaning there was no compute to hide the read
behind. That was true *at that operating point*. Measured 2026-08-28 at 256
samples x 2048 tokens, per-layer compute is ~350 s (calibration forwards ~230 s +
token-independent AWQ grid search ~120 s) against ~120 s of streaming -- 3x more
compute than I/O. At the production operating point the read hides almost
completely, worth ~2.6 h over 78 layers.

WHY RELEASE
-----------
The same day, the smoke's save phase degraded from 240 s to 600+ s per shard. The
cause was not slow writes; it was the pod's memory cgroup pinned at its limit:

    memory.max     = 1825361100800   (1700 GiB)
    memory.current = 1825333997568   (within 27 MB of the limit)
    file           = 1654 GiB of page cache
    memory.events  = high 0, max 0, oom 0      <- nothing ever FAILED
    pgsteal_kswapd unchanged; pgsteal_direct +836,096 pages/30 s
    memory.pressure: full avg10=23.45          <- 23% fully stalled, 1138 s total

`memory.high` is unset, so there is no gentle background throttling: 100% of
reclaim is DIRECT reclaim, performed synchronously inside whichever thread wants
memory. Nothing crashes, everything just runs ~23% slower.

Page cache from streamed layers is what filled it. A prefetcher that only ever
adds pages would reach that state sooner and stall harder. So every file is
released with POSIX_FADV_DONTNEED once the last subgraph that needs it has passed,
which bounds the resident working set to roughly the prefetch window (~2 layers,
~37 GiB) instead of the whole 1403 GiB model.

MECHANISM
---------
`posix_fadvise` only. WILLNEED asks the kernel to read ahead asynchronously --
no user-space copy, no buffer of our own, no thread to manage. DONTNEED drops
clean pages (dirty pages are left alone, so this can never lose a write).

Both are advisory: every call is best-effort and a failure is never fatal. Dropping
pages another rank still wants costs that rank a re-read and nothing else -- page
cache is per-node and all ranks walk the subgraph list in lockstep behind
collectives, so skew is bounded, but correctness never depends on it.
"""

from __future__ import annotations

import os
from typing import TYPE_CHECKING

from loguru import logger

if TYPE_CHECKING:
    import torch

__all__ = ["WeightPrefetcher", "subgraph_source_files", "plan_last_use"]

# Present on Linux; absent on macOS/Windows, where this whole module no-ops.
#
# The advice constants are resolved through getattr with the Linux values as
# fallbacks so the LOGIC (which files, when to release) stays exercisable on a
# non-Linux dev machine -- tests set `_HAS_FADVISE` and stub `_advise`. Only the
# syscall is platform-gated, not the planning it drives.
_HAS_FADVISE = hasattr(os, "posix_fadvise")
_WILLNEED = getattr(os, "POSIX_FADV_WILLNEED", 3)
_DONTNEED = getattr(os, "POSIX_FADV_DONTNEED", 4)


def _module_offload_files(module: "torch.nn.Module") -> set[str]:
    """Backing files for one module's offloaded parameters and buffers.

    Reads ``OffloadCache.offloaded_values`` (the raw name -> meta-tensor mapping)
    rather than iterating the cache as a Mapping: ``__getitem__`` ONLOADS, so
    ``for v in cache.values()`` would stream the entire layer off disk to decide
    what to prefetch, which is precisely the cost being avoided.
    """
    files: set[str] = set()
    for attr in ("_parameters", "_buffers"):
        cache = module.__dict__.get(attr)
        offloaded = getattr(cache, "offloaded_values", None)
        index = getattr(cache, "index", None)
        if offloaded is None or index is None:
            continue  # not offloaded, or offloaded somewhere without files (cpu/gpu)
        for tensor in offloaded.values():
            try:
                info = index.get(tensor)
            except TypeError:
                continue  # unhashable; nothing we can do
            if not info:
                continue
            path = info.get("safetensors_file")
            if path:
                files.add(str(path))
    return files


def subgraph_source_files(model: "torch.nn.Module", subgraph) -> set[str]:
    """Every file backing the offloaded weights of the modules in ``subgraph``.

    Convenience wrapper for a single subgraph. Prefer :func:`build_file_index` when
    planning a whole walk -- calling this per subgraph is quadratic (see there).
    """
    return build_file_index(model, [subgraph]).get(0, set())


def build_file_index(
    model: "torch.nn.Module", subgraphs: list
) -> dict[int, set[str]]:
    """subgraph index -> backing files, computed in ONE pass over the modules.

    Complexity is the whole point of this function. The obvious implementation --
    for each subgraph, scan every module and prefix-match it against that
    subgraph's targets -- is O(subgraphs x modules x targets). GLM-5.2 has 79
    subgraphs and ~60,000 modules (78 layers x 256 experts x 3 projections =
    59,904 leaf Linears alone), and a single MoE subgraph calls ~771 of them, so
    that product is ~3.7e9 string comparisons: hours of Python before the run
    starts, for a page-cache hint.

    Inverting it makes the cost O(modules x depth). Targets are collected into a
    lookup once, then each module name is matched by testing its own name and its
    ancestor prefixes -- at most ~8 dict lookups per module.
    """
    target_owners: dict[str, set[int]] = {}
    for index, subgraph in enumerate(subgraphs):
        for node in subgraph.graph.find_nodes(op="call_module"):
            target_owners.setdefault(node.target, set()).add(index)

    files_by_subgraph: dict[int, set[str]] = {}
    if not target_owners:
        return files_by_subgraph

    for name, module in model.named_modules():
        owners: set[int] = set(target_owners.get(name, ()))
        # A module is also claimed by a target that is one of its ancestors: the
        # AST autowrapper can leave a called submodule without a node of its own.
        parts = name.split(".")
        for depth in range(len(parts) - 1, 0, -1):
            ancestor = ".".join(parts[:depth])
            owners |= target_owners.get(ancestor, set())
        if not owners:
            continue
        files = _module_offload_files(module)
        if not files:
            continue
        for index in owners:
            files_by_subgraph.setdefault(index, set()).update(files)
    return files_by_subgraph


def plan_last_use(model: "torch.nn.Module", subgraphs: list) -> dict[str, int]:
    """file -> index of the LAST subgraph that reads it.

    A shard can back weights in more than one layer, so releasing after the first
    subgraph that touches it would evict pages a later subgraph still needs and
    turn the optimization into extra reads.
    """
    last_use: dict[str, int] = {}
    for index, files in build_file_index(model, subgraphs).items():
        for path in files:
            if last_use.get(path, -1) < index:
                last_use[path] = index
    return last_use


class WeightPrefetcher:
    """Issues WILLNEED ahead of the walk and DONTNEED behind it.

    Disabled instances are inert, so the pipeline can call the methods
    unconditionally.
    """

    def __init__(
        self,
        model: "torch.nn.Module",
        subgraphs: list,
        enabled: bool = False,
        depth: int = 1,
    ):
        self.enabled = bool(enabled) and _HAS_FADVISE and len(subgraphs) > 0
        self.depth = max(1, int(depth))
        self._model = model
        self._subgraphs = subgraphs
        self._files_by_subgraph: dict[int, set[str]] = {}
        self._last_use: dict[str, int] = {}
        self._prefetched: set[str] = set()
        self._released: set[str] = set()

        if enabled and not _HAS_FADVISE:
            logger.warning(
                "weight prefetch requested but os.posix_fadvise is unavailable on "
                "this platform; continuing without it"
            )
        if not self.enabled:
            return

        # ONE pass over named_modules for the whole walk. Doing this per subgraph
        # would be quadratic -- see build_file_index.
        self._files_by_subgraph = build_file_index(model, subgraphs)
        for index, files in self._files_by_subgraph.items():
            for path in files:
                if self._last_use.get(path, -1) < index:
                    self._last_use[path] = index
        if not self._last_use:
            # Model is not disk-offloaded (or is offloaded to cpu/gpu), so there
            # are no files to advise about. Stay silent-but-inert rather than
            # pretending to prefetch.
            self.enabled = False
            logger.info(
                "weight prefetch inactive: no disk-backed offloaded weights found"
            )
            return
        logger.info(
            f"weight prefetch active: {len(self._last_use)} source file(s), "
            f"depth={self.depth}"
        )

    def _files(self, index: int) -> set[str]:
        if index < 0 or index >= len(self._subgraphs):
            return set()
        # Precomputed in __init__; a subgraph with no offloaded weights of its own
        # simply has no entry.
        return self._files_by_subgraph.get(index, set())

    @staticmethod
    def _advise(path: str, advice: int) -> bool:
        try:
            fd = os.open(path, os.O_RDONLY)
        except OSError:
            return False
        try:
            # length 0 means "to end of file"
            os.posix_fadvise(fd, 0, 0, advice)
            return True
        except OSError:
            return False
        finally:
            os.close(fd)

    def prefetch(self, subgraph_index: int) -> int:
        """Ask the kernel to start reading the next ``depth`` subgraphs' files.

        Call this BEFORE the current subgraph's compute so the read overlaps it.
        """
        if not self.enabled:
            return 0
        wanted: set[str] = set()
        for offset in range(self.depth):
            wanted |= self._files(subgraph_index + offset)
        # Re-advising a file already resident is wasted work; re-advising one we
        # released is not, so drop it from the released set when we ask again.
        todo = wanted - self._prefetched
        count = 0
        for path in sorted(todo):
            if self._advise(path, _WILLNEED):
                self._prefetched.add(path)
                self._released.discard(path)
                count += 1
        return count

    def release_through(self, subgraph_index: int) -> int:
        """Drop clean page cache for files no later subgraph needs."""
        if not self.enabled:
            return 0
        count = 0
        for path, last in sorted(self._last_use.items()):
            if last > subgraph_index or path in self._released:
                continue
            if self._advise(path, _DONTNEED):
                self._released.add(path)
                self._prefetched.discard(path)
                count += 1
        return count
