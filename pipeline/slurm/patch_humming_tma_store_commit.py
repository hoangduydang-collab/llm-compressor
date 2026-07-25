#!/usr/bin/env python3
"""Commit Humming's TMA C stores so the store-group waits actually wait.

Symptom. With ``gemm_type=grouped_contiguous`` (TMA-C epilogue), whole
(m_block, n_block) output tiles are intermittently written with garbage
(~20-25% of launches at BM=32/BK=256; pipeline/m3_humming_grouped_dp_tile_probe.py
and pipeline/m3_humming_grouped_tile_forensics.py). The forensics ratio
test shows bad tiles are PARTIALLY correct -- some rows match the
reference exactly (per-row ratio 1.0000) while adjacent row segments hold
huge pipeline-data values -- i.e. the output tile was read from shared
memory while something was overwriting it. At serving scale this was the
grouped arm's early-EOS pathology (1,594 OSL mismatches vs ~10, window
20260725T122256Z).

Root cause. PTX bulk async-group semantics: ``cp.async.bulk.wait_group N``
only waits on operations that were batched into bulk async-groups by
``cp.async.bulk.commit_group``. Humming defines the commit wrapper
(``tma_commit_store_group`` in utils/ptx/tma.cuh) but NEVER CALLS IT, so
every ``tma_wait_store_group`` in the codebase is a no-op:

- kernel/humming_ws.cuh:184 (and kernel/humming.cuh:110): the math warps
  "wait" for the C store to finish reading smem, then release the producer
  (``consumer.arrive``). The epilogue ``reduce`` buffer lives in a UNION
  with the producer's stage buffers (utils/storage.cuh), so the producer's
  next-block A/B/scale TMA loads overwrite the very smem the in-flight
  ``cp.async.bulk.tensor.2d.global.shared::cta`` C store is still reading.
  Whole-tile, timing-dependent corruption; reproduces with stream-K on and
  off; vanishes with ``use_tma=False`` (the legacy writer reads smem
  synchronously). CUTLASS's mandatory pattern is tma_store_arrive() =
  commit_group, then tma_store_wait<0>() -- humming implemented both
  halves and forgot the arrive.

- epilogue/gmem_writer.cuh:139: stream-K slice 0 "waits" for its plain
  store to complete before releasing the tile lock, so the adder slice's
  ``cp.reduce.async.bulk.tensor`` can race the unfinished store -- the
  bitwise nondeterminism observed at BM=8/16.

The fix. Commit each issued store/reduce into a bulk async-group right
after issuance (per issuing thread), making every existing wait
meaningful. Threads that issued nothing have no committed groups and their
wait returns immediately, unchanged.

Verified by pipeline/m3_humming_grouped_tile_forensics.py before/after:
10-11/48 launches bad before, expected 0 after (see results window).

Worth reporting upstream to inclusionAI/humming together with the
missing ``fence.proxy.async`` (patch_humming_tma_store_fence.py), the
grouped_contiguous last-expert shape_m bound
(patch_humming_grouped_expert_bounds.py, still present in 0.1.11), and
the ``pack-quantized`` input-schema gap.

Cache note: the JIT signature includes every ``.cuh`` mtime, so editing
this header automatically invalidates cached cubins.

This patcher edits an isolated side-install, never the shared venv, and is
idempotent. The resulting file hash is declared in
``pipeline/m3_humming_w4a8.py`` so the preflight integrity gate reports
the patch rather than silently tolerating a modified distribution.
"""

from __future__ import annotations

import argparse
import hashlib
import sys
from pathlib import Path

DEFAULT_SITE = Path("/mnt/nfs/hoangduy/venvs/humming-0.1.10-site")
RELATIVE_TARGET = "humming/include/humming/epilogue/gmem_writer.cuh"

COMMIT_NOTE = (
    "        // llmc M3 Humming TMA-store commit patch: cp.async.bulk.wait_group\n"
    "        // only tracks COMMITTED bulk async-groups; without this commit every\n"
    "        // tma_wait_store_group in the kernel is a no-op and the producer\n"
    "        // overwrites the union-aliased reduce smem mid-store.\n"
)

ANCHOR = (
    "      if constexpr (!kUseStreamK) {\n"
    "        tma_store_2d(smem_ptr + smem_offset, tensor_map_ptr, col_offset2, row_offset);\n"
    "      } else if (slice_count == 1 || slice_id == 0) {\n"
    "        tma_store_2d(smem_ptr + smem_offset, tensor_map_ptr, col_offset2, row_offset);\n"
    "        if (slice_count > 1) tma_wait_store_group<0>();\n"
    "      } else {\n"
    "        tma_reduce_add_2d(smem_ptr + smem_offset, tensor_map_ptr, col_offset2, row_offset);\n"
    "        if (slice_id != slice_count - 1) tma_wait_store_group<0>();\n"
    "      }"
)

PATCHED = (
    "      if constexpr (!kUseStreamK) {\n"
    "        tma_store_2d(smem_ptr + smem_offset, tensor_map_ptr, col_offset2, row_offset);\n"
    + COMMIT_NOTE
    + "        tma_commit_store_group();\n"
    "      } else if (slice_count == 1 || slice_id == 0) {\n"
    "        tma_store_2d(smem_ptr + smem_offset, tensor_map_ptr, col_offset2, row_offset);\n"
    "        tma_commit_store_group();\n"
    "        if (slice_count > 1) tma_wait_store_group<0>();\n"
    "      } else {\n"
    "        tma_reduce_add_2d(smem_ptr + smem_offset, tensor_map_ptr, col_offset2, row_offset);\n"
    "        tma_commit_store_group();\n"
    "        if (slice_id != slice_count - 1) tma_wait_store_group<0>();\n"
    "      }"
)


def target_path(site: Path) -> Path:
    return site / RELATIVE_TARGET


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def classify(source: str) -> str:
    """Return ``patched``, ``unpatched``, or ``unknown`` for one file body."""

    if PATCHED in source:
        return "patched"
    if ANCHOR in source:
        return "unpatched"
    return "unknown"


def apply_patch(site: Path, apply: bool) -> tuple[str, str]:
    """Apply or check the patch. Returns ``(status, sha256)``."""

    path = target_path(site)
    if not path.is_file():
        raise SystemExit(f"target not found: {path}")
    source = path.read_text(encoding="utf-8")
    status = classify(source)

    if status == "unknown":
        raise SystemExit(
            f"neither patched nor expected-unpatched content in {path}; "
            "refusing to guess"
        )
    if status == "patched":
        return "already patched", sha256(path)
    if not apply:
        return "NOT patched", sha256(path)

    if source.count(ANCHOR) != 1:
        raise SystemExit(
            f"expected exactly one anchor in {path}, found {source.count(ANCHOR)}"
        )
    path.write_text(source.replace(ANCHOR, PATCHED), encoding="utf-8")
    return "patched", sha256(path)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--site", type=Path, default=DEFAULT_SITE)
    parser.add_argument(
        "--check",
        action="store_true",
        help="report status without writing; exit nonzero if unpatched",
    )
    args = parser.parse_args(argv)

    status, digest = apply_patch(args.site, apply=not args.check)
    print(f"{RELATIVE_TARGET}: {status}")
    print(f"sha256: {digest}")
    if args.check and status != "already patched":
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
