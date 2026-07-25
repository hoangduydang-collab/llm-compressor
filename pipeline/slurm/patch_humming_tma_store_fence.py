#!/usr/bin/env python3
"""Add the missing ``fence.proxy.async`` before Humming's TMA C stores.

Symptom. With ``gemm_type=grouped_contiguous`` (TMA-C epilogue), whole
(m_block, n_block) output tiles are intermittently written with garbage:
pipeline/m3_humming_grouped_dp_tile_probe.py measured ~20-25% of launches
producing one wrong tile (a full 128-column block of some expert, values
~1.5-2.2e7 vs reference ~5), nondeterministically, with ``use_stream_k``
both on and off -- and 0/8 failures with ``use_tma=False``. At serving
scale this surfaced as the grouped arm's early-EOS pathology (1,594 OSL
mismatches vs ~10 for the other arms in window 20260725T122256Z).

Root cause. The epilogue writes each C tile to shared memory with ordinary
generic-proxy stores, syncs the math threads (``bar.sync``), and issues
``cp.async.bulk.tensor.2d.global.shared::cta`` -- an async-proxy read of
that shared memory. The PTX memory model requires a cross-proxy fence,
``fence.proxy.async.shared::cta``, between generic-proxy writes and an
async-proxy read of the same locations; plain barriers do not order across
proxies. CUTLASS issues exactly this fence (``tma_store_fence()``) before
every TMA store. Humming has no ``fence.proxy.async`` anywhere in its
kernel tree, so the TMA engine can read stale shared memory -- whole-tile,
timing-dependent corruption. The dense path rarely shows it (long
epilogues, different timing) and the indexed path cannot use TMA-C at all
(``static_assert(!kIsIndexedGemm)`` in write_tma), which is why the bug
survived in the wild: grouped_contiguous + TMA-C is the undertested corner.

The fix. Make ``tma_store_2d`` and ``tma_reduce_add_2d`` (the only
smem->gmem bulk-tensor wrappers) issue ``fence.proxy.async.shared::cta``
before the copy. Fencing inside the wrapper is correct by cumulativity:
the tile's smem writes happen-before the issuing thread via the epilogue's
math-thread barrier, so the fence makes them async-proxy visible. Cost is
one fence per tile store, noise compared to the store itself.

Verified by pipeline/m3_humming_grouped_tile_forensics.py: 10/48 launches
bad before the patch, expected 0 after (see results window).

Worth reporting upstream to inclusionAI/humming alongside the
``pack-quantized`` input-schema gap and the grouped_contiguous
last-expert ``shape_m`` bound (still present in 0.1.11).

Cache note: the JIT signature includes every ``.cuh`` mtime, so editing
this header automatically invalidates cached cubins.

This patcher edits an isolated side-install, never the shared venv, and is
idempotent. The resulting file hash is declared in
``pipeline/m3_humming_w4a8.py`` so the preflight integrity gate reports the
patch rather than silently tolerating a modified distribution.
"""

from __future__ import annotations

import argparse
import hashlib
import sys
from pathlib import Path

DEFAULT_SITE = Path("/mnt/nfs/hoangduy/venvs/humming-0.1.10-site")
RELATIVE_TARGET = "humming/include/humming/utils/ptx/tma.cuh"

FENCE = (
    "  // llmc M3 Humming TMA-store proxy-fence patch: generic-proxy smem\n"
    "  // writes must be fenced before an async-proxy read (PTX memory model;\n"
    "  // CUTLASS tma_store_fence()). Without it the TMA engine intermittently\n"
    "  // reads stale smem and stores whole garbage tiles.\n"
    "  asm volatile(\"fence.proxy.async.shared::cta;\" ::: \"memory\");\n"
)

ANCHOR_STORE = (
    "CUDA_INLINE void tma_store_2d(void *smem_ptr, const void *desc_ptr, "
    "uint32_t crd0, uint32_t crd1) {\n"
    "  uint64_t gmem_int_desc = reinterpret_cast<uint64_t>(desc_ptr);\n"
    "  uint32_t smem_int_ptr = cast_smem_ptr_to_uint(smem_ptr);\n"
    "\n"
    "  asm volatile(\"cp.async.bulk.tensor.2d.global.shared::cta.bulk_group\""
)

PATCHED_STORE = (
    "CUDA_INLINE void tma_store_2d(void *smem_ptr, const void *desc_ptr, "
    "uint32_t crd0, uint32_t crd1) {\n"
    "  uint64_t gmem_int_desc = reinterpret_cast<uint64_t>(desc_ptr);\n"
    "  uint32_t smem_int_ptr = cast_smem_ptr_to_uint(smem_ptr);\n"
    "\n" + FENCE +
    "  asm volatile(\"cp.async.bulk.tensor.2d.global.shared::cta.bulk_group\""
)

ANCHOR_REDUCE = (
    "CUDA_INLINE void tma_reduce_add_2d(void *smem_ptr, const void *desc_ptr, "
    "uint32_t crd0, uint32_t crd1) {\n"
    "  uint64_t gmem_int_desc = reinterpret_cast<uint64_t>(desc_ptr);\n"
    "  uint32_t smem_int_ptr = cast_smem_ptr_to_uint(smem_ptr);\n"
    "\n"
    "  asm volatile(\"cp.reduce.async.bulk.tensor.2d.global.shared::cta.add.bulk_group\""
)

PATCHED_REDUCE = (
    "CUDA_INLINE void tma_reduce_add_2d(void *smem_ptr, const void *desc_ptr, "
    "uint32_t crd0, uint32_t crd1) {\n"
    "  uint64_t gmem_int_desc = reinterpret_cast<uint64_t>(desc_ptr);\n"
    "  uint32_t smem_int_ptr = cast_smem_ptr_to_uint(smem_ptr);\n"
    "\n" + FENCE +
    "  asm volatile(\"cp.reduce.async.bulk.tensor.2d.global.shared::cta.add.bulk_group\""
)

PAIRS = [
    ("tma_store_2d", ANCHOR_STORE, PATCHED_STORE),
    ("tma_reduce_add_2d", ANCHOR_REDUCE, PATCHED_REDUCE),
]


def target_path(site: Path) -> Path:
    return site / RELATIVE_TARGET


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def classify(source: str) -> str:
    """Return ``patched``, ``unpatched``, or ``unknown`` for one file body."""

    if all(patched in source for _, _, patched in PAIRS):
        return "patched"
    if all(anchor in source for _, anchor, _ in PAIRS):
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

    for name, anchor, patched in PAIRS:
        if source.count(anchor) != 1:
            raise SystemExit(
                f"expected exactly one {name} anchor in {path}, "
                f"found {source.count(anchor)}"
            )
        source = source.replace(anchor, patched)
    path.write_text(source, encoding="utf-8")
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
