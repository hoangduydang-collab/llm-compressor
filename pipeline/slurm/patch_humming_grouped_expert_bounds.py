#!/usr/bin/env python3
"""Fix Humming's grouped_contiguous last-expert row count (arm 3 root cause).

Symptom. Serving MiniMax-M3 with ``VLLM_HUMMING_MOE_GEMM_TYPE=grouped_contiguous``
produces fluent but semantically wrong output: at temperature 0 the model read
the prompt "What is the weather in Paris?" as being about "Skik", while CUTLASS
and Humming-indexed answered the identical request correctly. No crash, no IMA,
no NaN -- and the ten fixed "2+2" qualification smokes all passed, because that
prompt is too overdetermined to expose it.

Root cause. For ``GROUPED_CONTIGUOUS`` the scheduler derives every expert's row
count from differences of ``expert_offset`` **except the last**, which it derives
from the scalar kernel argument ``shape_m``::

    smem.expert_tokens[kNumExperts - 1] = shape_m - smem.expert_offset[kNumExperts - 1];

``shape_m`` is ``a.size(0)`` (csrc/launcher/launcher.cpp:83), so the kernel is
implicitly requiring the caller to size the A tensor to exactly the number of
valid permuted rows. vLLM cannot honour that: ``HummingGroupedExperts.main_apply``
passes ``buffers["quanted_gate_up_input"]``, whose shape is ``(M * topk, K)``,
and the true valid total lives in device memory (``expert_first_token_offset``),
so slicing it on the host would need a D2H sync and break CUDA-graph capture.

Under expert parallelism the gap is large. With 16 local experts of 128 and
topk=4 only about ``M/2`` of ``M*4`` rows are ever filled, so the final expert is
told it owns thousands of phantom rows. That inflates ``m_blocks``, which moves
the stream-K/data-parallel partition boundary, which corrupts the tail experts'
output tiles.

Measured with pipeline/m3_humming_grouped_bounds_probe.py -- same weights, same
offsets, same activations, the only variable being ``a.size(0)``:

    valid rows 261, buffer rows 2048 (7.85x oversized)
    last expert: true 16 rows, kernel inferred 1803
    experts 0-12  max|diff| = 0
    experts 13,14,15  100% of rows wrong, max|diff| = 0.03125

The fix. The scheduler already loads ``kNumExperts + 1`` offsets into shared
memory (``expert_offset`` is declared ``uint32_t[kNumExperts + 1]`` in
utils/storage.cuh), so the exact total is available on-device. Use it, making the
last expert consistent with every other expert instead of depending on the
caller's buffer size:

    smem.expert_tokens[kNumExperts - 1] =
        smem.expert_offset[kNumExperts] - smem.expert_offset[kNumExperts - 1];

This is backward compatible: when a caller does size A exactly (humming's own
usage, where ``shape_m == expert_offset[kNumExperts]``) the two expressions are
equal. It costs one extra shared-memory read, no sync, and is CUDA-graph safe.

Worth reporting upstream to inclusionAI/humming alongside the ``pack-quantized``
input-schema gap.

Cache note: the JIT signature includes ``Compiler.cuh_last_update_time()`` (a map
of every ``.cuh`` path to its mtime), so editing this header automatically
invalidates every cached cubin. No manual cache-namespace bump is required.

This patcher edits an isolated side-install, never the shared venv, and is
idempotent. The resulting file hash is declared in ``pipeline/m3_humming_w4a8.py``
so the preflight integrity gate reports the patch rather than silently
tolerating a modified distribution.
"""

from __future__ import annotations

import argparse
import hashlib
import sys
from pathlib import Path

DEFAULT_SITE = Path("/mnt/nfs/hoangduy/venvs/humming-0.1.10-site")
RELATIVE_TARGET = "humming/include/humming/scheduler.cuh"

ANCHOR = (
    "        smem.expert_tokens[kNumExperts - 1] = "
    "shape_m - smem.expert_offset[kNumExperts - 1];"
)

PATCHED = (
    "        // llmc M3 Humming grouped_contiguous exact-total patch: derive the\n"
    "        // last expert's row count from the loaded offsets rather than from\n"
    "        // shape_m (== a.size(0)), which vLLM oversizes to (M * topk, K).\n"
    "        smem.expert_tokens[kNumExperts - 1] = "
    "smem.expert_offset[kNumExperts] - smem.expert_offset[kNumExperts - 1];"
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
