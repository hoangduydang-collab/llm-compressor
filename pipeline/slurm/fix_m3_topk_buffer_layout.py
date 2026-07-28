#!/usr/bin/env python3
"""Fix the vLLM 0.26.0 MiniMax-M3 topk_indices_buffer layout regression.

THE DEFECT
vLLM 0.26.0 transposed the shared indexer top-k buffer to token-major:

    # 0.24.0 (head-major)
    torch.empty(num_index_heads, padded_num_tokens, sparse_topk_blocks, ...)
    # 0.26.0 (token-major)
    torch.empty(padded_num_tokens, num_index_heads, sparse_topk_blocks, ...)

That is correct for the rewritten SM100 MSA top-k, which consumes it as
``sparse_topk_select(..., output=buf[:num_tokens], max_score_layout="THK")``.
But the two impls selected on every NON-SM100 GPU were not updated and still
index the buffer head-major:

    indexer  MiniMaxM3IndexerTritonImpl:  minimax_m3_index_topk(..., out=buf[:, nd:, :])
    attend   MiniMaxM3SparseImpl:         minimax_m3_sparse_attn_decode(..., topk[:, :nd, :])

vLLM hardcodes sparse_num_index_heads = 4 and shards it, so at TP8
num_index_heads = max(1, 4 // 8) = 1. Then ``buf[:, nd:, :]`` with nd >= 1 slices
the HEAD axis away entirely and yields a ZERO-element view whose base pointer is
still advanced by one head stride. _topk_index_kernel's grid comes from
``score.shape``, not from ``out``, so it launches a full grid writing through that
view; the attend then reads garbage block ids and dereferences them outside the KV
cache -> CUDA illegal memory access.

WHY IT HIDES AT CONCURRENCY 1
Pure-prefill batches have nd == 0 (``buf[:, 0:, :]`` is the whole buffer) and pure
decode passes ``out=buf`` unsliced. In both cases writer and reader share the same
wrong-but-mutually-consistent convention, and [H,T,K] / [T,H,K] hold identical
element counts, so it works by accident. Only a MIXED prefill+decode batch -- the
one case with nd > 0 AND num_prefills > 0 -- breaks the consistency. Measured:
conc 1 clean for 30 requests / 89 s, then 0/10 within ~5 s at conc 10.

THE FIX
Restore head-major on non-SM100 (what the Triton impls require, i.e. 0.24.0
behaviour) and leave upstream's token-major untouched on SM100, where MSA is
selected and token-major is correct. Gating on the platform is sound here because
MSA is never selected off SM100.

SCOPE
Deliberately a standalone fixer rather than a new target in
patch_vllm_m3_serve.py: that patcher is re-run per serve by live benchmark
windows, and adding a target to it mid-window has already cost one replicate.
Fold this into the patcher once no window is running.

Idempotent, fail-closed, and a no-op on 0.24.0 (already head-major).
Usage:
    python fix_m3_topk_buffer_layout.py --venv /mnt/nfs/hoangduy/venvs/serve-026 [--check]
"""

from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

MARKER = "llmc M3 topk_indices_buffer layout fix"

TOKEN_MAJOR = """            self.topk_indices_buffer = torch.empty(
                padded_num_tokens,
                num_index_heads,
                sparse_cfg["sparse_topk_blocks"],
                dtype=torch.int32,
            )
"""

HEAD_MAJOR = """            self.topk_indices_buffer = torch.empty(
                num_index_heads,
                padded_num_tokens,
                sparse_cfg["sparse_topk_blocks"],
                dtype=torch.int32,
            )
"""

REPLACEMENT = f'''            # {MARKER} (0.26.0 regression).
            # 0.26.0 made this shared buffer token-major [T, H, K] for the rewritten
            # SM100 MSA top-k (sparse_topk_select max_score_layout="THK"), but the
            # Triton indexer and Triton attend -- the only impls selected off SM100 --
            # still index it head-major (buf[:, nd:, :] / topk[:, :nd, :]). At TP8
            # num_index_heads == 1, so those slices drop the head axis and produce an
            # empty view with a shifted base pointer; the attend then dereferences
            # garbage block ids outside the KV cache. Only MIXED prefill+decode
            # batches (nd > 0 and num_prefills > 0) hit it, which is why conc 1 is
            # clean. Keep token-major on SM100, where it is the correct layout.
            from vllm.platforms import current_platform as _llmc_platform

            if _llmc_platform.is_device_capability_family(100):
                _llmc_topk_shape = (
                    padded_num_tokens,
                    num_index_heads,
                    sparse_cfg["sparse_topk_blocks"],
                )
            else:
                _llmc_topk_shape = (
                    num_index_heads,
                    padded_num_tokens,
                    sparse_cfg["sparse_topk_blocks"],
                )
            self.topk_indices_buffer = torch.empty(
                *_llmc_topk_shape,
                dtype=torch.int32,
            )
'''

# NVIDIA path only. Checked 2026-07-28: amd/model.py allocates head-major in BOTH
# 0.24.0 and 0.26.0 (num_index_heads first, un-padded max_num_batched_tokens), so the
# transposition landed only on the NVIDIA side alongside the SM100 MSA rewrite. AMD is
# already correct and we do not serve on it, so including it here would only produce a
# spurious "moved" failure against a form we never intend to patch.
TARGETS = ("vllm/models/minimax_m3/nvidia/model.py",)


def patch_text(text: str) -> tuple[str, str]:
    """Return (new_text, status). status in {applied, already, not_applicable, moved}."""
    if MARKER in text:
        return text, "already"
    if TOKEN_MAJOR in text:
        return text.replace(TOKEN_MAJOR, REPLACEMENT, 1), "applied"
    # NOT APPLICABLE vs LAYOUT MOVED. 0.24.0 already allocates head-major, so the
    # token-major anchor is legitimately absent and there is nothing to fix. Only
    # treat a missing anchor as fatal when neither known form is present, which
    # means the code we patch really did move.
    if HEAD_MAJOR in text:
        return text, "not_applicable"
    return text, "moved"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--venv", required=True)
    ap.add_argument("--check", action="store_true", help="report only, do not write")
    args = ap.parse_args()

    site = Path(args.venv) / "lib/python3.12/site-packages"
    if not site.is_dir():
        print(f"FATAL: no site-packages under {args.venv}")
        return 2

    rc = 0
    touched = 0
    for rel in TARGETS:
        path = site / rel
        if not path.is_file():
            print(f"  skip (absent)      {rel}")
            continue
        text = path.read_text()
        new, status = patch_text(text)
        if status == "moved":
            print(f"  FATAL (moved)      {rel}: neither known allocation form found")
            rc = 1
            continue
        if status == "not_applicable":
            print(f"  ok (head-major)    {rel}: already correct, nothing to do")
            continue
        if status == "already":
            print(f"  ok (marked)        {rel}: fix already applied")
            continue
        # applied
        if args.check:
            print(f"  WOULD PATCH        {rel}: token-major -> platform-gated")
            touched += 1
            continue
        backup = path.with_suffix(path.suffix + ".llmc-topk-bak")
        if not backup.exists():
            shutil.copy2(path, backup)
        path.write_text(new)
        print(f"  PATCHED            {rel}  (backup {backup.name})")
        touched += 1

    if rc:
        print("\nRESULT: FAILED -- the allocation moved; re-read the source before patching")
        return rc
    verb = "would patch" if args.check else "patched"
    print(f"\nRESULT: ok ({verb} {touched} file(s))")
    return 0


if __name__ == "__main__":
    sys.exit(main())
