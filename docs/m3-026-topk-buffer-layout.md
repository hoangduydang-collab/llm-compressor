# MiniMax-M3 on vLLM 0.26.0: the `topk_indices_buffer` layout regression

**Status 2026-07-28: ROOT-CAUSED and FIXED.** This was the blocker that made *any*
0.26.0 M3 serve die under concurrency, including plain k=0 with no speculative
decoding. It is distinct from the Model-Runner-V2 indexer assert, which is still
open — see `m3-dspark-blockers-026.md`.

Evidence: `/mnt/nfs/hoangduy/results/m3-026-ima-bisect/20260728T090527Z` (3-arm
bisect), `/mnt/nfs/hoangduy/results/m3-026-ima-pin/20260728T092223Z` (the pin),
`.../m3-026-ima-bisect/VERIFY-20260728T094139Z` (fix verification).

## The defect

vLLM 0.26.0 transposed the shared indexer top-k buffer in
`vllm/models/minimax_m3/nvidia/model.py`:

```python
# 0.24.0 -- head-major [H, T, K]
torch.empty(num_index_heads, padded_num_tokens, sparse_cfg["sparse_topk_blocks"], ...)
# 0.26.0 -- token-major [T, H, K]
torch.empty(padded_num_tokens, num_index_heads, sparse_cfg["sparse_topk_blocks"], ...)
```

Token-major is *correct* for the rewritten SM100 MSA top-k, which consumes it as
`sparse_topk_select(..., output=buf[:num_tokens], max_score_layout="THK")`. But the
two impls selected on every **non-SM100** GPU still index it head-major, unchanged
from 0.24.0:

| consumer | expression |
|---|---|
| `MiniMaxM3IndexerTritonImpl` | `minimax_m3_index_topk(..., out=buf[:, nd:, :])` |
| `MiniMaxM3SparseImpl` (attend) | `minimax_m3_sparse_attn_decode(..., topk[:, :nd, :])` |

vLLM hardcodes `sparse_num_index_heads: 4` and shards it:
`num_index_heads = max(1, 4 // tp_size)`. **At TP8 that is 1.** So `buf[:, nd:, :]`
with `nd >= 1` slices the *head* axis away entirely and yields a **zero-element
view whose base pointer is still advanced by one head stride**.
`_topk_index_kernel` takes its grid from `score.shape`, not from `out`, so it
launches a full grid writing through that view; the attend then reads garbage block
ids and dereferences them outside the KV cache → CUDA illegal memory access.

The bug exists at **any** TP, not only TP8 — at TP1, `buf[:, 1:, :]` is
`[8192, 3, 16]`, non-empty but still the wrong axis. A fix must correct the layout,
not merely avoid the empty slice.

## Why concurrency 1 is clean

Pure-prefill batches have `nd == 0`, so `buf[:, 0:, :]` is the whole buffer; pure
decode passes `out=buf` unsliced. In both cases writer and reader share the same
wrong-but-*mutually consistent* convention, and `[H,T,K]` / `[T,H,K]` hold identical
element counts — it works by accident. Only a **mixed prefill+decode** batch (the
one case with `nd > 0` *and* `num_prefills > 0`) breaks the consistency.

Measured: conc 1 clean for 30 requests / 89 s at `Running: 1 reqs`, then 0/10 within
~5 s at conc 10, on batches `{1 decode, 8191 prefill}`.

## This is not our stack

Nothing about Humming, our GPTQ checkpoint, or the ABI overlay is required. **Any M3
serve on Hopper with concurrency on 0.26.0 hits this**, because the Triton indexer is
selected whenever `sm100=False`:

```
MiniMax M3 indexer: selected Triton (no fmha_sm100) [topk_blocks=16, indexer_kv_dtype=bf16, sm100=False]
```

`amd/model.py` allocates head-major in *both* versions, so the transposition landed
only on the NVIDIA side, alongside the SM100 MSA rewrite. The plausible reading is
that upstream's M3 coverage is Blackwell, where token-major is the correct layout.

## The fix

`pipeline/slurm/fix_m3_topk_buffer_layout.py` first shipped the fix standalone:
restore head-major off SM100 (what the Triton impls require), while retaining
upstream's token-major layout on SM100 (where MSA is selected). It is idempotent
and fail-closed, distinguishes *not applicable* (0.24.0, already head-major)
from a *moved anchor*, and was verified as a no-op against `quant` (0.24.0) and
`serve` (0.23.1).

The fix is now also a required target in
`pipeline/slurm/patch_vllm_m3_serve.py` (commit `ccef9936`), so the normal
per-serve overlay preserves it after a `serve-026` rebuild. The standalone
patcher remains useful for diagnosis and explicit repair. It was initially kept
separate because changing the live per-serve patch set mid-window had already
cost the `L0-hum-k0-r3` replicate on 2026-07-28.

### Verification

| | before | after |
|---|---|---|
| warm phase | 0/10, engine dead in 5.3 s | **10/10**, engine alive |
| burst phase | never reached | **10/10**, engine alive |

Reproducer: `pipeline/diag/m3_026_ima_burst.py` (10 concurrent 8k SPEED-Bench
prompts; gates on completed count *and* `/health`, never on HTTP status alone).

## How it was found, and what was falsified

The traceback initially pointed at Humming's MoE input-quant kernel
(`humming/ops/input.py:217`). That was a **sticky-error artifact**: a Triton launch
raising `Triton Error [CUDA]` is also what a pending asynchronous fault looks like at
the next CUDA API call. `CUDA_LAUNCH_BLOCKING=1` re-attributed it to
`_topk_index_kernel` via `index_topk.py:736`.

Falsified along the way — recorded so the search does not circle back:

- **`ll_bf16`**, 0.26.0's new dimension-ungated cuteDSL router GEMM, is numerically
  correct at M3's `(K=6144, N=128)` for M=1..16 on both dispatch backends
  (`m3-ll-bf16-probe/20260728T083753Z`). Untuned ≠ broken.
- **The packed KV layout + new in-kernel fp8 K/V dequant** are not involved:
  byte-identical fault frames at `kv_cache_dtype=fp8` and `=auto`.
- **Cudagraphs are not involved**: the enforce-eager arm crashes too, which also
  clears our `LLMC_M3_CAPTURE_SYNC` breakable-capture patch a second time.
- **The Humming merge is not involved**: `serve-026`'s in-venv humming is
  byte-identical (`diff -rq`, no differing files) to the patched 0.24.0 side-install,
  including the file in the traceback.
- **MoE workspace undersizing** is not involved: `_resize_cache` asserts
  `prod(v) <= x.numel()`, so it would raise rather than fault.
- The whole 0.26.0 MSA rewrite (`nvidia/indexer_msa.py` 251→358 lines, new
  `nvidia/ops/index_decode_score.py`) is **dead code on H100**.

## Correction to earlier notes

`venvs/serve` is **vLLM 0.23.1rc1**, not 0.24.0. The qualified 0.24.0 baseline is
`venvs/quant` (which is also `run_vllm_http_serve_smoke.sh`'s `SERVE_VENV` default).
Version-labelled diffs taken against `venvs/serve` during this investigation were
0.23.1-based; the load-bearing ones were re-checked against `quant` and held —
including that `index_topk.py` is identical between 0.23.1 and 0.24.0, so the
faulting kernel really is byte-identical between the version that works and the
version that faults, on the same torch 2.11.0+cu130 and triton 3.6.0.

## Upstream-reportable (user-gated — outward-facing, do not file without say-so)

`nvidia/model.py` allocates `topk_indices_buffer` token-major for the SM100 MSA
top-k, but `MiniMaxM3IndexerTritonImpl` and `MiniMaxM3SparseImpl` — the impls
selected on all non-SM100 CUDA GPUs — still slice it head-major, so any mixed
prefill+decode batch corrupts the top-k and faults. Reproducer: MiniMax-M3, TP8,
H100, 10 concurrent ~8k prompts; fails within seconds. `amd/model.py` is unaffected.
