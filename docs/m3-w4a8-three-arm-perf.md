# MiniMax-M3 W4A8 kernel-arm perf — CUTLASS vs Humming indexed vs Humming grouped

**Window:** `20260725T122256Z` · **Date reported:** 2026-07-25 · **Scope:** numbers
only — this document records measurements and their provenance. It makes **no
adoption decision** (per the task's decision rule).

## Setup

All three arms serve the **same checkpoint**
(`artifacts/m3-awq-gptq-prepared/gptq-checkpoint-vllm-w123-abi-overlay`, in-house
GPTQ W4AFP8, group-128, per-token FP8 activations) on **1 node, 8×H100, TP8/EP8,
vLLM 0.24.0** with the M3 serve overlay. The only variable is the MoE GEMM
backend/kernel:

| Arm | Backend | `VLLM_HUMMING_MOE_GEMM_TYPE` | Port |
|---|---|---|---|
| `cutlass-w4afp8` | vLLM CUTLASS W4A8 MoE | — | 8000 |
| `humming-w4afp8-indexed` | humming 0.1.10 (patched side-install) | `indexed` | 8005 |
| `humming-w4afp8-grouped` | humming 0.1.10 (patched side-install) | `grouped_contiguous` | 8010 |

Humming arms run the four-patch side-install
(`/mnt/nfs/hoangduy/venvs/humming-0.1.10-site`; see `DECLARED_PATCH_SHA256` in
`pipeline/m3_humming_w4a8.py`) with `VLLM_HUMMING_USE_F16_ACCUM=0`. Arms ran on
dedicated exclusive nodes; the shared window pins serve defaults and repo state.

### Grouped-arm attempt history (why the first two runs don't count)

| Attempt | Launched | Outcome |
|---|---|---|
| 1 | 12:23Z | **Invalid** — kernel's last-expert row bound read from `a.size(0)` (oversized by vLLM); failed suite preflight. Fixed by `patch_humming_grouped_expert_bounds.py`. Evidence: `evidence/m3-arm3-grouped-bounds/`. |
| 2 | 13:16Z | **Confounded** — humming never calls `cp.async.bulk.commit_group`, so every TMA store-group wait was a no-op; intermittent whole-tile output corruption → early-EOS (1,621 OSL-mismatch log lines vs ~14–18 on the other arms; 169/640 short requests at reasoning conc-64). Latency/throughput optimistic, not comparable. Fixed by `patch_humming_tma_store_commit.py` (+ PTX-required `patch_humming_tma_store_fence.py`); verified 0/96 bad launches, clean sweep, determinism restored (`m3-arm3-commit-verify/20260725T162957Z`). Evidence: `evidence/m3-arm3-tma-commit/`. |
| 3 | 16:47Z | **Valid** — all four patch gates passed pre-launch; suite rc=0 at 18:28Z; **22 OSL-mismatch lines** (in family with CUTLASS 14 / indexed 18); reasoning avg OSL ≥ 7993 at every concurrency. Numbers below. |

Raw runs: `/mnt/nfs/hoangduy/results/m3-perf-w4a8-three-arm/20260725T122256Z/`
(attempts 1–2 archived as `perf-humming-w4afp8-grouped.attempt{1,2}-*`; provenance
in `arm-provenance.txt`). Per-run summaries:
`/mnt/nfs/hoangduy/projects/benchmarks/results/minimax-m3-inhouse-<arm>/vllm/perf/{reasoning,agentic}/20260725T122256Z/`.

## Results

Metrics come from each run's `perf_summary.json` (aiperf export). "warm" = real
agentic serving (cache-warm); "cold" = cache-defeated control. vLLM 0.24 does not
emit `cached_tokens`, so cache-hit columns are unavailable on all arms alike
(`docs/m3-benchmark-arms.md`, known measurement gap).

### Reasoning (ISL 1000 / OSL 8000)

| conc | metric | CUTLASS | Humming indexed | Humming grouped |
|---|---|---|---|---|
| 1 | TPOT p50 (ms) | 9.73 | 7.29 | 8.49 |
| 1 | TTFT p50 (ms) | 129.51 | 136.93 | 145.23 |
| 1 | TTFT p95 (ms) | 140.16 | 145.96 | 147.09 |
| 1 | out tok/s | 102.48 | 136.67 | 117.40 |
| 1 | avg OSL | 7999.90 | 7999.90 | 7999.90 |
| 4 | TPOT p50 (ms) | 11.91 | 8.82 | 9.90 |
| 4 | TTFT p50 (ms) | 252.58 | 236.91 | 247.61 |
| 4 | TTFT p95 (ms) | 286.85 | 266.71 | 299.25 |
| 4 | out tok/s | 334.86 | 451.42 | 402.53 |
| 4 | avg OSL | 7999.95 | 7997.85 | 7995.02 |
| 16 | TPOT p50 (ms) | 15.28 | 12.24 | 13.27 |
| 16 | TTFT p50 (ms) | 444.23 | 449.99 | 508.34 |
| 16 | TTFT p95 (ms) | 676.21 | 695.44 | 759.83 |
| 16 | out tok/s | 1042.50 | 1298.67 | 1196.60 |
| 16 | avg OSL | 7999.93 | 7996.55 | 7994.02 |
| 64 | TPOT p50 (ms) | 22.28 | 19.42 | 20.63 |
| 64 | TTFT p50 (ms) | 1230.14 | 1331.92 | 1036.46 |
| 64 | TTFT p95 (ms) | 2229.71 | 2332.71 | 1420.40 |
| 64 | out tok/s | 2848.60 | 3265.06 | 3044.24 |
| 64 | avg OSL | 7994.00 | 7999.83 | 7993.61 |

### Agentic, cache-warm

| conc | metric | CUTLASS | Humming indexed | Humming grouped |
|---|---|---|---|---|
| 1 | TTFT p50 (ms) | 190.21 | 192.88 | 177.88 |
| 1 | TTFT p95 (ms) | 216.47 | 215.11 | 201.07 |
| 1 | out tok/s | 86.43 | 109.53 | 97.29 |
| 4 | TTFT p50 (ms) | 314.32 | 308.88 | 312.87 |
| 4 | TTFT p95 (ms) | 421.34 | 441.69 | 430.35 |
| 4 | out tok/s | 238.43 | 293.02 | 271.39 |
| 16 | TTFT p50 (ms) | 501.47 | 591.99 | 587.77 |
| 16 | TTFT p95 (ms) | 818.93 | 860.82 | 878.78 |
| 16 | out tok/s | 634.50 | 728.89 | 690.09 |
| 32 | TTFT p50 (ms) | 591.37 | 676.95 | 678.24 |
| 32 | TTFT p95 (ms) | 1070.79 | 1195.26 | 1220.98 |
| 32 | out tok/s | 945.83 | 1060.30 | 989.74 |

(avg OSL 99.4–99.5 on every arm/concurrency.)

### Agentic, cache-defeated (cold)

| conc | metric | CUTLASS | Humming indexed | Humming grouped |
|---|---|---|---|---|
| 1 | TTFT p50 (ms) | 554.27 | 568.89 | 579.12 |
| 1 | TTFT p95 (ms) | 581.62 | 576.15 | 599.75 |
| 1 | out tok/s | 67.85 | 80.67 | 73.11 |
| 4 | TTFT p50 (ms) | 624.94 | 644.80 | 665.30 |
| 4 | TTFT p95 (ms) | 1818.40 | 1918.08 | 1934.85 |
| 4 | out tok/s | 164.52 | 183.59 | 173.62 |
| 16 | TTFT p50 (ms) | 1569.20 | 2130.86 | 3371.53 |
| 16 | TTFT p95 (ms) | 4294.42 | 5466.77 | 5394.97 |
| 16 | out tok/s | 229.28 | 236.37 | 192.66 |
| 32 | TTFT p50 (ms) | 3041.65 | 3552.39 | 3434.73 |
| 32 | TTFT p95 (ms) | 5724.30 | 6063.93 | 6767.27 |
| 32 | out tok/s | 223.01 | 218.40 | 203.68 |

(avg OSL 100.0 on every arm/concurrency.)

## Reading notes (observations, not decisions)

- **Decode (TPOT):** both Humming arms beat CUTLASS at every reasoning
  concurrency; indexed is the fastest throughout (e.g. conc-1
  9.73 / 7.29 / 8.49 ms; conc-64 22.28 / 19.42 / 20.63 ms).
- **Prefill (TTFT):** CUTLASS is generally fastest at low-to-mid concurrency;
  the grouped arm's reasoning conc-64 TTFT (p50 1036, p95 1420 ms) is the
  outlier in the other direction.
- **Cold agentic conc-16** is the grouped arm's worst cell (TTFT p50 3372 ms vs
  1569/2131). Single run per cell; no variance estimate.
- Comparisons are arm-to-arm within this window only; not calibrated against
  any public benchmark recipe.

## Caveats

- One run per (arm, mode, concurrency) — no repeats, no error bars.
- The grouped arm rejoined the window later the same day (attempt 3, 16:47Z)
  rather than starting simultaneously; arms run on dedicated exclusive nodes,
  so there is no cross-arm contention, but time-of-day effects are uncontrolled.
- Humming numbers are for the **patched 0.1.10** side-install. Upstream 0.1.11
  (packed-K dequant layout) is patched and awaiting qualification
  (`pipeline/slurm/run_humming_0111_packedk_qual_srun.sh`); its numbers, when
  measured, belong in a new window, not this table.
