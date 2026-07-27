# M3 W4A8: humming 0.1.10 vs 0.1.11 packed-K — suite-native perf A/B (path 1)

**Window:** `m3-perf-w4a8-packedk-ab/20260726T033158Z` — four arms, one
checkpoint (`artifacts/m3-awq-gptq-prepared/gptq-checkpoint-vllm-w123-abi-overlay`),
1 node 8×H100 TP8/EP8 each, vLLM 0.24.0, suite-native workflows (reasoning +
agentic warm/cold, pinned OSL), `PERF_STRICT=1`. All four arms rc=0; Humming
attestation valid everywhere.
**Launcher:** `pipeline/slurm/run_perf_eval_w4a8_packedk_ab_srun.sh`; the two
0.1.11 arms were relaunched into the same window by
`relaunch_perf_eval_w4a8_0111_arms_srun.sh` after the original launch died on
the serving preflight's stale `EXPECTED_HUMMING_VERSION` pin (fixed in
`c4c02fd8`; `relaunch-0111.rc=0` is the authoritative rc for that pair —
`controller.rc=1` records the pre-fix exits).
**Scope:** numbers only. The packed-K adoption call is not made here.

## Arms and validity

| arm | humming | gemm | packed-K | OSL-mismatch total |
|---|---|---|---|---|
| indexed-0110 | 0.1.10 | indexed | off | 5 |
| grouped-0110 | 0.1.10 | grouped_contiguous | off | 6 |
| indexed-0111 | 0.1.11 | indexed | **on** | 4 |
| grouped-0111 | 0.1.11 | grouped_contiguous | **on** | 6 |

OSL-mismatch requests (`osl_mismatch_count`, summed over every reasoning +
agentic cell) sit almost entirely in reasoning conc-64 and are uniform across
arms — no early-EOS pathology; the window is unconfounded.

## Run-to-run noise floor

The re-run 0.1.10 arms vs the same arms in yesterday's window
(`20260725T122256Z`) give a same-config repeat across different nodes/days:

- **Reasoning TPOT p50: reproducible to ±0.02 ms (≤0.2%)** at every
  concurrency (e.g. indexed 7.29/7.29, 8.82/8.82, 12.22/12.25, 19.43/19.42;
  grouped 8.48/8.49 … 20.64/20.65).
- Interactivity (per-user decode tps): ±0.5–1%.
- TTFT p50: wobbles up to ~6–8% in individual cells (queueing-sensitive).

So TPOT/decode deltas ≥ ~0.5% are real; single-cell TTFT deltas under ~8%
should not be read.

## Reasoning (pinned OSL) — TPOT p50 ms / aggregate tok/s

| conc | idx-0110 | idx-0111pk | Δ | grp-0110 | grp-0111pk | Δ |
|---|---|---|---|---|---|---|
| 1 | **7.29** / 136.6 | 7.59 / 131.2 | **+4.1% slower** | **8.48** / 117.5 | 8.58 / 116.1 | +1.2% slower |
| 4 | 8.82 / 451.5 | 8.90 / 447.6 | +0.9% slower | 9.91 / 402.1 | 9.89 / 403.1 | flat |
| 16 | 12.22 / 1302.0 | 12.23 / 1300.3 | flat | 13.28 / 1197.7 | **13.04** / 1219.9 | **−1.8% faster** |
| 64 | 19.43 / 3262.5 | **19.21** / 3298.6 | **−1.1% faster** | 20.64 / 3045.6 | **20.11** / 3127.6 | **−2.6% faster** |

## Agentic — per-user decode tok/s (aiperf `output_token_throughput_per_user`), TTFT p50 ms in parentheses

### Warm

| conc | idx-0110 | idx-0111pk | grp-0110 | grp-0111pk |
|---|---|---|---|---|
| 1 | **138.7** (196) | 133.1 (191) | 118.3 (191) | 117.2 (196) |
| 4 | 106.5 (316) | 104.4 (308) | 95.8 (309) | 94.8 (322) |
| 16 | 73.0 (592) | 72.6 (563) | 67.9 (589) | 67.9 (589) |
| 32 | 49.1 (735) | 49.1 (738) | 46.5 (733) | 46.0 (701) |

### Cold

| conc | idx-0110 | idx-0111pk | grp-0110 | grp-0111pk |
|---|---|---|---|---|
| 1 | **139.5** (574) | 133.5 (568) | 118.9 (566) | 118.0 (579) |
| 4 | 83.0 (660) | 81.7 (653) | 77.6 (899) | 74.8 (722) |
| 16 | 29.2 (1949) | 28.0 (2079) | 19.7 (3090) | **21.2** (3276) |
| 32 | 9.9 (3204) | 9.9 (3627) | 8.9 (3243) | 8.9 (3251) |

## Reading (descriptive)

- **Packed-K trades low-concurrency decode speed for high-load throughput.**
  Consistent across the suite and the AA-style sweep
  (`docs/m3-w4a8-packedk-aa-sweep.md`, window `20260726T040130Z`):
  - conc-1 decode: indexed −4% (7.29→7.59 ms TPOT; agentic per-user decode
    138.7→133.1), grouped −1% — all far above the ±0.2% noise floor.
  - conc-16/64 reasoning: grouped −1.8%/−2.6% TPOT, indexed flat/−1.1%;
    system output tok/s +1–3%. Grouped cold conc-16 per-user decode +7.5%
    (19.7→21.2). The AA sweep's 10k-input conc-10 cells showed the same
    direction at +7–10%.
  - Mechanism consistent with the 0.1.11 packed-K sm90 tuner capping
    block_m at 128 (0.1.10 picked BM=184 at small m), while the packed-K
    B-loader raises dequant throughput where weight-loading dominates.
- **Indexed remains faster than grouped in every cell**, both versions —
  unchanged ordering from `docs/m3-w4a8-three-arm-perf.md` (CUTLASS anchor
  lives there, window `20260725T122256Z`).
- Within-pair comparisons here are same-window, same-config, single run per
  cell; the noise-floor section bounds what that single run can claim.

**Raw:** `benchmarks/results/minimax-m3-inhouse-humming-w4afp8-<arm>/vllm/perf/{reasoning,agentic}/20260726T033158Z/`;
controller artifacts at `/mnt/nfs/hoangduy/results/m3-perf-w4a8-packedk-ab/20260726T033158Z/`.
