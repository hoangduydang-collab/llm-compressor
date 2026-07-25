# MiniMax-M3 W4A8 kernel comparison: CUTLASS vs Humming (indexed)

**Status:** measured; **no adoption decision applied by design**

**Date:** 2026-07-25 · `RUN_TS=20260725T074535Z`

**Configuration:** full-stack agent (`FULL_STACK_AGENT_PROTOCOL.md`)

Prerequisite: [`M3_HUMMING_W4A8_QUALIFICATION_REPORT.md`](M3_HUMMING_W4A8_QUALIFICATION_REPORT.md)
(Humming correctness/attestation). Kernel landscape:
[`M3_HOPPER_W4A8_KERNEL_INVESTIGATION.md`](M3_HOPPER_W4A8_KERNEL_INVESTIGATION.md)
arm 2.

## Design

One variable: `M3_W4A8_BACKEND`. Everything else identical.

| | CUTLASS (control) | Humming (indexed) |
| --- | --- | --- |
| Checkpoint | `gptq-checkpoint-vllm-w123-abi-overlay` | **same** |
| Node | 1 exclusive 8×H100, `gpu-h101` | 1 exclusive 8×H100, `gpu-h115` |
| Topology | TP8 + EP, graphs on, `max_model_len=40960`, `kv_cache_dtype=fp8` | same |
| Port | 8000 | 8005 |
| Effective argv | — | differs **only** by `--quantization humming` |
| Suite | `run_performance.sh` `PERF_STRICT=1` | same, same `RUN_TS` |

Control re-measured fresh in this window. **Do not compare these absolute
numbers to earlier perf passes** — serve defaults changed since. Cross-arm
comparison within this run is the valid claim.

Humming arm attestation before benchmarking: `valid=true`, `backend=humming`,
`gemm_type=indexed`, `indexed_marker=true`, `grouped_marker=false`,
`reason_codes=[]`. The control's `serve.log` contains **zero** Humming markers.

**Confound check (required by the `completion_tokens` regression discipline):**
different kernels change numerics, which can change generation length and fake a
throughput delta. Verified per point — ISL identical, OSL drift ≤0.3%, request
counts identical at all 12 points. The deltas below are not a length artifact.

## Results

`C` = CUTLASS, `H` = Humming. `out_tok/s` and `interactivity_tps`: higher is
better. `TPOT`/`TTFT`: lower is better.

### Reasoning (ISL 1000 / OSL ~8000)

| conc | out_tok/s C → H | TPOT ms C → H | TTFT p50 ms C → H |
| --- | --- | --- | --- |
| 1 | 102.50 → 136.59 **+33.3%** | 9.73 → 7.29 **−25.0%** | 134.99 → 131.80 −2.4% |
| 4 | 335.08 → 450.37 **+34.4%** | 11.90 → 8.82 **−25.8%** | 246.53 → 244.51 −0.8% |
| 16 | 1043.23 → 1294.59 **+24.1%** | 15.26 → 12.22 **−19.9%** | 415.23 → 465.47 +12.1% |
| 64 | 2845.03 → 3271.68 **+15.0%** | 22.28 → 19.37 **−13.0%** | 1265.97 → 1329.71 +5.0% |

### Agentic warm (ISL ~12.5k / OSL ~100)

| conc | out_tok/s C → H | interactivity C → H | TTFT p50 ms C → H |
| --- | --- | --- | --- |
| 1 | 86.56 → 109.09 **+26.0%** | 102.83 → 138.51 **+34.7%** | 187.00 → 195.23 +4.4% |
| 4 | 240.34 → 291.81 **+21.4%** | 80.40 → 106.49 **+32.5%** | 287.98 → 317.43 +10.2% |
| 16 | 641.56 → 723.29 **+12.7%** | 58.30 → 72.47 **+24.3%** | 491.53 → 602.22 **+22.5%** |
| 32 | 933.69 → 1044.20 **+11.8%** | 38.42 → 47.29 **+23.1%** | 525.42 → 677.88 **+29.0%** |

### Agentic cold

| conc | out_tok/s C → H | interactivity C → H | TTFT p50 ms C → H |
| --- | --- | --- | --- |
| 1 | 67.87 → 80.65 **+18.8%** | 103.15 → 138.73 **+34.5%** | 553.43 → 568.01 +2.6% |
| 4 | 165.69 → 183.15 **+10.5%** | 65.86 → 86.63 **+31.5%** | 628.59 → 647.94 +3.1% |
| 16 | 230.59 → 235.76 +2.2% | 24.13 → 27.21 +12.8% | 1884.16 → 1779.49 −5.6% |
| 32 | 221.62 → **218.91 −1.2%** | 9.58 → 10.05 +4.9% | 2711.60 → **3548.80 +30.9%** |

### KV collapse

Identical for both arms — collapse concurrency 16 (warm) and 4 (cold). The
collapse point is not kernel-dependent in this envelope.

## What the numbers say

1. **Humming wins decode, consistently and by a lot.** TPOT is 13–26% lower
   across every reasoning point, and interactivity is 23–35% higher across every
   agentic point. This is the strongest signal in the run.
2. **The decode win shrinks as concurrency rises** (reasoning out_tok/s +33% at
   conc-1 → +15% at conc-64).
3. **Humming loses TTFT, and the loss grows with load** — reasoning +12.1% at
   conc-16; agentic warm +4.4% → +29.0% from conc-1 to 32; agentic cold +30.9%
   at conc-32.
4. **One outright regression:** agentic cold conc-32 output throughput,
   −1.2%. Agentic cold throughput crosses over between conc-16 (+2.2%) and
   conc-32 (−1.2%).

The shape — decode strongly favoured, prefill penalised under load — is
*consistent with* Humming's WGMMA/TMA kernels being tuned for the few-tokens-
per-expert decode regime while CUTLASS handles large prefill batches better.
**That is a hypothesis, not a finding: no profiling was done.** The kernel
investigation's own discipline applies — do not attribute end-to-end deltas to
GEMM speed without checking routing, sorting, activation quantization,
collectives, and CUDA-graph behaviour.

## Limitations — read before quoting any number

- **No adoption decision.** Per explicit instruction, no pass/fail threshold was
  applied. Adoption is an open call.
- **No profiling.** The mechanism behind the decode/prefill split is unverified.
- **`cache_hit_observed=False` for both arms.** Per-request prefix-cache hit rate
  was not reported (known vLLM 0.24 behaviour: APC is on but `cached_tokens`
  never appears in usage). The warm-vs-cold TTFT delta remains usable; a
  per-request cache-hit figure does not exist in this run.
- **Not a public-benchmark score.** This is an internal paired serving
  comparison; nothing here is comparable to any published leaderboard recipe.
- **Indexed only.** Humming grouped/automatic scheduling untested — it is the
  investigation's arm 3 and is a plausible response to the prefill deficit.
- **Single run per point.** No repeat, so no variance estimate. The small deltas
  (agentic cold conc-16/32) are within plausible run-to-run noise; the large
  decode deltas are not.
- **Two nodes, not one.** Arms ran concurrently on `gpu-h101` and `gpu-h115`.
  Same model and exclusive allocation, but not the same physical hardware.

## Execution provenance — a defect worth recording

The first controller invocation passed the Humming arm's env through
`"${array[@]}"` as an assignment prefix. Words produced by expansion are **not**
parsed as assignments, so bash tried to execute
`VLLM_HUMMING_MOE_GEMM_TYPE=indexed` as a command name and the arm died
instantly (exit 127) while CUTLASS ran on unaffected. Fixed by routing through
`env`.

The Humming arm was then relaunched standalone into the **same** `ROOT`/`RUN_TS`
so the pair stays valid (`humming-arm-relaunch.txt`,
`humming-arm-command.txt`). Consequence: two writers raced
`perf-humming-w4afp8-indexed.rc`, and the original controller's stale `127` won
because it was written later (09:32:42) than the real arm's completion
(09:24:25). That file has been corrected to `0` with the full story in
`rc-provenance.txt`. `controller.rc=1` is left as-is — it correctly records that
the *controller's own* second arm failed.

The measurements themselves are unaffected: the arm that produced them served on
`gpu-h115`, attested `indexed`, and reported `suite rc=0`.

Also note the controller's `wait` loop is sequential, so a fast-failing arm is
not reported until the slow arm finishes. That is why the dead arm went
unnoticed for minutes, and why arm liveness should be checked with `squeue`
rather than trusted from the controller log.

## Artifacts

- Controller root: `/mnt/nfs/hoangduy/results/m3-perf-cutlass-humming/20260725T074535Z/`
- Suite results: `benchmarks/results/minimax-m3-inhouse-{cutlass-w4afp8,humming-w4afp8-indexed}/vllm/perf/{reasoning,agentic}/20260725T074535Z/`
- Attestation: `perf-humming-w4afp8-indexed/backend-attestation.json`
- Controller: `pipeline/slurm/run_perf_eval_cutlass_humming_srun.sh`

## Candidate next steps

- **Humming grouped/auto scheduling** (investigation arm 3) — directly targets
  the prefill deficit; cheapest way to test whether the TTFT loss is a
  scheduling artifact rather than intrinsic.
- **Profile both arms** at representative routed-token counts to convert the
  decode/prefill hypothesis into a finding.
- **Repeat the two crossover points** (agentic cold conc-16/32) to separate a
  real regression from noise.
- **Marlin W4A8-INT8** (arm 4) remains untested and is a different activation
  format, not an FP8 comparison.
