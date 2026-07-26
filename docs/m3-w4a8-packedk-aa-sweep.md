# M3 W4A8: AA-style sweep over the packed-K A/B arms (path 2)

**Window:** `m3-aa-sweep-w4a8-packedk/20260726T040130Z` — all four arms rc=0,
controller rc=0, every cell `status=ok` with zero request errors.
**Runner:** `benchmarks/performance/aa/run_aa_sweep.py` (aiperf 0.8.0),
launched by `pipeline/slurm/run_aa_sweep_w4a8_packedk_srun.sh` →
`aa_sweep_arm.sh`. Matrix: input 1k/10k × concurrency 1/10, temp 0.6,
thinking on, **natural output length** (AA answer-token floor is *checked*,
not forced). 8 requests per conc-1 cell, 10 per conc-10 cell.
**Serving:** same checkpoint
(`artifacts/m3-awq-gptq-prepared/gptq-checkpoint-vllm-w123-abi-overlay`),
1 node 8×H100 TP8/EP8, vLLM 0.24.0, Humming attestation valid on all four
arms. Nodes: gpu-h115/116/117 (compute) + gpu-h125 (debug; identical
8×H100/192-CPU). Repo commit `784950bf`.

**Scope:** numbers only — no adoption decision. These are AA-*style* numbers
for arm-to-arm comparison, explicitly **not** comparable to the public AA
leaderboard (self-hosted endpoint, natural OSL below AA floors — see caveats).

## Arms

| arm | humming | gemm | packed-K |
|---|---|---|---|
| indexed-0110 | 0.1.10 | indexed | off |
| grouped-0110 | 0.1.10 | grouped_contiguous | off |
| indexed-0111 | 0.1.11 | indexed | **on** |
| grouped-0111 | 0.1.11 | grouped_contiguous | **on** |

Both sites carry the same four declared patches (schema, grouped bounds, TMA
fence, TMA commit-group); the only intended variable within each gemm pair is
the humming release, whose sole serving-relevant change for us is the packed-K
weight layout (plus its BM=128 sm90 tuning heuristic).

## Results

TTFT and e2e in ms; "out speed" = per-user decode rate (`output_speed_tps`
p50); "agg" = aggregate output tok/s across the cell.

### Per-user decode speed (the cleanest cross-arm metric here)

| cell | indexed-0110 | grouped-0110 | indexed-0111 | grouped-0111 |
|---|---|---|---|---|
| 1k/c1 | **137.4** | 118.1 | 132.2 | 116.5 |
| 10k/c1 | **137.2** | 118.5 | 132.1 | 116.5 |
| 1k/c10 | 91.7 | 83.9 | **92.0** | 83.0 |
| 10k/c10 | 66.4 | 56.6 | **71.0** | 62.5 |

Per-request decode-rate std within a conc-1 cell is ~0.16 tok/s, so the
conc-1 deltas are far above measurement noise.

### Full matrix

| cell | metric | indexed-0110 | grouped-0110 | indexed-0111 | grouped-0111 |
|---|---|---|---|---|---|
| 1k/c1 | TTFT p50 / p95 | 128 / 135 | 126 / 139 | 134 / 165 | 126 / 162 |
| | agg out tok/s | 132.7 | 115.5 | 129.0 | 113.0 |
| | natural OSL avg | 493 | 647 | 739 | 495 |
| 1k/c10 | TTFT p50 / p95 | 433 / 434 | 442 / 552 | 442 / 442 | 321 / 469 |
| | agg out tok/s | 705.8 | 459.3 | 635.1 | 597.9 |
| | natural OSL avg | 570 | 552 | 563 | 499 |
| 10k/c1 | TTFT p50 / p95 | 564 / 583 | 557 / 594 | 567 / 586 | 548 / 559 |
| | agg out tok/s | 120.0 | 101.9 | 113.4 | 99.4 |
| | natural OSL avg | 501 | 383 | 417 | 341 |
| 10k/c10 | TTFT p50 / p95 | 1830 / 3555 | 2152 / 3537 | 2765 / 3610 | 2171 / 3687 |
| | agg out tok/s | 404.6 | 351.1 | 408.4 | 323.0 |
| | natural OSL avg | 416 | 339 | 396 | 417 |

AA floor cleared: **False in every cell** (floors 1000/1500 tokens; natural
OSL ran ~340–740).

## Reading (descriptive, not a decision)

- **Conc-1 decode: packed-K is consistently *slower*.** indexed 137.4→132.2
  (−3.8%) and 137.2→132.1; grouped 118.1→116.5 (−1.4%) and 118.5→116.5.
  Input length doesn't matter (1k and 10k agree), which points at the decode
  GEMM itself — consistent with packed-K's sm90 tuner capping block_m at 128
  (0.1.10's heuristic picked BM=184 tunings for these shapes).
- **High-load decode: packed-K helps at 10k/c10.** indexed 66.4→71.0 (+7%),
  grouped 56.6→62.5 (+10%); at 1k/c10 it's flat (91.7→92.0, 83.9→83.0). The
  dequant-throughput win appears where weight-loading pressure is highest.
- **Indexed remains ahead of grouped everywhere** on decode speed, at both
  versions — same ordering as the suite-native window 20260725T122256Z.
- TTFT differences are mostly within run noise given natural-OSL queueing
  (e.g. the 10k/c10 TTFT p50 spread); don't read single TTFT cells strongly.

## Caveats

- **Natural OSL varies per arm/cell** (340–740 tokens), so aggregate tok/s
  and e2e are OSL-confounded; per-user decode speed and TTFT are the
  comparable columns. The suite-native A/B window (20260726T033158Z) pins
  OSL and is the controlled comparison for the packed-K adoption question.
- Single run per cell, 8–10 requests each; no run-to-run variance anchor
  within this window.
- Not AA-leaderboard-comparable: self-hosted endpoint, AA floors not cleared
  in any cell, synthetic random-token prompts.

**Raw:** `benchmarks/results/minimax-m3-inhouse-humming-w4afp8-<arm>/self-hosted/perf/aa-sweep/20260726T040130Z/`
(`aa_sweep_summary.json`, `AA_SWEEP_SUMMARY.md`, per-cell aiperf artifacts);
controller logs at `/mnt/nfs/hoangduy/results/m3-aa-sweep-w4a8-packedk/20260726T040130Z/`.
