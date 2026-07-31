# Goal 7 · Native Humming W4A8 Serving — Field Note

> **Done · 2026-07-26** · markdown twin of
> [`goal-7-native-humming-w4a8.html`](goal-7-native-humming-w4a8.html) — keep in sync.
> [← Program overview](../automatic-quantization-pipeline-progress.md)

**Contents:** [Objective](#objective) · [Sub-tasks](#sub-tasks) ·
[Result](#result) · [Boundary](#boundary) · [Evidence](#evidence)

A faster serving kernel (Humming W4A8) for the in-house 4-bit model on H100,
qualified with fail-closed checks and adopted as the default — about **34%
faster for a single user** (137 vs 102 tokens/s).

## Objective

The in-house GPTQ model served correctly but left speed on the table. Adopt the
faster kernel **only** after it passes the same evidence bar as everything else
we ship.

## Sub-tasks

<!-- EXTENSION POINT: append sub-tasks here AND in the HTML twin; IDs from PROJECT_GOALS.md goal 7. -->

- [x] 7a · Qualified — backend attestation, correctness, stability `wk Jul 20–26`
- [x] 7b · Adopted — ~34% faster for a single user; now the default kernel `wk Jul 20–26`

## Result

The qualified stack keeps the model's 4-bit weights packed in GPU memory and
computes on 8-bit activations, with all production serving features on. A
fail-closed **attestation** proves at serve time that the intended kernel
actually ran — the fast path cannot be silently swapped out. A faster kernel
that returns wrong numbers would be a regression, not a win; qualification
(correctness, stability) came before the benchmark comparison counted.

**Before / after, by concurrency** (26 July paired rerun — same model, same
node, only the kernel changes; TPOT ms with total output tok/s in parentheses):

| Concurrency | Before (CUTLASS) | After (Humming) | Server-throughput gain |
|---|---|---|---|
| 1 | 9.73 (102) | **7.29 (137)** | **+34%** |
| 4 | 11.91 (335) | **8.82 (452)** | **+35%** |
| 16 | 15.28 (1,042) | **12.22 (1,302)** | **+25%** |
| 64 | 22.28 (2,849) | **19.43 (3,262)** | **+14%** |

The gain holds at long context: with 100k-token inputs at concurrency 1,
per-user speed is 131 tok/s on the new kernel vs 100 on the old (+31%).

**How the single-user number moved:**

| Stage | TPOT (conc 1) | What changed |
|---|---|---|
| 0 · Original stack | 10.3–10.5 ms | The kernel behind the first published results |
| 1 · Scheduling fix | 9.7 ms | A GPU-stream defect root-caused and fixed |
| 2 · Humming indexed (adopted) | **7.29 ms** | Hopper-native W4A8 kernel — this goal |
| 3 · Newer variant (optional) | 7.59 ms | Evaluated, slightly slower — not adopted |

## Boundary

This goal changed *how fast* the model serves, not *what* it answers: checkpoint
quality is Goal 3's contract, and the paired evaluation plus the full
performance comparison belong to Goal 2.

## Evidence

- Qualification: [`M3_HUMMING_W4A8_QUALIFICATION_REPORT.md`](../../M3_HUMMING_W4A8_QUALIFICATION_REPORT.md)
- Benchmark: [`m3-two-axis-perf.md`](../m3-two-axis-perf.md), [`M3_OFFICIAL_PERF_RESULTS.html`](../../M3_OFFICIAL_PERF_RESULTS.html)
- Design: [`2026-07-25-minimax-m3-humming-native-w4a8-backend-design.md`](../superpowers/specs/2026-07-25-minimax-m3-humming-native-w4a8-backend-design.md)
- Contract: [`PROJECT_GOALS.md`](../../PROJECT_GOALS.md)
