# Goal 7 · Native Humming W4A8 Serving — Field Note

> **Done · 2026-07-26** · web version:
> [`goal-7-native-humming-w4a8.html`](goal-7-native-humming-w4a8.html).
> [← Program overview](../automatic-quantization-pipeline-progress.md)

**Contents:** [Objective](#objective) · [Sub-tasks](#sub-tasks) ·
[Result](#result) · [Detailed results](#detailed-results) ·
[Boundary](#boundary) · [Evidence](#evidence)

A faster serving kernel (Humming W4A8) for the in-house 4-bit model on H100,
qualified with fail-closed checks and adopted as the default — about **34%
faster for a single user** (137 vs 102 tokens/s).

## Objective

The in-house GPTQ model served correctly but left speed on the table. Adopt the
faster kernel **only** after it passes the same evidence bar as everything else
we ship.

## Sub-tasks


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
node, only the kernel changes). TPOT is the steady time per output token — lower is faster; the parenthesized number is the whole server’s output rate, tok/s:

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

## Detailed results

**All kernel variants measured, model fixed.** TPOT (steady time per output
token, lower is faster) with total server output in parentheses. Four
variants were benchmarked against the old kernel; the adopted one wins decode
at every load level, and the newer "packed-K" build's small win at 64 users
was not worth its single-user loss.

| Users | Old (CUTLASS) | Adopted (Humming 0.1.10) | Humming grouped | 0.1.11 packed-K |
|---|---|---|---|---|
| 1 | 9.73 ms (102) | **7.29 ms (137)** | 8.48 ms (117) | 7.59 ms (131) |
| 4 | 11.91 ms (335) | **8.82 ms (452)** | 9.91 ms (402) | 8.90 ms (448) |
| 16 | 15.28 ms (1,042) | **12.22 ms (1,302)** | 13.28 ms (1,198) | 12.23 ms (1,300) |
| 64 | 22.28 ms (2,849) | 19.43 ms (3,262) | 20.64 ms (3,046) | **19.21 ms (3,299)** |

**The gain holds at every prompt length.** Speed per user (first-token
latency in parentheses); the kernel ordering never changes from a 1k to a
100k-token prompt.

| Prompt × users | Old (CUTLASS) | Adopted (Humming 0.1.10) |
|---|---|---|
| 1k × 1 | 102.2 (114 ms) | **137.3 (114 ms)** |
| 1k × 10 | 71.9 (403 ms) | **92.2 (423 ms)** |
| 10k × 1 | 102.0 (532 ms) | **137.1 (554 ms)** |
| 10k × 10 | 56.6 (1.7 s) | **75.8 (2.7 s)** |
| 100k × 1 | 100.3 (5.2 s) | **130.9 (5.4 s)** |

**The one thing the old kernel still does better:** first-token latency under
heavy load (16+ users), where the old kernel prepares long prompts 12–29%
faster. Decoding — where users spend nearly all their time — is 23–35% faster
on the new kernel at every operating point, which is why it is the default.

**Four defects found and fixed on the way.** Qualification surfaced four real
bugs in the vendor kernel library (the largest: stored results were never
actually committed to memory, making all result-waits no-ops). All four are
fixed in our build and reported; none are fixed in the vendor's releases yet,
so every serve run verifies at startup — fail-closed — that the patched
kernel is the one actually running.

**Same kernel, second job:** it also serves the speculative-decoding draft
model under load, worth a further +2.25% per user at 10 concurrent users
([Goal 2, speculative decoding](goal-2-temporary-evaluation-pipeline.md#speculative-decoding-results)).

## Boundary

This goal changed *how fast* the model serves, not *what* it answers: checkpoint
quality is Goal 3's contract, and the paired evaluation plus the full
performance comparison belong to Goal 2.

## Evidence

- Qualification: [`M3_HUMMING_W4A8_QUALIFICATION_REPORT.md`](../../M3_HUMMING_W4A8_QUALIFICATION_REPORT.md)
- Benchmark: [`m3-two-axis-perf.md`](../m3-two-axis-perf.md), [`M3_OFFICIAL_PERF_RESULTS.html`](../../M3_OFFICIAL_PERF_RESULTS.html)
- Design: [`2026-07-25-minimax-m3-humming-native-w4a8-backend-design.md`](../superpowers/specs/2026-07-25-minimax-m3-humming-native-w4a8-backend-design.md)
- Contract: [`PROJECT_GOALS.md`](../../PROJECT_GOALS.md)
