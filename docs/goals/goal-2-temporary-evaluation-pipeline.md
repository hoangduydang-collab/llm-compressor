# Goal 2 · Evaluation Pipeline — Field Note

> **Work in progress** · markdown twin of
> [`goal-2-temporary-evaluation-pipeline.html`](goal-2-temporary-evaluation-pipeline.html) — keep in sync.
> [← Program overview](../automatic-quantization-pipeline-progress.md)

**Contents:** [Purpose](#purpose) · [Quality results](#quality-results) ·
[Performance results](#performance-results) · [Harness of record](#harness-of-record) ·
[Historical](#historical--wk-jul-1319) · [Evidence](#evidence)

The fail-closed paired harness is built, the migration onto the team's official
pipeline is done, and the in-house GPTQ model holds a seven-task validation.
Open: the seven-task run for the fixed AWQ model.

## Purpose

Comparable model-to-model evidence: every checkpoint gets the same prompts,
examples, decoding settings, and scoring. These are paired fidelity signals for
internal ship decisions — deliberately **not** public-leaderboard scores.

<!-- EXTENSION POINT: append sub-tasks here AND in the HTML twin; IDs from PROJECT_GOALS.md goal 2. -->

- [x] 2a · Paired harness + smoke gate live `wk Jul 20–26`
- [x] 2b · GPTQ validated on all seven tasks — 97–101% recovery `wk Jul 20–26`
- [x] 2c · Serving-performance report (ten arms) `wk Jul 20–26`
- [x] 2d · Second model family — GLM-5.2, three arms `wk Jul 20–26`
- [x] 2e · Second quant track — 2-bit vs baseline A/B `wk Jul 27–Aug 02`
- [x] 2f · Collaborator guide, live-verified `wk Jul 27–Aug 02`
- [ ] 2g · Seven-task run for the fixed AWQ model

## Quality results

**Per-model verdicts (as of 31 July 2026):**

| Model | Verdict | Why |
|---|---|---|
| In-house GPTQ (4-bit) | ✅ Serve — default | 97–101% recovery on all seven tasks; token spend within 2% of baseline |
| In-house AWQ r6 (4-bit) | ✅ Serve | GPQA 98.7%, IFEval 98.6% recovery; seven-task run queued (2g) |
| In-house AWQ r7 (4-bit) | ⚠️ Usable, watch it | GPQA 104.4% but IFEval 95.7% with one-sided losses |
| Vendor MXFP8 (8-bit) | ✅ Reference | 97.5–100.6% recovery; strongest external control |
| In-house AWQ r5 | ❌ Do not serve | Runaway reasoning: GPQA recovery 71.7%, 2.2× token spend |
| Community AWQ (cyankiwi) | ❌ Disqualified | Runaway generations; 55.6% of GPQA answers hit the token budget |

**Newest (23–24 July) · deep-reasoning A/B at 64k tokens.** The hardest test:
long reasoning with a 64k-token budget, scored greedy and paired per question.
This caught the AWQ r5 defect and confirmed its fix (r6). A broken quant fails
by *thinking forever*, not just scoring lower — exhaustion rate and token spend
are the early-warning metrics.

| Metric | BF16 | AWQ r5 | AWQ r6 | AWQ r7 |
|---|---|---|---|---|
| GPQA recovery | — | 71.7% | **98.7%** | **104.4%** |
| GPQA exhausted | 12.6% | 38.9% | 14.7% | **10.6%** |
| GPQA token spend | 1.00× | 2.19× | 1.13× | **0.93×** |
| IFEval recovery | — | 90.9% | **98.6%** | 95.7% |
| IFEval token spend | 1.00× | 3.69× | **1.17×** | 1.25× |

**21–23 July · seven-task validation.** Recovery relative to the unquantized
baseline (100% = no quality loss). This run made the in-house GPTQ model the
default.

| Task | BF16 (abs.) | GPTQ | MXFP8 | AWQ r5 |
|---|---|---|---|---|
| GSM8K | 0.949 | **100.1%** | **100.3%** | 97.3% |
| MMLU | 0.750 | 97.4% | **100.4%** | 93.8% |
| ARC-Challenge | 0.469 | **101.1%** | 98.7% | 98.7% |
| HellaSwag | 0.705 | **99.4%** | **100.3%** | 97.8% |
| TruthfulQA-mc2 | 0.640 | 98.3% | **100.6%** | 96.7% |
| IFEval | 0.874 | **98.3%** | 97.5% | 93.0% |
| GPQA-Diamond | 0.813 | **99.4%** | 98.1% | 70.2% |

Full data (every arm, per-question flips, sampling probes):
[`M3_OFFICIAL_QUALITY_RESULTS.html`](../../M3_OFFICIAL_QUALITY_RESULTS.html).
Same protocol on a second family:
[`GLM52_OFFICIAL_EVAL_RESULTS.html`](../../GLM52_OFFICIAL_EVAL_RESULTS.html).

## Performance results

Speed per user (the number a person feels) across the serving-ready models,
from the 26 July single-controller rerun. The in-house GPTQ + fast kernel leads
every concurrency level.

| Concurrency | GPTQ · fast kernel | AWQ r7 | Community AWQ | MXFP8 | BF16 (16 GPUs) |
|---|---|---|---|---|---|
| 1 | **137** | **136** | 118 | 107 | 81 |
| 4 | **451** | **453** | 415 | 349 | 246 |
| 16 | **1,300** | **1,303** | 1,268 | 968 | 711 |
| 64 | **3,267** | **3,262** | 2,923 | 2,177 | 1,700 |

*Output speed per user, tok/s · 1k in / 8k out.*

**4-bit vs the unquantized baseline · economics:**

| Metric | BF16 · 16 GPUs, 2 nodes | In-house 4-bit · 8 GPUs, 1 node | Advantage |
|---|---|---|---|
| Model weights | 796 GB | **225 GB** | 3.5× smaller |
| Speed per user (conc 1) | 81 tok/s | **137 tok/s** | 1.7× faster on half the GPUs |
| GPU-hours per 1M tokens | 2.61 | **0.68** | 3.8× cheaper |

The fast kernel itself (~34% single-user gain, qualified with fail-closed
checks) is [Goal 7](goal-7-native-humming-w4a8.md). Full data:
[`M3_OFFICIAL_PERF_RESULTS.html`](../../M3_OFFICIAL_PERF_RESULTS.html),
[`m3-two-axis-perf.md`](../m3-two-axis-perf.md).

## Harness of record

MiniMax-M3 runs on the team's **official evaluation pipeline**
(`AICloud/benchmarks`, served on a pinned inference stack); the results above
come from it. The same protocol produced the three-arm GLM-5.2 evaluation (the
community W4AFP8 checkpoint is clean; our earlier AWQ pathology does not
replicate there) and the 2-bit A/B on the sub-4-bit track (Goal 4).

## Historical · wk Jul 13–19

An earlier interim harness produced the program's first four-way comparison
(in-house GPTQ / community AWQ / vendor MXFP8 / BF16 on seeded subsets, three
seeds). Its numbers use a different protocol and are **not comparable** to the
official-pipeline results above; kept for the record in the HTML twin.

## Evidence

- Quality: [`M3_OFFICIAL_QUALITY_RESULTS.html`](../../M3_OFFICIAL_QUALITY_RESULTS.html) (2b) · [`GLM52_OFFICIAL_EVAL_RESULTS.html`](../../GLM52_OFFICIAL_EVAL_RESULTS.html) (2d)
- Performance: [`M3_OFFICIAL_PERF_RESULTS.html`](../../M3_OFFICIAL_PERF_RESULTS.html), [`m3-two-axis-perf.md`](../m3-two-axis-perf.md) (2c)
- Collaborators: [`M3_COLLABORATOR_GUIDE.md`](../../M3_COLLABORATOR_GUIDE.md) (2f)
- Harness qualification: [`M3_PRODUCTION_EVAL_HANDOFF.md`](../../M3_PRODUCTION_EVAL_HANDOFF.md) · Contract: [`PROJECT_GOALS.md`](../../PROJECT_GOALS.md)
