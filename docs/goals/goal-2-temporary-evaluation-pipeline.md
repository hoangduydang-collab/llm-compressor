# Goal 2 · Evaluation Pipeline — Field Note

> **Work in progress** · web version:
> [`goal-2-temporary-evaluation-pipeline.html`](goal-2-temporary-evaluation-pipeline.html).
> [← Program overview](../automatic-quantization-pipeline-progress.md)

**Contents:** [Purpose](#purpose) · [Quality results](#quality-results) ·
[Speculative decoding](#speculative-decoding-results) ·
[Performance results](#performance-results) · [Harness of record](#harness-of-record) ·
[Historical](#historical--wk-jul-1319) ·
[Appendix (detailed tables)](#appendix--detailed-results-tables) · [Evidence](#evidence)

The fail-closed paired harness is built, the migration onto the team's official
pipeline is done, and the in-house GPTQ model holds a seven-task validation.
Open: the seven-task run for the fixed AWQ model.

## Purpose

Comparable model-to-model evidence: every checkpoint gets the same prompts,
examples, decoding settings, and scoring. These are paired fidelity signals for
internal ship decisions — deliberately **not** public-leaderboard scores.


- [x] 2a · Paired harness + smoke gate live `wk Jul 20–26`
- [x] 2b · GPTQ validated on all seven tasks — 97–101% recovery `wk Jul 20–26`
- [x] 2c · Serving-performance report (ten arms) `wk Jul 20–26`
- [x] 2d · Second model family — GLM-5.2, three arms `wk Jul 20–26`
- [x] 2e · Second quant track — 2-bit vs baseline A/B `wk Jul 27–Aug 02`
- [x] 2f · Collaborator guide, live-verified `wk Jul 27–Aug 02`
- [ ] 2g · Seven-task run for the fixed AWQ model
- [x] 2h · Speculative-decoding tuning study — 1.21–2.53× decode speedup, no quality cost `wk Jul 27–Aug 02`

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

**Newest (31 July) · 2-bit vs baseline A/B on a second quant track (Qwen3-30B
MoE).** The same paired protocol, applied to the program's first 2-bit
checkpoint (Goal 4). Verdict: **not usable at the smoke-tier tuning budget** —
the pipeline's gates blocked it exactly as designed. A longer-tuning retry is
queued (4c).

| Metric | BF16 | 2-bit (W2A16) | Delta |
|---|---|---|---|
| GPQA-Diamond (exact match) | 0.54 | 0.24 | **−0.30** |
| IFEval (strict accuracy) | 0.84 | 0.45 | **−0.39** |
| GPQA answers hitting the token cap | 52% | 81% | +29 pts |
| IFEval answers hitting the token cap | 0% | 32% | +32 pts |
| IFEval token spend | 1.00× | **3.44×** | |

*Failure mode: the 2-bit model falls into verbatim repetition loops — the same
"thinking forever" signature the harness caught on AWQ r5. Both arms' GPQA
absolute scores are depressed by the 4,096-token budget; the paired deltas are
the decision signal.*

**23–24 July · deep-reasoning A/B at 64k tokens (MiniMax-M3).** The hardest test:
long reasoning with a 64k-token budget, deterministic decoding, scored side by side per question.
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

**22–23 July · second model family (GLM-5.2, three arms).** The same paired
protocol on a different 753B-parameter model, comparing the vendor's FP8
release and a community 4-bit (W4AFP8) checkpoint that was *also produced with
AWQ* — a direct test of whether the M3 AWQ failure is a general AWQ problem.
It is not: both quants recover 96–105% on all seven tasks, every difference
inside the noise band.

| Task | BF16 (abs.) | FP8 (vendor) | W4AFP8 (community, AWQ) |
|---|---|---|---|
| GSM8K | 0.958 | 98.5% | 99.3% |
| IFEval | 0.882 | 100.0% | 99.6% |
| GPQA-Diamond | 0.717 | 99.3% | **101.4%** |
| MMLU | 0.448 | 105.3% | 99.1% |
| ARC-Challenge | 0.466 | 101.8% | 102.4% |
| HellaSwag | 0.755 | 100.1% | 100.7% |
| TruthfulQA-mc2 | 0.598 | 101.1% | 96.2% |

The three early-warning signals that convicted M3's broken AWQ recipe all come
back healthy on GLM-5.2 — so "AWQ breaks reasoning models" is the wrong
lesson; the M3 failure was one model × recipe interaction, since fixed:

| Signal | M3 · broken AWQ (r5) | GLM-5.2 · AWQ-produced W4AFP8 |
|---|---|---|
| GPQA recovery | 70.2% | **101.4%** |
| GPQA answers lost / gained | 53 / 5 (one-sided) | 15 / 17 (symmetric) |
| GPQA token spend | 2.19× | **0.86×** |
| GPQA budget exhaustion | 38.9% (baseline 12.6%) | **17.2%** (baseline 21.2%) |
| IFEval token spend | 3.69× | **0.89×** |

Full data (every arm, per-question flips, sampling probes):
[`M3_OFFICIAL_QUALITY_RESULTS.html`](../../M3_OFFICIAL_QUALITY_RESULTS.html) ·
[`GLM52_OFFICIAL_EVAL_RESULTS.html`](../../GLM52_OFFICIAL_EVAL_RESULTS.html).
More quality tables: [appendix](#appendix--detailed-results-tables).

## Speculative decoding results

**Newest (27–29 July) · how much faster the 4-bit model gets with a draft
model.** Speculative decoding pairs the full model with a small "drafter" that
proposes several tokens per step; the full model then verifies them all at
once. Every token is still checked by the full model, so **output quality is
untouched by construction** — the only question is speed. It is the single
largest decode-speed lever measured in this program, larger than any kernel or
quantization change below.

The size of the win is set by *what the model is writing* — predictable content
(code, structured output) drafts easily, creative prose does not — and not by
prompt length or load:

| Traffic | Users | Best draft depth | Speed per user, with vs without | Speedup |
|---|---|---|---|---|
| Code / structured | 1 | 6 tokens | **346** vs 137 tok/s | **2.53×** |
| Code / structured | 10 | 5 tokens + fast drafter kernel | **155** vs 75 tok/s | **2.07×** |
| Mixed real chat | 1 | 3 tokens | **250** vs 138 tok/s | **1.81×** |
| Creative writing | 1 | 2 tokens | **186** vs 137 tok/s | **1.36×** |
| Creative writing | 10 | 2 tokens | **97** vs 81 tok/s | **1.21×** |

*Per-user decode speed on the in-house 4-bit model, one 8-GPU node, with an
off-the-shelf community drafter (quantized to 4-bit in-house).*

Server capacity rises alongside per-user speed — at ten users on code the same
8 GPUs go from ~706 to ~1,428 total output tok/s — so this is not a
latency-for-throughput trade. And the drafter accepts our 4-bit model's
guidance exactly as well as the vendor's own 8-bit release, while our stack is
22–40% faster in absolute terms in every measured cell.

**The three tuning levers**, in order of value:

| Lever | Recommendation | Worth |
|---|---|---|
| Draft depth (tokens proposed per step) | Tune per traffic class: 5–6 for code, 2 for creative writing — the reference default (3) is wrong for both | up to ~12% |
| Drafter kernel | Switch the drafter to the Humming kernel only under load | +2.3% at 10 users; nothing at 1 |
| Drafter precision | Serve the 4-bit drafter — free in draft quality, never hurts | ~2.6% single-user, <1% under load |

**Caveats.** Prefill-dominated traffic (long prompts, short answers) gains
almost nothing — speculation accelerates the writing, not the reading. Numbers
come from NVIDIA's SPEED-Bench (clean subset) and support configuration
decisions, not leaderboard comparison. All conclusions were hardened by a
replicated re-measurement (35 serves on one node, 29 July) that corrected
three earlier readings.

Full study:
[`M3_OFFICIAL_SPECDEC_RESULTS.html`](../../M3_OFFICIAL_SPECDEC_RESULTS.html);
complete write-up: [`m3-specdec-eagle3.md`](../m3-specdec-eagle3.md).

## Performance results

Serving throughput across the ready models, from the 26 July paired rerun. At
concurrency 1 the number is one user’s speed — what a person actually feels:
**137 tok/s** on the lead model vs 81 for the unquantized baseline. The
in-house GPTQ + fast kernel leads every load level.

| Concurrency | GPTQ · fast kernel | AWQ r7 | Community AWQ | MXFP8 | BF16 (16 GPUs) |
|---|---|---|---|---|---|
| 1 | **137** | **136** | 118 | 107 | 81 |
| 4 | **451** | **453** | 415 | 349 | 246 |
| 16 | **1,300** | **1,303** | 1,268 | 968 | 711 |
| 64 | **3,267** | **3,262** | 2,923 | 2,177 | 1,700 |

*Total server output, tok/s · 1k-token input / 8k-token output; at concurrency 1 this equals one user’s speed.*

**4-bit vs the unquantized baseline · economics:**

| Metric | BF16 · 16 GPUs, 2 nodes | In-house 4-bit · 8 GPUs, 1 node | Advantage |
|---|---|---|---|
| Model weights | 796 GB | **225 GB** | 3.5× smaller |
| Speed per user (conc 1) | 81 tok/s | **137 tok/s** | 1.7× faster on half the GPUs |
| GPU-hours per 1M tokens | 2.61 | **0.68** | 3.8× cheaper |

**Against the market.** On the matched single-user workload shape, our 4-bit
model decodes faster than every commercial MiniMax-M3 endpoint but one
(Nebius, whose lead comes from a serving-stack tier — newer silicon plus
speculative decoding — that would stack on our checkpoint too). Provider
numbers carry live multi-tenant load, so read this as same-ballpark placement,
not a ranking.

| Endpoint | Speed per user (tok/s) |
|---|---|
| Nebius (FP8) | 197 |
| **Ours · in-house 4-bit + fast kernel** | **137** |
| Ours · vendor MXFP8 | 107 |
| Together AI | 97 |
| MiniMax first-party | 95 |
| Novita | 94 |
| SiliconFlow | 92 |
| Ours · BF16 baseline (16 GPUs) | 81 |
| Parasail | 66 |

Two operational headlines from the same window: the in-house model holds a
**1-second p95 first-token SLO up to 16 concurrent agentic users** (the
vendor MXFP8 and community arms miss it), and it still decodes at **131 tok/s
with a 100k-token prompt**.

The fast kernel itself (~34% single-user gain, qualified with fail-closed
checks) is [Goal 7](goal-7-native-humming-w4a8.md). Full data:
[`M3_OFFICIAL_PERF_RESULTS.html`](../../M3_OFFICIAL_PERF_RESULTS.html),
[`m3-two-axis-perf.md`](../m3-two-axis-perf.md). More performance tables:
[appendix](#appendix--detailed-results-tables).

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

## Appendix · detailed results tables

Everything above is the decision view; the tables below carry the full detail
behind it, grouped to match the main sections. Sources are the same official
reports listed under [Evidence](#evidence).

### A1 · Quality detail (MiniMax-M3)

**Score and budget exhaustion at the 64k budget, all arms.** "Exhausted" =
the answer hit the 64k generation ceiling without finishing; even the
baseline has a ~12% hard core on GPQA, so the signal is the excess above it.

| Arm | GPQA score | GPQA exhausted | IFEval score | IFEval exhausted |
|---|---|---|---|---|
| BF16 (baseline) | 0.803 | 12.6% | 0.893 | 0.9% |
| In-house GPTQ | 0.803 | 11.6% | 0.874 | 1.1% |
| Vendor MXFP8 | 0.828 | 7.6% | 0.885 | 1.3% |
| In-house AWQ r5 | 0.576 | 38.9% | 0.811 | 9.2% |
| In-house AWQ r6 | 0.793 | 14.7% | 0.880 | 1.5% |
| In-house AWQ r7 | 0.838 | 10.6% | 0.854 | 1.9% |
| Community AWQ (cyankiwi) | 0.455 | 55.6% | 0.782 | 13.7% |

**Token spend per arm (× BF16, same prompts, 64k budget).** A healthy quant
spends what the baseline spends; a broken one thinks in circles.

| Arm | GPQA spend | IFEval spend |
|---|---|---|
| In-house GPTQ | 1.01× | 1.02× |
| Vendor MXFP8 | 0.80× | 1.08× |
| In-house AWQ r5 | 2.19× | 3.69× |
| In-house AWQ r6 | 1.13× | 1.17× |
| In-house AWQ r7 | 0.93× | 1.25× |
| Community AWQ (cyankiwi) | 3.06× | 5.09× |

**Answer-level churn (right→wrong / wrong→right vs the baseline).** Direction
separates noise from damage: GPTQ and MXFP8 flip roughly symmetrically; the
broken AWQ r5 loses one-sidedly, most extreme on GPQA.

| Task (questions) | GPTQ ✗/✓ | MXFP8 ✗/✓ | AWQ r5 ✗/✓ |
|---|---|---|---|
| GPQA (198) | 8 / 7 | 10 / 7 | **53 / 5** |
| IFEval (541) | 34 / 27 | 31 / 19 | 57 / 24 |
| MMLU (14,042) | 971 / 695 | 622 / 666 | 1,416 / 761 |
| GSM8K (1,319) | 26 / 27 | 17 / 21 | 59 / 25 |
| HellaSwag (10,042) | 199 / 153 | 118 / 137 | 400 / 241 |
| ARC (1,172) | 37 / 43 | 38 / 31 | 60 / 53 |

**Every metric through the AWQ fix (r5 → r6/r7), 64k budget.**

| Metric | BF16 | AWQ r5 | AWQ r6 | AWQ r7 |
|---|---|---|---|---|
| GPQA score | 0.803 | 0.576 | 0.793 | **0.838** |
| GPQA recovery | — | 71.7% | 98.7% | **104.4%** |
| GPQA exhausted | 12.6% | 38.9% | 14.7% | **10.6%** |
| GPQA token spend | 1.00× | 2.19× | 1.13× | **0.93×** |
| GPQA flips ✗/✓ | — | 50 / 5 | 11 / 9 | 9 / 16 |
| IFEval score | 0.893 | 0.811 | **0.880** | 0.854 |
| IFEval recovery | — | 90.9% | **98.6%** | 95.7% |
| IFEval exhausted | 0.9% | 9.2% | **1.5%** | 1.9% |
| IFEval token spend | 1.00× | 3.69× | **1.17×** | 1.25× |
| IFEval flips ✗/✓ | — | 62 / 18 | 29 / 22 | 37 / 16 |

**Sampling probe on the hardest (previously stuck) GPQA questions.** Re-runs
the exact 50 questions where each arm had hit the budget, under greedy and
under the vendor's recommended sampling. The fixed recipe (r6) is the only
arm that improves on both decoding modes and is nearly always right when it
finishes; the community AWQ stays stuck either way.

| Arm | Greedy non-termination | Sampled non-termination | Greedy correct | ≥1 correct in 3 draws | Correct when it finishes |
|---|---|---|---|---|---|
| BF16 (control) | 64.0% | 62.7% | 36% | 44% | 86% |
| GPTQ (control) | 52.2% | 58.0% | 30% | 57% | 90% |
| AWQ r5 | 76.0% | 49.3% | 24% | 60% | 83% |
| AWQ r6 | **46.0%** | **44.0%** | **52%** | **66%** | **95%** |
| Community AWQ | 94.0% | 87.3% | 6% | 12% | 32% |

### A2 · Quality detail (GLM-5.2, second family)

**Answer-level churn.** Both GLM-5.2 quants flip symmetrically — churn, not
directional damage (compare M3's broken AWQ at 53 lost / 5 gained).

| Task (questions) | FP8 ✗/✓ | W4AFP8 ✗/✓ |
|---|---|---|
| GPQA (198) | 19 / 18 | 15 / 17 |
| IFEval (541) | 25 / 25 | 27 / 25 |
| GSM8K (1,319) | 30 / 11 | 30 / 21 |
| MMLU (14,042) | 1,156 / 1,489 | 1,784 / 1,730 |
| HellaSwag (10,042) | 244 / 252 | 279 / 331 |
| ARC (1,172) | 45 / 55 | 49 / 62 |

**Token spend and budget exhaustion at 64k.** About 1 in 5 GPQA questions
never finishes under this protocol on *any* arm, baseline included — the
censoring binds all arms equally.

| Task · arm | Mean tokens | Median | Exhausted | Spend × BF16 |
|---|---|---|---|---|
| GPQA · BF16 | 18,985 | 5,381 | 21.2% | 1.00× |
| GPQA · FP8 | 18,718 | 4,871 | 21.7% | 0.99× |
| GPQA · W4AFP8 | 16,410 | 4,627 | **17.2%** | **0.86×** |
| IFEval · BF16 | 5,781 | 1,299 | 6.5% | 1.00× |
| IFEval · FP8 | 5,475 | 1,331 | 5.9% | 0.95× |
| IFEval · W4AFP8 | 5,144 | 1,250 | **5.4%** | **0.89×** |
| GSM8K · BF16 | 1,132 | 428 | 0.9% | 1.00× |
| GSM8K · FP8 | 1,225 | 439 | 1.1% | 1.08× |
| GSM8K · W4AFP8 | 1,272 | 444 | 1.1% | 1.12× |

### A3 · Speculative-decoding detail

**Draft-depth sweep, code/structured content.** "Accepted" = tokens kept per
draft-verify step. Speed keeps rising until the extra drafting work outweighs
the extra accepted tokens; the knee is at depth 5–6.

| Depth | Accepted (1 user) | Speed, 1 user | Speedup | Accepted (10 users) | Speed per user, 10 users | Speedup |
|---|---|---|---|---|---|---|
| off | — | 136.8 tok/s | 1.00× | — | 74.8 tok/s | 1.00× |
| 5 | 3.82 | 334.1 | 2.43× | 3.86 | **153.5** | **2.05×** |
| 6 | 4.11 | **341.6** | **2.50×** | 4.08 | 149.7 | 1.94× |
| 7 | 4.22 | 339.2 | 2.48× | 4.31 | 152.4 | 1.97× |

*Speedups in the two sweep tables are measured on decode latency, so they can
differ by a few percent from the headline table's speed-per-user ratios.*

**Draft-depth sweep, creative writing.** Acceptance is much lower, so the
optimum is shallow — depth 2, not the reference default of 3.

| Depth | Accepted (1 user) | Speed, 1 user | Speedup | Accepted (10 users) | Speed per user, 10 users | Speedup |
|---|---|---|---|---|---|---|
| off | — | 136.8 tok/s | 1.00× | — | 80.5 tok/s | 1.00× |
| 1 | 1.51 | 178.0 | 1.29× | 1.51 | 95.1 | 1.17× |
| 2 | 1.72 | **185.6** | **1.36×** | 1.75 | **97.0** | **1.18×** |
| 3 (default) | 1.82 | 187.9 | 1.33× | 1.82 | 94.7 | 1.15× |

*At 1 user, depths 2 and 3 are tied within noise; depth 2 is the
recommendation because it also wins under load.*

**Server-side view at each tier's best depth.** Speculation raises total
server output too — not a latency-for-throughput trade.

| Workload | Users | Depth | Speed per user | vs off | Server tok/s | vs off |
|---|---|---|---|---|---|---|
| Code | 1 | off | 136.8 | — | 132.1 | — |
| Code | 1 | 6 | **341.6** | 2.50× | 316.3 | 2.40× |
| Code | 10 | off | 74.8 | — | 706.4 | — |
| Code | 10 | 5 | **153.5** | 2.05× | 1,427.8 | 2.02× |
| Creative | 1 | off | 136.8 | — | 132.4 | — |
| Creative | 1 | 2 | 185.6 | 1.36× | 175.1 | 1.32× |
| Creative | 10 | off | 80.5 | — | 764.9 | — |
| Creative | 10 | 2 | 97.0 | 1.21× | 892.5 | 1.17× |

**Drafter kernel (replicated, code content, depth 5).** Only one variant
clears its own noise, and only under load: running the drafter entirely on
the Humming kernel is +2.25% per user at 10 users. At a single user, nothing
beats the default.

| Drafter kernel | Speed, 1 user | vs default | Speed per user, 10 users | vs default |
|---|---|---|---|---|
| Default (Machete + Marlin) | 339.3 | — | 152.0 | — |
| Humming on the output layer only | 334.2 | −1.50% | 153.7 | +1.11% (noise) |
| Humming on all nine layers | 341.5 | +0.66% (noise) | **155.5** | **+2.25%** |
| Machete everywhere (padded) | 340.7 | +0.43% (noise) | 153.9 | +1.20% (noise) |

**Drafter precision (4-bit vs full-precision drafter), 12-cell A/B.** The
4-bit drafter is free in draft quality (acceptance change −0.21% mean, signs
flipping cell to cell) and consistently, if modestly, faster: step cost lower
in 12 of 12 cells (mean −1.8%), per-user speed higher in 11 of 12. At the
deployment settings it is worth ~2.6% per user single-stream and under 1%
under load.

**Content type, not prompt length, decides the win.** Acceptance moves +75%
between content tiers but less than 0.05 over a 32× range of prompt length
(1k → 32k tokens), and is flat from 1 to 64 users.

| Content tier | Accepted length (depth 3) | 1st / 2nd / 3rd draft token survives |
|---|---|---|
| Code, sorting, structured output | 3.11 | 84% / 70% / 57% |
| Creative writing | 1.78 | 47% / 21% / 10% |

### A4 · Performance detail

**Kernel comparison, model fixed (in-house GPTQ).** Decode latency per token
(TPOT, lower is better) with total server output in parentheses. The Humming
indexed kernel wins decode at every load; the newer packed-K variant trades
single-user speed for loaded throughput and was not adopted.

| Users | CUTLASS | Humming 0.1.10 (adopted) | Humming grouped | 0.1.11 packed-K |
|---|---|---|---|---|
| 1 | 9.73 ms (102) | **7.29 ms (137)** | 8.48 ms (117) | 7.59 ms (131) |
| 4 | 11.91 ms (335) | **8.82 ms (452)** | 9.91 ms (402) | 8.90 ms (448) |
| 16 | 15.28 ms (1,042) | **12.22 ms (1,302)** | 13.28 ms (1,198) | 12.23 ms (1,300) |
| 64 | 22.28 ms (2,849) | 19.43 ms (3,262) | 20.64 ms (3,046) | **19.21 ms (3,299)** |

**Server output rate per GPU (the efficiency axis).** Total output tok/s with
per-GPU rate in parentheses — quant arms run 8 GPUs, BF16 runs 16.

| Workload · users | GPTQ + fast kernel | AWQ r7 | Community AWQ | MXFP8 | BF16 |
|---|---|---|---|---|---|
| Reasoning · 64 | **3,267 (408)** | 3,262 (408) | 2,923 (365) | 2,177 (272) | 1,700 (106) |
| Agentic warm · 1 | **112 (14.0)** | 109 (13.7) | 104 (13.0) | 91 (11.3) | 70 (4.4) |
| Agentic warm · 16 | **727 (90.9)** | 727 (90.9) | 662 (82.8) | 576 (72.0) | 490 (30.6) |
| Agentic warm · 32 | **1,036 (129.5)** | 1,036 (129.6) | 931 (116.4) | 825 (103.2) | 708 (44.3) |

**Agentic multi-turn serving (speed per user, with p95 first-token latency).**
Warm = the conversation prefix is cached, the production case. The in-house
arms hold the 1-second p95 first-token SLO to 16 users; without caching, no
arm holds it past a single user — cache reuse is mandatory at any precision.

| Users (warm) | GPTQ + fast kernel | AWQ r7 | Community AWQ | MXFP8 | BF16 (16 GPUs) |
|---|---|---|---|---|---|
| 1 | **138.9 (p95 209 ms)** | 138.2 (225 ms) | 127.5 (212 ms) | 108.1 (206 ms) | 80.8 (226 ms) |
| 4 | 106.5 (487 ms) | **107.8 (433 ms)** | 101.0 (423 ms) | 84.0 (439 ms) | 60.3 (420 ms) |
| 16 | 72.2 (893 ms) | **73.2 (898 ms)** | 69.0 (1,085 ms ✗) | 55.6 (1,062 ms ✗) | 41.4 (783 ms) |
| 32 | **49.1 (1,314 ms ✗)** | 47.3 (1,231 ms ✗) | 48.1 (1,836 ms ✗) | 38.8 (1,647 ms ✗) | 27.3 (808 ms) |

*✗ = misses the 1-second p95 first-token SLO. BF16 buys its low first-token
latency with twice the GPUs.*

**Prompt-length sweep (speed per user; first-token latency in parentheses).**
Decode speed barely moves from a 1k to a 100k-token prompt on the in-house
arm; a 100k prompt takes ~5.4 s to first token.

| Prompt × users | GPTQ + fast kernel | AWQ r7 | MXFP8 | BF16 |
|---|---|---|---|---|
| 1k × 1 | **137.3 (114 ms)** | 137.0 (117 ms) | 107.7 (118 ms) | 80.9 (129 ms) |
| 1k × 10 | **92.2 (423 ms)** | 90.6 (428 ms) | 71.1 (638 ms) | 50.5 (491 ms) |
| 10k × 1 | **137.1 (554 ms)** | 137.1 (549 ms) | 107.6 (836 ms) | 80.6 (583 ms) |
| 10k × 10 | **75.8 (2.7 s)** | 70.4 (2.2 s) | 55.8 (4.4 s) | 44.3 (2.4 s) |
| 100k × 1 | 130.9 (5.4 s) | **133.4 (5.4 s)** | 104.9 (8.3 s) | 78.4 (5.9 s) |

*The community AWQ arm is omitted here: on long prompts it decodes to the
token cap (its quality pathology made performance-visible), so its cells are
not a comparable workload.*

## Evidence

- Quality: [`M3_OFFICIAL_QUALITY_RESULTS.html`](../../M3_OFFICIAL_QUALITY_RESULTS.html) (2b) · [`GLM52_OFFICIAL_EVAL_RESULTS.html`](../../GLM52_OFFICIAL_EVAL_RESULTS.html) (2d)
- Performance: [`M3_OFFICIAL_PERF_RESULTS.html`](../../M3_OFFICIAL_PERF_RESULTS.html), [`m3-two-axis-perf.md`](../m3-two-axis-perf.md) (2c)
- Speculative decoding: [`M3_OFFICIAL_SPECDEC_RESULTS.html`](../../M3_OFFICIAL_SPECDEC_RESULTS.html), [`m3-specdec-eagle3.md`](../m3-specdec-eagle3.md) (2h)
- Collaborators: [`M3_COLLABORATOR_GUIDE.md`](../../M3_COLLABORATOR_GUIDE.md) (2f)
- Harness qualification: [`M3_PRODUCTION_EVAL_HANDOFF.md`](../../M3_PRODUCTION_EVAL_HANDOFF.md) · Contract: [`PROJECT_GOALS.md`](../../PROJECT_GOALS.md)
