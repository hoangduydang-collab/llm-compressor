# Sep 14 - Sep 18

## Duy

### What I worked on

- **Confirmed AA's actual sampling config with AA directly.** Their published
  methodology documents a generic default (0.6 / 1.0) *and* a rule that a lab's
  recommended config takes precedence, without saying per model which applies.
  GLM-5.3 is scored at Z.ai's **1.0 / 0.95**. Both AA benchmarks gained ~10
  points on it and got ~3x cheaper.
- **The third arm now has numbers.** Native EP-GPTQ → W4AFP8 finished, qualified,
  and has been benchmarked. Last week's tables were two-arm.
- Three benchmarks ran: AA GPQA Diamond, AA-LCR v1.1, and the cheap full7 suite.

Detail: [AA-LCR](2026-09-16-glm53-aa-lcr-v11-zai-sampling.md) ·
[EP GPTQ + first full7](2026-09-14-glm53-ep-gptq-w4afp8-and-full7.md) ·
[full7 rerun](2026-09-17-glm53-full7-zai-sampling.md).

### Key result 1, the confirmed config is worth ~10 points on both AA benchmarks

| Benchmark | Generic 0.6/1.0 | **Z.ai 1.0/0.95** | AA published | Gap to AA |
|---|---:|---:|---:|---:|
| **GPQA Diamond**, ours | 81.82% | **91.52%** ±0.87 | ~91.7% | 9.9 pp → **0.2 pp** |
| **GPQA Diamond**, PhalaCloud | 79.39% | **91.11%** ±0.85 | ~91.7% | 12.3 pp → **0.6 pp** |
| **AA-LCR v1.1**, ours | 71.33% | **79.00%** | 80% | 8.7 pp → **1.0 pp** |

GPQA is the formal AA protocol, 198 items × 5 repeats = 990 completions per arm,
all HTTP 200. AA-LCR is 100 × 3. Public-methodology reproductions, not official
AA runs — but the sampling axis is now confirmed rather than inferred.

**Why it moves so much: truncation, and it is a sampling effect not a budget
one.** Low temperature with an untruncated tail lets a minority of traces run
away into the output cap and return empty.

| GPQA, per arm | ours 0.6/1.0 | **ours 1.0/0.95** | phala 0.6/1.0 | **phala 1.0/0.95** |
|---|---:|---:|---:|---:|
| Hit the 131,072 cap | 120/990 (12.1%) | **4/990 (0.4%)** | 155/990 (15.7%) | **14/992 (1.4%)** |
| Avg completion tokens | 26,463 | **14,757** | 30,745 | **16,332** |
| Wall clock, one 8×H100 | 37.1 h | **11.2 h** | 46.2 h | **14.3 h** |

Same on AA-LCR: 32/300 cap hits → 2/300, generated tokens −62%, while
non-truncated accuracy barely moved (79.48% → 79.53%). The whole gain is about
not running away — which confirms last week's hypothesis that truncation
explained most of the gap to AA's 91.7%.

Two consequences worth keeping: a creator's recommended config can be worth ten
points on a reasoning benchmark, so it is a first-class part of a published
score; and **raising the cap is the wrong lever** — a 262,144 AA-LCR rerun
converted zero items and scored *lower* on more tokens.

### Key result 2, the GPTQ arm

**Checkpoint.** Quantization job 15 h 04 m, clean, 371 GiB. Qualification passed:
splitting the work 8 ways gives **answers identical to the 4-way run** at
**half the peak memory** (51.8%) — the claim that was unproven last week. It also
produces the servable format directly, skipping the ~3.5 h of post-processing the
AWQ path needs.

**Quality.** On the cheap greedy full7 it won the only task that separated the
arms by more than 2 pp:

| | GPTQ | AWQ | Phala |
|---|---:|---:|---:|
| **GPQA Diamond CoT** | **67.68%** | 61.11% | 55.56% |

Elsewhere on full7 it is inside noise (GSM8K −0.38, IFEval +0.74, MMLU +0.17 vs
AWQ). Both AA arms are **in flight**: GPQA on `gpu04`, AA-LCR on the two-node
serve. Those are the numbers a three-way verdict should rest on — not the full7
rerun, which excluded GPQA.

### The cheap suite could not see the change

Three arms, 6 tasks, 7 h 59 m total, all clean. GPQA excluded (its node was busy).

| Task | ours | gptq | phala | greedy ours/gptq/phala |
|---|---:|---:|---:|---|
| GSM8K | 97.35 ±0.44 | **97.57** ±0.42 | 96.82 ±0.48 | 97.65 / 97.27 / 97.19 |
| IFEval | **90.76** ±1.25 | 88.91 ±1.35 | 90.20 ±1.28 | 89.65 / 90.39 / 90.76 |
| MMLU | 86.63 ±0.28 | **86.84** ±0.27 | 86.81 ±0.27 | 86.67 / 86.84 / 86.66 |
| ARC Challenge | 69.37 ±1.35 | 68.86 ±1.35 | **70.31** ±1.34 | 68.77 / 68.94 / 69.80 |
| HellaSwag | 89.20 ±0.31 | 88.97 ±0.31 | **89.35** ±0.31 | 89.37 / 89.00 / 89.29 |
| TruthfulQA MC2 | **62.88** ±1.46 | 61.77 ±1.45 | 62.54 ±1.46 | 62.99 / 61.92 / 62.50 |

Every score landed inside its own error bar — no arm moved, in either direction.

That is a limit of the suite, not a result about the models: most of its tasks
are multiple-choice, where sampling cannot change the answer, and the rest are
far too short to run into the output cap. So full7 stays useful as a cheap
regression check, but **AA GPQA and AA-LCR are the instruments for anything
sampling-related** — worth one node saved per question asked.

The rerun also made the sampling config an explicit, recorded setting in that
suite rather than an implicit default, so every future run states the decoding it
actually used.

### Blocking: GPU resources

- **Single-node pool is full** — 6 of 72 free, zero fully-free nodes. This binds
  full7 and the AA GPQA arms, which are one node per arm. `gpu07` was taken by
  another namespace minutes after our last arm released it.
- **AA-LCR is not capacity-blocked** — it runs on Zhou Yu's two nodes, the only
  setup with enough memory to hold a full-length answer for this benchmark. Its
  constraint is serial, not scarce: one checkpoint at a time, ~2 h 34 m per arm
  plus reload.
- Quantization still needs its own predictable 8-GPU allocation.

### Plan for next week

- **Land GPTQ's two AA arms** (both in flight) and **run AA-LCR for PhalaCloud** —
  the last missing cell. That completes the three-way on both AA benchmarks.
- **Write and publish the technical blog.** The AA-comparable three-way table is
  the material, and the sampling-config finding is the story worth telling.
- **HLE text-only** (2,158 questions) — carried over, not started; still needs
  the AA equality-checker judge, the one dependency outside our cluster.
- **Re-add GPQA to full7** for a three-way on the cheap instrument. Needs the
  gated GPQA dataset licence accepted on our Hugging Face account.
- Still open on quantization: direct native export, expert activation-scale
  alignment, and the fact that no BF16 GLM-5.3 fits an 8×H100 node — so a defect
  shared by all three arms stays invisible on every benchmark above.
