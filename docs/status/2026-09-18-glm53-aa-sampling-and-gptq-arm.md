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

**Checkpoint.** Full EP8 job 15 h 04 m, exit 0, 371 GiB. Qualification passed:
EP4 and EP8 give **identical greedy outputs, 100% top-1**, with EP8 peak memory
**51.8% of EP4** — the memory claim that was unproven last week. Skips
`to_sglang`, indexer repatch and MTP graft entirely (AWQ needed ~3.5 h of that).

**Quality.** On the cheap greedy full7 it won the only task that separated the
arms by more than 2 pp:

| | GPTQ | AWQ | Phala |
|---|---:|---:|---:|
| **GPQA Diamond CoT** (flexible) | **67.68%** | 61.11% | 55.56% |

Elsewhere on full7 it is inside noise (GSM8K −0.38, IFEval +0.74, MMLU +0.17 vs
AWQ). Both AA arms are **in flight**: GPQA on `gpu04`, AA-LCR on the two-node
serve. Those are the numbers a three-way verdict should rest on — not the full7
rerun, which excluded GPQA.

### The full7 rerun moved nothing, which is the useful finding

Three arms, 6 tasks, 7 h 59 m total, all gates passed. Every per-arm delta vs its
greedy run sits inside stderr. **Structural:** only GSM8K and IFEval are
generative — the other four are teacher-forced loglikelihood and lm-eval ignores
`gen_kwargs` there, so 4 of 6 tasks cannot respond to sampling at all, and full7
generates too short to have a runaway tail. **Do not re-run full7 for sampling
reasons.**

It did close a provenance gap: the suite had no decoding channel at all, so
published full7 numbers were greedy by *inheritance* from gsm8k's yaml while the
profile declared a temperature that reached only other runners.
`GENERAL_TEMPERATURE` / `GENERAL_TOP_P` now exist and are recorded in the payload.

### Blocking: GPU resources

- **Single-node pool is full** — 6 of 72 free, zero fully-free nodes. This binds
  full7 and the AA GPQA arms, which are one node per arm. `gpu07` was taken by
  another namespace minutes after our last arm released it.
- **AA-LCR is not capacity-blocked** — it runs on Zhou Yu's two nodes as a
  standing TP=16 serve (pool 598,848), the only place a 131,072-cap trace fits.
  Its constraint is serial: one checkpoint at a time, ~2 h 34 m per arm plus
  reload.
- Quantization still needs its own predictable 8-GPU allocation.

### Plan for next week

- **Land GPTQ's two AA arms** (both in flight) and **run AA-LCR for PhalaCloud** —
  the last missing cell.
- **HLE text-only** (2,158 questions) — carried over, not started; still needs
  the AA equality-checker judge, the one dependency outside our cluster.
- **Re-add GPQA to full7** for a three-way on the cheap instrument. Needs
  `HF_TOKEN` plus the `Idavidrein/gpqa` licence accepted on that account.
- Still open on quantization: direct native export, expert activation-scale
  alignment, and the fact that no BF16 GLM-5.3 fits an 8×H100 node — so a defect
  shared by all three arms stays invisible on every benchmark above.
