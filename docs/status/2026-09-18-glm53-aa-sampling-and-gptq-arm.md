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
- **On AA GPQA Diamond all three arms tie, but our EP-GPTQ gets there on the
  fewest tokens** — 0.41 pp spread, 14.6% fewer tokens than PhalaCloud, 3.7 h
  faster. Consistent with the token-efficiency edge seen on the other
  benchmarks. Full numbers in key result 3.
- **AA-LCR repeats it**: EP-GPTQ 78.33% vs AWQ 79.00%, tied on a paired test
  (p ≈ 0.87), on **18.2% fewer tokens** — the largest token gap of the three
  benchmarks. Only PhalaCloud's AA-LCR cell is left.
- Three benchmarks ran: AA GPQA Diamond, AA-LCR v1.1, and the cheap full7 suite.

**The three arms, since two of them are ours.** `ours (AWQ)` and
`ours (EP-GPTQ)` are both our own GLM-5.3 W4AFP8 checkpoints, produced by two
different quantization paths. `PhalaCloud` is the third-party W4AFP8 release we
benchmark against. So "ours beat PhalaCloud" is a two-against-one comparison,
and the AWQ-vs-EP-GPTQ contrast is an internal one between our own paths.

**How the three key results divide up**, so no number needs hunting: **1** is
the sampling axis — one checkpoint, two configs, both AA benchmarks. **2** is
our new EP-GPTQ checkpoint and the cheap full7 suite. **3** is the comparison
between checkpoints, score and token cost together — three-way on AA GPQA
Diamond, two-way on AA-LCR (PhalaCloud's cell still to run) — plus the
cross-benchmark summary of EP-GPTQ's token edge.

Detail: [AA-LCR](2026-09-16-glm53-aa-lcr-v11-zai-sampling.md) ·
[EP GPTQ + first full7](2026-09-14-glm53-ep-gptq-w4afp8-and-full7.md) ·
[full7 rerun](2026-09-17-glm53-full7-zai-sampling.md).

### Key result 1, the confirmed config is worth ~10 points on both AA benchmarks

This section is the **sampling axis**: the same checkpoint scored twice, at the
generic config and at Z.ai's. Only the two arms that have been run at both
configs appear here. The three-way comparison between checkpoints is key result
3.

| Benchmark, arm | Generic 0.6/1.0 | **Z.ai 1.0/0.95** | AA published | Gap to AA |
|---|---:|---:|---:|---:|
| **GPQA Diamond**, ours (AWQ) | 81.82% | **91.52%** ±0.87 | ~91.7% | 9.9 pp → **0.2 pp** |
| **GPQA Diamond**, PhalaCloud | 79.39% | **91.11%** ±0.85 | ~91.7% | 12.3 pp → **0.6 pp** |
| **AA-LCR v1.1**, ours (AWQ) | 71.33% | **79.00%** | 80% | 8.7 pp → **1.0 pp** |

GPQA is the formal AA protocol, 198 items × 5 repeats = 990 completions per arm,
all HTTP 200. AA-LCR is 100 × 3. Public-methodology reproductions, not official
AA runs — but the sampling axis is now confirmed rather than inferred.

EP-GPTQ is absent here because it only ran at Z.ai's config; its scores
(91.21% GPQA, 78.33% AA-LCR) are in key result 3.

**Why it moves so much: truncation, and it is a sampling effect not a budget
one.** Low temperature with an untruncated tail lets a minority of traces run
away into the output cap and return empty.

| GPQA Diamond, same checkpoint twice | ours AWQ 0.6/1.0 | **ours AWQ 1.0/0.95** | PhalaCloud 0.6/1.0 | **PhalaCloud 1.0/0.95** |
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

### Key result 2, our EP-GPTQ arm

**Checkpoint.** Quantization job 15 h 04 m, clean, 371 GiB. Qualification passed:
splitting the work 8 ways gives **answers identical to the 4-way run** at
**half the peak memory** (51.8%) — the claim that was unproven last week. It also
produces the servable format directly, skipping the ~3.5 h of post-processing our
AWQ path needs.

**Quality.** On our own cheap suite our EP-GPTQ won the only task that has ever
separated the arms by more than 2 pp — GPQA Diamond CoT, **67.68%** against our
AWQ's 61.11% and PhalaCloud's 55.56%. Everything else on that suite is inside
noise, for all three arms:

| Task | ours (AWQ) | ours (EP-GPTQ) | PhalaCloud |
|---|---:|---:|---:|
| GSM8K | 97.35 ±0.44 | **97.57** ±0.42 | 96.82 ±0.48 |
| IFEval | **90.76** ±1.25 | 88.91 ±1.35 | 90.20 ±1.28 |
| MMLU | 86.63 ±0.28 | **86.84** ±0.27 | 86.81 ±0.27 |
| ARC Challenge | 69.37 ±1.35 | 68.86 ±1.35 | **70.31** ±1.34 |
| HellaSwag | 89.20 ±0.31 | 88.97 ±0.31 | **89.35** ±0.31 |
| TruthfulQA MC2 | **62.88** ±1.46 | 61.77 ±1.45 | 62.54 ±1.46 |

Three arms, 7 h 59 m, GPQA excluded (its node was busy). Every score sits inside
its own error bar, so this suite is a regression check rather than a way to rank
the arms — and key result 3 shows that its one apparent separation does not
survive a stronger instrument.

### Key result 3, comparing checkpoints: scores tie, token cost does not

All three arms have now run the **formal AA GPQA Diamond protocol** — 198 items
× 5 repeats = 990 completions each, 131,072-token cap, Z.ai's 1.0 / 0.95, one
8×H100 node, all HTTP 200. This is the single place the three-way comparison
lives:

| Arm | Score | Mean completion | p50 | p90 | p99 | Cap-hits | Total generated | Infer time |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| **ours (EP-GPTQ)** | 91.21% ±0.88 | **13,944** | 6,685 | 38,645 | 82,580 | 5 (0.5%) | **13.80 M** | **10.66 h** |
| **ours (AWQ)** | **91.52%** ±0.87 | 14,757 | 7,987 | 38,888 | 86,655 | 4 (0.4%) | 14.61 M | 11.25 h |
| PhalaCloud | 91.11% ±0.85 | 16,332 | 7,940 | 42,687 | **131,072** | 14 (1.4%) | 16.20 M | 14.32 h |

**Scores tie.** Total spread is **0.41 pp against ±0.9 error bars** — our two
paths are indistinguishable from each other, and both are indistinguishable
from PhalaCloud. The full7 GPQA-CoT gap in key result 2 (67.68 / 61.11 / 55.56)
therefore must **not** be read as a quality ranking: different cap, CoT harness,
and 1 repeat instead of 5. Where the instrument is strong enough to publish
from, the arms are level.

**Token cost does not tie, and this is the axis that actually separates them.**
Our EP-GPTQ spends **14.6% fewer tokens than PhalaCloud and 5.5% fewer than our
own AWQ** at an indistinguishable score, finishing 3.7 h sooner than PhalaCloud
on the same hardware. On a served endpoint that is a direct cost saving, and it
makes EP-GPTQ the arm to prefer between our two paths. This matches the
token-efficiency edge already seen on the other benchmarks, so treat it as a
repeated property of the EP-GPTQ path rather than a one-off.

PhalaCloud is also the only arm whose **p99 sits on the cap** — both of ours top
out around 83–87k — and it runs away 3× more often. Its tail, not its average,
is what makes it the slowest arm.

**AA-LCR v1.1 repeats both findings**, two-way for now (PhalaCloud is the
remaining cell). The collaborator swapped only the checkpoint on this endpoint,
so serve, sampling, cap and judge were identical:

| Arm | Score | Mean completion | p50 | p90 | p99 | Cap-hits | Total generated | Wall clock |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| **ours (EP-GPTQ)** | 78.33% ±2.4 | **5,202** | 2,091 | **12,855** | **45,726** | **1 (0.3%)** | **1.56 M** | **2 h 09 m** |
| **ours (AWQ)** | **79.00%** ±2.4 | 6,362 | 2,214 | 15,649 | 49,895 | 2 (0.7%) | 1.91 M | 2 h 34 m |

Here the tie can be *shown*, not inferred from error bars: both arms cover the
same 300 units, so a paired McNemar applies — 218 both correct, 17 EP-GPTQ only,
19 AWQ only, 46 neither, **p ≈ 0.87**. Note 36 of 300 units (12%) flip between
arms while the net difference is 2 units, so **compare future arms paired, not
by headline delta** — an unpaired run cannot resolve a difference this size.

And the token edge is **−18.2%**, the largest of the three benchmarks, against
−5.5% on GPQA (13,944 mean) and −3.2% on full7 (~47 mean). AA-LCR shows the
biggest gap on the *shorter* mean, so this is a repeated property whose size is
benchmark-dependent, **not** a "scales with output length" law; the saving also
moves position — GPQA's sits in the median (p50 −16%, tail flat), AA-LCR's in
the tail (p90 −18%, p50 −6%).

Two structural notes from the same data: **98–99% of every completion is
reasoning tokens** (the answer itself is 200–300), and the length distribution
is heavily right-skewed (p50 ~7k against a ~15k mean), so a minority of long
traces drives most of the spend on all three arms.

**Provenance.** Token figures are cross-checked two ways that agree to within
0.0%: SGLang's prometheus counters diffed across the eval, and the harness's own
`response_stats_cache`. Cap-hits come from `finish_reason: "length"` — the
server's statement about why it stopped — not from a token count equalling the
cap. Sampling was read back from each arm's executed `run_config.yml`
(1.0 / 0.95, `max_new_tokens` 131072, `n_samples` 5, `limit_samples: null`),
identical across all three. PhalaCloud's aggregate covers 992 attempts against
990 scored items — two retries — which moves its mean by under 0.5%. On AA-LCR,
both our arms serve under the same `--served-model-name glm-5.3-w4afp8`, so they
are told apart by run id and by `run-manifest.json`'s
`endpoint_deployment_identity.model_path`, and the GPTQ Job asserts that path
contains `gptq-W4AFP8` before generating.

### Plan for next week

- **GPQA Diamond is done three-way, and AA-LCR is now two-way** (key results
  1–3 above). Remaining on AA: **run AA-LCR for PhalaCloud** — the last missing
  cell. That completes the three-way on both AA benchmarks. Note the serve
  currently hosts our EP-GPTQ checkpoint, so the PhalaCloud cell needs another
  checkpoint swap or a second endpoint; a 131,072-cap AA-LCR run needs a KV pool
  ≥ ~247k tokens, which the two-node TP=16 serve has (598,848) and a single-node
  TP=8 serve (~164,800) does not.
- **Write and publish the technical blog.** The AA-comparable three-way table is
  the material, and the sampling-config finding is the story worth telling.
- **HLE text-only** (2,158 questions) — carried over, not started; still needs
  the AA equality-checker judge, the one dependency outside our cluster.
- **Re-add GPQA to full7** for a three-way on the cheap instrument. Needs the
  gated GPQA dataset licence accepted on our Hugging Face account.
- Still open on quantization: direct native export, expert activation-scale
  alignment, and the fact that no BF16 GLM-5.3 fits an 8×H100 node — so a defect
  shared by all three arms stays invisible on every benchmark above.
