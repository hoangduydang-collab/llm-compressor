# GLM-5.3 W4AFP8 — AA GPQA Diamond analysis (2026-09-11)

Post-hoc read of the Sep-11 formal AA run (`results-formal-198-c8`). CPU-only
jobs against the harness sqlite caches. No 256k GPU rerun was launched.

This note is the analysis packet. Formal scores and provenance live in
[`2026-09-11-glm53-aa-gpqa-and-three-objectives.md`](2026-09-11-glm53-aa-gpqa-and-three-objectives.md)
and [`GLM53_AA_QUALITY_RERUN_HANDOFF.md`](../GLM53_AA_QUALITY_RERUN_HANDOFF.md).

## What we believe now

1. **Headline gap to AA 91.7% is truncation, not a 10-point quality hole.**
   Every cap-hit scored 0. Uncapped accuracy is 93.1% (ours) / 94.1% (phala).
   Cap-hits bound ours at 870/990 = 87.9% even if every remaining item is
   perfect — below 91.7% — so truncation is *sufficient* to explain the gap,
   with the usual caveat that harder items truncate more.
2. **Most cap-hits are long rumination, not MiniMax-style collapse.** Tight
   zlib loops are 15/120 ours and 17/155 phala. The rest still look like GPQA
   reasoning at the tail.
3. **Runaway is per-attempt, not per-question.** Correct finished siblings of
   the same stem finish at median ~12k tokens; none reach 100k.
4. **~34–38% of failures stated gold and did not score it** (Lotfi et al.
   overthinking), after dropping hedge-only “maybe / what if” hits and
   *keeping* format-rehearsal `Answer: X` (the letter is the model’s current
   pick, not a dummy C).

## Setup

| | |
|---|---|
| Task | NVIDIA `gpqa_diamond_aa_v3` (`nvidia-simple-evals==26.3`) |
| Items | 198 Diamond × 5 repeats = **990** per arm |
| Decoding | temp 0.6, top_p 1.0, thinking on, `max_new_tokens` **131,072** |
| Serve (formal run) | SGLang 0.5.17, tp=8 / 8×H100, ctx 164,800, no specdec |
| Identity | **Question stem**, not full-prompt hash. AA shuffles A/B/C/D per attempt, so prompt-hash “114 unique questions” is shuffled-prompt identity, not 114 Diamond items. |
| Gold | Official `Idavidrein/gpqa` `gpqa_diamond`: `Question` → `Correct Answer`, mapped onto that attempt’s shuffled letter. **990/990** joined by normalized stem (`gpqa_question`). |

Caches (read-only):

```
/mnt/cephfs/hoangduy/results/glm53-quality-paired/20260902t0431z/
  client-{ours,phala}/aa-gpqa-v3/results-formal-198-c8/
    gpqa_diamond_aa_v3/cache/cache.sqlite/cache.db
```

Each cache row already joins `score`, `completion_tokens`, and the full
prompt. There is no missing join key (Packet A, job `hd-aa-artifact-inventory`).

Code: `pipeline/aa_cap_loop_screen.py`, `pipeline/aa_overthinking.py`,
tests under `pipeline/tests/`. CPU jobs under `pipeline/k8s/hd-aa-*.yaml`.
Cluster JSON/logs: `/mnt/cephfs/hoangduy/runlogs/hd-aa-*`. Local copies of
logs: `docs/evidence/2026-09-11-aa-*.log`.

---

## 1. Formal scores and the truncation join

Evidence: handoff Packet A; `docs/evidence/2026-09-11-aa-artifact-inventory.log`.

| Arm | n | pass@1 | Cap-hits (ct ≥ 131072) | Capped correct | Uncapped acc | Mean completion tokens |
|---|---:|---:|---:|---:|---:|---:|
| ours | 990 | **81.82%** | 120 (12.12%) | **0** | **93.10%** | 26,463 |
| phala | 990 | 79.39% | 155 (15.66%) | **0** | **94.13%** | 30,745 |
| AA published GLM-5.3 (max) | — | ~91.7% | — | — | — | — |

- Ours is +2.42 vs phala (≈1.3× SE of the difference: directional, not a
  proven win).
- If every cap-hit had instead scored 1, ours would be at most
  (810+120)/990 = 93.9%. The *binding* arithmetic for the AA gap is the
  other way: 120 guaranteed zeros cap the score at 87.9% < 91.7%.
- Earlier status note that “per-item scores were absent” is **false**. The
  sqlite caches have them.

Live two-node serve (not used for this formal run; relevant to a future
raised-cap rerun): tp=16, KV pool **598,848**, EAGLE on. Packet B (full 990
at a raised cap) is **not authorized**.

---

## 2. Cap-hit loops vs rumination

Detectors in `pipeline/aa_cap_loop_screen.py`.

**Hard loop label** (either fires):

- zlib of the last 8k chars **≤ 0.08** (M3-style collapses compress to ~0.004;
  GLM-5.3 cap-hit median is ~0.31)
- `health._periodic_suffix` on the tail (period ≤ 16, ≥4 repeats)

**Reported only, not the label:** `sample_output_check.judge` distinct-4gram
< 0.30. That threshold was fit on ~100-char M3 collapses and **over-fires on
131k GPQA CoT** (first screen labeled 120/120 and 155/155 as “loop”).

| | ours | phala |
|---|---:|---:|
| Cap-hits | 120 | 155 |
| zlib-tail median (all cap-hits) | 0.31 | 0.34 |
| Tight zlib ≤ 0.08 | **15** | **17** |
| Of those, verbatim paragraph-loop (same suffix ≥3×) | **10 / 15** | **12 / 17** |
| Tight but advancing tail | 5 | 5 |
| `periodic_suffix` period ≤ 16 | **0 / 120** | **0 / 155** |

Lowest-zlib example (ours `q=c618f94ac003`, zlib 0.026): the same ~217-char
paragraph pasted ~36 times (astronomy transit).

**Question overlap (prompt-hash keys in the overlap job; tight set is 1:1
with unique questions):**

| | ∩ | ours only | phala only |
|---|---:|---:|---:|
| All cap-hits | 39 | 75 | 112 |
| Tight zlib ≤ 0.08 | **0** | 15 | 17 |

Tight-loop items do **not** overlap across arms. The same Diamond stem can
loop on one checkpoint and ruminate on the other.

Evidence: `docs/evidence/2026-09-11-aa-cap-loop-screen.log`,
`-screen-report.log`, `-all-tight-tails.log`, `-question-overlap.log`.

### 2.1 The 160k diagnostic does not test extra budget

`results-diag-allcap-q008-160k`: `max_new_tokens: 160000`, n=986.

| | n |
|---|---:|
| `completion_tokens == 131072` | 115 |
| `completion_tokens == 160000` | **1** |
| Mean score | 0.8215 |

The serve still stopped at 131k for almost every long trace. Uncapped
accuracy in that dump is still ~93.7%. Do not read it as “we gave them 160k
and they still failed.”

---

## 3. Per-stem length: truncation is a runaway, not “this item needs 131k”

Group by **stem** (shuffle-invariant). 198 stems × 5 attempts.

| | ours | phala |
|---|---:|---:|
| Stems with ≥1 cap-hit | 70 / 198 | 98 / 198 |
| Cap-hits on those stems | mostly 1 of 5 (ours 41×1, 16×2, …; phala 56×1, 30×2, …) | |
| Correct finished siblings of cap-hit stems | 205 | 300 |
| … median tokens | **12.2k** | **12.8k** |
| … max | 89.5k | 91.2k |
| … ≥ 100k | **0** | **0** |
| Cap-hit stems with zero finished-correct siblings | 8 / 70 | 6 / 98 |

Finished-only sample std of length (same stem, drop ct ≥ 131k):

| Stem class | ours median std | phala median std | CV |
|---|---:|---:|---:|
| Never-cap | 1.6k | 0.7k | ~0.3 |
| Has a cap-hit (finished sibs only) | 5.2k | 3.6k | ~0.3 |
| Has a cap-hit, counting 131k as a value | ~54k | ~55k | ~1.1 |

Including the 131k attempts inflates std because the stem is a **mix of short
finishes and one runaway**, not because all five attempts are long.

Evidence: `docs/evidence/2026-09-11-aa-capped-stem-sibling-tokens.log`,
`-aa-stem-token-std.log`.

### 3.1 256k rerun (designed, not launched)

Diagnostic only; headline cap stays 131,072.

- Rerun **attempts**, not all 5 repeats of unique problems.
- Skip the **10** ours verbatim paragraph-loops; keep the 5 zlib-tight-but-
  advancing tails. **n = 110** ours at 256k / concurrency 2.
- Do not GPU-launch until explicitly authorized.

---

## 4. Overthinking (Lotfi et al. 2026, arXiv:2606.00206)

Paper §4.2 / Table 12: on an *incorrect* attempt, if gold appears in the CoT
but is not the final answer → **overthinking**. The other three labels
(logical / arithmetic / formatting) need an LLM judge; we left those as
`not_overthinking`.

### 4.1 Gold

| Source | Result |
|---|---|
| Score=1 sibling vote (first pass) | ours 120/180 fails labeled (12 stems never scored); phala 164/204 (8 stems) |
| Official GPQA Diamond `Correct Answer` | **180/180** and **204/204**; join `gpqa_question` on all 990×2 attempts |

Sibling inference is only a fallback. The gold run did not need it.

Job: `hd-aa-overthinking-gpqa-gold`.
Evidence: `docs/evidence/2026-09-11-aa-overthinking.log` (sibling),
`-aa-overthinking-gpqa-gold.log` (official).

### 4.2 What “found gold” means

Not a bare mention of “A”. Same high-precision extractors NVIDIA uses for the
**final** line, scanned left-to-right on the whole trace:

- `Answer: X`
- `\boxed{X}`
- `(the) answer is X` / `answer is (X)`
- `X is the correct answer`

Loose fallbacks (`C)`, trailing letter) are **out** — they fire on option
restatement.

An attempt is raw-regex overthinking iff it failed **and** gold appears in
that list **and** (last commit ≠ gold **or** it stated gold then hit the
131k cap).

### 4.3 Raw regex rates (official gold)

| Arm | Fails | Overthinking | Of cap-hit fails | Of finished fails |
|---|---:|---:|---:|---:|
| ours | 180 | 68 (**37.8%**) | 47/120 | 21/60 |
| phala | 204 | 82 (**40.2%**) | 67/155 | 15/49 |

The sibling-gold pass looked higher (49% / 44% of *labeled* fails) because it
dropped never-scored stems. Those hard items rarely state gold, so adding
them **lowered** the rate.

### 4.4 Consideration vs decision vs format rehearsal

The unanchored `(?:the )?answer is ([A-D])` matches inside hedges:

| Model wrote | Regex sees |
|---|---|
| maybe the correct **answer is A** | `answer is A` |
| what if **the answer is D** | `the answer is D` |

**Format rehearsal** is different: the prompt requires last line
`Answer: A/B/C/D`. Mid-CoT the model writes e.g. “Need craft final … last
line **Answer: B**.” That is template talk, but the **letter is not a dummy**.

On the **first** format-rehearsal `Answer: X` in a trace (all attempts, not
just failures):

| | ours (856/990) | phala (846/990) |
|---|---:|---:|
| A / B / C / D | 267 / 209 / 193 / 187 | 274 / 203 / 185 / 184 |
| X == gold | **91.6%** | **92.3%** (chance ~25%) |
| X == last committed letter | **96.4%** | **96.2%** |
| X == A (prompt’s `e.g. Answer: A`) | 31% | 32% |

So they fill in the letter they currently think is right. Uniform A/B/C/D
*among overthinking traces* is expected from AA’s shuffle and does not by
itself prove that; the 92% gold match on all format rehearsals does.

First gold-hit bucket **among the 68 / 82 raw overthinking labels**:

| First gold-hit | ours (68) | phala (82) |
|---|---:|---:|
| Format rehearsal | 47 | 66 |
| Hedged consideration | 13 | 8 |
| Unhedged `answer is X` | 4 | 5 |
| `Answer: X` / `\boxed{X}` (no format/hedge window) | 4 | 3 |

Jobs: `hd-aa-overthinking-windows`, `-hedge`, `-format-letter`.

### 4.5 Agreed definition and final rates

**Keep:** format-rehearsal `Answer: X` and unhedged commits (`Answer: X`,
`the answer is X` without maybe/what-if).

**Drop:** hedge-only (every gold hit is `maybe` / `what if` / `if the expected` …
and never a format or unhedged commit).

| | ours | phala |
|---|---:|---:|
| Fails | 180 | 204 |
| Raw regex | 68 / 37.8% | 82 / 40.2% |
| **Keep format + unhedged** | **62 / 34.4%** | **77 / 37.8%** |
| Hedge-only dropped | 6 | 5 |
| Among cap-hit fails | 44/120 = **36.7%** | 63/155 = **40.7%** |
| Among finished fails | 18/60 = **30.0%** | 14/49 = **28.6%** |

The 19/180 figure was the other extreme (treat format talk as fake). That is
**not** the agreed rule. Format rehearsal is counted.

Keep-set overlap (an attempt can have more than one keep type): ours 54
format / 15 decision-form / 7 unhedged `answer is`; phala 71 / 20 / 6.

Job: `hd-aa-overthinking-keep-ratio`.
Log: `/mnt/cephfs/hoangduy/runlogs/hd-aa-overthinking-keep-ratio.log`.

Residual failures (not overthinking under this rule) still need a GPT-5-style
judge for logical vs arithmetic vs formatting. We did not run that.

---

## 5. Implications (not yet executed)

- A raised-cap **attempt** rerun (110 ours @ 256k, skip 10 paragraph-loops)
  tests whether rumination finishes given budget. It does not change the
  AA-comparable headline, which must stay at 131,072.
- Tight paragraph-loops will not be saved by more tokens; skip them.
- Overthinking is a large slice of *failures* (~1/3), including many
  cap-hits that stated gold then ran on. A decoding-time “wait/but”
  penalty (Lotfi) is a separate intervention from raising the cap.
- Do not mix this AA number with last week’s in-house 61%/56% pair
  (greedy, 32k, homemade extract).

## 6. Job / artifact index

| Job | What |
|---|---|
| `hd-aa-artifact-inventory` | Per-item score × `completion_tokens` join |
| `hd-aa-cap-loop-screen` (+ report, diag, tail-sample, all-tight-tails, question-overlap) | Loop vs rumination |
| `hd-aa-capped-stem-sibling-tokens`, `hd-aa-stem-token-std` | Per-stem length |
| `hd-aa-overthinking` | Sibling-gold overthinking (superseded) |
| `hd-aa-overthinking-gpqa-gold` | Official GPQA gold + raw regex |
| `hd-aa-overthinking-windows` | Snippets around first gold hit |
| `hd-aa-overthinking-hedge` | Hedge / format / decision split |
| `hd-aa-overthinking-format-letter` | Format-rehearsal letter vs gold |
| `hd-aa-overthinking-keep-ratio` | Agreed keep/drop rates |
