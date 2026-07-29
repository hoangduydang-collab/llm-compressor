# M3 EAGLE3 speculative decoding — waves 1 and 2

Wave 1: window `m3-specdec-eagle3/20260727T061506Z`, 4 arms.
Wave 2: window `m3-specdec-eagle3/20260727T064934Z-wave2`, 6 arms.
Phase D: window `m3-specdec-eagle3/20260727T073533Z-phaseD`, 2 arms × 10 cells.
Phase E: window `m3-specdec-eagle3/20260727T084526Z-format`, 2 arms × 4 cells.
Phase F: window `m3-specdec-eagle3/20260727T092342Z-bf16ref`, 2 arms × 4 cells.
Phase G: window `m3-specdec-eagle3/20260727T102751Z-int4drafter`, 3 A/B arms + 1 probe.
Phase H: window `m3-specdec-eagle3/20260727T105919Z-kopt`, 2 arms × 5 serves.
Phase I: window `m3-specdec-eagle3/20260727T134635Z-kernel`, 1 arm × 5 serves (3 served,
2 killed by an upstream vLLM defect — see phase I).
Phase I.2: window `m3-specdec-eagle3/20260727T154725Z-kernel`, 1 arm × 3 serves — the
Humming lm_head fix + re-run of the two killed cells.
All arms rc=0 except phase I (rc=1, fail-closed on the two Humming cells); every gate
passed. Design + decision rule:
`M3_SPECDEC_EAGLE3_PLAN.md`.

**Answer to the question that prompted this ("can spec-dec give 2.5–3.5× at conc 1?"):
no, not on production traffic — and the honest answer is a range, not a point.
Measured at conc 1 with the recipe's k=3, temp 0.6, natural output: **1.81× on
real ShareGPT prompts**, 1.36× on creative writing, 2.15× on code. 2.5× is only
reached by workloads forced past their natural stopping point (2.31× on the
pinned-8k reasoning shape) or by greedy decoding (2.05×). What moves the number is
**what the model is writing**, not how long the prompt is and not the load.**

## What was held fixed

Target, kernel, topology and every serve flag are identical to the
`20260726T132617Z` window's `gptq-hum-idx-0110` arm: in-house GPTQ W4AFP8 ABI
overlay, Humming indexed 0.1.10, TP8/EP8 on one 8×H100 node,
`MAX_MODEL_LEN=131072`, `kv_cache_dtype=fp8`, prefix caching on,
`LLMC_M3_CAPTURE_SYNC=sync`. The only per-arm difference is
`--speculative-config`. The control is re-measured **in this window**, and it
reproduces the earlier one to 0.2% (137.5 / 136.9 here vs 137.3 / 137.1 on 07-26),
so every ratio below is a same-window ratio.

Drafter: `Inferact/MiniMax-M3-EAGLE3` @ `44cafa5ace418d8b22e2958df0c6aa1f2476842c`
(6.53 GB, `LlamaForCausalLMEagle3`, 1 layer, hidden 6144, full 200,064 draft
vocab), loaded by our own vLLM 0.24.0 with no code change. MTP was not an option:
`config.json` declares `num_mtp_modules: 7` but neither the BF16 nor the vendor
MXFP8 release ships a single MTP tensor.

## Output speed — tok/s (AA p50), × = vs the in-window control

| cell | control | k=1 | k=3 | k=5 |
|---|---|---|---|---|
| 1k × conc 1 | 137.5 | 194.7 (1.42×) | **236.0 (1.72×)** | 228.8 (1.66×) |
| 10k × conc 1 | 136.9 | 200.9 (1.47×) | **239.0 (1.75×)** | 231.5 (1.69×) |
| 1k × conc 10 | 94.2 | 121.7 (1.29×) | 137.2 (1.46×) | 139.1 (1.48×) |
| 10k × conc 10 | 67.3 | 83.3 (1.24×) | **108.3 (1.61×)** | 92.2 (1.37×) |

## Aggregate output tok/s — spec-dec is not a latency-for-throughput trade here

| cell | control | k=1 | k=3 | k=5 |
|---|---|---|---|---|
| 1k × conc 1 | 133.7 | 185.9 | 215.0 | 216.4 |
| 10k × conc 1 | 118.5 | 159.5 | 186.9 | 176.9 |
| 1k × conc 10 | 570.0 | 679.8 | **1000.1 (1.75×)** | 994.7 |
| 10k × conc 10 | 391.8 | 453.5 | **641.3 (1.64×)** | 513.4 |

Natural OSL is comparable across arms (410–625 tokens; 582 vs 608 in the
1k×10 cell), so these aggregates are not the OSL-confounded kind that the
two-axis report warns about. **Concurrency 10 gains throughput as well as
latency** — at that load 8×H100 is nowhere near compute-saturated for this MoE
(the same node does 3,267 tok/s at conc 64), so the extra verified tokens ride in
spare capacity.

TTFT is unaffected at conc 1 (118 → 131 ms at k=3) and roughly flat at conc 10
(2,094 → 2,173 ms at 10k; the 1k×10 cell reads 461 → 293 ms, which is scheduling
noise, not a mechanism).

## Acceptance — and why the workload changes the answer

| arm | mean accepted length | avg draft acceptance | per-position |
|---|---|---|---|
| k=1 | 1.69 | 69.1% | 0.69 |
| k=3 | 2.45 | 48.2% | 0.70, 0.46, 0.29 |
| k=5 | 2.55 | 31.0% | 0.64, 0.40, 0.25, 0.16, 0.10 |

Divide accepted length by measured speedup to get what a step costs:

| arm | accepted length ÷ speedup | step cost vs plain decode |
|---|---|---|
| k=1 | 1.69 / 1.42 | 1.19× |
| k=3 | 2.45 / 1.72 | 1.42× |
| k=5 | 2.55 / 1.66 | 1.54× |

The per-draft-token cost is **front-loaded, not flat** — an earlier version of this
line called it "a consistent ~0.11–0.12×", which is true only of the middle
positions. Marginal cost from the table above:

| draft token | marginal step cost | running average per token |
|---|---|---|
| 1st | **+19%** | 19% (k=1) |
| 2nd–3rd | +11.5% each | **14% (k=3)** |
| 4th–5th | +6% each | 10.8% (k=5) |

The first draft token costs nearly double a middle one — a fixed
drafter-invocation overhead paid once per step regardless of depth. **Use ~14% per
draft token for planning at k=3**, not a flat 11–12%. Today's measurements agree:
step inflation at k=3 is +43.0% on W4AFP8 (14.3%/token) and +38.8% on MXFP8
(12.9%/token) at 8k-low conc 1.

This cost structure is why **k=3 is the optimum and k=5 is a net loss**: k=5 buys
+0.10 accepted length (2.45 → 2.55) for +0.12× step cost, and its per-position
rates have collapsed to 0.10 by the fifth token. Nothing deeper than k=3 is worth
serving.

Not to be confused with `M3_SPECDEC_EAGLE3_PLAN.md`'s "a 4-token step costs only
1.21× a 1-token step" — that is the *verify forward alone* (≈7% per extra position,
excluding the drafter's own passes), and it was a pre-measurement prior.

**The workload matters more than k.** On the greedy probe's 8 hand-picked prompts
(English questions, temp 0, forced continuation) the same k=3 arm accepted
**0.862 / 0.740 / 0.600 → mean length 3.20–3.35**, against 2.45 on AA's synthetic
random-token prompts at temp 0.6.

> **SUPERSEDED.** This section originally divided that 3.2 by the measured 1.42×
> step cost to infer "**≈2.25×** on real prompts." Wave 2 measured the real-prompt
> number directly and got **1.81×**, so the inference was wrong and is struck. Its
> error was attributing the probe's high acceptance to prompt *naturalness*. Wave 2
> decomposed the gap: naturalness is worth −1% (2.45 synthetic → 2.473 natural),
> temp 0 is worth +4% (→ 2.575), and the real driver was the probe's **forced
> continuation** (+33%, → 3.286). Do not cite 2.25×.

## Gates

- Humming attestation `valid: true` on all four arms; no fallback to CUTLASS.
- Spec-dec provably active on every k>0 arm (`speculative_config` in the engine
  banner, `SpecDecodingLogging` reporting drafted > 0).
- Greedy equivalence vs control at temp 0: k=1 **7/8**, k=3 **7/8**, k=5 **6/8**
  prompts identical for ≥120 characters (one k=1 output identical for its full
  651 characters). Divergence appears after a few hundred characters, which is the
  floating-point signature of different batch shapes; a broken multi-query verify
  path would diverge within the first tokens on every prompt. M3's sparse
  attention is wired for spec-dec rather than bypassed — the drafter shares the
  target's `topk_indices_buffer`.
- AA runner rc=0, all 16 cells `status=ok`, no Xid, no worker restart.

## Verdict against the packet's decision rule

Rule was: adopt for a latency tier if conc-1 ≥ +40%, greedy passes, and conc-10
regression ≤ 10%. k=3 delivers **+72–75%** at conc 1, passes the greedy gate, and
*improves* conc-10 aggregate by 64–75%. **→ ADOPT k=3 for the latency tier.**

Scope limit, stated because the data does not cover it: this sweep went no higher
than **concurrency 10**. Spec-dec normally turns into a throughput loss once the
batch saturates compute, and our own high-load numbers live at conc 32–64. Do not
enable it globally on that basis — the conc 32/64 measurement is wave 2.
*(Wave 2 measured conc 16/32/64 and found no crossover; this limit is now lifted —
see "Verdict after wave 2".)*

## Wave 2 — measured (window `20260727T064934Z-wave2`, 6 arms)

Same target, kernel, topology and serve flags as wave 1. Each phase carries its own
in-window control, so every × below is a same-window ratio. Speed = per-request
output tok/s; total = server output tok/s.

### Phase A — real prompts (aiperf `--public-dataset sharegpt`), natural output

This is the production number. Prompts are real user turns (ISL ≈ 227 tokens), the
model stops when it wants to, `ignore_eos` absent.

| temp | conc | control speed | k3 speed | × | control total | k3 total | × | accepted len | per-position |
|---|---|---|---|---|---|---|---|---|---|
| **0.6** | **1** | 137.9 | 249.8 | **1.81×** | 132.7 | 222.3 | 1.68× | 2.473 | 0.690 / 0.459 / 0.324 |
| 0.6 | 10 | 77.7 | 128.8 | 1.66× | 701.5 | 1002.0 | 1.43× | 2.503 | 0.700 / 0.479 / 0.324 |
| 0 | 1 | 140.3 | 287.0 | 2.05× | 136.3 | 256.1 | 1.88× | 2.575 | 0.714 / 0.506 / 0.355 |
| 0 | 10 | 82.9 | 143.2 | 1.73× | 738.7 | 1173.6 | 1.59× | 2.570 | 0.720 / 0.503 / 0.348 |

Temperature is a **minor** effect on acceptance (+4%, 2.473 → 2.575) but a larger
one on speed (+13%, 1.81× → 2.05×). The extra speed is not better drafting — it is
a cheaper sampler. Argmax costs less than sampling, and k=3 pays that cost on up to
4 positions per step against the control's 1, so removing it helps k=3 ~4× as much.
Treat 2.05× as an upper bound that production sampling does not get.

### Phase B — load sweep (1k in / 8k pinned out, temp 0.6)

The question was where spec-dec stops paying for itself.

| conc | control speed | k3 speed | × | control total | k3 total | × | accepted len | per-position |
|---|---|---|---|---|---|---|---|---|
| 16 | 81.8 | 166.6 | 2.04× | 1300.9 | 2530.7 | **1.95×** | 3.427 | 0.889 / 0.801 / 0.736 |
| 32 | 65.5 | 135.5 | 2.07× | 2079.3 | 4097.6 | **1.97×** | 3.477 | 0.899 / 0.820 / 0.759 |
| 64 | 51.5 | 95.4 | 1.85× | 3263.0 | 5834.6 | **1.79×** | 3.470 | 0.897 / 0.818 / 0.759 |

**There is no crossover anywhere in reach on one node.** k=3 nearly doubles server
throughput at every load tested, and the wave-1 scope limit ("spec-dec normally
turns into a throughput loss once the batch saturates compute") does not bind here.
Two measurements explain why:

- **Acceptance is flat in concurrency** — 3.427 → 3.477 → 3.470 from conc 16 to 64,
  and 2.473 → 2.503 on phase A. Larger batches do not degrade draft quality, so the
  only possible crossover mechanism is compute saturation.
- **The node is not saturated, because steps are prefill-bound.** At conc 16 the
  control's TTFT is 482.55 ms against k=3's 220.58 ms — k=3 *improves* TTFT 2.2×.
  That is the tell: the control is spending each step admitting prefill at a small
  effective decode batch, leaving idle compute that verified draft tokens consume
  for free. Spec-dec is not trading latency for throughput here; it is filling a
  hole in the schedule.

Only the conc-64 cell shows the beginnings of a squeeze (1.97× → 1.79×), and it is
still a large win. A genuine crossover would need load past conc 64 or a smaller
node.

### Phase C — like-for-like against the two-axis report

Same pinned-output reasoning shape the two-axis perf tables use, so these ratios
can be applied to those numbers directly.

| conc | control speed | k3 speed | × | control total | k3 total | × | accepted len | per-position |
|---|---|---|---|---|---|---|---|---|
| 1 | 137.2 | 317.3 | **2.31×** | 136.7 | 311.1 | 2.28× | 3.286 | 0.855 / 0.749 / 0.682 |
| 4 | 113.3 | 257.1 | 2.27× | 451.4 | 980.1 | 2.17× | 3.441 | 0.889 / 0.807 / 0.745 |

### What actually drives acceptance

Ranked, all measured at conc 1 against the phase-A natural temp-0.6 baseline of
2.473:

| factor | change | accepted len | effect |
|---|---|---|---|
| **output shape** | natural → pinned to 8k (`ignore_eos`) | 2.473 → 3.286 | **+33%** |
| temperature | 0.6 → 0 | 2.473 → 2.575 | +4% |
| prompt naturalness | synthetic random → real ShareGPT | 2.45 → 2.473 | +1% (noise) |
| prompt length | 227 → 1k → 10k tokens | flat | 0% |
| concurrency | 1 → 64 | flat | 0% |

**Output shape dominates, and this has reach beyond spec-dec.** `ignore_eos` with a
large `min_tokens` forces the model to keep writing past where it would have
stopped, and continuation of its own text is far easier to draft than fresh
reasoning. That is why the pinned shape reports 2.31× where real traffic gets
1.81×.

The consequence is a **measurement bias in the entire two-axis perf report**, whose
reasoning and agentic suites both pin output this way
(`run_perf_agentic.sh:119`, `ignore_eos:true` + `min_tokens`). Any feature that
predicts tokens — speculative decoding, MTP, prompt-lookup, n-gram drafting — is
systematically flattered by that shape. Absolute throughput numbers from those
suites remain valid; **ratios between a token-predicting arm and a control do not
transfer to production traffic.** Phase A is the shape to quote for that class of
claim.

## Verdict after wave 2

The wave-1 scope limit is lifted. Against the decision rule ("enable k=3 by default
only up to the highest concurrency where aggregate output tok/s is ≥ control"):
**k=3 satisfies it at every concurrency measured, 1 through 64**, by margins of
43–197% on server throughput. → **Enable k=3 by default, not just for the latency
tier.**

Caveats worth carrying:
- The gain on production traffic is **1.81×**, not the 2.3× the pinned suites show.
- Expect **1.16×–2.17× depending on workload mix and load** (phase D, below): code
  at low concurrency is the ceiling, creative writing under load is the floor.
- k=5 remains a net loss (wave 1); nothing deeper than k=3 is worth serving. Phase
  D's per-position rates suggest k=3 is also too deep for high-entropy traffic —
  untested, and the one open question against this verdict.
- Untested past conc 64 on a single node.

## Phase D — length × entropy on nvidia/SPEED-Bench (complete)

Window `20260727T073533Z-phaseD`, 2 arms, 10 cells each, all rc=0. Isolates
*content domain* from *prompt length* using NVIDIA's purpose-built spec-dec
benchmark: fixed-ISL buckets (1k/8k/32k) crossed with entropy tier, temp 0.6, no
`ignore_eos`, `max_tokens` 2048.

**Conc 1** — the full 3×2 grid:

| cell (conc 1) | ISL | control | k3 | × | accepted len | per-position | ITL control → k3 | TTFT control → k3 |
|---|---|---|---|---|---|---|---|---|
| 1k **low** entropy (code, sorting) | 1011 | 137.31 | 294.77 | **2.15×** | 3.100 | 0.847/0.691/0.562 | 7.283 → 3.406 | 128.9 → 127.5 |
| 8k **low** entropy | 8080 | 136.74 | 296.90 | **2.17×** | 3.106 | 0.844/0.696/0.565 | 7.313 → 3.378 | 420.8 → 420.1 |
| 32k **low** entropy | 32389 | 133.91 | 289.23 | **2.16×** | 3.064 | 0.838/0.681/0.544 | 7.468 → 3.475 | 1217.9 → 1245.2 |
| 1k **high** entropy (creative writing) | 990 | 137.38 | 186.22 | **1.36×** | 1.850 | 0.507/0.237/0.106 | 7.279 → 5.455 | 133.1 → 136.0 |
| 8k **high** entropy | 8183 | 136.70 | 178.95 | **1.31×** | 1.779 | 0.472/0.211/0.096 | 7.315 → 5.729 | 456.1 → 466.9 |
| 32k **high** entropy | 32062 | 133.97 | 185.00 | **1.38×** | 1.892 | 0.523/0.252/0.117 | 7.464 → 5.485 | 1649.0 → 1690.1 |

The control is an almost exact internal check across all six cells (133.91–137.38
tok/s, ITL 7.279–7.468 ms) — the baseline is indifferent to both length and subject
matter, so the entire spread is drafter behaviour.

**Content domain is the dominant axis: 1.78 → 3.11 accepted length, +75%** — larger
than output shape (+33%) and far larger than temperature (+4%). ShareGPT's mixed
traffic at 1.81× sits between the two extremes, where a blend should.

**Length is flat over a 32× range, now for the right reason.** Within a tier,
1k → 32k moves accepted length by −0.036 (low) and +0.042 (high) — both smaller
than the tier gap by a factor of ~30. Wave 1 also found length flat, but only on
synthetic random tokens, where one could argue there was nothing worth copying.
These are real code and real prose at up to 32k, the ideal setup for a drafter to
lift spans out of context, and it still buys nothing — because EAGLE3's drafter
conditions on the target's hidden state, not on retrievable prompt text. The
prefill penalty stays near-constant in absolute terms (−0.7 ms at 8k-low,
+27.3 ms at 32k-low), so it shrinks *relatively* as prompts grow; per-user
throughput at 32k is unchanged, and only the server-aggregate ratio slips
(2.10× → 1.97× at 8k → 32k low) as prefill takes a larger share of the step.

**Conc 10** — the 2×2 load crossing (1k/8k, both tiers):

| cell (conc 10) | control | k3 | × per-user | server total × | accepted len | per-position |
|---|---|---|---|---|---|---|
| 1k-low | 80.71 | 152.91 | **1.89×** | 1.91× | 3.064 | 0.832/0.679/0.552 |
| 8k-low | 73.03 | 133.18 | **1.82×** | 1.79× | 3.131 | 0.850/0.704/0.577 |
| 1k-high | 85.40 | 104.69 | **1.23×** | 1.18× | 1.867 | 0.511/0.243/0.114 |
| 8k-high | 78.05 | 90.49 | **1.16×** | 1.14× | 1.816 | 0.486/0.226/0.105 |

**Acceptance is invariant to concurrency, third confirmation.** Every cell moves by
<0.04 between conc 1 and conc 10 (3.100→3.064, 3.106→3.131, 1.850→1.867,
1.779→1.816) — twice at 8× the load here, after phase B's flat 3.427/3.477/3.470
across conc 16/32/64. The speedup does shrink under load (2.15→1.89 low,
1.36→1.23 high), but that is compute sharing in the control, not worse drafting.

**The floor of the whole study is 8k-high at conc 10: 1.16× per-user, 1.14×
server.** With per-position rates 0.486/0.226/0.105, k=3 is spending three draft
slots to win 0.82 extra tokens. Still a gain, so it does not change the
enable-by-default verdict — but it is the cell where k=1 or k=2 would plausibly
beat k=3, and it is the one worth measuring before hard-coding k=3 for all
traffic.

One anomaly, recorded not explained: at 8k-high conc 10 the control's TTFT is
1448.2 ms against k3's 661.7 ms. The direction matches phase C's conc-16 inversion
(482.55 → 220.58 ms) and the same prefill-bound mechanism would predict it, but a
single cell at one concurrency is not enough to claim the magnitude is real.

> **Later correction (2026-07-29, unified run):** the magnitude *is* real and
> reproducible, but the effect is narrower than "prefill-bound at load" — see
> [The k=0 / 8k-high / conc-10 TTFT effect](#the-k0--8k-high--conc-10-ttft-effect).

### Output-budget censoring (measured, affects interpretation)

`max_tokens=2048` truncated a large share of responses, so these are **not**
natural-stopping lengths. Share of requests hitting the cap exactly
(control / k3; a few land at 2046–2047 from tokenizer re-count versus the server's
own accounting, so treat these as slight underestimates):

| cell | conc 1 | conc 10 |
|---|---|---|
| 1k-low | 15% / 20% | 30% / 28% |
| 8k-low | 60% / 60% | 52% / 53% |
| 32k-low | 65% / 70% | — |
| 1k-high | 80% / 82.5% | 78% / 70% |
| 8k-high | 87.5% / 90% | 93% / 93% |
| 32k-high | 90% / 95% | — |

The ratios and the tier contrast survive this: censoring is within a few points
between arms in every cell, both arms emit essentially the same token counts, and
at conc 1 the speed ratio is just the ITL ratio. Nor is this the wave-2 shape
inflation — `ignore_eos` forces generation *past* its natural stop (which is what
made drafting easy, +33%), whereas truncation stops reading *early*, so acceptance
over the retained prefix (~70–200k tokens per cell) is genuine natural-generation
acceptance. What the data does **not** support is any claim about per-tier natural
response length; a higher budget would be needed for that.

The conc-10 sweep was scoped to 1k and 8k by design
(`pipeline/slurm/specdec_phaseD_arm.sh:118`); 32k ran conc-1 only, so the two
32k conc-10 cells are absent rather than failed.

## Phase E — is the drafter compatible with *our* 4-bit target? (complete)

Window `20260727T084526Z-format`, 2 arms (`mxfp8-k{0,3}`), 4 cells each, all rc=0.
EAGLE3 consumes the *target's hidden states*, so quantization can cost acceptance
twice over — once by shifting the verify distribution, once by feeding the drafter
off-distribution input. This phase holds everything but the weight format constant
and reads acceptance directly. The W4AFP8 side is phase D's 8k cells: hash-identical
prompt bytes, same seed 42, temp 0.6, `max_tokens` 2048, TP8/EP8 on one node,
kv_cache_dtype fp8, block_size 128, gpu_util 0.9. The serve banners were diffed —
the only non-default args that differ are the checkpoint and `quantization: humming`.

**MXFP8 is the right reference, and BF16 is not.** The drafter's README states all
its published numbers are measured against `MiniMaxAI/MiniMax-M3-MXFP8` at
`tensor-parallel-size=4`. Training used `inference.vllm.tp_size=4` on GB300 nodes
(~744 GiB per engine): BF16 M3 is 796 GiB of safetensors and cannot fit that, while
MXFP8 is 414 GiB and fits with ~330 GiB left for KV. So the hidden states the
drafter learned from were MXFP8's. (`base_model: MiniMax-M3-preview` is lineage
only; the "bf16" in its training section is the *draft trainer's* dtype.) MXFP8 is
therefore the drafter's on-distribution target, which makes this phase — not a BF16
arm — the one that answers whether drafter finetuning has headroom.

### Accepted length — the answer

| cell | conc | W4AFP8 (ours, 4-bit) | MXFP8 (vendor, 8-bit) | delta | rel |
|---|---|---|---|---|---|
| 8k-low | 1 | 3.106 | 3.147 | +0.041 | +1.33% |
| 8k-low | 10 | 3.131 | 3.138 | +0.007 | +0.22% |
| 8k-high | 1 | 1.779 | 1.804 | +0.025 | +1.41% |
| 8k-high | 10 | 1.816 | **1.803** | **−0.013** | **−0.71%** |

Mean +0.56%, and **the sign flips** — on 8k-high at conc 10 our 4-bit target
*out-accepts* the vendor's 8-bit target. A difference that changes direction across
cells is measurement noise, not a quantization penalty. → **Our W4AFP8 target costs
no measurable drafter acceptance versus the drafter's own reference format.** On the
training-distribution-mismatch argument there is nothing for drafter finetuning to
recover.

Per-position rates agree cell-for-cell (e.g. 8k-low conc 1: 0.844/0.696/0.565 vs
0.851/0.709/0.587). An earlier read of the two conc-1 cells alone suggested a
consistent ~1.35% penalty *and* a compounding-down-the-chain pattern; both
dissolved once the conc-10 cells landed, and neither is claimed here.

### Speed — W4AFP8 is faster at equal acceptance

Both formats ran TP8 on one 8×H100 node with the same flag set, so unlike the BF16
arm these absolute numbers *are* comparable. Each format still carries its own k=0
control, so no ratio crosses formats.

| cell | conc | W4AFP8 k0 → k3 (×) | MXFP8 k0 → k3 (×) | our k3 advantage |
|---|---|---|---|---|
| 8k-low | 1 | 136.74 → 296.90 (2.17×) | 107.01 → 242.56 (2.27×) | **+22%** |
| 8k-low | 10 | 73.03 → 133.18 (1.82×) | 54.68 → 95.03 (1.74×) | **+40%** |
| 8k-high | 1 | 136.70 → 178.95 (1.31×) | 107.01 → 147.24 (1.38×) | **+22%** |
| 8k-high | 10 | 78.05 → 90.49 (1.16×) | 60.63 → 67.58 (1.11×) | **+34%** |

The speedup *ratios* are a wash (mean 1.62× vs 1.63×, no consistent direction), but
the absolute throughput is not: our W4AFP8 + Humming stack is **22–40% faster than
MXFP8 in every cell, with and without the drafter**. MXFP8's marginally better
ratios at conc 1 are an artefact of its slower baseline step — the drafter's largely
format-independent cost is a smaller fraction of a slower step. Decomposing conc-1
8k-low via `speedup = accepted / step-cost-ratio`: drafting inflates the W4AFP8 step
by 43.5% (7.313 → 10.49 ms) and the MXFP8 step by 39.3% (9.345 → 13.01 ms). Quoting
a cross-format ratio would credit MXFP8 for being slow.

### Cross-check against the vendor's published numbers

The drafter's README reports SPEED-Bench low-entropy at 16k (n=64, greedy draft,
`--enforce-eager`): **2.776 accepted, per-position 0.747/0.576/0.453.** Our W4AFP8 at
8k-low measured **3.106, 0.844/0.696/0.565** — above the vendor's own figure on their
own benchmark family. Not directly comparable (different ISL bucket, greedy vs
temp 0.6, eager vs graphs, unknown masked-data handling), but it rules out our
target underperforming the drafter's advertised behaviour.

Harness comparability: these are **not** comparable to published SPEED-Bench
scores. ~42–56% of the public parquet is masked
(`FULL BENCHMARK DATA SHOULD BE FETCHED FROM THE SOURCE USING SPECDEC_BENCH`) and
aiperf's loader does not filter it, so `pipeline/stage_speedbench.py` stages the
clean subset and the launcher gates on its hashes; the `mixed` tier is 512/512
masked and absent entirely; and the serving stack is our own W4AFP8 + Humming.

## Phase F — is drafter retraining worth it? (complete)

The question behind this phase: *if the drafter were trained against our target
instead of someone else's, how much acceptance would we get back?* The decision rule
was "if the unquantized baseline's acceptance is noticeably higher, it's worth
training/finetuning the draft model."

One correction to the framing before the numbers, because it changes what BF16 *is*.
BF16 is **not** the ceiling for this drafter. `Inferact/MiniMax-M3-EAGLE3` was both
measured and trained against **MXFP8**, not BF16 — its README states the measurement
target outright (`MiniMaxAI/MiniMax-M3-MXFP8`, TP=4, `--enforce-eager`), and training
is pinned by arithmetic: `inference.vllm.tp_size=4` on GB300 gives ~744 GiB per
engine, which BF16 M3 (796 GiB) cannot fit and MXFP8 (414 GiB) can. EAGLE3 drafters
consume the target's hidden states, so **MXFP8 is the on-distribution reference and
phase E was already the decisive arm.** Phase F still earns its keep: it rules out
the one alternative phase E could not, namely that 4-bit and 8-bit are *equally*
degraded relative to an unquantized target and phase E's null result is two damaged
arms agreeing with each other.

### Accepted length across the whole 4 → 8 → 16 bit range

Same drafter, same prompts (sha256-identical staged SPEED-Bench), same seed, same
k=3. Each figure is a Prometheus counter delta, `1 + Δaccepted/Δdrafts`:

| cell | conc | W4AFP8 (ours, 4-bit) | MXFP8 (8-bit) | BF16 (16-bit) | spread | ordering |
|---|---|---|---|---|---|---|
| 8k-low  |  1 | 3.106 | **3.147** | 3.128 | 0.041 (1.3%) | 8 > 16 > 4 |
| 8k-low  | 10 | 3.131 | 3.138 | **3.140** | 0.009 (0.3%) | 16 > 8 > 4 |
| 8k-high |  1 | 1.779 | **1.804** | 1.796 | 0.025 (1.4%) | 8 > 16 > 4 |
| 8k-high | 10 | **1.816** | 1.803 | 1.809 | 0.013 (0.7%) | 4 > 16 > 8 |

**The answer to the decision rule is no.** BF16 beats our 4-bit target by +0.71%,
+0.29% and +0.96% in three cells and *loses* to it by 0.39% in the fourth. Every one
of the four cells produces a **different ordering of the three formats**, and the
widest spread across a 4× bit-width range is 0.041 accepted tokens. A quantity that
cannot even rank 4-bit, 8-bit and 16-bit consistently is not measuring a quantization
penalty; it is measuring noise. Retraining or finetuning the drafter against our
target has no acceptance headroom to recover, because there is no deficit.

Per-position acceptance says the same thing more finely — position 0 across all six
low-entropy cells lands in 0.844–0.851, and across all six high-entropy cells in
0.472–0.486. The drafter's per-position behaviour is indistinguishable whether the
target's weights carry 4, 8 or 16 bits.

Two of my own earlier readings died here, and both were mine to retract: a "~1.35%
consistent penalty" claimed from phase E's first two cells (the conc-10 cells flipped
the sign) and an "MXFP8 > BF16 > W4AFP8" ordering that held twice and then broke
twice. The general shape of this study is that any effect under ~1.5% on accepted
length has not survived a fourth cell.

### Speed — and a cross-node drafting penalty

BF16 M3 is 796 GiB of weights, so it cannot be served on one 8×80 GiB node: this arm
runs **TP16 over ray across two nodes** while the W4AFP8 and MXFP8 arms are TP8
single-node. Its absolute latency is therefore *not* comparable to theirs, which is
why each arm carries its own k=0 control and only within-format ratios are quoted:

| cell | conc | W4AFP8 (TP8) | MXFP8 (TP8) | BF16 (TP16, 2 nodes) |
|---|---|---|---|---|
| 8k-low  |  1 | 2.17× | 2.26× | 2.06× |
| 8k-low  | 10 | 1.80× | 1.72× | 1.85× |
| 8k-high |  1 | 1.28× | 1.35× | 1.23× |
| 8k-high | 10 | 1.13× | 1.09× | 1.16× |

What *is* comparable across arms is the **absolute cost of drafting** — the k=3 step
minus the same arm's k=0 step, in ms:

| cell | conc | W4AFP8 (TP8) | MXFP8 (TP8) | BF16 (TP16) |
|---|---|---|---|---|
| 8k-low  |  1 | 3.18 | 3.67 | **6.17** |
| 8k-high |  1 | 2.88 | 3.13 | **5.46** |
| 8k-low  | 10 | 10.13 | 15.24 | 16.09 |
| 8k-high | 10 | 7.73 | 10.88 | 12.15 |

At conc 1 the cross-node arm pays **1.9× the drafting overhead** of the single-node
arms for the same drafter doing the same work. The mechanism is that every drafter
forward adds an inter-node all-reduce, and at k=3 there are three of them per step.
**Practical consequence: spec-dec's gain degrades on multi-node TP serving**, so a
speedup measured single-node should not be promised on a cross-node topology.

The two single-node arms also let the overhead be decomposed, since drafting cost
splits into a target-side part (the verify forward covers k+1 positions instead of 1,
so it scales with the target's own step time) and a format-independent drafter part.
Solving the two arms simultaneously at 8k-low conc 1 gives an extended verify worth
**+24% of a base step** for 3 extra positions (~8%/position) and a drafter cost of
**1.42 ms for 3 forwards** (~0.47 ms each). Repeating the solve on 8k-high gives +12%
and 1.98 ms — the same order, but a 2× disagreement in the split, because it divides
two small differences. Treat it as "roughly half the drafting overhead is the drafter
itself, roughly half is the widened verify," not as a precise partition. It does
independently corroborate the ~7%/position verify prior in
`M3_SPECDEC_EAGLE3_PLAN.md:22`.

## Phase G — does quantizing the *drafter* buy back drafting overhead? (complete)

Phases D–F varied the target's weight format and found acceptance indifferent to it.
This phase varies the **drafter** and asks a different question: not "does acceptance
survive?" but "can drafting be made cheaper?" Phase D pins the cost — at 8k-low conc 1
a k=0 step is 7.313 ms and a k=3 step is 10.49 ms, so drafting costs 3.18 ms/step and
is what caps the speedup at 2.17×.

Instrument: `Sebesky/MiniMax-M3-EAGLE3-RTN-INT4`, an RTN INT4 quantization of the
exact drafter we already use. Weight-only **W4A16** (no `input_activations` in any
config group), which is the right choice rather than a limitation — the drafter sees
1 token per user per forward, so its activations are kilobytes and quantizing them
would add work without saving bandwidth.

### The published checkpoint does not load, and its embedding quantization was moot

`llama_eagle3.py:158` constructs the draft `VocabParallelEmbedding` **without passing
`quant_config`**, while `ParallelLMHead` at line 292 passes
`quant_config=get_draft_quant_config(vllm_config)`. So vLLM always builds an
unquantized draft embedding and has no `weight_packed` parameter to load into. The
probe arm confirms it: `KeyError: 'embed_tokens.weight_packed'` at
`llama_eagle3.py:284`, on all 8 ranks, during drafter weight load.

Quantizing that tensor could never have paid anyway: when the draft embedding matches
the target's, vLLM **deletes it and shares the target's table** (phase D's serve log
logs exactly that), so an INT4 embedding adds error and no speed.
`pipeline/prepare_int4_drafter.py` therefore restores the bf16 embedding, drops the
`group_embed` group, and carries all 34 other published tensors byte-for-byte —
asserted tensor-by-tensor, not assumed. The result loads, shares the embedding on all
8 ranks, keeps its own INT4 lm_head, and reports **28.78 GiB** of model weights
against the bf16 drafter's 29.26 GiB — a 0.48 GiB drop against 0.44 GiB predicted from
the parameter count.

Kernel split, from vLLM's own chooser: the 8 attention/MLP/fc linears get **Machete**;
`lm_head` gets **Marlin**, because Machete's `can_implement` returns
`(False, 'Output features size must be divisible by 128')` and TP8 gives
200064/8 = 25008 per rank (`25008 % 128 == 48`). That matters — lm_head is 154 M of
the 254 M parameters read per drafter forward, so **~60% of the drafter's weight
traffic runs on the older kernel.** Padding the vocab to 200704 (→ 25088, divisible)
would move it onto Machete; not tested.

### Result — no acceptance cost, and a consistent but small speed win

Each arm served INT4, ran the grid, tore down, served bf16, and ran the identical
grid on the same node in the same allocation. 12 cells:

| k | cell | conc | Δ accepted | Δ ITL | Δ step cost |
|---|---|---|---|---|---|
| 3 | 8k-low  |  1 | −0.10% | −2.10% | −2.20% |
| 3 | 8k-high |  1 | −2.05% | −0.08% | −2.13% |
| 3 | 8k-low  | 10 | +0.57% | −1.88% | −1.32% |
| 3 | 8k-high | 10 | +0.80% | −1.50% | −0.72% |
| 4 | 8k-low  |  1 | −2.15% | −1.11% | −3.24% |
| 4 | 8k-high |  1 | +1.26% | −3.67% | −2.45% |
| 4 | 8k-low  | 10 | −0.55% | +0.13% | −0.43% |
| 4 | 8k-high | 10 | −0.39% | −0.65% | −1.03% |
| 5 | 8k-low  |  1 | +2.89% | −5.28% | −2.54% |
| 5 | 8k-high |  1 | −2.84% | −0.96% | −3.77% |
| 5 | 8k-low  | 10 | +0.25% | −1.00% | −0.75% |
| 5 | 8k-high | 10 | −0.16% | −1.25% | −1.40% |

**Acceptance: no cost.** Mean −0.21%, range −2.84% to +2.89%, signs flipping. RTN INT4
on a 1-layer drafter is free in accuracy terms. Note the noise band here (±3%) is
wider than the target-format comparison's (±1.5%) — each half is a separate engine on
a separate serve, so these are noisier than same-serve numbers.

**Speed: real, consistent, and about half the predicted size.** Step cost is lower in
**12 of 12 cells**, mean −1.83%. ITL is lower in 11 of 12, mean −1.61%. The bf16
half's same-serve repeat cell puts the drift floor at ±0.3% (+0.30%, −0.19%, +0.28%
across the three arms), so the effect clears the noise floor comfortably — but the
bandwidth model predicted −4.3% at k=3 8k-low and the measurement is −2.20%. Roughly
half. The likeliest reason is the Marlin lm_head above: if the layer holding 60% of
the weight traffic doesn't reach the bandwidth roof, neither does the saving. A
per-invocation cost that INT4 cannot touch (launch, all-reduce) accounts for the rest.

### Per-user output tok/s — bf16 drafter vs INT4 drafter (the default metric)

| k | workload | conc | bf16 drafter | INT4 drafter | gain |
|---|---|---|---|---|---|
| 3 | 8k-low  |  1 | 299.3 | 305.8 | +2.15% |
| 3 | 8k-low  | 10 | 141.0 | 143.5 | +1.74% |
| 3 | 8k-high |  1 | 183.0 | 183.4 | +0.19% |
| 3 | 8k-high | 10 |  90.3 |  92.1 | +2.00% |
| 4 | 8k-low  |  1 | 318.9 | 321.9 | +0.95% |
| 4 | 8k-low  | 10 | 148.0 | 148.6 | +0.41% |
| 4 | 8k-high |  1 | 173.2 | 181.0 | +4.51% |
| 4 | 8k-high | 10 |  88.7 |  89.9 | +1.26% |
| 5 | 8k-low  |  1 | 322.0 | 340.1 | +5.61% |
| 5 | 8k-low  | 10 | 153.8 | 155.7 | +1.22% |
| 5 | 8k-high |  1 | 171.6 | 168.1 | −2.04% |
| 5 | 8k-high | 10 |  83.8 |  85.1 | +1.50% |

**Mean +1.63%**, positive in 11 of 12 cells; server tok/s agrees at +1.59% mean. In
absolute terms this is 2–6 tok/s/user on an 84–340 tok/s base.

> **Noise-floor correction (phase I).** The ±0.24% figure originally quoted here came
> from the bf16 half's *same-serve* repeat, but the int4-vs-bf16 comparison is between two
> *different* serves, so that was the wrong yardstick. Phase I measured the cross-engine
> floor for the first time — three fresh-engine measurements of one identical config give
> **sd 1.22%, range 2.4%**, roughly 10× larger. The +1.63% mean is therefore *inside* the
> cross-engine floor. What carries the conclusion is sign consistency, not magnitude: 11
> of 12 cells positive is p = 0.0063 under a null of zero effect. Read this as "small and
> consistently positive", not as a measured 1.63%. The two outliers (+5.61%, −2.04%) are
> noise excursions either way.

Note that `output_token_throughput_per_user` is aiperf's mean of per-request `1/ITL`,
not `1/mean(ITL)`, so it is not exactly the reciprocal of the ITL column above — and in
the k=5 / 8k-high / conc 1 cell the two aggregations disagree on sign (ITL improves
0.96%, per-user rate worsens 2.04%). Where they conflict, per-user is the reported
metric and the disagreement is itself evidence the cell is inside the noise band.

For scale: on the same code workload, moving k=3 → k=6 (phase H) takes per-user speed
305.8 → 341.6, i.e. **+11.7%** — about seven times the drafter-quantization gain.

### What the drafter swap is worth at the *deployment* k

Only two cells of the grid above are configurations we would actually run: 8k-low at
k=5 (phase H's low-entropy optimum) and 8k-high at k=2 (its high-entropy optimum).
Read that way the answer is smaller than the +5.61% headline, and one half of it is
missing entirely.

**8k-low k=5 — the +5.61% is mostly an acceptance fluctuation, not a drafting win.**

| 8k-low k=5 | acceptance | step cost | ITL | per-user |
|---|---|---|---|---|
| conc 1  | 3.805 → 3.915 (**+2.89%**) | −2.54% | −5.28% | +5.61% |
| conc 10 | 3.853 → 3.863 (+0.25%)     | −0.75% | −1.00% | +1.22% |

`ITL = step / accepted`, so the conc-1 figure is `(1−0.0254)/(1+0.0289) = −5.28%`: over
half of it comes from acceptance, which is the one quantity an INT4 drafter is not
supposed to move. Because acceptance is concurrency-invariant (confirmed four times
here), the two concurrencies are replicates of a single value — bf16 measured
3.805/3.853, INT4 measured 3.915/3.863, overlapping scatter about ~3.86 with bf16's
conc-1 point the low outlier. Pooled the gap is +1.57%, inside the ±3% two-serve band.
**Quote the step-cost saving instead: ~2.5% (conc 1) and ~0.8% (conc 10), i.e. ~+2.6%
and ~+0.8% per-user at acceptance parity.**

**8k-high k=2 — not measured.** Phase G ran k=3/4/5 only, so no bf16-vs-INT4 pair
exists at the high-entropy optimum. Scaling k=3's measured saving (0.217 ms of step for
3 drafter forwards ≈ 0.072 ms/forward) to 2 forwards on a ~9.3 ms step predicts ~1.5% —
a *prediction*, and a weak one, since phase G's other finding was that the predicted
k-scaling did not materialise. Closing this hole costs one A/B arm.

Also note that under the per-user metric the two entropy tiers disagree about which k
wins at 8k-high conc 1: ITL and server tok/s pick k=2, per-user picks k=3 (187.9 vs
185.6, +1.2%), while at conc 10 k=2 wins by 2.4%. Treat 8k-high as a k=2/k=3 tie under
per-user.

The predicted scaling with k did **not** appear: at 8k-low conc 1 the step saving is
−2.20% / −3.24% / −2.54% for k=3/4/5 — flat within cell-to-cell variance, where more
drafter forwards should have meant a bigger absolute saving.

One retraction. An early cross-window read against phase D showed −7.2% ITL at 8k-low
conc 10 and I flagged it as suspicious because the bandwidth model predicts the effect
should *shrink* at load (the drafter's weight read is fixed per step while total
drafting overhead grows from 3.2 to 10.1 ms). The same-node number is **−1.88%**. The
cross-window figure was node contamination, and node variance in this study runs
1–2%.

**Verdict:** adopt the INT4 drafter — it is free in acceptance and worth ~1.8% of step
cost grid-wide, ~2.5% at the low-entropy deployment k and (predicted, unmeasured) ~1.5%
at the high-entropy one — but it is a minor lever. Choosing k per workload (phase H) is
worth 4–9%.

## Phase H — optimal draft depth per entropy tier (complete)

Phase G's k=3/4/5 sweep bracketed neither optimum, and it had a design limitation
worth naming: it ran each k on a *different* node (h107/h123/h108), so node variance
was folded into its k-trend. That was correct for the INT4-vs-bf16 A/B it was built
for — that comparison was same-node — but choosing k *is* a k-comparison. Phase H
puts every k for a tier on one node and measures the drift that replaces node variance
by re-serving the first spec k at the end of the window.

Each arm carries its own k=0 control, so every speedup below is within-window.

### Low entropy (8k-low, code/sorting) — the knee is at k≈5–6

| k | conc 1 accepted | ITL | speedup | | conc 10 accepted | ITL | speedup |
|---|---|---|---|---|---|---|---|
| 0 | — | 7.307 | 1.00× | | — | 13.431 | 1.00× |
| 5 | 3.820 | 3.014 | 2.425× | | 3.864 | **6.707** | **2.002×** |
| 6 | 4.107 | **2.970** | **2.460×** | | 4.077 | 6.925 | 1.939× |
| 7 | 4.224 | 2.979 | 2.453× | | 4.307 | 6.805 | 1.974× |

*drift control: k=5 re-served at end of window, ITL 3.014 → 3.032 (+0.59%) at conc 1,
6.707 → 6.656 (−0.76%) at conc 10.*

**This corrects the extrapolation I made from phase G.** Phase G saw k=3→4→5 each buy
~5% of ITL with no decay and I inferred the knee was above k=5. It isn't: k=5→6 buys
1.5% and k=6→7 buys −0.3%. The marginal gain collapsed immediately after k=5. At
conc 10 the sweep is non-monotone (k=6 worse than both k=5 and k=7) with a 3.3% spread
against a 0.76% drift floor, which is the signature of a plateau, not a trend.
**Practical answer: k=5 or 6; anything beyond 5 is within noise of it.**

### High entropy (8k-high, creative writing) — the optimum is k=2

| k | conc 1 accepted | ITL | speedup | | conc 10 accepted | ITL | speedup |
|---|---|---|---|---|---|---|---|
| 0 | — | 7.312 | 1.00× | | — | 12.453 | 1.00× |
| 1 | 1.507 | 5.654 | 1.293× | | 1.512 | 10.630 | 1.171× |
| 2 | 1.717 | **5.473** | **1.336×** | | 1.745 | **10.568** | **1.178×** |
| 3 | 1.815 | 5.486 | 1.333× | | 1.816 | 10.846 | 1.148× |

*drift control: k=1 re-served at end of window, ITL 5.654 → 5.736 (+1.44%) at conc 1,
10.630 → 10.697 (+0.63%) at conc 10.*

**k=2 wins at both concurrencies.** At conc 1, k=3 is statistically tied (+0.2%) and
k=1 is 3.3% behind. At conc 10 the recipe's k=3 is a clear 2.6% loss — load pushes the
optimum down, as the marginal-cost-per-position figures predict (12.1%/token at conc 1
vs 19.4%/token at conc 10).

### Output speed in tok/s (same windows, within-window k=0 control)

The ratios above are ITL-based; these are the absolute rates, per user and per server,
on the INT4 drafter at each tier's best k:

| workload | conc | k | per-user tok/s | vs k=0 | server tok/s | vs k=0 |
|---|---|---|---|---|---|---|
| code (8k-low) | 1 | 0 | 136.8 | — | 132.1 | — |
| code (8k-low) | 1 | **6** | **341.6** | **2.50×** | **316.3** | **2.40×** |
| code (8k-low) | 10 | 0 | 74.8 | — | 706.4 | — |
| code (8k-low) | 10 | **5** | **153.5** | **2.05×** | **1427.8** | **2.02×** |
| creative (8k-high) | 1 | 0 | 136.8 | — | 132.4 | — |
| creative (8k-high) | 1 | **2** | 185.6 | 1.36× | **175.1** | **1.32×** |
| creative (8k-high) | 1 | 3 | 187.9 | 1.37× | 174.7 | 1.32× |
| creative (8k-high) | 10 | 0 | 80.5 | — | 764.9 | — |
| creative (8k-high) | 10 | **2** | **97.0** | **1.21×** | **892.5** | **1.17×** |
| creative (8k-high) | 10 | 3 | 94.7 | 1.18× | 868.9 | 1.14× |

Full k sweeps in tok/s per user — 8k-low conc 1: 136.8 (k=0) → 334.1 / 341.6 / 339.2
(k=5/6/7); conc 10: 74.8 → 153.5 / 149.7 / 152.4. 8k-high conc 1: 136.8 → 178.0 /
185.6 / 187.9 (k=1/2/3); conc 10: 80.5 → 95.1 / 97.0 / 94.7.

**Why the two columns differ at conc 1, where there is only one user.** They have
different denominators, from aiperf's definitions: `output_token_throughput_per_user`
is `1 / ITL`, and ITL is the mean gap *between* output tokens, so it excludes the first
token and is a pure decode rate. `output_token_throughput` is total output tokens over
benchmark wall clock, which includes every request's prefill. At conc 1 the gap between
them is therefore **only TTFT amortization**, and reconstructing it confirms that —
`OSL / (TTFT + (OSL−1)·ITL)` reproduces the measured server rate to within ~1–3%:

| workload | k | decode | prefill | prefill share | per-user | server | ratio |
|---|---|---|---|---|---|---|---|
| 8k-low  | 0 | 12.40 s | 0.413 s | 3.2% | 136.8 | 132.1 | 0.965 |
| 8k-low  | 6 |  4.68 s | 0.417 s | **8.2%** | 341.6 | 316.3 | **0.926** |
| 8k-high | 0 | 14.90 s | 0.457 s | 3.0% | 136.8 | 132.4 | 0.968 |
| 8k-high | 3 | 10.85 s | 0.467 s | 4.1% | 187.9 | 174.7 | 0.930 |

The share **grows as decode gets faster** because TTFT does not move — it is 413–420 ms
across every k in the low-entropy arm, since spec-dec accelerates decode and never
touches prefill. That is why the server-level speedup (2.40×) is lower than the decode
speedup (2.50×): ordinary Amdahl on the prefill fraction. It also means the gap widens
for shorter outputs, and the whole benefit disappears for a workload whose cost is
prefill — which is exactly what the agentic shape in `docs/m3-two-axis-perf.md` is
(≈100 output tokens per ~12k-token prompt). **Quote the per-user rate for how fast text
feels, the server rate for capacity; the second is the honest end-to-end number.**

Two honest notes on the high-entropy tier. At conc 1, **k=2 and k=3 are tied**: ITL
favours k=2 by 0.2%, per-user tok/s favours k=3 by 1.2%, server tok/s favours k=2 by
0.2% — all inside that arm's 1.44% drift floor. (The metrics can disagree slightly
because ITL excludes the first token while the throughput metrics do not.) The tiebreak
comes from conc 10, where k=2 beats k=3 by 2.4% per user and 2.7% per server. So k=2 is
the recommendation because it never loses, not because it wins at conc 1.

Also recorded and not explained: the 8k-high conc-10 **k=0 control shows TTFT 1236 ms**
against 382–411 ms for every spec-dec config in the same arm. Decode is slower at k=0,
so requests queue longer, but that does not obviously account for a 3× gap. This is the
same unexplained TTFT behaviour flagged in phase D and it does not affect the
output-rate comparison, which is measured after the first token.

### The k=0 / 8k-high / conc-10 TTFT effect

The unified run (31 serves, 3 windows, gpu-h114) settles what this is. Across every
serve in the pooled set, TTFT splits into exactly two populations:

| serve class | 8k-high conc 10, TTFT (ms) | n |
|---|---|---|
| **k=0** (`L0-cut-*`, `L0-hum-*`) | 835.8, 934.8, 1135.7, 1172.0, 1431.6 | 5 / 5 elevated |
| **k≥1** (every k, kernel, drafter) | 415.4 – 449.9 | 9 / 9 normal |

Three things are now ruled out that earlier phases left open.

1. **It is not a property of the 8k-high workload.** The same tier at conc 1 is
   normal in the same serves (431–467 ms), and every k≥1 serve at 8k-high conc 10
   is normal.
2. **It is not a property of the k=0 configuration.** Those same five k=0 serves
   are normal in their other three cells — 8k-high c1 431–467 ms, 8k-low c10
   361–417 ms (in fact the *lowest* TTFT anywhere in the study).
3. **It is not an unreplicated cell.** It reproduces 5/5, across two prefill
   backends and two windows. It is also the noisiest cell in the study by far:
   sd ≈ 220 ms, ~20% CV, against ≈2% for every other TTFT cell.

So the effect requires the **conjunction** of k=0, the high-entropy tier, and conc 10.
No single factor produces it. Earlier text in this doc called it "a workload property";
that was wrong, and the correction is the conjunction.

**Confound that remains, stated because it is not separable from this data.** The
`L0-*` k=0 serves are the only ones that run all four cells in a single serve; every
k≥1 serve runs only its own tier's two cells. 8k-high/c10 is therefore the *last*
cell of a k=0 serve and an early cell everywhere else, so within-serve ordering is
perfectly collinear with k=0 here. Position-in-window is separately ruled out (window
`20260729T031914Z` opened with a k=1 serve that measured 438.9 ms in this cell), but
position-*within-serve* is not. Suggestively, the two CUTLASS values are the two
lowest and the three Humming values the three highest — consistent with the backend
mattering, on n=2 vs n=3.

**The one-line experiment that would settle it:** run a single k≥1 serve over all four
cells in the k=0 cell order. If its 8k-high/c10 TTFT is elevated, the cause is
within-serve position (accumulated KV/prefix-cache state), not k=0. Cost is one serve.
Not run, because TTFT is not a metric any decision in this study rests on — every
throughput comparison here is measured after the first token.

### The rule that predicts all of it

`ITL = step / accepted`, so raising k helps **iff the fractional gain in accepted
length exceeds the fractional increase in step cost** — Δa/a > Δs/s. That is algebra,
not a fit, and it calls every cell measured:

| tier | step | Δa/a | Δs/s | predicted | measured ITL |
|---|---|---|---|---|---|
| 8k-low c1 | 5→6 | 7.51% | 5.96% | better | −1.5% ✓ |
| 8k-low c1 | 6→7 | 2.85% | 3.14% | worse | +0.3% ✓ |
| 8k-high c10 | 2→3 | 4.07% | 6.82% | worse | +2.6% ✓ |

Because per-position acceptance falls geometrically while marginal step cost falls only
slowly, the crossing point is sharp and entropy-dependent: **k≈5–6 for code, k=2 for
creative writing** — and the recipe's one-size k=3 is wrong for both, leaving 4–9% on
the table depending on the workload. This is a larger lever than the drafter
quantization of phase G.

## Phase I — which W4A16 kernel should the drafter use? (complete)

Window `20260727T134635Z-kernel`, 1 arm × 5 serves on gpu-h113, all at k=5 / 8k-low /
conc 1 and 10. Arm rc=1: three of five serves completed, two were killed by an upstream
defect (below), which is itself the phase's second finding.

**The hypothesis.** Phase G's INT4 drafter delivered ~half its predicted bandwidth
saving, and I named the drafter's `lm_head` running on Marlin as the leading
explanation. The mechanism was concrete: `check_machete_supports_shape` requires
`out_features % 128 == 0`, TP8 gives 200064/8 = 25008 (`% 128 == 48`), so Machete takes
8 of 9 drafter linears and Marlin takes `lm_head` — 153.6 M of the 254.3 M params read
per rank per forward, **60% of the drafter's weight traffic**. vLLM warns about it
directly (`marlin_utils.py:237`): the padded layer's activations/outputs are
"padded/sliced on every forward". `marlin_padded_nk(25008, 6144, 128)` picks
(25024, 6144) — only +16 columns, so the cost is a per-forward pad/slice and an extra
launch, not bandwidth.

**Result: the hypothesis is refuted.** Cell D moved `lm_head` onto Machete by padding
the draft vocab to 200704 (25088/rank, `% 128 == 0`), verified at serve time —
`kernels=[Machete]`, Marlin gone from the process, pad warning absent.

| cell | conc | per-user | vs A | ITL ms | accepted | step ms | vs A |
|---|---|---|---|---|---|---|---|
| A-baseline (Machete+Marlin) |  1 | 337.3 | — | 2.994 | 3.866 | 11.577 | — |
| A-baseline | 10 | 152.9 | — | 6.728 | 3.868 | 26.023 | — |
| D-machete-all |  1 | 331.7 | −1.66% | 3.042 | 3.780 | 11.499 | −0.68% |
| D-machete-all | 10 | 155.8 | +1.85% | 6.602 | 3.844 | 25.379 | −2.47% |
| A-repeat (drift) |  1 | 329.2 | **−2.40%** | 3.057 | 3.759 | 11.492 | −0.74% |
| A-repeat | 10 | 153.2 | +0.14% | 6.720 | 3.796 | 25.510 | −1.97% |

The two concurrencies disagree in sign (−1.66%, +1.85%), and **the drift control is as
large as the effect**: re-serving the identical config at window end moved per-user
−2.40% at conc 1 — bigger than cell D's −1.66%.

Three independent measurements of the *same* config on the *same* node, each on a fresh
engine, pin the precision:

| measurement | per-user tok/s |
|---|---|
| phase H, k=5 | 334.1 |
| phase I, A-baseline | 337.3 |
| phase I, A-repeat | 329.2 |

mean 333.5, **sd 1.22%**, range 2.43%. Cell D's effects are 1.36 sd and 1.51 sd — **not
significant**. Moving 60% of the drafter's weight traffic off Marlin and eliminating a
per-forward pad/slice produces no measurable change in output speed. If the Marlin
`lm_head` were worth the ~2% needed to explain phase G's missing half, cell D would have
shown it consistently. It did not.

One suggestive detail, not a claim: the step-cost saving is larger at conc 10 (−2.47%)
than conc 1 (−0.68%), which is the shape conventional wisdom predicts — Marlin was built
for the memory-bound batch-1 regime and Machete's advantage grows with batch (the
drafter GEMM has 1 row at conc 1, 10 at conc 10). But most of that conc-10 figure is
drift (A-repeat alone moved step −1.97%), so it needs replication to claim.

**Second finding: `HummingLinearKernel` cannot serve a quantized `lm_head` at all.**
Cells B and C forced Humming onto `lm_head` via `VLLM_DISABLED_KERNELS`. Kernel
*selection* succeeded (`Using HummingLinearKernel for CompressedTensorsWNA16` appears in
both logs), then all 8 ranks died in weight prep:

```
AttributeError: 'ParallelLMHead' object has no attribute 'input_size'
```

`prepare_humming_layer` (`humming_utils.py:81`) is annotated `layer: LinearBase` and
needs three attributes `ParallelLMHead` does not have:

| needed | `ParallelLMHead` equivalent |
|---|---|
| `input_size_per_partition` / `input_size` | `embedding_dim` |
| `output_partition_sizes` (used twice) | `[num_embeddings_per_partition]` |
| `has_bias` | absent — only `LinearBase` sets it (`linear.py:260`) |

The irony is that the function's own comment names the class it crashes on — *"Use
hasattr rather than getattr's default arg, which is evaluated eagerly and would raise on
layers lacking input_size (e.g. ParallelLMHead)"* — and then the `else` branch
dereferences `layer.input_size` anyway. `can_implement` never checks layer type, so the
kernel claims an `lm_head` it cannot prepare. This fully explains why the path has never
run on CUDA: it is unreachable by default (Marlin precedes it and always succeeds) *and*
broken when reached. Fixing it was initially **not** pursued, because cell D appeared to
refute the hypothesis that the `lm_head` kernel is worth anything. It was fixed in
phase I.2 (below), which also revised that dismissal at conc 10. It is the study's
second upstream vLLM defect, alongside `llama_eagle3.py:158` omitting `quant_config`
(phase G).

**Retroactive correction to phase G's noise floor.** Phase G's drift control was a
*same-serve* repeat (±0.24% per-user), but its actual comparison — int4 vs bf16 — was
between two *different* serves. The right yardstick is the cross-engine floor, which
phase I measures for the first time at **sd 1.22%, range 2.4%** — roughly 10× larger. So
phase G's mean **+1.63% per-user is inside the cross-engine noise floor**, and its
conclusion does not rest on that magnitude. What survives is sign consistency: 11 of 12
cells positive, which under a null of zero effect is p = 0.0063. The INT4 drafter is
still worth adopting (it is free in acceptance and consistently non-negative), but
"+1.63%" should be read as "small and positive", not as a measured 1.63%.

**Also confirmed:** acceptance run-to-run variance at these sample sizes is ~2–3% at
conc 1 (A-repeat moved it −2.78% with nothing changed), which independently vindicates
treating phase G's +2.89% acceptance excursion at k=5/8k-low as noise rather than a
drafter effect.

**Verdict (as issued at the time; revised at conc 10 by phase I.2):** the drafter's
kernel assignment is not a lever *at conc 1*. Phase I.2 re-measured the killed Humming
cells after fixing the upstream defect and found the conc-10 half of this verdict was
wrong — cell D's +1.85% at conc 10 was dismissed against the conc-1 noise floor
(sd 1.22%), but the conc-10 floor turns out to be ~8× tighter (sd 0.16% across four
cross-window replicates), making it significant after all. See phase I.2 for the
corrected picture. The conc-1 conclusion stands, and the missing half of phase G's
predicted INT4 saving at conc 1 remains explained by per-invocation costs INT4 cannot
touch (kernel launch, all-reduce, Python/dispatch overhead).

## Phase I.2 — the Humming lm_head fix, and a conc-10 revision (complete)

Window `20260727T154725Z-kernel`, 1 arm × 3 serves on gpu-h113 (same node as phases
H/I), rc=0. Two purposes: prove the `prepare_humming_layer` defect is fixable with a
small patch, and re-run the two cells phase I lost to it.

### The fix

`pipeline/slurm/patch_vllm_humming_lmhead.py` (applied to the quant venv, commit
`f35989b4`) replaces the three `LinearBase`-only attribute reads with guarded
fallbacks that fire only when the `LinearBase` attributes are absent:

| read | fallback |
|---|---|
| `input_size_per_partition` → `input_size` | → `embedding_dim` |
| `output_partition_sizes` (both uses) | → `[num_embeddings_per_partition]` |
| `has_bias` | → `getattr(layer, "bias", None) is not None` |

Everything downstream was verified layer-type agnostic before writing it:
`VocabParallelEmbedding` runs the same WNA16 `create_weights` (same param names and
`input_dim`/`output_dim` tags), the extra `weight_shape` param flows through
`convert_humming` untouched, `prepare_layer_meta` takes shapes explicitly, and
`humming_gemm` allocates its output at the *valid* (unpadded) width
(`shape_n − pad_shape_n`, `ops/__init__.py:132`) so the vocab-parallel logits gather
stays aligned — Humming needs none of the vocab-padding games Machete needed in cell D.
The patch is inert for every layer vLLM served before; the controller gains a
fail-closed source gate for it.

**It works.** Both previously-dead cells came up on all 8 ranks with the gated kernel
sets, and acceptance — the numerics check; a garbage lm_head collapses it toward 1.0 —
stayed in the normal 3.74–3.89 band. This was the first Humming W4A16 forward ever run
through vLLM's MPLinear path.

### Results (all k=5 / 8k-low, INT4 drafter)

| cell | kernels | conc | per-user | vs A | ITL ms | accepted | step ms | vs A |
|---|---|---|---|---|---|---|---|---|
| A-baseline | Machete+Marlin | 1 | 335.2 | — | 3.007 | 3.820 | 11.487 | — |
| A-baseline | | 10 | 153.1 | — | 6.746 | 3.884 | 26.199 | — |
| B-hum-lmhead | Machete+Humming | 1 | 341.0 | +1.73% | 2.954 | 3.861 | 11.405 | −0.72% |
| B-hum-lmhead | | 10 | **157.3** | **+2.73%** | 6.524 | 3.885 | 25.348 | **−3.25%** |
| C-hum-all | Humming ×9 | 1 | 333.2 | −0.59% | 3.023 | 3.743 | 11.315 | −1.50% |
| C-hum-all | | 10 | 155.5 | +1.54% | 6.607 | 3.834 | 25.336 | **−3.30%** |

A-baseline reproduces phase H's k=5 cell across windows at +0.32% (conc 1) and −0.25%
(conc 10) — the cross-window check passes.

### The noise floor splits by concurrency — and that revises phase I

The identical baseline config now has **four** fresh-engine, cross-window replicates on
gpu-h113:

| replicate | conc-1 per-user | conc-10 per-user |
|---|---|---|
| phase H, k=5 | 334.1 | 153.5 |
| phase I, A-baseline | 337.3 | 152.9 |
| phase I, A-repeat | 329.2 | 153.2 |
| phase I.2, A-baseline | 335.2 | 153.1 |

Conc 1: mean 333.9, **sd 1.02%**, range 2.4% — the phase I floor, confirmed. Conc 10:
mean 153.2, **sd 0.16%**, range 0.4%. Batch-10 aggregation averages away the
single-stream jitter, and the conc-10 column is ~7× more precise than the conc-1 one.

Against the right floor, at **conc 10** every no-Marlin cell is a significant win:
B-hum-lmhead +2.73%, D-machete-all +1.85% (phase I, wrongly dismissed against the
conc-1 floor), C-hum-all +1.54%. The acceptance-adjusted view agrees and is cleaner:
step cost −3.25% (B), −3.30% (C), −2.47% (D) — B and C land on the *same* ~3.3%
kernel-level saving, with C's smaller per-user gain explained by its −1.27% acceptance
drift. At **conc 1** every effect stays inside the 1.0–1.2% floor: phase I's conc-1
verdict stands.

The mechanism is the conventional one, now with clean evidence: the three winning cells
share exactly one feature — **Marlin is off the `lm_head` at batch > 1**. Marlin was
built for the memory-bound batch-1 regime; at conc 10 the drafter GEMM has ~10 rows and
Marlin's weak batch scaling costs ~2–3% of the whole decode step through the one layer
holding 60% of drafter weight traffic. (These cells cannot split how much of that is the
per-forward pad/slice vs. Marlin's batch scaling, and nothing downstream needs the
split.) At batch 1 Marlin is at home, and the assignment doesn't matter.

**Verdict:** two regimes. At conc 1, leave the default (Machete ×8 + Marlin `lm_head`) —
no measurable effect. For loaded serving at conc ≈ 10, take Marlin off the drafter's
`lm_head`: the best measured cell is `VLLM_DISABLED_KERNELS=MarlinLinearKernel`
(+2.73% per-user, requires the `prepare_humming_layer` patch), and the patch-free
alternative is `LLMC_EAGLE3_LMHEAD_PAD=1024` for Machete-all (+1.85%). This does *not*
resurrect the phase G "missing half" hypothesis at conc 1 — the conc-1 null is
confirmed, and the overhead decomposition (drafter forward ≈ 4× its HBM floor) remains
the explanation there.

## Raw evidence

**Phase I.2** — `/mnt/nfs/hoangduy/results/m3-specdec-eagle3/20260727T154725Z-kernel/`:
`arm-kernel/{A-baseline,B-hum-lmhead,C-hum-all}/` with the same per-cell artifact set as
phase I (`wna16-kernels.txt`, `marlin-pad-warning.txt`, `kernel-census.txt`,
`accepted-*.txt`, `metrics/`, `speedbench/`), window-level `patch-checks.log` (including
the new humming-lmhead source gate), `aggregate.json`. Re-run cells were selected with
`CELLS=A-baseline,B-hum-lmhead,C-hum-all`.

**Phase I** — `/mnt/nfs/hoangduy/results/m3-specdec-eagle3/20260727T134635Z-kernel/`:
`arm-kernel/{A-baseline,B-hum-lmhead,C-hum-all,D-machete-all,A-repeat}/`. Each served
cell carries `wna16-kernels.txt` (the gated kernel set), `marlin-pad-warning.txt`,
`kernel-census.txt` (every "Using X for Y" line, to prove the env lever never touched a
non-WNA16 scheme), `kernel-env.txt`, `accepted-*.txt`, `metrics/*-{pre,post}.txt`, and
`speedbench/*/conc_*/profile_export_aiperf.json`. `B-hum-lmhead/serve.log` and
`C-hum-all/serve.log` hold the `ParallelLMHead' object has no attribute 'input_size'`
tracebacks on all 8 ranks, each preceded by `Using HummingLinearKernel for
CompressedTensorsWNA16` — selection succeeded, weight prep did not. Pre-flight gates in
`kernel-registry.txt` (priority order Machete > Marlin > Humming),
`humming-availability.txt`, `drafter-identity.txt` (asserts the 25008 % 128 == 48
premise), `patch-checks.log`, `speedbench-manifest.txt`. Aggregate:
`kernel-summary.json` via `pipeline/specdec_kernel_aggregate.py`.


**Wave 2** — `/mnt/nfs/hoangduy/results/m3-specdec-eagle3/20260727T064934Z-wave2/`:
6 arms (`natural-k{0,3}`, `load-k{0,3}`, `lowconc-k{0,3}`), each with `client.log`,
`serve.log`, `spec-boot.log`, `backend-attestation.json`, and per-cell
`metrics/<cell>-{pre,post}.txt` from which every acceptance figure above is a
counter delta. Window-level `aggregate.json`, `arm-provenance.txt`,
`actual-commit.txt`. Regenerate the tables with:

```
pipeline/specdec_wave2_aggregate.py --root <window> --out-json <window>/aggregate.json
```

**Phase H** — `/mnt/nfs/hoangduy/results/m3-specdec-eagle3/20260727T105919Z-kopt/`:
arms `kopt-low` (gpu-h113, k=0/5/6/7 on 8k-low) and `kopt-high` (gpu-h114, k=0/1/2/3
on 8k-high), each 5 serves in one allocation with per-config `k<n>/` subdirectories
holding `serve.log`, `spec-boot.log`, `backend-attestation.json`,
`model-loading-gib.txt`, per-cell aiperf artifacts and
`metrics/sb-<cell>-c<n>-{pre,post}.txt`. The trailing `k<n>-repeat/` config is the
end-of-window drift control. Window-level `drafter-identity.txt` (asserts W4A16 with
no activation quant and an unmodified derivation hash), `speedbench-manifest.txt`,
`arm-provenance.txt`, `actual-commit.txt`, `controller-done.txt`. Regenerate with the
inline analysis in `pipeline/specdec_int4drafter_aggregate.py`'s helpers.

**Phase G** — `/mnt/nfs/hoangduy/results/m3-specdec-eagle3/20260727T102751Z-int4drafter/`:
arms `int4-k{3,4,5}`, each holding both halves of its A/B as `int4/` and `fp/`
subdirectories (identical cell grids, `fp/` additionally carrying the
`8k-low-repeat` drift cell), plus `probe-published/` whose `probe-verdict.txt` records
`failed` and `probe-failure-excerpt.txt` the `KeyError: 'embed_tokens.weight_packed'`
traceback from the as-published checkpoint. Each half has `serve.log`,
`spec-boot.log`, `backend-attestation.json`, `drafter-config.json`,
`model-loading-gib.txt` (28.78 GiB INT4 vs 29.26 GiB bf16) and per-cell
`metrics/sb-<cell>-c<n>-{pre,post}.txt`. Window-level `drafter-identity.txt`,
`speedbench-manifest.txt` (hash-equal to phases D–F), `arm-provenance.txt`,
`actual-commit.txt`, `controller-done.txt`. Regenerate the table with:

```
pipeline/specdec_int4drafter_aggregate.py --root <window> --out-json <window>/aggregate.json
```

The measured drafter is derived, not the published artifact — see
`/mnt/nfs/hoangduy/hf_assets/derived/MiniMax-M3-EAGLE3-INT4-bf16embed/derivation-manifest.json`
for the input hashes, the three dropped tensors and the spliced embedding.

**Phase F** — `/mnt/nfs/hoangduy/results/m3-specdec-eagle3/20260727T092342Z-bf16ref/`:
arms `bf16-k{0,3}`, each 2 nodes × 8 GPUs (TP16 over ray), with `client.log`,
`serve.log`, `spec-boot.log`, `ray_runtime/gate.json`, per-cell aiperf artifacts and
`metrics/sb-<cell>-c<n>-{pre,post}.txt`. Window-level `speedbench-manifest.txt`
(hash-equal to phases D and E), `reference-windows.txt` naming both comparison
windows, `arm-provenance.txt`, `actual-commit.txt`, `controller-done.txt`. The
launcher gates the BF16 identity fail-closed (`torch_dtype=bfloat16`, no
`quantization_config`, weights >640 GiB so the TP16 rationale cannot go stale) and
re-probes every node's free GPU memory immediately before taking it — on the first
attempt that gate refused the launch when `gpu-h97` went from 633 GiB free to
252 GiB between an earlier probe and the launch.

**Phase E** — `/mnt/nfs/hoangduy/results/m3-specdec-eagle3/20260727T084526Z-format/`:
arms `mxfp8-k{0,3}`, each with `client.log`, `serve.log`, `quant-boot.log` (the
asserted native MXFP8 path), `spec-boot.log`, per-cell aiperf artifacts and
`metrics/sb-<cell>-c<n>-{pre,post}.txt`. Window-level `speedbench-manifest.txt`
(hash-equal to phase D's), `w4a8-reference-window.txt` naming the phase D window
used as the comparison arm, `arm-provenance.txt`, `actual-commit.txt`.

**Phase D** — `/mnt/nfs/hoangduy/results/m3-specdec-eagle3/20260727T073533Z-phaseD/`:
arms `phaseD-k{0,3}`, each with `client.log`, `serve.log`, `spec-boot.log`,
`backend-attestation.json`, per-cell `speedbench/<cell>/conc_<n>/` aiperf artifacts
(including `profile_export.jsonl`, the source of the censoring table) and
`metrics/sb-<cell>-c<n>-{pre,post}.txt` from which every acceptance and
per-position figure is a counter delta. Window-level `speedbench-manifest.txt`
records the staged prompt hashes and per-cell token statistics the launcher gated
on; `arm-provenance.txt`, `actual-commit.txt`, `controller-done.txt`.

**Wave 1** — `/mnt/nfs/hoangduy/results/m3-specdec-eagle3/20260727T061506Z/` — per arm
`serve.log`, `spec-boot.log`, `spec-metrics.log`, `backend-attestation.json`,
`greedy-probe.json`, `metrics-{pre,post}-aa.txt`, `aa-sweep.log`; window-level
`AGGREGATE.md`, `aggregate.json`, `arm-provenance.txt`, `actual-commit.txt`.
AA cells: `benchmarks/results/minimax-m3-specdec-{k0-control,k1,k3,k5}/self-hosted/perf/aa-sweep/20260727T061506Z/`.

Infrastructure note: `gpu-h97`, `gpu-h98` and `gpu-h101` were reported `idle` by
slurm while another user's out-of-band DeepSeek-V4 run held all 8 GPUs on each
(~32 GiB free). Two arms were refused by our serve preflight and one died mid-boot
when the foreign job took memory after preflight had passed; all three were
re-run on verified-clean nodes via `pipeline/slurm/relaunch_specdec_eagle3_arms.sh`.
No arm reported numbers from a contended node.
