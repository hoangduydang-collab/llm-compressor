# M3 EAGLE3 speculative decoding — waves 1 and 2

Wave 1: window `m3-specdec-eagle3/20260727T061506Z`, 4 arms.
Wave 2: window `m3-specdec-eagle3/20260727T064934Z-wave2`, 6 arms.
Phase D: window `m3-specdec-eagle3/20260727T073533Z-phaseD`, 2 arms × 10 cells.
Phase E: window `m3-specdec-eagle3/20260727T084526Z-format`, 2 arms × 4 cells.
Phase F: window `m3-specdec-eagle3/20260727T092342Z-bf16ref`, 2 arms × 4 cells.
All arms rc=0, every gate passed. Design + decision rule:
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

## Raw evidence

**Wave 2** — `/mnt/nfs/hoangduy/results/m3-specdec-eagle3/20260727T064934Z-wave2/`:
6 arms (`natural-k{0,3}`, `load-k{0,3}`, `lowconc-k{0,3}`), each with `client.log`,
`serve.log`, `spec-boot.log`, `backend-attestation.json`, and per-cell
`metrics/<cell>-{pre,post}.txt` from which every acceptance figure above is a
counter delta. Window-level `aggregate.json`, `arm-provenance.txt`,
`actual-commit.txt`. Regenerate the tables with:

```
pipeline/specdec_wave2_aggregate.py --root <window> --out-json <window>/aggregate.json
```

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
