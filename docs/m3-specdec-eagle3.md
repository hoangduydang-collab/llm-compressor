# M3 EAGLE3 speculative decoding — waves 1 and 2

Wave 1: window `m3-specdec-eagle3/20260727T061506Z`, 4 arms.
Wave 2: window `m3-specdec-eagle3/20260727T064934Z-wave2`, 6 arms.
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

Each extra draft token costs a consistent ~0.11–0.12× of a decode step, which is
why **k=3 is the optimum and k=5 is a net loss**: it buys +0.10 accepted length
(2.45 → 2.55) for +0.12× step cost, and its per-position rates have collapsed to
0.10 by the fifth token. Nothing deeper than k=3 is worth serving.

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
- Expect **1.36×–2.15× depending on workload mix** (phase D, below).
- k=5 remains a net loss (wave 1); nothing deeper than k=3 is worth serving.
- Untested past conc 64 on a single node.

## Phase D — length × entropy on nvidia/SPEED-Bench (in flight)

Window `20260727T073533Z-phaseD`, 2 arms. Isolates *content domain* from *prompt
length* using NVIDIA's purpose-built spec-dec benchmark: fixed-ISL buckets
(1k/8k/32k) crossed with entropy tier, temp 0.6, no `ignore_eos`, `max_tokens`
2048. The 1k and 8k conc-1 cells have landed:

| cell (conc 1) | ISL | control speed | k3 speed | × | accepted len | ITL control → k3 | TTFT control → k3 |
|---|---|---|---|---|---|---|---|
| 1k **low** entropy (code, sorting) | 1011 | 137.31 | 294.77 | **2.15×** | 3.100 | 7.283 → 3.406 | 128.9 → 127.5 |
| 8k **low** entropy | 8080 | 136.74 | 296.90 | **2.17×** | 3.106 | 7.313 → 3.378 | 420.8 → 420.1 |
| 1k **high** entropy (creative writing) | 990 | 137.38 | 186.22 | **1.36×** | 1.850 | 7.279 → 5.455 | 133.1 → 136.0 |
| 8k **high** entropy | 8183 | 136.70 | 178.95 | **1.31×** | 1.779 | 7.315 → 5.729 | 456.1 → 466.9 |

Two axes separate cleanly, and the control is an almost exact internal check across
all four cells (136.70–137.38 tok/s, ITL 7.279–7.315 ms) — the baseline is
indifferent to both length and subject matter, so the entire spread is drafter
behaviour.

**Content domain is the dominant axis: 1.78 → 3.11 accepted length, +75%** — larger
than output shape (+33%) and far larger than temperature (+4%). ShareGPT's mixed
traffic at 1.81× sits between the two extremes, where a blend should.

**Length is flat, now for the right reason.** Within a tier, an 8× longer prompt
moves acceptance by +0.006 (low) and −0.071 (high). Wave 1 also found length flat,
but only on synthetic random tokens, where one could argue there was nothing worth
copying. These prompts are real code and real prose at 8k, the ideal setup for a
drafter to lift spans out of context, and it still buys nothing — because EAGLE3's
drafter conditions on the target's hidden state, not on retrievable prompt text.
The 8k-low cell also shows **no prefill penalty at all** (420.8 → 420.1 ms).

### Output-budget censoring (measured, affects interpretation)

`max_tokens=2048` truncated a large share of responses, so these are **not**
natural-stopping lengths:

| cell | requests at the 2048 cap (control / k3) |
|---|---|
| 1k-low | 15% / 20% |
| 8k-low | 60% / 60% |
| 1k-high | 82.5% / 82.5% |
| 8k-high | 92.5% / 92.5% |

The ratios and the tier contrast survive this: censoring is identical between arms
in every cell, both arms emit the same token counts, and at conc 1 the speed ratio
is just the ITL ratio. Nor is this the wave-2 shape inflation — `ignore_eos` forces
generation *past* its natural stop (which is what made drafting easy, +33%),
whereas truncation stops reading *early*, so acceptance over the retained prefix
(~80k tokens per cell) is genuine natural-generation acceptance. What the data does
**not** support is any claim about per-tier natural response length; a higher budget
would be needed for that.

Remaining cells: both 32k cells at conc 1, and the 1k/8k crossing at conc 10.

Harness comparability: these are **not** comparable to published SPEED-Bench
scores. ~42–56% of the public parquet is masked
(`FULL BENCHMARK DATA SHOULD BE FETCHED FROM THE SOURCE USING SPECDEC_BENCH`) and
aiperf's loader does not filter it, so `pipeline/stage_speedbench.py` stages the
clean subset and the launcher gates on its hashes; the `mixed` tier is 512/512
masked and absent entirely; and the serving stack is our own W4AFP8 + Humming.

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

**Phase D** — `/mnt/nfs/hoangduy/results/m3-specdec-eagle3/20260727T073533Z-phaseD/`:
arms `phaseD-k{0,3}`, plus `speedbench-manifest.txt` recording the staged prompt
hashes and per-cell token statistics the launcher gated on.

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
