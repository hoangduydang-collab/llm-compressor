# Making DeepSeek-V4-Flash faster on SGLang — what we found, what we changed, what's next

**Audience:** collaborators picking this work up or reviewing it. Plain language,
every metric defined at first use, and every number links to the engineering
detail in
[`DSV4_FLASH_SGLANG_SERVING_PERF_REPORT.md`](DSV4_FLASH_SGLANG_SERVING_PERF_REPORT.md).

**Status as of 2026-08-24:** two levers measured, neither adopted. No quality
evaluation has been run on either. Nothing here is a production recommendation.

**Contents:** [1 The objective](#1-the-objective-and-how-the-two-metrics-are-weighted) ·
[2 Why we re-measured everything](#2-why-we-had-to-re-measure-everything-from-scratch) ·
[3 Where the time goes](#3-axis-1--where-the-time-actually-goes) ·
[4 The fix: prefill CUDA graphs](#4-the-fix-we-measured-prefill-cuda-graphs) ·
[5 The next free thing](#5-the-next-free-thing-to-try-fusing-prefill-into-decode-steps) ·
[6 Axis 2: the MoE kernel](#6-axis-2--the-moe-kernel) ·
[7 The measurement lesson](#7-the-lesson-we-most-want-you-to-inherit) ·
[8 The remaining ladder](#8-the-remaining-ladder-ranked-by-what-it-costs) ·
[9 How to read our numbers](#9-how-to-read-our-numbers)

> **Provenance of this narrative.** The sections below are in *logical* order,
> which is not the order we discovered things. Prefill CUDA graphs (§4) were
> measured first, as a fix for a prefill-overhead defect. The interference model
> in §3.2 came later, out of a problem with the *kernel* work in §6 — and it then
> turned out to explain the graph experiment's own results, which it had not been
> built from. We think that makes the model more trustworthy, not less, and we'd
> rather say so than present a tidier story than the one that happened.

---

## 1. The objective, and how the two metrics are weighted

Two metrics, weighted **roughly equally**:

| metric | what it means | rank |
|---|---|---|
| **Output speed** — per-user decode rate, `1 / ITL` | ITL (inter-token latency) is the gap between streamed tokens for **one** request. `1/ITL` is how fast text appears to a single user | co-equal |
| **TTFT** — time to first token | how long a user waits before anything appears | co-equal |

⚠️ **Revised 2026-08-24.** Earlier documents in this campaign — including the
prefill-graph pre-registration preserved verbatim in §7.2 — ranked output speed
**primary** and TTFT explicitly **subordinate**. That strict ranking is
withdrawn. **No measured conclusion moves:** the pre-committed ITL veto passed
anyway, and as it turns out *neither* measured lever trades one metric for the
other — §6's kernel improves output speed with TTFT a wash, and §7's graphs
improve both. What changes is the emphasis: §7's TTFT half is a first-class
result, not the consolation half. Most measurement effort still went into output
speed, because it is the harder of the two to move.

**Target deployment:** DeepSeek-V4-Flash-0731 (MXFP4-quantized experts) on
SGLang v0.5.17, one node of 8×H100, tensor-parallel 8, `max-running-requests 64`,
1 Mi context, KV cache in FP8. No speculative decoding.

---

## 2. Why we had to re-measure everything from scratch

**The immediate predecessor is this same model on vLLM**, not the MiniMax-M3
work. CA production cut this pool from vLLM `0.26.1rc1.dev77` to SGLang v0.5.17
on **2026-08-17**, so `opprof`'s `ds-v4-flash/` tree is now historical — it
describes an engine serving no traffic. Both of its baselines already disown this
engine in their own words (*"Nothing here transfers from or to SGLang"*;
*"7 of 12 configuration dimensions differ"*). This is a **new campaign, not a
refresh**, and the trees are kept apart so no table ever mixes two engines'
shares.

**The engine swap reorders the problem rather than rescaling it.** On this same
architecture, prior SGLang data put **58.87%** of prefill on
`ncclDevKernel_AllReduce_Sum_bf16_RING_LL` where vLLM's own
`multimem_all_reduce_kernel` measured **8.89%**. Same model, same node class, a
different bottleneck ranking. Expect the target list **reordered, not rescaled**.

| carries from the vLLM campaign | does not carry |
|---|---|
| Model shape — 43 layers, 256 experts, `num_experts_per_tok` 6, `index_topk`/`index_n_heads` 512/64 | The **`humming` NO verdict** — it was arithmetic over vLLM-measured shares, and `moe_backend` lives in vLLM's `KernelConfig`; SGLang's `--moe-runner-backend` is a different flag in a different tree |
| DSpark per-position acceptance (81.1 / 67.1 / 59.8 / 59.1 / 56.9%, then 14.1 / 8.2) and `dspark_block_size` 5 — a property of the checkpoint's draft layers 40–42 | The comm-instability finding (BS 1 rounds up to 9131 ms on `vllm::cross_device_reduce_1stage`) — that is a vLLM kernel |
| Experts MXFP4, dense + attention FP8 | The BS 1 → 265.0 tok/s / ITL 9.2 ms figure — **should not be quoted at all**; its two columns disagree by ~2.4× and were never reconciled |
| Method — clock-locking prerequisite, steady-state gates, nsys 2026.4.1 pin | The profiled context window — the vLLM campaign pinned 262 144 against production's **1 048 576**, so long context was under-sampled ~4× |

🔴 **Two predecessors, opposite conclusions, and both wrong here.** The vLLM
campaign on *this* model concluded **NO** on `humming`. The MiniMax-M3 campaign —
a different model, a different quant scheme (GPTQ W4A8, so humming's **A8**
path), a different opponent (CUTLASS) and a different routing regime (128 experts
top-4 → 2.0 tokens/expert vs our 256 top-6 → **1.5**) — concluded **yes at every
concurrency**. The measured answer here is neither: a **U**, and only on the
newer library version (§6). Carrying the vLLM verdict forward would have left
**+22.3% at conc 1** unclaimed; carrying M3's forward and flipping the flag on
the shipped version would have shipped **−13.4% per-user at conc 64**.

So both axes below were measured on the real engine, the real model and the real
production argument list, with no profiler attached and GPU clocks left alone —
i.e. in production's own operating conditions.

---

## 3. Axis 1 — where the time actually goes

Two separate problems. One is ours and unusual; one is universal. **The first
makes the second roughly 2.6× worse than it has to be**, which is why they are
easy to confuse and important to separate.

### 3.1 Problem one, ours: prefill is running at 2.6× its own compute floor

Profiling the prefill path showed each 2048-token forward pass taking **196.6 ms
against a 75.3 ms compute floor** — a fixed **~121 ms per forward** that is not
arithmetic at all. It is launch overhead: thousands of small GPU kernel launches
that a CUDA graph is designed to collapse into one.

The cause is a rule in SGLang that **switches prefill CUDA graphs off for this
model family**. The rule's own source comment says the model *"is
BCG-compatible but introduces heavy memory pressure"* — so it is a **memory
guard, not a correctness guard**. That distinction matters and §4 returns to it.

### 3.2 Problem two, universal: on a busy server, ITL is not a decode metric

This is the part that surprised us, and it changed how we run every experiment.

SGLang, by default, does **not** mix prefill and decode work in the same step —
a step is either one or the other. So while the server prefills someone else's
prompt, **every request currently generating text is completely stopped.** Those
stops land inside the gaps between your tokens, which is exactly what ITL
measures. The result is an identity:

```
ITL  =  decode step time  +  (decode batch ÷ output length) × prefill forward time
                             └──────────── other people's prefill ────────────┘
```

The second term's coefficient is *request turnover*, so it depends entirely on
the traffic shape:

| traffic shape | coefficient |
|---|--:|
| short answers (2048-token prompt, 128-token reply) | 1 / 128 |
| reasoning (1000-token prompt, 4000-token reply) | 1 / 4000 |
| **ratio** | **31×** |

Measured on the same node, same kernel, same conditions — only the shape
differing:

| concurrency | ITL, short-answer traffic | ITL, reasoning traffic |
|--:|--:|--:|
| 1 | 7.67 ms | 7.59 ms |
| 16 | 20.36 | 9.63 |
| **64** | **53.45** | **14.35** |

The two agree to 1% at concurrency 1 — where there is no other request to
interrupt you — and diverge **3.7×** at concurrency 64. Decode work per token is
identical in both columns. The whole gap is other requests' prefill.

### 3.3 How the two problems compound

At concurrency 64 on short-answer traffic, of a measured **53.45 ms** ITL:

| component | |
|---|--:|
| actual decode work | ~10.5 ms |
| other requests' prefill — **real compute** | ~16.5 ms |
| other requests' prefill — **pure launch overhead** (§3.1) | **~26.5 ms** |

**80% of what we were calling "inter-token latency" was other requests' prefill,
and 62% of that was overhead that shouldn't exist.** Fix §3.1 and the universal
problem shrinks by more than half — without touching the scheduler, the topology
or the hardware.

### 3.4 This is a well-known phenomenon, and we are on the wrong side of the defaults

Worth saying plainly, because it means **nothing here needs inventing**:

* **Sarathi-Serve (OSDI '24)** names it **"generation stalls"** and makes 99th-
  percentile time-between-tokens its headline metric. Their fix — chunked
  prefills plus stall-free scheduling — buys 2.6–3.7× serving capacity under
  latency targets, and they report that stall-free batching alone is about half
  of the latency gain.
* **DistServe (OSDI '24)** opens on the same observation: colocating the two
  phases "leads to strong prefill-decoding interferences", and separating them
  onto different GPUs buys 7.4× more requests or 12.6× tighter latency targets.
* **vLLM makes the mitigation its default.** With chunked prefill on, its
  scheduler batches all pending decode work *before* any prefill, so "running
  generations always advance at least one token per step". An engine does not
  make something non-optional for a rare problem.
* **SGLang leaves it opt-in.** Its maintainers: *"By default, it does not mix
  prefill and decode"*, and `--enable-mixed-chunk` *"can help reduce the
  inter-token latency as described in that paper"*.

So our configuration sits on the unmitigated side of a switch the engine's own
developers describe as an inter-token-latency fix. That is §5.

**What is genuinely unusual about our case,** and worth flagging so "known
problem" doesn't become a reason to stop looking:

1. In the literature the stall is mostly *real prefill compute*. Ours is 2.61×
   its floor — we have an extra, removable inflation source stacked on the
   textbook one.
2. Our decode is exceptionally **starved** — 0.66–1.5 tokens per expert, against
   dense 7B–34B models in the published work. That makes the fusion mitigation in
   §5 *more* promising for us than for them, and nobody has measured that case.

*(How we know the model in §3.2 is right, rather than a story that fits: it
reproduces the ITL ladder's shape, it predicts TTFT it was never fitted to
within the measured p50–p95 band at 3 of 4 points, and it predicts — structurally,
not by fitting — that §4's lever has exactly zero ITL effect at batch 1 and a
monotonically growing effect above it. All three hold. Detail in the engineering
report.)*

---

## 4. The fix we measured: prefill CUDA graphs

**Lever:** `--cuda-graph-backend-prefill breakable`, plus
`--mem-fraction-static 0.80` to pay for it. Five runs, each one existing because
the previous left a hole.

### 4.1 What it buys

Per 2048-token forward pass, against the 75.3 ms compute floor:

| | ms/forward | × floor |
|---|--:|--:|
| today (graphs off) | 196.6 – 212.7 | 2.61× |
| **graphs on** | **89.8 – 105.0** | **1.21×** |

~107 of the ~121 ms fixed overhead is gone. Across the full serving grid — 11
shape × batch combinations, all passing an internal consistency gate:

| | batch 1 | batch 64 |
|---|--:|--:|
| **TTFT improvement** | **3.97×** | 2.45× — *shrinking* |
| **ITL improvement** | **1.00×** | **1.92×** — *growing* |

The two objectives move in **opposite directions** along batch size. Under equal
weighting that is a feature, not a trade: the lever pays in TTFT where
concurrency is low and in output speed where it is high, so it is never idle. Per-user output speed at batch 64:
**15.9 → 30.5 tokens/s.**

Batch 1 reading exactly 1.00× is not a disappointment — it is the control that
proves the mechanism. With one request in flight nothing is prefilling during its
decode, so there is no stall to remove.

### 4.2 The trade-off, stated in full

**It is paid for out of the KV cache**, and we pre-committed to a rule that a KV
loss must be shown not to bind. **That rule failed:**

| | KV pool (tokens) | ÷ 64 requests |
|---|--:|--:|
| today | 6 473 216 | 101 144 |
| graphs on at `mem-fraction-static 0.80` | 5 675 520 | **88 680 (−12.3%)** |

Above roughly **89 000 tokens of context**, the server runs out of KV pool before
it runs out of request slots — so the loss is *not* hidden by the concurrency
cap. At full 1 Mi context, maximum concurrency goes **6 → 5**.

We tried to get the win for free. **It did not work.** A variant using only 4
captured size buckets instead of 42 holds KV at ±0.0%, but any prompt whose last
chunk is small **loses the graph entirely and falls back to the slow path** —
because the engine refuses a graph when padding would more than double the token
count. Measured penalties up to **6.1× worse** on short prompts. Since almost
every real prompt has a partial last chunk, that variant is rejected.

The recommended configuration survives the top of a 1 Mi context window with
8.27 GiB of headroom — essentially unchanged from today's 8.52 GiB — because
funding the graph pool from KV leaves the activation reserve intact. That makes
`mem-fraction-static 0.80` a safety property here, not only a cost.

### 4.3 Two honest caveats

* **The 1.92× is measured on short-answer traffic, which is the shape that
  flatters this lever most** (§3.2 — that shape charges prefill to ITL 31× more
  heavily). Our model predicts the gain on reasoning traffic is closer to
  **1.13×** — still real, still ~3× larger than the kernel lever in §6, but not
  92%. **This is a prediction, not a measurement**, and testing it is the top
  item in §8.
* **No quality evaluation exists.** Graphing prefill can perturb results at the
  last bit in principle. Upstream ran a reasoning benchmark on their own
  graphs-on configuration and it held steady; **we have not run anything.**

---

## 5. The next free thing to try: fusing prefill into decode steps

`--enable-mixed-chunk` makes a single step carry both a prefill chunk *and* the
decode batch, so generating requests keep emitting tokens while someone else's
prompt is being processed. One flag, no extra hardware, and untested by us.

**The honest null hypothesis first:** if a fused step costs exactly what the two
separate steps cost, this changes *nothing* about average ITL — the same work
happens on the same GPU, just relabelled. Fusion helps only to the extent the
combined step is cheaper than the sum, which is a hardware-utilization claim.

**Why we expect it to be, here.** Our decode steps are extremely
memory-bandwidth-bound — at 0.66–1.5 tokens per expert, the MoE matrix
multiplications are almost pure weight-streaming with nearly no arithmetic reuse.
Prefill tokens fused into that step **ride weight loads the decode step is
already paying for**. This is the most favourable possible starting point for
the mitigation, and more favourable than the dense models it was published on.

**Bounds, rather than a prediction:**

| per prefill chunk | cost |
|---|--:|
| today, as a full stall | **196.6 ms** |
| fused, no utilization benefit (the null) | 196.6 — no change |
| fused, weight loads fully reused | → the compute floor, **~75–90 ms** |

That lower bound is **the same destination §4 reaches.** Two different routes to
one place: graphs remove the launch cost, fusion avoids launching a second
forward at all. **So they overlap — their gains must not be added together.**

What is near-certain either way is the **tail**: today a generating request eats
a ~196 ms hole whenever a prefill lands, and fusion bounds that. Our
short-answer-traffic tails are severe (95th-percentile TTFT 12.5 s against a
5.3 s median).

---

## 6. Axis 2 — the MoE kernel

**Lever:** `--moe-runner-backend`, choosing between `marlin` (SGLang's default
for this model) and `humming`, a Hopper-native MoE kernel library.

**Why this needed re-measuring** is §2, and this axis is where that argument
gets its proof: on MiniMax-M3 this kernel won at every concurrency; here the
curve is a **U**.

### 6.1 Design

Each arm runs **marlin → humming → marlin in a single pod**, so clock and
thermal drift between boots is *measured* rather than assumed — and the
comparison refuses any effect smaller than the drift it measured. Measured drift:
**0.00–0.49%** over two hours, which is what makes 2% effects readable.

Engagement is proven per phase, never assumed, because the engine's kernel
dispatch has **no error branch** — an unrecognised backend name silently returns
the default and the run looks successful while measuring the incumbent. So every
phase asserts the loaded method class on all 8 ranks, plus the loaded library
version and path, before any number is believed.

### 6.2 Results — and the headline is about the version, not the flag

Reasoning traffic (1000-token prompt / 4000-token reply), all points resolving at
20–282 standard errors:

| concurrency | marlin | humming **0.1.10** *(what SGLang ships)* | humming **0.1.13** | cost of the pin |
|--:|--:|--:|--:|--:|
| 1 | 7.597 ms | 6.308 | **6.208** | +1.6% |
| 8 | 8.137 | 8.549 | 8.420 | +1.5% |
| 16 | 9.637 | 10.575 | 9.854 | +7.3% |
| 32 | 11.633 | 12.846 | **10.835** | +18.6% |
| **64** | 14.353 | **16.562** | **13.711** | **+20.8%** |

Per-user output speed at production concurrency 64:

> marlin **69.72 tok/s** → shipped 0.1.10 **60.43 (−13.4%)** → 0.1.13 **73.02 (+4.7%)**

**Read that middle number carefully.** SGLang pins `humming-kernels==0.1.10`.
With that version, **turning the flag on is a regression at every concurrency
above 1.** Anyone who enables `--moe-runner-backend=humming` on stock
SGLang v0.5.17 at production concurrency today makes output speed **13.4%
worse**. The win exists only in **0.1.13**, so a version bump or a vendored wheel
is **mandatory, not an optimisation** — and the cost of the pin is *monotone in
concurrency*, worst exactly where production runs.

At concurrency 1 the win is large and has now been measured three times across
two shapes and two pods: **+22.3% output speed**, and 11 seconds off an
8000-token reasoning reply.

### 6.3 Why the curve is a U

Splitting the kernel work in two explains it completely:

* **The down-projection improves at every batch size, and improves
  monotonically** — up to **2.55× faster** at batch 64. This was predicted from
  reading the library's source before the run: the old version was running a
  work-splitting optimization that had nothing to split, as pure overhead, at 25%
  occupancy.
* **The gate/up projection is the regression.** The new version's tile shape is
  a loss of up to 1.57× through the middle of the range and only becomes a win at
  batch 64. Its tuning ladder's next rung needs a batch of **239**; we give it
  **1.5 tokens per expert**. It was tuned for a regime we are nowhere near.

**The regression is the tunable half.** Keeping the new down-projection setting
while forcing the gate/up tile back to the old one is a one-line override —
untested, one GPU, no upgrade required. It could plausibly remove the concurrency
8–16 loss without touching the win.

### 6.4 What is clear, and what is not

**Clear:**

* marlin is the right default *today*, because the version SGLang ships makes the
  alternative worse.
* humming **0.1.13** is faster at concurrency 1, 32 and 64, and slower at 8–16.
  All five points are statistically decisive.
* The pin costs up to **20.8%** of decode ITL. That is a single-instrument
  measurement, not an inference across datasets.
* Cross-pod reproducibility on this shape is **≤0.33%** — two boots, three hours
  apart. Our instrument is trustworthy at the precision we are quoting.

**Not clear — and this is the honest blocker on adoption:**

* **We have no stability or quality evidence for 0.1.13 at all.** We have
  performance and engagement proofs. We have not run a quality evaluation, a
  long-soak, or an error-rate check on *either* backend. The upstream changes are
  authored by a maintainer of another major inference engine, which is
  reassuring, and reassurance is not evidence.
* **Adopting 0.1.13 means carrying something.** Either a side-installed library
  or a landed upstream pin bump. Whichever it is, **the deployment must assert
  the loaded version**, because a silent fallback to 0.1.10 does not raise an
  error — it is just 20.8% slower.
* **The concurrency 8–16 loss is real** (2.4–3.6%, well above noise). Small, but
  do not quote the concurrency 1 and 64 numbers without it.
* **One traffic shape, one node.** And nothing about the `grouped` kernel
  scheduling mode, which is untested here.
* An unreleased upstream fix titled *"Fix SM90 indexed A16 large-M scheduling"*
  targets exactly the gate/up weakness in §6.3 and has not been evaluated.

---

## 7. The lesson we most want you to inherit

Not a number. **Our first attempt at axis 2 produced two findings, and both had
the wrong sign.**

| concurrency | first ladder said | corrected ladder says |
|--:|---|---|
| **16** | 0.916× — humming **faster** | 1.024× — humming **slower** |
| **32** | 1.064× — humming **slower** | 0.930× — humming **faster** |

Both wrong points were exactly the two it reported as findings. Both **cleared
their own stated drift and statistical bars.**

The cause is §3.2. That ladder ran on short-answer traffic, where **59–80% of the
measured "ITL" was other requests' prefill** — and the kernel under test *also*
changes prefill time, in the opposite direction from its decode effect. So the
experiment was measuring a blend of two effects with opposing signs and calling
it a decode result. On top of that, the second term makes ITL depend on the
*actual* decode batch, which on that shape is a queue-derived quantity (28 of a
nominal 64), making the metric **11× more sensitive** to run-to-run load
variation.

**Clearing a drift estimate and a standard error was not sufficient.** What fixed
it was choosing a shape where the metric measures what its name says, running the
control backend twice in one pod so drift is measured, and writing the decision
rule down *before* the data landed. Everything quoted in §6 was produced that
way; the earlier ladder's headline is retired.

A corollary worth carrying: **under colocation, ITL is not a property of the
decode engine — it is a property of the traffic mix.** That is why every ITL
number in this document has a shape attached, and why you should distrust any
that doesn't.

---

## 8. The remaining ladder, ranked by what it costs

Deliberately ordered by cost, not by attractiveness. **Every item is untested.**

| # | lever | what it does to the problem | cost | expected prize |
|--:|---|---|---|---|
| 1 | **Fuse prefill into decode** (`--enable-mixed-chunk`) | stops charging prefill to ITL, no new hardware | **one flag** | unknown mean (bounded 0 → same as #2); tail almost certain |
| 2 | **Prefill CUDA graphs** (§4) | makes the prefill work genuinely cheaper | one flag + **12.3% of KV** | measured: 1.92× ITL on short-answer traffic, predicted ~1.13× on reasoning |
| 3 | **Gate/up tile override** (§6.3) | targets the kernel's residual weakness | **one line**, 1 GPU | could erase the concurrency 8–16 loss |
| 4 | **humming 0.1.13** (§6) | faster MoE kernel | a carried dependency + version assertion | +4.7% at concurrency 64, +22.3% at 1 |
| 5 | **Upstream large-M scheduling fix** | same target as #3, upstream | version bump, unreleased | unquantified |
| 6 | **Prefill/decode disaggregation** | moves prefill onto *other* GPUs entirely | **separate GPU pools + network fabric**; changes the topology | removes whatever remains after #1/#2 |
| 7 | **Speculative decoding (DSpark)** | a different axis entirely — fewer decode steps | a study, not an arm; per-traffic-class tuning | Largest known prize. Acceptance rates **carry** from the vLLM campaign (checkpoint property); the **1.21–2.53×** figure is M3's and is indicative only |

**Two notes on the ordering, because it is the substance of the recommendation.**

**Why disaggregation is #6 and not #1.** It is the textbook fix and it works —
but it buys output speed with *hardware*, and its prize **shrinks as you do the
cheap items first**. At concurrency 64 today, disaggregation would remove ~43 ms
of inflation; after prefill graphs, ~20 ms. It also changes the topology, which
would invalidate both of our measurement baselines. It remains the only thing
that makes ITL a property of the decode engine rather than of the traffic mix —
which is a real architectural argument, just not a first move.

**Why speculative decoding is listed last but is probably worth the most.** It is
untouched here, and it is the one lever whose *tuning inputs actually carry*
across the engine move — DSpark acceptance is a property of this checkpoint's own
draft layers 40–42, so the vLLM campaign's per-position rates (81.1 / 67.1 / 59.8
/ 59.1 / 56.9%, then a cliff to 14.1 / 8.2) and its `dspark_block_size` of **5**
transfer, and SGLang v0.5.17 exposes `spec_accept_length`, `spec_accept_rate`,
`spec_num_draft_tokens` and `spec_num_steps` in `/metrics` so acceptance is
directly observable. ⚠️ Note the vLLM pool ran **k=7** where the acceptance cliff
says **5**.

⚠️ **What does not carry is the size of the end-to-end win.** The only figure we
have is **1.21–2.53× per-user decode** on MiniMax-M3 (vLLM, goal 2h) — indicative
of the order of magnitude, and nothing more: draft depth per traffic class was the
dominant lever there and would have to be re-derived here.

---

## 9. How to read our numbers

* **Every ITL figure has a traffic shape attached, and it matters** — up to 3.7×
  at the same concurrency (§3.2). A number without its shape is not usable.
* **Measured / modelled / predicted are marked separately.** §4.1, §4.2 and §6.2
  are measured. §3.2's identity is a model, validated three ways. §4.3's 1.13%
  and §5's bounds are **predictions** and are labelled as such.
* **Nothing here is a public benchmark score.** These are internal paired
  comparisons, valid for configuration-to-configuration decisions and not
  comparable to any published leaderboard result.
* **No adoption decision has been applied to any number**, and **no quality
  evaluation has been run on either lever.** Measurement and adoption are
  separate decisions, and one of them has not been made.
* **Raw evidence is retained** — per-request records for every point, full
  resolved server configuration for every boot (so nothing rests on the flag we
  *passed*), and complete engagement proofs. Statistics in the engineering report
  were recomputed from the per-request records rather than trusted from summaries.
* **We have not filed anything upstream.** The finding that SGLang's pinned
  kernel version costs up to 20.8% of decode speed is reportable and would
  likely be welcome; opening that conversation is a decision for the team, not
  something we have done.

---

**Engineering detail, full tables, method defects and provenance:**
[`DSV4_FLASH_SGLANG_SERVING_PERF_REPORT.md`](DSV4_FLASH_SGLANG_SERVING_PERF_REPORT.md).

**External sources for §3.4:** Sarathi-Serve, OSDI '24
([paper](https://www.usenix.org/system/files/osdi24-agrawal.pdf)) · DistServe,
OSDI '24 ([paper](https://www.usenix.org/system/files/osdi24-zhong-yinmin.pdf)) ·
[vLLM optimization docs](https://docs.vllm.ai/en/stable/configuration/optimization/) ·
[SGLang discussion #1163](https://github.com/sgl-project/sglang/discussions/1163) ·
[SGLang PD-disaggregation docs](https://docs.sglang.io/docs/advanced_features/pd_disaggregation)
