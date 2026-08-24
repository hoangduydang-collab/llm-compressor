# Faster serving for DeepSeek-V4-Flash

**Status: measured, not adopted.** Two independent speed-ups have been measured on
real production hardware. Neither is switched on.

Updated 2026-08-24. Engineering detail:
[`DSV4_FLASH_SGLANG_SERVING_PERF_REPORT.md`](../DSV4_FLASH_SGLANG_SERVING_PERF_REPORT.md)
· collaborator brief:
[`DSV4_FLASH_SGLANG_COLLABORATOR_BRIEF.md`](../DSV4_FLASH_SGLANG_COLLABORATOR_BRIEF.md)

**Contents:** [01 What we are optimising](#01--what-we-are-optimising) ·
[02 Results](#02--results) · [03 What it costs](#03--what-it-costs) ·
[04 Risks](#04--risks) ·
[05 Why this needed measuring from scratch](#05--why-this-needed-measuring-from-scratch) ·
[06 What's next, by cost](#06--whats-next-by-cost) · [07 Evidence](#07--evidence) ·
[Appendix — where the time actually goes](#appendix--where-the-time-actually-goes)

---

## 01 · What we are optimising

This is a **serving-speed** study on a model we did not quantize ourselves
(DeepSeek-V4-Flash), running on the SGLang inference server, 8 H100 GPUs. It is
separate from the in-house quantization programme; nothing here changes a model
file. Every gain comes from how the model is *served*.

Two user-visible metrics, and the priority between them is fixed:

| metric | what a user experiences | priority |
|---|---|---|
| **Output speed** | how fast text streams out once it starts, per user | 🥇 **first** |
| **Time to first token** | how long the user stares at nothing before text appears | 🥈 second |

**"Second" is meant literally:** we do not accept a change that makes text appear
sooner but then stream more slowly.

---

## 02 · Results

Two levers. Both measured on the production configuration, in production's own
operating conditions — no profiler attached, GPU clocks left alone.

### Lever 1 — swap the mixture-of-experts kernel: `marlin` → `humming`

The model's experts are 4-bit, and the GPU code that multiplies them is
selectable via SGLang's `--moe-runner-backend` flag:

* **`marlin`** — what we run today, and SGLang's default for this model.
* **`humming`** — an alternative Hopper-native kernel library
  ([inclusionAI/humming](https://github.com/inclusionAI/humming), open source).
  **Two versions matter.** SGLang v0.5.17 pins `humming-kernels==0.1.10`;
  **0.1.13** contains a re-tuning by an outside contributor and is the fast one.

| | marlin → **humming 0.1.13** |
|---|--:|
| Output speed, **single user** | **+22.3%** (131.7 → 161.1 tokens/s) |
| A long reasoning answer (8,000 tokens) | **11 seconds faster** — 61.1 s → 50.0 s |
| Output speed at **full load** (64 concurrent users) | **+4.7%** (69.7 → 73.0 tokens/s) |
| Time to first token | unchanged |

🔴 **But the cheap version of this — just flipping the flag — is a regression.**
With the version SGLang actually ships, `humming` is **slower than marlin** at
every load above a single user:

| 64 concurrent users | output speed per user | vs today |
|---|--:|--:|
| `marlin` — today's default | 69.7 tokens/s | — |
| `humming` **0.1.10** — what SGLang pins | **60.4** | **−13.4%** ❌ |
| `humming` **0.1.13** — the re-tuned version | **73.0** | **+4.7%** ✅ |

The win is real but it lives entirely in 0.1.13, and the gap between the two
versions is **up to 20.8%** of output speed — worst precisely at the load
production runs at. Setting `--moe-runner-backend=humming` without also
upgrading the library would make the product measurably worse.

### Lever 2 — enable prefill GPU graphs

The server currently disables a GPU optimisation for this model family, which
leaves a large fixed overhead on every prompt-processing pass. Removing it:

| | today | with graphs on |
|---|--:|--:|
| Prompt-processing time per pass | 196.6 ms | **89.8 ms (2.2× faster)** |
| **Time to first token**, 2,000-token prompt, 1 user | 408 ms | **103 ms (4.0× faster)** |
| **Time to first token**, 32,000-token prompt, 1 user | 3,355 ms | **1,583 ms (2.1× faster)** |
| **Output speed** at 64 concurrent users, short-answer traffic | 15.9 tok/s | **30.5 tok/s (1.9×)** |
| Full 1-million-token context | works | **works** — per-pass 424 → 285 ms |

The output-speed gain here is a side effect worth understanding, because it is
the larger half at production load: when the server processes someone else's
prompt, **every user currently mid-answer is briefly stopped.** Making prompt
processing faster shortens those interruptions.

⚠️ **The 1.9× figure depends on the traffic mix.** It was measured on
short-prompt/short-answer traffic, where interruptions are most frequent. On
long-reasoning traffic our model predicts closer to **1.13×** — still a real gain,
but not 90%. **That prediction is not yet measured**, and measuring it is the top
item in §06.

---

## 03 · What it costs

Lever 1 costs a **dependency**, not performance: 0.1.13 is not what SGLang pins,
so adopting it means either carrying our own copy of the library or getting the
upstream project to move its pin.

Lever 2 costs **memory that would otherwise hold conversation history**, and that
converts directly into a capacity limit:

| | today | with graphs on |
|---|--:|--:|
| Conversation-history capacity | 6.47 M tokens | **5.68 M (−12.3%)** |
| Max simultaneous users at very long context (1 M tokens) | 6 | **5** |
| Above ~89,000 tokens of context | capacity limited by user cap | **limited by memory instead** |

Below ~89,000 tokens of context this costs nothing observable, because the
server's 64-user limit binds first. Above it, the cost is real. **We attempted a
version of this lever with no memory cost and it failed on measurement** — it
degrades badly on the short prompts typical of chat traffic, so it is rejected.

We also verified the safety side: with graphs on, the server still survives a
full 1-million-token context with essentially the same memory headroom as today
(8.27 GB vs 8.52 GB).

> ### Two things we need from you
>
> 1. **Accept carrying a non-standard kernel version, or push for an upstream
>    pin bump.** Mandatory for lever 1's win — there is no third option that
>    keeps the gain. A carried copy is ongoing maintenance; an upstream bump is
>    slower but cleaner, and the 0.1.13 changes were authored by a maintainer of
>    another major inference engine, so a bump is not a fork.
> 2. **Tell us the traffic mix to optimise for.** A genuine input we are missing.
>    Lever 2 is worth 1.9× on short-prompt traffic and an estimated 1.13× on
>    long-reasoning traffic, and its memory cost only bites above ~89,000 tokens
>    of context. Both the configuration and the expected gain change with the
>    answer.

---

## 04 · Risks

* **No quality evaluation exists on either lever, and that gates shipping
  both.** Both change how numbers are computed on the GPU. The upstream project
  ran a reasoning-benchmark check on its own graphs-on configuration and results
  held steady; we have run nothing. Neither lever should ship before that is
  closed.
* **Silent degradation on lever 1.** If the 0.1.13 library is ever missing at
  start-up, the server does **not** fail — it quietly falls back to 0.1.10 and
  runs 20.8% slower. Any deployment must assert the loaded version at start-up.
  We have that check built and used it in every experiment.
* **Capacity reduction at long context** from lever 2 (§03).
* **One quoted figure is a prediction, not a measurement** — lever 2's gain on
  long-reasoning traffic (§02). Named as such wherever it appears.
* **A small measured loss inside lever 1.** At moderate load (8–16 concurrent
  users) `humming` 0.1.13 is 2.4–3.6% *slower* than marlin. Real, small, and it
  should be quoted alongside the wins rather than dropped.

---

## 05 · Why this needed measuring from scratch

The predecessor here is **the same model on a different inference server.**
Production moved this pool from vLLM to SGLang on **2026-08-17**, so our earlier
DeepSeek-V4-Flash measurements now describe an engine serving no traffic. Same
checkpoint, same 4-bit expert format, same class of machine — the server changed.

That turns out to **reorder** the problem rather than rescale it:

> ### Changing the engine changes which thing is the bottleneck
>
> On this same model, SGLang spends **58.87%** of prompt-processing time inside a
> single cross-GPU communication step, where vLLM's equivalent step measured
> **8.89%**. Same model, same class of machine, a different top bottleneck. So
> the earlier list of things worth optimising is *reordered*, not rescaled —
> which is why it could not simply be re-run.

| survives the engine change | does not |
|---|---|
| The model's own shape — 43 layers, 256 experts, 6 consulted per token | The earlier **"don't use `humming`"** verdict — it was arithmetic over vLLM-measured time shares, and the setting itself lives in a different codebase |
| Speculative-decoding acceptance rates — a property of the checkpoint's own draft layers | The earlier cross-GPU instability finding — that was a vLLM code path |
| The 4-bit expert / 8-bit dense-and-attention format | The earlier single-user throughput figure — its own two columns disagreed by **2.4×** and were never reconciled, so it should not be quoted at all |
| Our measurement method — clock locking, validity gates, profiler version | |

🔴 **Two predecessors, opposite conclusions, and both wrong here.** Our vLLM work
on *this* model concluded **no** to the alternative kernel. Our earlier work on a
*different* model (MiniMax-M3, a different quantization scheme and a different
rival kernel) concluded **yes, at every load level**. The measured answer on this
engine is neither: yes at single-user and at full load, no in the middle, and
only with the newer library version. **Carrying the first verdict forward would
have left +22.3% single-user speed unclaimed. Carrying the second forward and
flipping the flag would have shipped −13.4% at production load.**

One more gap the earlier campaign could not close: it profiled at a
**262,144-token** context where production runs **1,048,576**, so long-context
behaviour was under-sampled roughly four-fold — which is why §02 and §03 test the
top of the real window. The appendix shows the mechanism behind both levers.

---

## 06 · What's next, by cost

Ordered by what it costs us, not by how attractive it is. Time estimates are
scaled from a measured comparison run that took **2.05 hours** on one 8-GPU node.

| # | next step | cost | what it would settle |
|--:|---|---|---|
| 1 | **Measure lever 2 on long-reasoning traffic**, and both levers together | ~half a day, one 8-GPU node | Replaces §02's predicted 1.13× with a measurement, and tells us whether the two levers add up — nobody has run them together, and production would use both |
| 2 | **Try one more free server setting** (`--enable-mixed-chunk`) that lets the server process prompts *without* fully stopping users mid-answer | ~3 hours, one node | This is the standard fix for the interruption problem — vLLM enables it by default and SGLang does not. One setting, no new hardware. It may overlap with lever 2 rather than add to it |
| 3 | **A one-line kernel tuning change** targeting the 8–16-user loss in §04 | ~1 hour, 1 GPU | Could remove lever 1's only regression — see the appendix for why one line is plausibly enough |
| 4 | **Quality evaluation** | scoped separately, needs the evaluation pipeline | Unblocks shipping either lever |
| 5 | **Speculative decoding** | a study, not an experiment | ⚠️ **Probably the largest prize on the table and completely untouched here.** Unusually, its acceptance rates **do** carry from the vLLM work, because they are a property of this checkpoint's own draft layers — 81% of first guesses accepted, decaying with depth, best at **5** tokens ahead where vLLM production ran 7. SGLang already exposes every metric needed to tune it. For the *size* of the end-to-end gain we only have a different model to go on, where it measured 1.2–2.5× faster output at no quality cost by design — indicative, not transferable |
| 6 | **Splitting prompt-processing and text-generation onto separate GPU pools** | multiple nodes + network configuration; changes the deployment shape | The textbook fix for the interruption problem, and it works — but it buys speed with **hardware**, and its value *shrinks* once items 1–2 land. Price it against what is left, not against today |

---

## 07 · Evidence

Every figure above traces to one of two engineering documents, which carry the
full tables, the statistical confidence for each point, and what is explicitly
*not* established:

* [`DSV4_FLASH_SGLANG_SERVING_PERF_REPORT.md`](../DSV4_FLASH_SGLANG_SERVING_PERF_REPORT.md)
  — measurements, method, limitations.
* [`DSV4_FLASH_SGLANG_COLLABORATOR_BRIEF.md`](../DSV4_FLASH_SGLANG_COLLABORATOR_BRIEF.md)
  — the narrative version for engineers picking the work up.

Confidence, in plain terms: each comparison ran the current configuration
**twice**, before and after the change, in a single session — so drift in machine
conditions is *measured* rather than assumed. It came out at **under 0.5%**,
against effects of 2–22%. Repeating an entire run three hours later on a fresh
boot reproduced it to within **0.33%**. Every run also proves, from the server's
own logs, that the intended code path actually executed on all 8 GPUs — because
the server does not error on a mistyped setting, it silently runs the old path.

**No performance figure here has had a pass/fail threshold applied to it.**
Measuring and deciding are deliberately kept separate.

---

## Appendix — where the time actually goes

*More technical than the sections above. This is what the profiling showed, why
each lever exists, and what the profiling cannot tell us.*

**Why profile at all, when we could just benchmark?** Because a benchmark tells
you *that* something is slow and profiling tells you *which code* is slow — and
because our earlier profiling was done on vLLM, which shares almost no serving
machinery with SGLang. It had to be redone. It also turned out to be the cheap
screening tool: a profiling run costs a fraction of a full benchmark ladder, and
it **correctly predicted the direction of the end-to-end result in 9 of 10
cases** (§A.4).

### A.1 Lever 2's origin: prefill is running at 2.6× its own compute floor

Profiling each prompt-processing pass (2,048 tokens) against the arithmetic that
pass actually has to do:

| | ms per pass | × the compute floor |
|---|--:|--:|
| Today (graphs off) | 196.6 – 212.7 | **2.61×** |
| Graphs on | 89.8 – 105.0 | **1.21×** |

The compute floor is 75.3 ms. So today roughly **121 ms per pass is not
arithmetic at all** — it is the cost of *launching* thousands of small GPU
operations one at a time, which is exactly what a CUDA graph exists to collapse
into a single launch. Turning graphs on removes ~107 of those 121 ms.

The reason they are off is a rule in SGLang that disables prefill graphs for this
model family. Its own source comment says the model *"is BCG-compatible but
introduces heavy memory pressure"* — i.e. it is a **memory guard, not a
correctness guard**, which is why paying for it out of conversation-history
memory (§03) is a legitimate answer rather than a workaround.

### A.2 Lever 1's origin: expert-multiplication time per decode step

Kernel-level time for the expert multiplications in one decode step, measured
directly on the GPU. Lower is better; the last column is the ratio we care about.

| batch | `marlin` | `humming` 0.1.10 | `humming` 0.1.13 | 0.1.13 ÷ marlin |
|--:|--:|--:|--:|--:|
| 1 | 42.60 µs | 20.25 | **16.65** | **0.391×** |
| 8 | 38.29 | 47.07 | 47.37 | 1.237× |
| 16 | 48.88 | 67.04 | 70.04 | 1.433× |
| 64 | 91.76 | 134.82 | **83.36** | **0.908×** |

*(Batch 32 is omitted — it failed its own quality-control check on both sides and
is not quotable.)*

**This is the shape of the whole story: a U.** 0.1.13 wins at both ends and loses
in the middle. And 0.1.10 does *not* have that shape — it degrades steadily
(0.475× → 1.229× → 1.371× → 1.469×), which is why the version SGLang ships is a
regression at load and the re-tuned one is not. In absolute terms, the version
bump alone recovers **2.2 ms per decode step at batch 64** — six times the margin
by which 0.1.13 then beats marlin.

### A.3 Why the U — and why one line might fix the losing half

The expert computation is two separate matrix multiplications, and splitting them
shows two opposite stories. Ratios are 0.1.13 against 0.1.10:

| batch | "gate/up" multiplication | "down" multiplication |
|--:|--:|--:|
| 1 | 0.916× | **0.680×** |
| 8 | *1.338×* | **0.638×** |
| 16 | *1.571×* | **0.524×** |
| 64 | 0.878× | **0.392×** |

* **The "down" multiplication improves everywhere, and improves steadily with
  batch size** — up to **2.55× faster**. This was predicted from reading the
  library's source *before* the run: the old version ran a work-splitting
  optimisation that had nothing to split, as pure overhead, at 25% GPU occupancy.
* **The "gate/up" multiplication is the entire regression.** Its new tile shape
  is a loss of up to 1.57× through the middle of the range. The reason is
  concrete: that tuning ladder's next step up needs a batch of **239**, and our
  model at 64 concurrent users gives each expert about **1.5 tokens**. It was
  tuned for a regime we are nowhere near.

**So the regression is the tunable half.** Keeping 0.1.13's "down" setting while
forcing "gate/up" back to the old tile is a one-line override — which is why
item 3 in §06 costs an hour rather than a study. There is also an unreleased
upstream fix titled *"Fix SM90 indexed A16 large-M scheduling"* aimed at exactly
this, not yet evaluated.

### A.4 What the profiling can and cannot tell us

**Can:** the direction. Against the end-to-end serving results, the kernel-level
ratio predicted the correct **sign** in 9 of 10 cases across both library
versions — including correctly predicting that the two versions differ in
*direction* at batch 64. That makes profiling a cheap screen before spending a
node on a benchmark ladder.

**Cannot:** the size, or anything user-facing. Two hard limits, both respected in
every figure quoted in §02:

1. **Magnitudes are unusable** — the kernel ratio over-predicts the end-to-end
   effect by roughly 2–3× (it says 1.433× at 16 users where the real serving
   result is 1.024×). Expert multiplication is one term in a decode step, not the
   whole step.
2. **The two instruments run the GPUs differently.** Profiling pins GPU clocks at
   a fixed frequency; the serving benchmarks leave them free, as production does.
   Kernel milliseconds therefore **cannot be multiplied into user-facing
   latency**, and no figure in this document does so. Everything in §02 is
   measured end-to-end on the serving benchmark.
