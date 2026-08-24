# Faster serving for DeepSeek-V4-Flash

**Status: measured, decisions pending.** Two independent speed-ups have been
measured on real production hardware. Neither is switched on. Both are blocked on
the same missing piece — a quality check — and one needs a dependency decision.

Updated 2026-08-24. Engineering detail:
[`DSV4_FLASH_SGLANG_SERVING_PERF_REPORT.md`](../DSV4_FLASH_SGLANG_SERVING_PERF_REPORT.md)
· collaborator brief:
[`DSV4_FLASH_SGLANG_COLLABORATOR_BRIEF.md`](../DSV4_FLASH_SGLANG_COLLABORATOR_BRIEF.md)

**Contents:** [01 What we are optimising](#01--what-we-are-optimising) ·
[02 Results](#02--results) · [03 What it costs](#03--what-it-costs) ·
[04 Decisions we need](#04--decisions-we-need) · [05 Risks](#05--risks) ·
[06 Why this needed measuring from scratch](#06--why-this-needed-measuring-from-scratch) ·
[07 What's next, by cost](#07--whats-next-by-cost) · [08 Evidence](#08--evidence)

---

## 01 · What we are optimising

This is a **serving-speed** project on a model we did not quantize ourselves
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

### Lever 1 — a faster expert-computation kernel

| | measured |
|---|--:|
| Output speed, **single user** | **+22.3%** (131.7 → 161.1 tokens/s) |
| A long reasoning answer (8,000 tokens) | **11 seconds faster** — 61.1 s → 50.0 s |
| Output speed at **full load** (64 concurrent users) | **+4.7%** (69.7 → 73.0 tokens/s) |
| Time to first token | unchanged |

🔴 **But the obvious cheap version of this is a regression, not a win.** The
kernel library has two relevant versions. The one the inference server ships by
default is **slower than what we run today** at every load above a single user:

| 64 concurrent users | output speed per user | vs today |
|---|--:|--:|
| Today's default kernel | 69.7 tokens/s | — |
| Faster kernel, **version the server ships** | **60.4** | **−13.4%** ❌ |
| Faster kernel, **newer version** | **73.0** | **+4.7%** ✅ |

So the win is real but it lives entirely in the newer version, and the gap
between the two versions is **up to 20.8%** of output speed — worst precisely at
the load production runs at. Turning the feature on without also upgrading the
library would make the product measurably worse. That is decision 2 below.

### Lever 2 — enabling prefill GPU graphs

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
but not 90%. **That prediction is not yet measured**, and measuring it is the
top item in §07. Which figure applies to us depends on decision 4.

---

## 03 · What it costs

Lever 1 costs a **dependency**, not performance: the newer kernel library is not
what the inference server pins, so adopting it means either carrying our own copy
or getting the upstream project to move its version.

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

---

## 04 · Decisions we need

| # | decision | why it is yours |
|--:|---|---|
| **1** | **Fund a quality evaluation** for both levers | Neither has one. This is the single gate on shipping either. The upstream project ran one for its own graphs-on configuration and results held steady — we have run nothing. It is cheap relative to the performance work already done |
| **2** | **Accept carrying a non-standard kernel version**, or push for an upstream version bump | Mandatory for lever 1's win. A carried dependency is ongoing maintenance; an upstream bump is slower but cleaner. The relevant upstream changes were authored by a maintainer of another major inference engine, so a bump is not a fork |
| **3** | **Report the finding upstream?** | We can show that the inference server's pinned kernel version costs up to 20.8% of output speed on this model. Reporting it is outward-facing and likely welcome, but it is a public statement about a third-party project and we have not made it |
| **4** | **Tell us the traffic mix to optimise for** | This is a genuine input we are missing. Lever 2 is worth 1.9× on short-prompt traffic and an estimated 1.13× on long-reasoning traffic, and its memory cost only bites above ~89,000 tokens of context. Both the configuration and the expected gain change with the answer |

---

## 05 · Risks

* **No quality data on either lever.** Both change how numbers are computed on
  the GPU; neither has been checked for output quality. We would not recommend
  shipping either without decision 1.
* **Silent degradation on lever 1.** If the newer kernel library is ever missing
  at start-up, the server does **not** fail — it quietly falls back to the slower
  version and runs 20.8% slower. Any deployment must assert the loaded version at
  start-up. We have that check built and used it in every experiment.
* **Capacity reduction at long context** from lever 2 (§03). Acceptable or not
  depending on decision 4.
* **One quoted figure is a prediction, not a measurement** — lever 2's gain on
  long-reasoning traffic (§02). Named as such wherever it appears.
* **A small measured loss inside lever 1.** At moderate load (8–16 concurrent
  users) the new kernel is 2.4–3.6% *slower*. Real, small, and it should be
  quoted alongside the wins rather than dropped.

---

## 06 · Why this needed measuring from scratch

We already had a large body of serving-performance work — but on a **different
inference server** and a **different model**. It does not transfer, and this is
not a technicality:

**On the previous model, the faster kernel won at every load level. On this one
it loses in the middle of the range, and the version the server ships loses
almost everywhere.** Had we assumed the earlier result carried over and flipped
the switch, we would have shipped a 13% slowdown at production load.

Four things differ simultaneously — the inference server, the model's numeric
format, the kernel we are competing against, and how work is distributed across
the model's experts. Each one is enough on its own to change the answer. That is
where the measurement time went, and it is why the results above are stated for
*this* configuration only.

---

## 07 · What's next, by cost

Ordered by what it costs us, not by how attractive it is. Time estimates are
scaled from a measured comparison run that took **2.05 hours** on one 8-GPU node.

| # | next step | cost | what it would settle |
|--:|---|---|---|
| 1 | **Measure lever 2 on long-reasoning traffic**, and both levers together | ~half a day, one 8-GPU node | Replaces §02's predicted 1.13× with a measurement, and tells us whether the two levers add up — nobody has run them together, and production would use both |
| 2 | **Try one more free server setting** that lets the server process prompts *without* fully stopping users mid-answer | ~3 hours, one node | This is the standard fix for the interruption problem — a competing inference server enables it by default and ours does not. One setting, no new hardware. It may overlap with lever 2 rather than add to it |
| 3 | **A one-line kernel tuning change** targeting the 8–16-user loss in §05 | ~1 hour, 1 GPU | Could remove lever 1's only regression |
| 4 | **Quality evaluation** (decision 1) | scoped separately, needs the evaluation pipeline | Unblocks shipping either lever |
| 5 | **Speculative decoding** | a project, not an experiment | ⚠️ **Probably the largest prize on the table and completely untouched here.** On the previous model it measured **1.2–2.5× faster output** with no quality cost by design — an order of magnitude more than either lever above. Its tuning will not transfer automatically, for the reasons in §06 |
| 6 | **Splitting prompt-processing and text-generation onto separate GPU pools** | multiple nodes + network configuration; changes the deployment shape | The textbook fix for the interruption problem, and it works — but it buys speed with **hardware**, and its value *shrinks* once items 1–2 land. Price it against what is left, not against today |

---

## 08 · Evidence

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
Measuring and deciding are deliberately kept separate; §04 is the deciding half,
and it has not happened.
