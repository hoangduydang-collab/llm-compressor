# DeepSeek-V4-Flash on SGLang: MoE kernel A/B (marlin vs humming) and prefill CUDA graphs

**Status:** measured; **no adoption decision applied by design.** Two independent
serving levers are priced here. Neither is switched on anywhere.

**Date:** 2026-08-24 · serving arms `2026-08-23`, operator arms `2026-08-18/20`,
prefill-graph arms `2026-08-20 → 08-22`

**Configuration:** full-stack agent (`FULL_STACK_AGENT_PROTOCOL.md`)

**Scope note — this is a different track from goals 1–7.** Those cover
MiniMax-M3 quantization and evaluation. This report covers the **serving stack**
for an already-quantized third-party model (DeepSeek-V4-Flash-0731, MXFP4
experts) on SGLang v0.5.17, 8×H100 TP8. It is filed here because the discipline,
the harness, and the humming kernel are the same ones goal 7 qualified on M3 —
and because the M3 result and this one now disagree in an instructive way (§4.5).
No `PROJECT_GOALS.md` sub-task is claimed by this document; whether this track
becomes a numbered goal is a program decision, not a reporting one.

**Contents:** [Headline](#headline) · [1 Design](#1-design) ·
[2 Serving results](#2-serving-results-the-primary-objective) ·
[3 Profiling results](#3-profiling-results-nsys-operator-level) ·
[4 What the numbers say](#4-what-the-numbers-say) ·
[5 Limitations](#5-limitations--read-before-quoting-any-number) ·
[6 Method defects worth recording](#6-method-defects-worth-recording) ·
[7 Prefill CUDA graphs (BCG)](#7-the-second-lever-prefill-cuda-graphs-bcg) ·
[8 Should BCG be re-tested on the benchmarks pipeline?](#8-should-bcg-be-re-tested-on-the-benchmarks-pipeline) ·
[9 Candidate next steps](#9-candidate-next-steps) ·
[Artifacts](#artifacts-and-provenance)

**Objective ranking, fixed for the whole campaign:** per-user decode speed
(`1 / ITL`) is **primary**; TTFT is explicitly **subordinate** — a TTFT win that
costs output speed is not a win. Both levers below are scored against that
ranking, and one of them (§7) turns out to have been documented around the
subordinate half of its own effect.

---

## Headline

**Lever 1 — the MoE runner backend.** `--moe-runner-backend=humming` with
humming-kernels **0.1.13** beats the marlin default on per-user output speed at
concurrency 1, 32 and 64, and loses slightly at 8–16. With the version SGLang
actually pins (**0.1.10**) the same flag is a **regression at every concurrency
above 1**.

| conc | marlin ITL | humming **0.1.10** (SGLang's pin) | humming **0.1.13** | 0.1.10 × | 0.1.13 × | **cost of the pin** |
|--:|--:|--:|--:|--:|--:|--:|
| 1 | 7.597 ms | 6.308 | **6.208** | 0.830× | **0.818×** | +1.6% |
| 8 | 8.137 | 8.549 | 8.420 | 1.049× | 1.036× | +1.5% |
| 16 | 9.637 | 10.575 | 9.854 | 1.096× | 1.024× | +7.3% |
| 32 | 11.633 | 12.846 | **10.835** | 1.106× | **0.930×** | **+18.6%** |
| **64** | 14.353 | **16.562** | **13.711** | **1.154×** | **0.955×** | **+20.8%** |

Ratios below 1.000 favour humming. Per-user output speed at production
concurrency 64: marlin **69.72 tok/s** → pinned 0.1.10 **60.43 (−13.4%)** →
0.1.13 **73.02 (+4.7%)**. At concurrency 1: **131.70 → 161.07 tok/s (+22.3%)**.

**Lever 2 — prefill CUDA graphs (BCG).** Forcing `--cuda-graph-backend-prefill
breakable` past SGLang's DSV4 auto-disable rule makes prefill **2.0–2.2× faster
per forward** (2.61× → **1.21×** of the compute floor) and, at batch 64, ITL
**1.92× better** — but it is funded from 12.3% of
the KV cache, and its measured ITL win sits on a prefill-heavy shape that
inflates exactly the quantity BCG removes (§7, §8).

**The two levers have never been measured together**, and production would ship
both. That is the largest gap this report leaves open (§8.5).

---

## 1. Design

### 1.1 One variable, and a paired A–B–A boot order

Every serving arm is generated from **one template**
(`_serving_paired.tmpl.yaml` via `_gen_paired_arm.py`), so the arms are
structurally identical past the backend flag. Shared by construction:

* same node (`ca-gpu01`), same digest-pinned `lmsysorg/sglang:v0.5.17`, same
  weights snapshot `7872f01b…`;
* same production argv — TP 8, EP 1, `chunked-prefill-size` 2048,
  `max-running-requests` 64, KV `fp8_e4m3`, page size 256,
  `mem_fraction_static` 0.874, context 1 Mi, **no speculative decoding**;
* **no profiler, no clock lock, no `SYS_ADMIN`** — DVFS free-running on both
  sides, i.e. production's own clock regime;
* the aiperf load generator in the server pod over **loopback**, so no DNAT hop
  or veth pair lands inside ITL;
* same aiperf 0.11.0 from the same PVC venv, same harness revision
  (`HARNESS_VERSION` = `duy-branch` / `4529676`), same shapes, same request
  counts.

**The A–B–A structure is the load-bearing part.** Each arm runs
`marlin → humming → marlin` **in one pod**, so cross-boot clock and thermal
drift is *measured* rather than assumed, and the comparator refuses any effect
smaller than the drift it measured. Measured drift on the reasoning shape:
**0.00–0.49%** across 2.05 h — small enough that 2% effects are readable.

### 1.2 Engagement, proven per phase — never assumed

`fp8.py:376-412` dispatches FP4-packed experts to the MXFP4 method classes with
**no `else: raise`**. An unrecognised backend value silently returns the generic
method, so the run looks successful while measuring the incumbent. Engagement is
therefore proven, per phase, from that phase's own `server.log`:

* marlin phases: `Mxfp4MarlinMoEMethod` on **8/8 ranks**;
* humming phases: `Mxfp4HummingMoEMethod` on **8/8 ranks**, zero marlin;
* **plus a version proof**, because "humming ran" and "0.1.13 ran" are different
  claims and the image ships 0.1.10: `humming.__version__` *and*
  `os.path.dirname(humming.__file__)` are both asserted in-band before the
  server starts, or the pod exits 1;
* **plus an inverted proof for the 0.1.10 arm**: that arm runs the image's own
  pin with no side-install and no `PYTHONPATH`, and the manifest fails if any
  humming `.so` maps out of `/tmp/humming-*`.

Engagement was 8/8 in every phase of every arm reported here.

### 1.3 Confound checks

| confound | how it was closed |
|---|---|
| **Generation length** — different kernels change numerics, which changes output length and fakes a throughput delta | `ignore_eos` + `min_new_tokens`; all 4550 records in the reasoning ladder carry `output_sequence_length` **exactly 4000** |
| **Clock/thermal drift between boots** | measured by the A–B–A bracket: **0.00–0.49%** |
| **Cross-pod reproducibility** (needed for the three-way pin arithmetic) | two pods, two boots, ~3 h apart, marlin phases agree to **≤0.33%** |
| **Client-side load-generator stalls** | aiperf stalls are frequent and *asymmetric between phases* (conc 32: 59 / 33 / **126**). The two marlin passes settle it: a 2–4× swing in stall count moves ITL by **≤0.12%**, against effects of 2.4–7.0% |
| **KV pool differing between phases** | phase B reads `max_total_num_tokens` 6 411 264 vs marlin's 6 473 216 (−1.0%) — humming's own workspace, reversible. At conc 64 the worst-case in-flight demand is 320 K tokens against a 6.41 M pool: **20× headroom** |
| **Instrument validity** | the marlin arm reproduces the campaign's independently measured BS-1 baseline on five quantities, from a *different client*, worst deviation **1.4%** |

### 1.4 Metric definitions

Named against the aiperf 0.11.0 fields so nothing is guessed. Same definitions
as `docs/m3-two-axis-perf.md`, so the M3 and DSV4 tracks are on one vocabulary.

| reported as | aiperf field | definition |
|---|---|---|
| **ITL** (ms) | `inter_token_latency` | `(t_last_chunk − t_first_chunk) / tokens_after_first` off the SSE stream. **Decode-only by construction** — prefill lands in TTFT and can never contaminate a *measured* ITL |
| **per-user output speed** (tok/s) | `output_token_throughput_per_user` | `1 / ITL` — one user's decode rate. The primary objective |
| TTFT (ms) | `time_to_first_token` | first streamed token |
| **effect** | derived | `abs(1 − B / mean(A1, A2))` — humming against the time-centred marlin mean |
| **drift** | derived | `abs(A1 − A2)` relative to the smaller, i.e. the conservative direction |
| expert-GEMM ms/rank-step | nsys | total `gpu_ms` of the decode-step instantiation set ÷ rank-steps. **An operator quantity; never combined arithmetically with ITL** (§3.5) |

---

## 2. Serving results (the primary objective)

Reasoning shape **ISL 1000 / OSL 4000**, thinking ON, temp 0.6, `ignore_eos` +
`min_new_tokens`, concurrency 1 / 8 / 16 / 32 / 64. 10 measured waves + 1 warmup
wave per point. One pod per arm, `marlin → humming → marlin`, ~2 h elapsed each,
**zero aiperf failures**.

### 2.1 marlin vs humming 0.1.13 — the U-shape is real

| conc | marlin (mean of 2) | humming 0.1.13 | ITL × | effect | drift | stat ± | σ | per-user Δ |
|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| **1** | 7.593 ms | **6.208** | **0.818×** | 18.23% | 0.00% | ±0.13% | 139 | **+22.3%** |
| 8 | 8.128 | 8.420 | 1.036× | 3.60% | 0.25% | ±0.10% | 35 | −3.5% |
| 16 | 9.621 | 9.854 | 1.024× | 2.42% | 0.10% | ±0.12% | 20 | −2.4% |
| **32** | 11.650 | **10.835** | **0.930×** | 6.99% | 0.00% | ±0.14% | 49 | **+7.5%** |
| **64** | 14.357 | **13.711** | **0.955×** | 4.51% | 0.14% | ±0.16% | 28 | **+4.7%** |

All five points resolve at **20–139σ**. `stat ±` and `σ` were recomputed
independently from the per-request records and reproduce the in-pod comparator
to four decimal places.

**The curve is a U, not a monotone loss:** humming wins at both ends of the
ladder and loses only in the middle. Per-user output speed, the primary
objective, on the shape production actually serves:

| conc | marlin tok/s | humming 0.1.13 tok/s | Δ |
|--:|--:|--:|--:|
| 1 | 131.70 | **161.07** | **+22.3%** |
| 8 | 123.04 | 118.77 | −3.5% |
| 16 | 103.96 | 101.50 | −2.4% |
| 32 | 85.88 | **92.33** | **+7.5%** |
| 64 | 69.72 | **73.02** | **+4.7%** |

Production runs `--max-running-requests=64`, so the concurrency that matters
most is a win.

**Drift, three independent ways, all small:**

| | measured | over |
|---|--:|---|
| between phases (`A1 ↔ A2`) | 0.00–0.25% | 2.05 h |
| within a point, least-squares over its own span | ≤1.11% | 226–548 s |
| sampling (SEM on the A/B ratio) | 0.10–0.16% | — |

The within-point fit came free from the 10 waves per point. It shows A1 trending
*faster* at conc 16/32/64 (−0.77%, −0.62%, −1.11%) while A2 does not —
consistent with the machine settling after the pod's first boot rather than a
monotone thermal ramp.

### 2.2 The concurrency-1 arm, on the longer reasoning shape

Measured separately at OSL 8000, 10 requests + 1 warmup per arm. Absolute ITL is
not comparable to §2.1 (different OSL); the ratio is, and it agrees.

| reasoning, conc 1 | marlin | humming 0.1.13 | delta |
|---|--:|--:|--:|
| ITL | 7.61 ms | **6.22 ms** | **0.817×** |
| per-user tok/s | 131.46 | **160.77** | **+22.3%** |
| request latency (8000 tok) | 61.08 s | **49.99 s** | **−11.09 s (−18.2%)** |
| TTFT | 231.3 ms | 235.1 ms | +1.6% — a wash |
| output length | 8000 / 8000 | 8000 / 8000 | identical |
| ITL std / avg | 0.13% | 0.16% | both far below the effect |

And on the short calibration shape (ISL 2048 / OSL 128 / conc 1), the same
direction plus a **prefill** win:

| calibration, conc 1 | marlin | humming 0.1.13 | delta |
|---|--:|--:|--:|
| ITL | 7.40 ms | **6.15 ms** | **0.831×** |
| per-user tok/s | 135.22 | **162.62** | **+20.3%** |
| TTFT | 501.0 ms | **438.4 ms** | **−12.5%** |
| prefill tok/s | 4103 | **4681** | **+14.1%** |

Concurrency 1 has now been measured **three times** across two shapes and two
pods (0.831× / 0.818× / 0.818×). It is the campaign's most solid single number.

### 2.3 What SGLang's `humming-kernels==0.1.10` pin costs

The headline table's third column. This arm exists because the "the pin costs
us X%" sentence was, until it ran, an **inference chaining two datasets**
(serving measured marlin → 0.1.13; operator measured 0.1.10 → 0.1.13). Phase B
here runs the **image's own pin** — no `pip install --target`, no `PYTHONPATH`,
just the flag — which is also the actionable configuration: if the pinned
version already beat marlin, the win would cost a flag flip and no dependency
change.

**It does not.** Five points, 44–282σ, drift 0.01–0.48%:

* **the flag alone is a de-optimisation** above conc 1 — 1.049× to 1.154× slower
  than marlin, i.e. **−13.4% per-user output speed at conc 64**;
* **the pin cost is monotone in concurrency** — +1.5% at conc 8 rising to
  **+20.8% at conc 64** — so it is worst exactly where production runs;
* therefore **a pin bump or a vendored wheel is mandatory, not an
  optimisation.** Any guidance that recommends `--moe-runner-backend=humming`
  without the version bump is actively harmful at production concurrency.

The three-way arithmetic is licensed by the cross-pod marlin anchor: two pods,
two boots, ~3 h apart, marlin phases agreeing to **≤0.33%** (per point: 0.10 /
0.23 / 0.33 / 0.29 / 0.06%). That is a result in its own right — on this shape a
full boot-and-ladder reproduces to a third of a percent — and it retires the
"free-running DVFS across boots" worry the earlier single-arm A/B had to list as
unexcluded.

⚠️ **0.1.10 vs 0.1.13 is still mediated by that anchor**, not measured in one
pod. A four-phase `marlin → 0.1.10 → 0.1.13 → marlin` arm would remove the
mediation; it was not run because the anchor check is cheaper and, having
passed, sufficient.

---

## 3. Profiling results (nsys, operator-level)

Five-point BS ladder, 5 measured rounds per point, GPU clocks **pinned at
1980 MHz**, one variable against the previous humming arm: the library,
`0.1.10 → 0.1.13`, via an isolated `pip install --target` side-install. Metric:
total `gpu_ms` of the **decode-step instantiation set, per decode step** —
per-step rather than per-window, because a window labelled "decode" does not
hold a fixed number of steps and charging that drift to the backend is wrong.

### 3.1 The config prediction — 10/10, made before the GPUs were touched

`_probe_humming_heuristic.py` replays both SM90 tuning heuristics on our shapes
with no GPU. It predicted the exact tile and warp shape for all ten
(GEMM × batch) points **in advance**, read back afterwards out of the kernel
signatures:

| BS | GEMM | 0.1.10 tile | 0.1.13 tile | predicted? |
|--:|---|---|---|:--|
| 1 | gate/up | (8,128,256) | (8,128,256) | ✅ unchanged, as predicted |
| 1 | down | (8,128,256) | **(8,128,128)** | ✅ |
| 8–64 | gate/up | (8,128,256) | **(8,256,128)** | ✅ |
| 8–64 | down | (8,128,256) | **(8,512,64)** | ✅ |

Warp shape moved with it, (8,32,64) → (8,64,64) wherever the tile moved. This
matters beyond bookkeeping: a source replay that predicts the on-GPU config for
ten points sight-unseen is what licenses the mechanism in §3.3.

### 3.2 Expert-GEMM cost per layer-step, and per rank-step

µs per layer-step, both expert GEMMs. Aggregate call counts agree within **0.8%**
on every point against marlin (0.0% at BS 64), which is what licenses comparing
totals at all.

| BS | marlin | h 0.1.10 | h 0.1.13 | **0.1.13 vs marlin** | **0.1.13 vs 0.1.10** |
|--:|--:|--:|--:|--:|--:|
| 1 | 42.60 | 20.25 | **16.65** | **0.391×** | 0.822× |
| 8 | 38.29 | 47.07 | 47.37 | 1.237× | 1.006× |
| 16 | 48.88 | 67.04 | 70.04 | 1.433× | 1.045× |
| 32 | 64.07 | 89.22 | 74.37 | *1.161×* | *0.834×* |
| 64 | 91.76 | 134.82 | **83.36** | **0.908×** | **0.618×** |

*Italics = BS 32, which failed its own QC on both arms (§3.4) and is not
quotable.*

Absolute expert-GEMM decode-step work in one rank-step (one decode forward on
one of the 8 ranks):

| BS | marlin ms | h 0.1.10 ms | h 0.1.13 ms | Δ vs marlin | Δ vs 0.1.10 |
|--:|--:|--:|--:|--:|--:|
| 1 | 1.832 | 0.871 | **0.716** | **−1.116** | −0.155 |
| 8 | 1.647 | 2.024 | 2.037 | +0.390 | +0.013 |
| 16 | 2.102 | 2.883 | 3.012 | +0.910 | +0.129 |
| 32 | 2.755 | 3.837 | 3.198 | +0.443 | −0.639 |
| 64 | 3.946 | 5.797 | 3.584 | **−0.361** | **−2.213** |

**BS 64 recovered 2.213 ms per rank-step from the version bump alone** — six
times the margin by which it now beats marlin. The 0.1.10 arm was not measuring
humming's ceiling; it was measuring an untuned heuristic. **The shape changed,
not just the level:** 0.1.10 degraded monotonically against marlin
(0.475 → 1.229 → 1.371 → 1.393 → 1.469); 0.1.13 is a U.

### 3.3 Mechanism: the down-proj won everywhere, gate/up caused the U

Splitting by GEMM is what makes the U legible. µs per layer-step:

| BS | gate/up 0.1.10 | gate/up 0.1.13 | ratio | down 0.1.10 | down 0.1.13 | **ratio** |
|--:|--:|--:|--:|--:|--:|--:|
| 1 | 12.22 | 11.19 | 0.916× | 8.03 | 5.46 | **0.680×** |
| 8 | 25.29 | 33.84 | *1.338×* | 21.79 | 13.89 | **0.638×** |
| 16 | 34.22 | 53.75 | *1.571×* | 31.44 | 16.48 | **0.524×** |
| 32 | 42.34 | 53.45 | *1.262×* | 46.19 | 20.92 | **0.453×** |
| 64 | 62.86 | 55.21 | 0.878× | 71.96 | 28.18 | **0.392×** |

Two clean, opposite stories:

* **down-proj (N=4096, K=256) improves at every point, monotonically with
  batch** — 0.680× → 0.392×, a 2.55× speed-up at BS 64. Predicted from source
  before the run: `use_stream_k` was unconditionally on at K=256 with
  `block_k=256`, so `K_BLOCKS = 1` and **stream-K had nothing to split** (pure
  overhead), and `num_ctas_per_sm` defaulting to 1 at 512 threads left 25%
  occupancy. 0.1.13's `(8,512,64)` runs stream-K **off** at 2–3 CTAs/SM.
* **gate/up (N=512, K=4096) is the U.** 0.1.13's wider-N/shallower-K
  `(8,256,128)` is a **loss** of up to 1.571× through BS 8–32 and only becomes a
  win at BS 64. Whatever upstream's tile-M ladder was tuned for, our
  **1.5-tokens-per-expert** regime is not it — the ladder's second rung needs
  BS ≥ 239, and DSV4-Flash at BS 64 with 256 experts and top-6 routing gives
  1.5 tokens per expert.

So the net U is one monotone win plus one badly non-monotone regression, **and
the regression is the tunable half.** Forcing gate/up back to `(8,128,256)`
while keeping the new down-proj config is a one-line `tuning_config` override —
still untested, 1 GPU, no upgrade needed.

### 3.4 QC: BS 32 is unusable, on both sides

| BS | verdict | calls_dev | primary dev | quotable? |
|--:|---|--:|--:|:--|
| 1 | PASS | 0.0% | 3.688% (cap 20) | ✅ |
| 8 | WINDOW_MISMATCH | 3.755% | 1.702% (cap 20) | scale-invariant only |
| 16 | PASS | 0.985% | 1.437% (cap 5) | ✅ |
| 32 | WINDOW_MISMATCH | **35.532%** | **9.523%** (cap 5) | ❌ **no** |
| 64 | WINDOW_MISMATCH | 3.226% | 4.350% (cap 5) | scale-invariant only |

A 35.5% swing in the window's own call count is not jitter — the work inside the
window changed between rounds (one round is 22% heavier than the rest). The
marlin arm carries the same defect at the same point, pre-registered. A 6-round
re-run on **both** arms is the fix; it remains on the backlog.

### 3.5 🔴 Why no end-to-end number comes out of the profiling data

Two ITL percentages derived from these tables were **withdrawn**. The campaign's
own baseline document forbids the operation in advance:

> These numbers must NOT be combined arithmetically with the coming nsys tables.
> The sweep pins GPU clocks to 1980 MHz; this boot leaves DVFS alone… There is
> no "ms per operator as production sees it" obtainable by multiplying one by
> the other.

Different clock regime, and the nsys boot also shrinks the KV pool. The
prohibition was written down in advance, in the same repo, and violated anyway.
The **operator** arithmetic in §3.2 is unaffected — it never leaves the nsys
tables, and it was verified two independent ways at BS 64 (0.8% apart on the
baseline, 0.05% on the challenger).

And the whole-window total is not decidable either — for a measurement reason,
not a policy one. At BS 64, humming **wins the decode-step GEMM 0.908× and
loses the window 1.036×**. The entire +747.6 ms sits inside `moe:expert-gemm`,
with every other category agreeing within ~55 ms. The cause is
**prefill-shaped siblings riding inside a window labelled decode**:

| BS | marlin | h 0.1.10 | h 0.1.13 |
|--:|--:|--:|--:|
| 1 | 0.0% | 9.1% | 7.8% |
| 8 | 22.3% | 28.5% | 9.8% |
| 16 | 22.0% | 23.3% | 13.0% |
| 32 | 30.2% | 25.9% | 27.4% |
| 64 | 28.9% | 26.0% | **48.6%** |

*Share of the `moe:expert-gemm` category that is NOT decode-step work.*
Contamination differs by 20 points between arms at the same point, so a window
total compares two differently-contaminated windows and blames the backend. Its
cause is the profiling harness (priming caches 1792 of 2048 prompt tokens, so
every round re-prefills 256 tokens; contamination ≈ `256/OSL`), not either
backend.

**This is why the campaign moved to the serving harness.** ITL is decode-only by
construction, so prefill can never contaminate a *measured* ITL — what it
contaminates is the nsys window. The fix was never a better nsys arm.

### 3.6 The operator ratio predicts the SIGN of the serving effect — 9 of 10

| conc/BS | operator × | serving × | sign |
|--:|--:|--:|---|
| **0.1.13** | | | |
| 1 | 0.391× | 0.818× | ✅ |
| 8 | 1.237× | 1.036× | ✅ |
| 16 | 1.433× | 1.024× | ✅ |
| 32 | 1.161× *(QC fail)* | 0.930× | ❌ — and not quotable |
| 64 | 0.908× | 0.955× | ✅ |
| **0.1.10** | | | |
| 1 | 0.475× | 0.830× | ✅ |
| 8 | 1.229× | 1.049× | ✅ |
| 16 | 1.371× | 1.096× | ✅ |
| 32 | 1.393× *(QC fail)* | 1.106× | ✅ |
| 64 | 1.469× | 1.154× | ✅ |

**Nine of ten agree on sign, and the single miss is the point that failed its
own QC on both arms.** The operator sweep also correctly predicted that the two
versions **differ in sign at BS 64** — the one place where getting the version
row wrong gives a wrong answer rather than a rounding error.

⚠️ **Magnitudes remain unusable.** Operator says 1.433× at conc 16 where serving
says 1.024×; 0.391× at conc 1 where serving says 0.818×. It over-predicts every
effect by roughly 2–3×, which is unsurprising: expert GEMM is one term in a
decode step, and the two instruments run under different clock regimes.

---

## 4. What the numbers say

1. **humming 0.1.13 is the better backend at production concurrency**, by 4.5%
   on ITL (+4.7% per-user) at conc 64, and by 22.3% at conc 1. It is slightly
   worse at conc 8–16 (2.4–3.6%) — real at 20–35σ, but real is not the same as
   important.
2. **The win lives entirely in the version, not the flag.** With SGLang's pinned
   0.1.10 the same flag is −13.4% per-user at conc 64. The pin costs up to
   **20.8% of decode ITL**, monotone in concurrency.
3. **The mechanism is understood and half of it is still on the table.** The
   down-proj improvement is monotone and large (2.55× at BS 64); the gate/up
   regression is a mis-targeted tile ladder for our 1.5-tokens-per-expert
   regime. A one-line tuning override could plausibly remove the 8–16 loss
   without touching the win.
4. **Operator profiling earns its place as a cheap sign oracle.** 9/10 on sign
   across two versions, at a fraction of a serving ladder's cost — but only for
   sign, and only where its own QC passes.
5. **This diverges from the M3 result, and the divergence is the interesting
   part** (§4.5).

### 4.5 Why M3 and DSV4-Flash disagree about humming

Goal 7 adopted humming on M3 because it beat CUTLASS by ~34% at concurrency 1
and stayed ahead at every concurrency up to 64. Here humming loses at conc 8–16
and wins only 4.7% at 64. The two are not in conflict; they are different
regimes, and the difference is arithmetic:

| | M3 (goal 7) | DSV4-Flash |
|---|---|---|
| quant scheme | GPTQ **W4A8** (INT4 weights, dynamic E4M3 activations) | **MXFP4** experts, A16 |
| humming path | indexed **A8** | indexed **A16** |
| opponent | CUTLASS | **marlin** |
| experts × top-k | **128 × 4** | **256 × 6** |
| tokens per expert at BS 64 | 2.0 | **1.5** |

Two things follow. First, **the opponent changed**: marlin is a stronger
baseline than CUTLASS at these shapes, so the same kernel wins by less. Second,
**the quant path changed**: the A16 indexed path is the one upstream re-tuned in
0.1.13, and its tile-M ladder assumes far more tokens per expert than 1.5. That
also explains the shape of the residual loss (§3.3) and predicts that
`--enable-dp-attention`-style changes to tokens-per-expert would move it —
whereas expert parallelism would *not*, since tokens-per-expert is invariant to
how experts are sharded.

**So "humming is faster" is not a portable claim.** It is a claim about a quant
path, an opponent, and a tokens-per-expert regime, and all three moved between
the two models.

---

## 5. Limitations — read before quoting any number

* **No adoption decision.** No pass/fail threshold was applied to any number
  here. Measurement and adoption stay separate decisions, as they did throughout
  the M3 work.
* **One shape for the ladder** (ISL 1000 / OSL 4000), one node, one boot per
  phase. Decode-dominated but not pure decode — 1 prefill token per 4 decode
  tokens.
* **No quality check on either lever.** Different MoE kernels change numerics;
  graphing prefill is bitwise-perturbing in principle. Upstream ran GSM8K for
  their BCG config (stable at 0.953); we have not run anything. **Nothing here
  is a public-benchmark score** — it is an internal paired serving comparison.
* **conc 8–16 is a measured loss** for humming 0.1.13. Small, but do not quote
  the conc-1 and conc-64 numbers without it.
* **BS 32 is void in the operator data** on both arms, so the U's left shoulder
  is located by BS 8/16 only. **The crossover between BS 1 and BS 8 is still
  unlocated** (BS 2/4 were never run).
* **Still the `INDEXED` gemm type** in every humming arm — no TMA, no warp
  specialisation. The re-tuning is a win *inside* the structurally weaker path.
  `SGLANG_HUMMING_MOE_GEMM_TYPE=grouped` is untested here.
* **No roofline closure.** `ncu dram__bytes.sum` on both backends at BS 1/64 is
  still the outstanding measurement that would say whether BS 64 has headroom
  left for anyone.
* ⚠️ **The loaded-library `/proc/*/maps` proof did not fire on the 0.1.10 arm**
  ("no humming `.so` mapped yet" — the probe ran ~2 s after engagement, where
  the 0.1.13 arm found 40 mappings). Engagement and the version probe still
  prove backend and version independently, so the arm stands, but its proof is
  one layer thinner. The probe should run after the first request, not right
  after boot.
* **TTFT at conc 64 is 3.36 s avg / 6.29 s p95 on this shape for *both*
  backends.** The ranked objective subordinates it; a deployment cannot.
* **Nothing about DSpark speculative decoding**, which on the M3 evidence
  (1.21–2.53× per-user, goal 2h) remains a substantially larger lever for output
  speed than either lever in this report.
* **Nothing about the two levers combined** (§8.5).

---

## 6. Method defects worth recording

Recorded because each one produced a wrong answer that cleared its own checks.

### 6.1 🔴 A shape choice inverted two signs

The first serving ladder used the calibration shape **ISL 2048 / OSL 128**. It
reported findings at conc 16 and 32 — and **both had the wrong sign**:

| conc | static ladder said | reasoning ladder says | |
|--:|---|---|---|
| 1 | 0.818× | 0.818× | ✅ agree |
| 8 | 1.043× | 1.036× | ✅ agree in sign |
| **16** | **0.916× — humming FASTER** | **1.024× — humming SLOWER** | 🔴 wrong sign, having cleared both its 5.66% drift bar and its ±2.35% stat bar |
| **32** | **1.064× — humming SLOWER** | **0.930× — humming FASTER** | 🔴 wrong sign |
| 64 | 0.989× (unresolvable) | 0.955× | ✅ agree in sign |

**Mechanism: above conc 1 that shape's ITL is not a decode metric.** 16 prefill
tokens per decode token means each decode stream is repeatedly preempted by
other requests' 2048-token prefill chunks. Same backend, same node, same
free-running DVFS:

| conc | static ITL (marlin) | reasoning ITL (marlin) | ratio |
|--:|--:|--:|--:|
| 1 | 7.67 ms | 7.59 ms | 0.99× |
| 8 | 12.12 | 8.14 | 0.67× |
| 16 | 20.36 | 9.63 | 0.47× |
| 32 | 31.73 | 11.65 | 0.37× |
| 64 | **53.45** | **14.35** | **0.27×** |

Decode work per token is identical on both shapes, so the 3.7× gap at conc 64
*is* the prefill interference. The two shapes agree to 1% at conc 1 — where
there is no other request to preempt — which is what licenses the comparison.

**Transferable lesson, and it is the most useful thing in this report:**
clearing a one-degree-of-freedom drift estimate and a standard error was **not
sufficient**. The static ladder's apparent 3.5–6.5% "drift" was largely the
difference of two numbers each carrying ±1.7–3.0% of sampling noise. Sampling
noise and drift are different things, and only the reasoning shape's ~0.01%
noise floor turned `A1 − A2` into an actual drift measurement. **A metric's name
does not tell you what it measures on a given shape.** §8 applies this directly
to the BCG result.

### 6.2 A threshold gate that flagged the campaign's best result as unquotable

The harness compares each calibration run against a stored baseline and returns
a verdict. On the first humming arm it printed `GATE 5 SUSPECT — worst deviation
22.0% (>10%) … Do NOT quote an A/B built on top of this.` **The gate was wrong,
and every deviation it flagged was humming being faster** — the baseline was
measured with marlin. On a marlin arm a deviation indicts the instrument; on any
other backend a deviation is the *expected outcome*, i.e. the thing being
measured.

Fixed by making the verdict backend-aware: on the baseline backend it keeps the
10% instrument band; on any other backend it prints the deltas as a *backend
delta*, says to read their sign, and validates the instrument **internally** on
the run's own ITL spread (`std/avg ≤ 2%`), which needs no marlin number.
**A tolerance against a fixed reference silently becomes a filter against the
effect under study the moment the reference stops describing the arm.**

### 6.3 The comparator compared against the wrong version's operator row

The paired template hardcoded 0.1.13's operator ratios regardless of which
humming version ran. Since the two versions **disagree in sign at BS 64**
(§3.6), that is a wrong answer, not a rounding error. Fixed by parameterising
the ratio table by version. The 0.1.10 pod was already running, so its in-pod
report mislabels conc 64 as DISAGREE — flagged at both the write-up and the data
directory so nobody reads that block later and believes it.

### 6.4 A misattribution that survived three documents

The KV pool was 1.0% smaller in humming phases, and the campaign had recorded
this three times as "the side-install's host memory" — because the side-install
was the only thing anyone had varied. The 0.1.10 arm has **no side-install** and
read the same 6 411 264 anyway. It is humming *the backend* reserving more
workspace, and it is fully reversible. Quantitatively dead (20× headroom), but
the attribution was wrong at three sources and is now fixed at all three.
**Running the arm that varies the other thing is what separated them.**

### 6.5 Two instrument traps in the harness itself

* 🔴 **`run_perf_*.sh` catches aiperf failures with `|| echo "WARN: aiperf
  failed"` and exits 0.** A wholly failed sweep looks successful. Every arm here
  asserts the per-point CSV exists and parses, independently of the exit code.
* **The aiperf CSV's `std` is the last column**, not column 13 (which is p99).
  An off-by-one there silently substitutes a percentile for a standard
  deviation, which is exactly the field the significance test consumes.

### 6.6 A cost model that was wrong by 2.3× — in the informative direction

The reasoning ladder was predicted at 6.3 h and took **2.05 h**, because the
estimate was built from static-shape ITLs and reasoning-shape ITLs are up to
3.7× lower at high concurrency (§6.1). The estimation error and the sign errors
have the same root cause, which is why it is recorded as a finding rather than
an apology.

---

## 7. The second lever: prefill CUDA graphs (BCG)

**Status: measured, five runs, recommendation stands with a priced caveat.**
Independent of §2 — this lever touches the prefill CUDA-graph backend, not the
MoE runner. Measured with the campaign's own in-pod SSE probe (not aiperf), which
is the whole subject of §8.

### 7.1 The question

Prefill runs at **2.54× its compute floor** because prefill CUDA graphs are
`disabled` by a DSV4-specific rule in SGLang. Forcing them on is much faster but
costs KV cache — and KV is shared with decode, which is objective 1. So: is the
trade worth it, and is the cost avoidable?

| arm | flags | what it tests |
|---|---|---|
| **A_eager** | *(none)* | production exactly as it runs today |
| **B** | `--cuda-graph-backend-prefill breakable --mem-fraction-static 0.80` | the known-good config |
| **C** | `--cuda-graph-config '{"prefill":{"backend":"breakable","bs":[512,1024,1536,2048]}}'` | 4 buckets instead of 42 — the attempt to get the win **without** paying KV |
| **D** | 42 buckets at `mem-fraction-static 0.874` | the unconfounded control: graphs on, KV untouched |

### 7.2 The gain

Per 2048-token forward, against the **75.3 ms** compute floor:

| arm | ms/forward | × floor |
|---|--:|--:|
| A_eager | 196.6 – 212.7 | 2.61× |
| B / C | 89.8 – 105.0 | **1.21×** |

Overhead above the floor goes **121.3 → 14.5 ms**, so ~107 of a predicted
~116 ms fixed per-forward tax is removed — the lever and the prior diagnosis
agree to ~8%.

Full serving grid, A vs B, all 11 cells, **all passing the closure gate**
(`tok/s(wall) == BS·OSL / (TTFT + (OSL−1)·ITL)`, measured 0.986–0.998 against a
0.98–1.02 band):

| ISL | BS | A TTFT ms | B TTFT ms | **TTFT ×** | A ITL ms | B ITL ms | **ITL ×** |
|--:|--:|--:|--:|--:|--:|--:|--:|
| 2048 | 1 | 408.4 | 103.0 | **3.97×** | 7.50 | 7.49 | 1.00× |
| 2048 | 4 | 732.3 | 237.1 | 3.09× | 10.55 | 8.45 | 1.25× |
| 2048 | 8 | 1298.1 | 521.9 | 2.49× | 12.20 | 10.18 | 1.20× |
| 2048 | 16 | 2094.9 | 802.5 | 2.61× | 21.20 | 13.51 | 1.57× |
| 2048 | 32 | 3754.2 | 1486.7 | 2.53× | 33.33 | 20.02 | 1.66× |
| 2048 | 64 | 6984.1 | 2852.7 | 2.45× | 62.94 | 32.80 | **1.92×** |
| 8192 | 1 | 1011.3 | 377.4 | 2.68× | 7.50 | 7.50 | 1.00× |
| 8192 | 8 | 4009.6 | 1602.0 | 2.50× | 27.97 | 16.59 | 1.69× |
| 8192 | 32 | 12674.1 | 5860.6 | 2.16× | 84.40 | 52.64 | 1.60× |
| 32768 | 1 | 3355.3 | 1582.5 | 2.12× | 7.59 | 7.58 | 1.00× |
| 32768 | 8 | 10731.4 | 6719.6 | 1.60× | 68.14 | 47.68 | 1.43× |

🔴 **The two objectives move in opposite directions along batch size:**

| | BS 1 | BS 64 |
|---|--:|--:|
| **TTFT gain** (objective 2) | 3.97× | 2.45× — *shrinking* |
| **ITL gain** (objective 1) | **1.00×** | **1.92×** — *growing* |

At BS 1 this is a **pure TTFT lever with exactly zero ITL effect** — which is
also the control that confirms the mechanism, since with one request in flight
nothing is prefilling during its decode. At BS 64 it is substantially an
**output-speed** lever: per-user 15.9 → 30.5 tok/s. Since objective 1 outranks
objective 2 and production sets `--max-running-requests=64`, **the more valuable
half of this lever at production's operating point is the half the original
write-up was not organised around.**

Objective 1 was also pre-committed as a veto: an ITL regression at any BS would
have been disqualifying regardless of the TTFT win. The A/B showed arm C at
0.92× at one cell; a dedicated n=20 re-measurement of every BS-1 cell put the
widest spread across all six arm × ISL cells at **0.002 ms**. The 0.92× was n=1
noise. **Rule 1 passes.**

### 7.3 The cost, and a pre-committed rule that failed

| | `max_total_num_tokens` | ÷ 64 |
|---|--:|--:|
| A_eager | 6 473 216 | 101 144 |
| B (mfs 0.80) | 5 675 520 | **88 680** |

The second pre-committed rule was that a KV loss is only acceptable if the
capacity table shows it does not bind at the ISL × BS production serves.
**It fails.** Above **ISL ≈ 89 k** arm B runs out of pool before it runs out of
`--max-running-requests`, so the loss is *not* masked by the concurrency cap. At
1 M context the ceiling goes **6 concurrent → 5**. The rule was written to be
failable and it failed; it does not veto the lever, it prices it.

### 7.4 Arm C — the point of the experiment — is rejected on measurement

Arm C was meant to make the trade free (fewer buckets → smaller capture pool →
no KV cost, and it does hold KV at ±0.0%). The A/B ladder ran only exact
multiples of 2048, so **every forward it measured was exactly 2048 tokens and
the padding path never executed once.** A dedicated padding probe closed that:

| ISL | B p50 ms | C p50 ms | **C/B** | B pads to | C pads to |
|--:|--:|--:|--:|---|---|
| 64 | 35.57 | 205.50 | **5.78×** | [64] | [512] |
| 128 | 33.58 | 204.87 | **6.10×** | [128] | [512] |
| 700 | 53.76 | 67.26 | 1.25× | [704] | [1024] |
| 2100 | 118.81 | 292.12 | **2.46×** | [2048, 64] | [2048, 512] |
| 512 | 45.87 | 46.03 | 1.004× | [512] | [512] |
| 2048 | 100.76 | 100.18 | 0.994× | [2048] | [2048] |
| 4096 | 187.59 | 188.12 | 1.003× | [2048, 2048] | [2048, 2048] |

The three zero-waste shapes land on 1.00×, so the effect is padding and not
arm-to-arm drift. **The mechanism is sourced, and it is not "wasted compute on
padding":** `_MAX_PREFILL_CUDA_GRAPH_PADDING_FACTOR = 2` means that if
`_pad_to_bucket(n) > 2n` the graph is **rejected and the forward runs eager** —
so arm C loses the graph entirely and falls back to the 196 ms path. The cap
predicts all 13 cells exactly. Since almost every real prompt has a partial last
chunk, arm C as configured is not recommendable.

### 7.5 Long context, headroom, and the upstream rule

The win holds well past 3% of the window, and **all arms survive the top of the
1 M window** (ISL 1 046 528 = 99.80%), prefill *and* the prefill → first-decode
transition, with zero OOM lines:

| ms/forward | 32 k | 128 k | 256 k | 512 k | 768 k | 1046 k |
|---|--:|--:|--:|--:|--:|--:|
| A_eager | 251.6 | 239.1 | 219.6 | 280.8 | 310.3 | 423.9 |
| B | 94.7 | 107.5 | 130.0 | 179.1 | 230.3 | 285.1 |
| **speed-up** | **2.66×** | **2.22×** | **1.69×** | **1.57×** | **1.35×** | **1.47×** |

Headroom after capture: A 8.52 GiB, **B 8.27**, C 3.55, E 3.49, **D 2.49** — all
survived. So `mem_fraction_static=0.80` is a **safety property, not only a
cost**: it funds the capture pool out of KV and leaves the activation reserve
essentially at production's level, where arm D looks free only because it takes
the pool out of the reserve.

**On the rule itself.** It is upstream code (`server_args.py:4452`,
`_disable_breakable_cudagraph_if_incompatible()`), not a local patch, and its own
comment says DSV4 *"is BCG-compatible but introduces heavy memory pressure"* —
so it is a **memory guard, not a correctness guard**. That reframes arm B: it is
not overriding a safety invariant, it **satisfies the rule's actual concern by
another means**. The two upstream OOM reports behind it differ from us on every
axis that drives capture-pool pressure (DeepSeek-V4-**Pro** on 2×8 H200, TP16
**DP16 with DP attention on**, DeepEP, EAGLE, `chunked-prefill 8192`,
`mem-fraction-static 0.88`) — and the thread's own recommended mitigation
(*"reduce `--mem-fraction-static` to 0.7–0.8"* plus a smaller graph batch) **is
exactly what arm B does**.

🔴 **But the rule is not stale-and-resolved, and reading it that way would be an
error.** Upstream [PR #25195](https://github.com/sgl-project/sglang/pull/25195)
enables BCG for DSV4 with DP attention and reports **+11.80% output throughput /
−13.27% median TPOT with GSM8K stable at 0.953** — same direction as our ITL
result, independently, *and with a quality check we have not run*. It **did not
relax this gate**: the rule is still present in our build, whose HEAD is dated
two months after that merge. **An upgrade does not give BCG by default; the
explicit flag stays required.**

### 7.6 What the BCG measurement does not answer

* **Steady-state ITL under realistic arrivals.** The 1.92× is a **burst**
  figure — the grid launches all BS requests simultaneously. Named in the
  original write-up as the highest-value remaining measurement.
* **Output quality.** Graphing prefill is bitwise-perturbing in principle
  (expert-count-dependent MoE GEMM tiling; `index_topk=512` selection can flip
  on last-bit differences). The padding machinery is correct and arm C's 2× cap
  means it falls back to eager rather than computing on padding — but
  "different" is measurable and "worse" is not, and no quality run exists.
* **Jitter.** B had the tightest repeat (0.0% vs C 7.8%, D 6.8%) at n=2 — a
  hypothesis, not a finding.
* 🔴 **Do not read a speed ranking among the graph arms** from §7.5. At ISL a
  multiple of 2048 all four arms hold a 2048 bucket and replay the *identical*
  graph, so the predicted between-arm spread is zero; observed spread at the top
  of the window (6.3%) is smaller than one arm's own repeat spread (7.8%). An
  earlier draft claimed "B is fastest at full context" from an n=1 ordering;
  that is withdrawn. B is chosen for **headroom** and for surviving the padding
  probe.

---

## 8. Should BCG be re-tested on the benchmarks pipeline?

**Yes — but as a different experiment, not a repeat.** The BCG numbers are good
measurements of what they measured; three specific things the aiperf pipeline can
settle that the in-pod probe structurally cannot, and one thing that would be
pure spend.

### 8.1 The strong reason: BCG's ITL win is measured on the shape that inflates it

This is the argument that makes the re-test worth GPU time.

BCG's ITL win **is prefill-preemption relief** — that is its stated and
controlled mechanism (§7.2: zero effect at BS 1, growing to 1.92× at BS 64,
because at BS 1 nothing is prefilling during a request's decode). And the shape
it was measured on is **ISL 2048 / OSL 128** — the exact shape §6.1 proved is
*not a decode metric* above concurrency 1, where marlin's own ITL reads 53.45 ms
against the reasoning shape's 14.35 ms at conc 64.

So BCG's headline output-speed number is measured on the shape that **maximises
the very quantity BCG removes.** Note the asymmetry with §6.1: for the MoE
kernel that shape inflated the *noise* and inverted two signs; for BCG it
inflates the *effect*. Both shapes are legitimate — they just describe different
traffic, and the campaign has already been burned once by not saying which.

**Pre-registered prediction, to be committed before the run:** on the reasoning
shape (ISL 1000 / OSL 4000 — 1 prefill token per 4 decode tokens instead of 16
per 1), BCG's ITL gain at conc 64 **shrinks from 1.92× to under 1.20×**, and
plausibly to ~1.00×. If it holds, BCG is a **TTFT lever with an output-speed
bonus that only appears on prefill-heavy traffic** — which changes how it is
sold and where it is enabled, without changing that it works. If it does *not*
hold, BCG is a large output-speed lever on every shape and immediately outranks
the MoE-kernel work (1.92× dwarfs 4.7%).

Either outcome is decision-relevant, which is the test for whether an experiment
is worth running.

### 8.2 Burst vs steady state

The grid launches all BS requests simultaneously and measures the resulting
burst. aiperf issues 11 waves with warmup, so it measures a loaded-but-not-
synchronised regime that is closer to production arrivals. The original write-up
names this its largest hole. Same instrument, no new tooling.

### 8.3 The confounded arm

Arm B moves **two** flags: `--cuda-graph-backend-prefill breakable` *and*
`--mem-fraction-static 0.80`. §7.5 argues convincingly that the memory flag is
the funding mechanism and not an independent speed lever, and arm D (graphs on,
KV untouched) exists — **but D was only ever measured for ms/forward at long
context, never for ITL/TTFT on a serving grid.** So no arm isolates the graph
flag from the memory funding on the metric that matters. One extra phase fixes
it, and it is worth having before anyone quotes "BCG costs 12.3% of KV" as if
the 12.3% were intrinsic to graphing.

### 8.4 What A–B–A pairing does *not* buy here — stated so it is not oversold

BCG's effects are 1.2–4.0×. Measured drift on this node is ≤0.5%. **Pairing is
not needed to resolve a 92% effect**, and claiming otherwise would misuse the
lesson from §6.1. What pairing actually buys is narrower and still worth it:

1. the two levers end up on **one instrument**, so "+4.7% from humming" and
   "+92% from BCG" become comparable numbers rather than two dialects;
2. it makes the **small** cells quotable — particularly the BS-1 ITL 1.00×,
   which is the mechanism control and currently rests on a separate n=20 probe
   run with a different client.

### 8.5 🔴 The cell nobody has measured: both levers at once

Production would ship both. They act on ITL through **different mechanisms** —
expert-GEMM time versus prefill head-of-line blocking — so naive addition
predicts roughly `1.92× × 1.05×` at conc 64, and **nobody has checked for
interaction.** There are concrete reasons to expect one:

* **Their costs stack on the same budget.** BCG at `mfs 0.80` takes 12.3% of KV;
  humming takes another ~1% of the pool for workspace (6 473 216 → 6 411 264).
  Rule 2 already **fails** for BCG alone above ISL ≈ 89 k (§7.3), so the
  combination is where that failure gets worse, and it is measurable rather than
  arguable.
* **BCG reduces prefill interference, which changes the decode batch
  composition** the MoE kernel sees — and humming's residual weakness is
  precisely tile selection versus tokens-per-expert (§3.3). Removing preemption
  could move the effective batch distribution and therefore the sign of the
  8–16 loss.

This is the single highest-value untested configuration in the report, and it is
**one extra phase in the same pod.**

### 8.6 Concrete proposal

One 4-phase paired arm, reasoning shape, conc 1 / 8 / 16 / 32 / 64:

    A_eager  →  B_bcg  →  C_bcg+humming0.1.13  →  A_eager

plus a second, much cheaper pass on a prefill-heavy shape (ISL 2048 / OSL 128,
or a chat-like 8192 / 256) so both regimes are on the record and §8.1's
prediction is scored on both. Optionally a fifth phase for arm D (graphs on at
`mfs 0.874`) to settle §8.3.

**Cost.** The 3-phase humming reasoning ladder took **2.05 h** of 8×H100 for 5
points, so 4 phases is ~2.7 h and the prefill-heavy pass is cheap (OSL 128
dominates the wall time downward). Call it **half a day of one node** for both
shapes.

**Build on what exists — this needs no new harness.** `_gen_paired_arm.py`
already parameterises workflow, shape, concurrency ladder and humming source
from one template, with the version gate, the shape gate, the engagement proof
and the drift-aware comparator all in place. Adding a *server-flag* dimension to
the phase table is a generator change, not a new instrument. Reusing it also
keeps every gate that caught the defects in §6.

⚠️ **If `--enable-mixed-chunk` ever enters this design, do not treat it as
additive with BCG.** Both attack the same term — the ~121 ms per-forward launch
tax — by different routes: BCG graphs it away, fusion avoids launching a second
forward at all and lets prefill tokens ride the decode step's weight loads. Both
bottom out at the prefill compute floor (~75–90 ms), so their gains **overlap**.
A phase table that includes both must measure the combination, not sum the arms.

**And add the quality gate.** Upstream ran GSM8K for their BCG config and got
0.953 stable; we have run nothing on either lever. The benchmarks repo *is* the
eval pipeline, so this is close to free, it is the only correctness check on a
config we would actually ship, and per this repo's evaluation-harness contract
it needs the fail-closed harness check (tokenizer/template hashes, reasoning
mode, task aliases, sampling params, sample-manifest hash) recorded before GPU
launch. A paired subset is valid for config-to-config decisions without being
comparable to a public leaderboard score — do not conflate those.

### 8.7 What should *not* be re-run

Spend without a question attached:

* **The ms/forward prefill ladder** — a per-forward server-side quantity, well
  measured, and not what aiperf is for.
* **The arm C padding probe.** Arm C is dead on a **sourced mechanism**
  (`_MAX_PREFILL_CUDA_GRAPH_PADDING_FACTOR = 2`) that predicts all 13 cells
  exactly. Re-measuring a mechanism you can read is not evidence.
* **The capacity table** — it comes from `max_total_num_tokens` at boot, not
  from a benchmark.
* **Long context ranking among graph arms** — §7.6 already establishes the arms
  are indistinguishable there and that between-arm spread is smaller than
  within-arm repeat spread. More samples of a null.

---

## 9. Candidate next steps

Ranked by expected value on the primary objective, none of them started:

1. **DSpark speculative decoding.** Untouched on DSV4-Flash, and the M3 study
   (goal 2h) measured **1.21–2.53×** per-user decode with zero quality cost by
   construction, with draft depth per traffic class as the dominant lever. That
   is an order of magnitude more output speed than either lever in this report.
   ⚠️ M3's numbers were measured on vLLM; none of that tuning transfers
   mechanically to SGLang.
2. 🆕 **`--enable-mixed-chunk` — one flag, currently `False`.** SGLang does not
   fuse prefill and decode by default, so every prefill forward is a full
   generation stall for the whole decode batch; the maintainers describe this
   flag as reducing inter-token latency "for some workloads"
   ([discussion #1163](https://github.com/sgl-project/sglang/discussions/1163)),
   and it is the same mitigation vLLM applies by default (decode-priority
   scheduling). Our decode is the most favourable possible case for it —
   0.66–1.5 tokens per expert means the MoE GEMMs are near-pure weight
   streaming, so fused prefill tokens ride weight loads already being paid for.
   **Falsifiable:** the mean effect is bounded between zero (if the fused step is
   perfectly additive — work conservation says scheduling alone cannot reduce a
   mean) and ~the same gain BCG delivers. The tail effect is near-certain. See
   the substitution caveat in §8.6.
3. **The combined BCG + humming arm** (§8.5–§8.6), with the shape pair.
4. **The gate/up tuning-config ablation** — keep 0.1.13's down-proj config,
   force gate/up back to `(8,128,256)`. One line, 1 GPU, no upgrade.
5. **Upstream #65** (`a81c45e`, "Fix SM90 indexed A16 **large-M** scheduling",
   unreleased — `main` only). Large-M gate/up is exactly where 0.1.13 still
   loses (§3.3), so this is the cheapest candidate fix for the conc 8–16 deficit.
6. **A quality run on both levers.** Neither has one. Upstream has one for BCG
   and we do not.
7. **Report the pin cost upstream.** SGLang's `humming-kernels==0.1.10` pin
   costs up to **20.8% of decode ITL** on this model, worst at production
   concurrency; upstream commits #57/#58/#59 are by a vLLM maintainer, so a pin
   bump is not a fork. 🔴 **Outward-facing — no issue or merge request has been
   opened, and none should be without explicit sign-off.**
8. **Prefill/decode disaggregation — last, not first.** `disaggregation_mode` is
   `'null'` in our build (supported; `transfer_backend='mooncake'`,
   `ib_device` unset). It is the textbook fix for §6.1's interference and it
   removes the whole term from the decode node's ITL — but it buys output speed
   with **hardware**, needs fabric configuration, and changes the topology, which
   invalidates both regime baselines (the same objection already recorded against
   `--enable-dp-attention`). Its prize also **shrinks as the cheap items land**:
   ~43 ms of inflation at conc 64 today, ~20 ms after BCG. Price it against
   what's left, not against today's number. It remains the only lever that makes
   ITL a property of the decode engine rather than of the traffic mix.
9. **Backlog, cheap:** the BS-32 operator re-run at 6 rounds on both arms;
   BS 2/4 to locate the crossover; `ncu dram__bytes.sum` for roofline closure;
   the `grouped` gemm-type arm; the four-phase
   `marlin → 0.1.10 → 0.1.13 → marlin` arm to remove the anchor mediation.

---

## Artifacts and provenance

Raw evidence and the full analysis live in the **`AICloud/opprof`** repo, branch
`feature/ds-v4-flash-sglang-rebaseline` (this report's numbers were taken from
commit `327c935`), under
`docs/operator_analysis/ds-v4-flash-sglang/`:

| this report | source document |
|---|---|
| §2.1 | `decode/kernel-ab/serving-ladder-reasoning-aba.md` |
| §2.2 | `decode/kernel-ab/serving-ab-conc1.md` |
| §2.3, §3.6 (0.1.10 row) | `decode/kernel-ab/serving-pin-cost.md` |
| §3 | `decode/kernel-ab/humming-0113-results.md`, `humming-decode-loss-investigation.md` |
| §6.1 | `decode/kernel-ab/serving-ladder-static-aba.md` (headline superseded, numbers retained) |
| §7 | `prefill/prefill-graph-ab.md`, `prefill/recon-kernel-table.md` |
| adoption state of every flag | `adoption/README.md`, `adoption/moe-runner-backend/README.md` |

**Data directories** (`docs/operator_analysis/ds-v4-flash-sglang/data/`, curated
per its own `README.md`): `serving-paired-reasoning-aba/`,
`serving-paired-reasoning-img0110/`, `serving-marlin-c1/`,
`serving-humming0113-c1/`, `recon-kernel-humming0113-decode/`,
`prefill-graph-ab/`. Each serving directory holds `run.log` (every gate
verdict), `server.log` (the engagement proof, trimmed of decode/prefill batch
spam), `server_info.json` (all resolved server args — nothing rests on the flag
we *passed*), `HARNESS_VERSION`, `profile_export_aiperf.csv`, `grid.csv`,
`points.csv`, `perf_summary.json`.

⚠️ **`profile_export.jsonl` is load-bearing** for the reasoning arms — the
within-point drift fits and every standard error in §2 were recomputed from the
per-request records. The full 206–207 MB trees are archived outside any repo at
`_bench_data_archive/pvc-reasoning-aba-20260823.tgz` and
`pvc-reasoning-img0110-20260823.tgz` (74.5 M / 74.9 M, `gzip -t` verified, 177
files each). ⚠️ `server_metrics_export.*` is **not usable** for the serving arms:
the Prometheus scrape was unreachable throughout. No figure here comes from it.

**Manifests.** `opprof/k8s/experiments/ds-v4-flash/serving/` (paired arms,
generated by `_gen_paired_arm.py` from `_serving_paired.tmpl.yaml`) and
`…/profiling/` (nsys arms, generated by `_gen_kernel_arm.py`; prefill graphs:
`44-prefill-graph-ab-sglang.yaml`).

**Benchmark harness.** `AICloud/benchmarks`, branch `duy-branch`, revision
`4529676` (collaborator-managed), aiperf 0.11.0 from the PVC venv.
`HARNESS_VERSION` is stamped into every arm's output directory.

**Environment.** sglang 0.5.17, torch 2.11.0+cu130, sgl_kernel 0.4.5, driver
580.159.03, 8×H100 80 GB HBM3, TP8 DP1 EP1 PP1, KV `fp8_e4m3`, context
1 048 576, chunked prefill 2048, `max-running-requests` 64.
