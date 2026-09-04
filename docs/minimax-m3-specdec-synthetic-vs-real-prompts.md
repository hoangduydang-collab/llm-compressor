# MiniMax-M3 speculative decoding: prompt source did not change acceptance

This note isolates one question: does using real prompts instead of synthetic
random-token prompts materially change EAGLE3 acceptance on our MiniMax-M3 arm?

## Takeaway

At k=3 and temperature 0.6, acceptance was **2.450** on the synthetic sweep
and **2.473** on natural ShareGPT prompts: a **+0.9%** difference that the
primary study treats as noise. This is a cross-window comparison with different
acceptance instrumentation, not a paired statistical test. Prompt source alone
is therefore not a supported explanation for a differing spec-dec result.

## What was compared

| Arm | Prompt construction | Input length | Output policy |
|---|---|---:|---|
| Wave 1 synthetic | AA-style generator produces random tokens, not natural-language text | 1k and 10k | Natural stopping |
| Wave 2 real | aiperf ShareGPT loader sends the first user message from each conversation | Mean ≈227 | Natural stopping |

Both used the in-house MiniMax-M3 GPTQ W4AFP8 target, the
`Inferact/MiniMax-M3-EAGLE3` drafter, vLLM 0.24.0, Humming indexed 0.1.10,
FP8 KV cache, one 8×H100 node at TP8/EP8, and a k=0 versus k=3 comparison.
Neither arm used `ignore_eos` or forced a minimum output.

## Results

| Workload | Accepted length | Per-position acceptance | k=0 → k=3 tok/s/user | Decode speedup |
|---|---:|---|---:|---:|
| Synthetic AA-style, 1k input / conc-1 | 2.450* | 0.70 / 0.46 / 0.29 | 137.5 → 236.0 | 1.72× |
| Natural ShareGPT, conc-1 | 2.473 | 0.690 / 0.459 / 0.324 | 137.9 → 249.8 | 1.81× |

\*Wave 1's 2.450 is an arm-level mean from periodic `SpecDecodingLogging`;
Wave 2's 2.473 is a per-cell Prometheus counter delta. The matching controls
(137.5 versus 137.9 tok/s) reduce concern about major drift, but do not make
the two measurements a paired A/B.

The real-prompt support point at concurrency 10 was 2.503 accepted length and
1.66× per-user speedup (77.7 → 128.8 tok/s).

## What to compare next

Before attributing a difference to prompt source, compare these controls:

1. **Output policy.** Forced 8k continuation (`ignore_eos`) raised acceptance
   from 2.473 to 3.286 (**+33%**); this was the strongest observed effect.
2. **Temperature.** Greedy decoding increased acceptance to 2.575 (+4%).
3. **Draft depth.** The measurements above use k=3; a different k changes both
   accepted length and draft/verify cost.
4. **Target and drafter pair.** EAGLE3 conditions on target hidden states, so
   a changed checkpoint or drafter revision can change acceptance.
5. **Topology and serving stack.** Kernel, TP/EP shape, KV-cache policy, and
   load can change throughput even when acceptance is similar.

## Inspect the evidence

The original results are in
[`docs/m3-specdec-eagle3.md`](m3-specdec-eagle3.md); the pre-declared design is
[`M3_SPECDEC_EAGLE3_PLAN.md`](../M3_SPECDEC_EAGLE3_PLAN.md). The raw artifacts
are in a private archive on `138.252.188.36`; request an owner-provided mount
or export rather than assuming the former cluster path is available.

| Window | What to inspect |
|---|---|
| `20260727T061506Z` | Wave 1 `aggregate.json`, `arm-k3/spec-metrics.log`, and `aa-sweep.log` |
| `20260727T064934Z-wave2` | Wave 2 `aggregate.json`, `arm-natural-k3/metrics/`, and aiperf export |

Use `pipeline/specdec_aggregate.py` for Wave 1 and
`pipeline/specdec_wave2_aggregate.py` for Wave 2. Wave 1's migrated logs
re-aggregate acceptance but not AA throughput because its external
`aa_sweep_summary.json` was not migrated; use its preserved `aggregate.json`
for the reported speed cell.
