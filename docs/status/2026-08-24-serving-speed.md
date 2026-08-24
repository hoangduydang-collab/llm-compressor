# Aug 18 – Aug 24

## Duy

### What I worked on

Improving serving speed for DeepSeek-V4-Flash on SGLang — both output speed (how
fast text streams out for one user) and time-to-first-token, which I weight roughly
equally. Production moved this model from vLLM to SGLang on 17 Aug, so none of our
existing performance work for it still applies and the whole picture had to be
re-measured on the new engine.

**Two levers measured.** Prefill GPU graphs, which SGLang switches off for this
model family — five runs, including a variant that would have made the trade free
and lost on measurement. And the mixture-of-experts kernel: `marlin` (today's
default) against `humming` at two library versions, five concurrency levels each.
Every run measures the current configuration, then the change, then the current
configuration again in one session, so machine drift is measured rather than
assumed.

**Diagnosing why the first attempt was wrong.** Our first kernel comparison
produced two findings and both had the wrong sign, despite passing their own
checks. Finding out why and rebuilding the measurement was the most useful thing I
did. Also wrote the study up in three documents — the previous handoff of this
workstream was under-documented and it cost us time.

### Key results/outcomes

**The kernel win is real, but it is the library version, not the setting.**
`humming` 0.1.13 gives **+22.3%** output speed at a single user — 11 seconds off an
8,000-token answer — and **+4.7%** at our production load of 64 users.

**The cheap version of it is a regression.** SGLang pins `humming` 0.1.10, and with
that version the same setting is **13.4% slower** at production load; the pin costs
up to **20.8%**, worst exactly where production runs. So this needs an upstream
version bump or a dependency we carry, not a config change.

**Prefill graphs: prompt processing 2.2× faster**, time-to-first-token **2.1–4.0×**
better, output speed **1.9×** better at 64 users. Cost is **12.3% of the
conversation-history cache** — invisible below ~89,000 tokens of context, real
above it. That 1.9× was measured on the traffic shape that flatters it most; on
long-reasoning traffic I expect ~1.13×, which is next week's first measurement. The
time-to-first-token gains do not depend on that question, so the lever is a clear
win either way.

**The first kernel measurement was wrong, and I can now say why.** Under load on
short-answer traffic, **80% of what we were calling inter-token latency was
actually other requests' prompt processing** — when the server handles someone
else's prompt, every user mid-answer stops. The metric was not measuring what its
name says. This is a known and named phenomenon, a *generation stall*
([Sarathi-Serve](https://www.usenix.org/system/files/osdi24-agrawal.pdf),
[DistServe](https://www.usenix.org/system/files/osdi24-zhong-yinmin.pdf), both
OSDI '24), so the fix does not need inventing:
[vLLM ships the mitigation on by default](https://docs.vllm.ai/en/stable/configuration/optimization/),
[SGLang leaves it opt-in](https://github.com/sgl-project/sglang/discussions/1163)
and we have it off. The rebuilt measurement resolves every point at 20–139× its own
error bar and reproduced to 0.33% on a fresh boot three hours later.

**Nothing is adopted, and neither lever has a quality check** — that is the gate on
shipping either one.

**Two decisions I need:** the traffic mix to optimise for, since the graph lever is
worth 1.9× on short prompts against ~1.13× on reasoning traffic and its memory cost
only bites above ~89,000 tokens of context; and whether to carry a non-standard
kernel version or push an upstream bump, since there is no third option that keeps
the win.

**Speculative decoding** is probably the largest remaining lever here, and Alex is
already building DSpark training for this model, so I have deliberately left it
alone.

### Plan for the next two weeks

**Week 1 (25–29 Aug) — GLM-5.2 through our own quantization pipeline**

1. **Quantize GLM-5.2 ourselves, then validate it the way we validated
   MiniMax-M3** — quantization run, a paired quality evaluation against BF16, then
   a performance benchmark. We have evaluated *other people's* GLM-5.2
   quantizations but never produced one, so this is the first end-to-end pass on a
   family that is not M3, where our gates are still M3-specific. It also prepares
   **day-0 support for GLM-5.3**, which may need W4A8 produced in-house if no
   release exists at launch. Cheaper than it sounds: the earlier GLM-5.2 evaluation
   already built the harness and both comparators.

Then, as the week allows:

2. **The graph lever on reasoning traffic, both levers together, and a larger
   prompt-chunk size.** Our chunk size is an out-of-memory workaround rather than a
   tuning choice, and it makes a 32,000-token prompt take 17 passes instead of
   about 5 — on paper that alone captures ~80% of the graph lever's benefit, though
   the two compete for the same memory at long context.
3. **The free scheduler setting** that processes prompts without fully stopping
   users mid-answer.
4. **A one-line kernel tuning change** for the small loss at moderate load.

**GLM-5.2 is the only item with a clock I do not control**, so if the week is tight
items 2–4 slip rather than the quantization run.

**Week 2 (1–5 Sep)** — quality evaluation on both serving levers, fitted to Kyle's
and Zhou Yu's pipeline rather than a parallel one; the adoption recommendation; the
GLM-5.2 write-up; and anything that slipped. If the dependency decision lands,
prepare the upstream report on the pin cost — a public statement about a
third-party project, so not filed without sign-off.
