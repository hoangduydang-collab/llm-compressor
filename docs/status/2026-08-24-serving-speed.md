# Aug 18 – Aug 24

## Duy

### What I worked on

The goal was improving **serving speed** for DeepSeek-V4-Flash on SGLang — both
how fast text streams out for one user (output speed) and how long they wait for
the first word (time-to-first-token). I weight the two roughly equally; most of
the measurement effort went into output speed because it is the harder of the two
to move.

Production moved this model from vLLM to SGLang on 17 Aug, so none of our existing
performance work for it still applies and the whole picture had to be measured
again on the new engine.

**Profiling the new engine.** Where the time actually goes on SGLang, at the level
of individual GPU operations. Our earlier profiling was all on vLLM, which shares
almost none of its serving machinery, so this was a rebuild rather than a refresh.

**Prefill GPU graphs — five runs.** SGLang switches off a GPU optimisation for this
model family. Measured what turning it back on buys and what it costs, plus a
variant that would have made the trade free — it lost on measurement and is
rejected.

**The mixture-of-experts kernel — three full ladders.** The current default
(`marlin`) against the alternative (`humming`) at two library versions, five
concurrency levels each. Every run measures the current configuration, then the
change, then the current configuration again in the same session, so
machine-condition drift is measured rather than assumed.

**Diagnosing why the first attempt was wrong.** Our first kernel comparison
produced two findings and both had the wrong sign, despite passing their own
checks. Finding out why, and rebuilding the measurement, was the most useful thing
I did this week. Also wrote up the whole study in three documents, since the
previous handoff of this workstream was under-documented and it cost us time.

### Key results/outcomes

**The kernel win is real, but it is the library version, not the setting.**
`humming` 0.1.13 gives **+22.3%** per-user output speed at a single user — 11
seconds off an 8,000-token answer — and **+4.7%** at our production load of 64
users.

🔴 **The cheap version of it is a regression.** SGLang pins `humming` 0.1.10, and
with that version the same setting is **13.4% slower** at production load. The pin
costs up to **20.8%** of output speed, worst exactly where production runs. So this
is not a config change: it needs an upstream version bump or a dependency we carry.

**Prefill graphs: prompt processing 2.2× faster**, time-to-first-token **2.1–4.0×**
better depending on prompt size, and output speed **1.9×** better at 64 users. Cost
is **12.3% of the conversation-history cache**, which is invisible below ~89,000
tokens of context (our 64-user limit binds first) and real above it.

⚠️ **That 1.9× was measured on the traffic shape that flatters it most.** On
long-reasoning traffic I predict closer to **1.13×** — still real, not 90%. Not yet
measured; first thing next week. The time-to-first-token gains do not hinge on
that question, so this lever is a clear win either way.

**The first kernel measurement was wrong, and I can now say why.** Under load on
short-answer traffic, **80% of what we were calling "inter-token latency" was
actually other requests' prompt processing** — when the server handles someone
else's prompt, every user mid-answer is stopped. The metric was not measuring what
its name says. The rebuilt version resolves every point at 20–139× its own error
bar, and re-running a whole comparison three hours later reproduced it to 0.33%.
This is a **known and named** phenomenon — a *generation stall*
([Sarathi-Serve](https://www.usenix.org/system/files/osdi24-agrawal.pdf),
[DistServe](https://www.usenix.org/system/files/osdi24-zhong-yinmin.pdf), both
OSDI '24) — so the fix does not need inventing:
[vLLM ships the mitigation on by default](https://docs.vllm.ai/en/stable/configuration/optimization/),
[SGLang leaves it opt-in](https://github.com/sgl-project/sglang/discussions/1163)
and we have it off. **One setting, untested, no new hardware.**

**Nothing is adopted, and neither lever has a quality check.** Upstream ran a
reasoning benchmark on their own graphs-on configuration and it held; we have run
nothing. That is the gate on shipping either one.

**Two decisions I need:** (1) **the traffic mix to optimise for** — the graph lever
is worth 1.9× on short-prompt traffic against ~1.13× on reasoning traffic, and its
memory cost only bites above ~89,000 tokens of context, so both the configuration
and the expected gain depend on the answer; (2) **carry a non-standard kernel
version, or push an upstream bump** — there is no third option that keeps the win.

**Coordination — speculative decoding.** This is probably the largest remaining
lever on output speed, and **Alex is already building DSpark training for this same
model**, so I have deliberately left it alone rather than duplicate it. Worth
knowing that SGLang already reports every draft-acceptance metric needed to tune
it, so it should be directly observable on our side of the stack when he gets
there.

### Plan for the next two weeks

**Week 1 (25–29 Aug) — close the open measurement, then the free settings**

1. **Measure the graph lever on reasoning traffic, and both levers together** —
   settles the 1.9×-vs-1.13× question and whether the two add up, which nobody has
   tested and production would need. Half a day on one 8-GPU node.
2. **Try the free scheduler setting** that lets the server process prompts without
   fully stopping users mid-answer. ~3 hours; it may overlap with the graph lever
   rather than add to it, and the same run shows which.
3. **The one-line kernel tuning change** aimed at the small loss at moderate load.
   ~1 hour, one GPU.

**Week 2 (1–5 Sep) — quality gate, adoption recommendation, quant-pipeline
readiness**

4. **Quality evaluation on both levers**, the gate on shipping either. This depends
   on the evaluation pipeline rather than anything of mine, so I will fit it to
   Kyle's and Zhou Yu's work rather than build a parallel one.
5. **Write the adoption recommendation**, with the traffic-mix answer and the
   dependency decision folded in.
6. **Run GLM-5.2 through our own quantization pipeline.** We have evaluated *other
   people's* GLM-5.2 quantizations but never produced one ourselves. A full
   end-to-end run does two things: it confirms the pipeline works on a model family
   that is not MiniMax-M3 — our pre-quantization and serving gates are still
   M3-specific — and it gets us ready for **day-0 support of GLM-5.3**. If no W4A8
   release exists at launch we would have to produce one ourselves, and launch day
   is the wrong moment to discover a pipeline problem. This is the one item here
   that is not serving work, and it is the one with an external clock on it.

**Also:** if the dependency decision lands, prepare the upstream report that
SGLang's pinned kernel version costs up to 20.8% of output speed on this model. It
would likely be welcome, but it is a public statement about a third-party project,
so nothing gets filed without sign-off.
