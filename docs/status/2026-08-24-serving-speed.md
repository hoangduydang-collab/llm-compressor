# Aug 18 – Aug 24

## Duy

### What I worked on

The goal was improving **output speed** for DeepSeek-V4-Flash on SGLang — how fast
text streams out for one user, which we rank first; time-to-first-token second.
Production moved this model from vLLM to SGLang on 17 Aug, so none of our existing
performance work for it still applies, and the whole picture had to be measured
again on the new engine.

**Profiling the new engine.** Where the time actually goes on SGLang, at the level
of individual GPU operations, for both prompt-processing and text-generation. Our
earlier profiling was all on vLLM, which shares almost none of its serving
machinery with SGLang, so this was a rebuild rather than a refresh.

**Prefill GPU graphs — measured, five runs.** SGLang switches off a GPU
optimisation for this model family. I measured what turning it back on buys, what
it costs, and tested a variant that would have made the trade free — it lost on
measurement and is rejected.

**The mixture-of-experts kernel — measured, three full ladders.** The current
default (`marlin`) against the alternative (`humming`) at two library versions,
five concurrency levels each. Each run measures the current configuration, then
the change, then the current configuration again, in one session — so
machine-condition drift is measured rather than assumed.

**Diagnosing why the first attempt was wrong.** Our first kernel comparison
produced two findings and both had the wrong sign, despite passing their own
checks. Tracking down why, and rebuilding the measurement, was the most useful
thing I did this week.

**Documentation.** Three documents — an engineering report, a brief for whoever
picks the work up, and a status page — because the previous handoff of this
workstream was not documented well enough and it cost us time.

### Key results/outcomes

**The kernel win is real, but it is the library version, not the setting.**
`humming` 0.1.13 gives **+22.3%** per-user output speed at a single user — 11
seconds off an 8,000-token reasoning answer — and **+4.7%** at our production load
of 64 concurrent users.

🔴 **The cheap version of it is a regression.** SGLang pins `humming` 0.1.10, and
with *that* version the same setting is **13.4% slower** at production load. The
pin costs up to **20.8%** of output speed, worst exactly where production runs. So
this is not a config change — it needs a version bump upstream or a dependency we
carry ourselves.

**Prefill graphs: prompt processing 2.2× faster.** Time-to-first-token improves
**2.1–4.0×** depending on prompt size, and at 64 users output speed improves
**1.9×** on short-answer traffic. The full 1-million-token context still works.

**Caveat on that 1.9×:** it was measured on short-prompt traffic, which is the
shape that flatters this lever most. On long-reasoning traffic I predict closer to
**1.13×** — still real, not 90%. Not yet measured; it is the first thing next week.

**Cost of the graph lever: 12.3% of the conversation-history cache.** Below
~89,000 tokens of context that costs nothing observable, because our 64-user limit
binds first. Above it, the cost is real, and at full 1M context maximum
concurrency drops from 6 users to 5.

**Our first kernel measurement was wrong, and I can now say exactly why.** On
short-answer traffic under load, **80% of what we were calling "inter-token
latency" was actually other requests' prompt processing** — when the server
processes someone else's prompt, every user mid-answer is stopped. So the metric
was not measuring what its name says, and two of three findings came out
backwards. The rebuilt measurement, on a traffic shape where the metric means what
it says, resolves every point at 20–139× its own error bar, with machine drift
under 0.5%. Re-running an entire comparison three hours later on a fresh boot
reproduced it to within 0.33%.

**That interference is a well-known problem and we are on the wrong side of the
defaults.** It is the founding premise of two OSDI '24 papers; vLLM ships the
mitigation switched **on** by default, SGLang leaves it opt-in, and we have it
off. That is one setting, no new hardware, and untested by us — high on next
week's list. Nothing here needs inventing.

**Nothing has been adopted, and neither lever has a quality check.** Upstream ran
a reasoning benchmark on their own graphs-on configuration and it held steady; we
have run nothing. That is the gate on shipping either lever.

**What I need from the team:**
1. **The traffic mix we should optimise for.** The graph lever is worth 1.9× on
   short-prompt traffic against an estimated 1.13× on reasoning traffic, and its
   memory cost only bites above ~89,000 tokens of context. Both the right
   configuration and the expected gain depend on the answer.
2. **A decision on the kernel dependency** — carry a non-standard library version
   ourselves, or push for an upstream version bump. There is no third option that
   keeps the win.

**Coordination — speculative decoding.** This is the largest remaining lever on
output speed and **Alex is already building DSpark training for this same model**,
so I have deliberately left it alone rather than duplicating it. One useful
handover from my side: unlike almost everything else, the draft-model **acceptance
rates from our old vLLM work do carry over** to SGLang, because they are a
property of the checkpoint rather than the engine. They say the best draft depth is
**5 tokens ahead**, where the vLLM deployment was running **7**. SGLang also
already reports every acceptance metric needed to tune this, so it should be
directly observable rather than inferred.

### Plan for the next two weeks

**Week 1 (25–29 Aug) — close the open measurement, then try the free settings**

1. **Measure the graph lever on reasoning traffic, and both levers together.**
   Settles the 1.9× vs 1.13× question, and tells us whether the two levers add up
   — nobody has run them together and production would use both. Half a day on one
   8-GPU node.
2. **Try the one free scheduler setting** that lets the server process prompts
   without fully stopping users mid-answer. One setting, ~3 hours. It may overlap
   with the graph lever rather than add to it, which the same run will show.
3. **The one-line kernel tuning change** aimed at the small loss we measured at
   moderate load. ~1 hour on a single GPU.

**Week 2 (1–5 Sep) — quality gate and an adoption recommendation**

4. **Quality evaluation on both levers** — the gate on shipping either. This
   depends on the evaluation pipeline rather than anything of mine, so I will fit
   it to Kyle's and Zhou Yu's work instead of building a parallel one.
5. **Write the adoption recommendation**, with the traffic-mix answer folded in
   and the dependency decision reflected.
6. **Hand the speculative-decoding findings to Alex** — the acceptance rates, and
   the draft-depth 5-versus-7 discrepancy.

**Also:** if the dependency decision lands, prepare the upstream report that
SGLang's pinned kernel version costs up to 20.8% of output speed on this model. It
would likely be welcome — the relevant upstream changes came from a maintainer of
another major inference engine — but it is a public statement about a third-party
project, so nothing gets filed without sign-off.
