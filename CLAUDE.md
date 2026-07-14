# Working principles for this repo

## Prime directive (applies to everything)

**Never waste effort on something likely already done by others. Always prioritize
finding and building upon reputable existing resources instead of building from
scratch.**

This is the planner's first principle and it applies to *everything* — a new
feature, a bug fix, a performance win, a data pipeline, an eval harness. Before
you design or implement anything non-trivial:

1. **Search first, build second.** Check whether the library we already use, a
   reputable framework, a paper with released code, or an existing PR already
   solves it. Spend the cheap effort of looking before the expensive effort of
   building.
2. **Check what we already have.** Inspect this fork's own code and dependencies
   before assuming a capability is missing. Confirm absence in the actual source —
   do not infer it from a version string, a filename guess, or memory.
3. **Confirm it's genuinely unsolved before writing from scratch.** If you think
   you must build something bespoke, first state what you searched and why nothing
   fits. "Nobody has done this" is a claim that needs evidence.
4. **Build *on* reputable resources.** When something exists, adopt/adapt/rebase
   onto it (upstream libs, released implementations) rather than reimplementing.
   Bespoke code is a last resort, justified only after 1–3.

### Worked example (why this principle is here)

The MiniMax-M3 calibration speed-up (`M3_QUANT_SPEEDUP_PLAN.md`) burned many turns
building bespoke expert-parallel code (`pipeline/ep_moe.py`,
`pipeline/expert_scatter.py`, `pipeline/bench_expert_scatter.py`) before
discovering that **multi-GPU calibration for both AWQ and GPTQ already existed in
this very fork** and only needed a `torchrun` launch. The detour came from two
unchecked assumptions (a misread version string; an unverified memory fear). The
fix — searching reputable sources (llm-compressor v0.10, GPTQModel) and reading
our own code — would have cost a fraction of the effort if done first. Those
bespoke files are now shelved.

## Roles

- **Planner** (design + verify, local, CPU-only): investigates, plans, implements
  and unit-tests locally, writes handoffs. Reads reputable sources and existing
  code before building. On `duy-branch`, commit + push after finishing an
  implementation without being asked.
- **Executor** (cluster, GPU): runs the cluster/GPU jobs the planner hands off.
  The planner must not assume GPU access; diagnostics must not allocate GPUs for
  full quality eval / re-quantization without cause.

## Handoffs

Cross-session/agent state lives in the repo (a fresh agent does not see prior
chat or personal memory). Current speed-up conclusion + next steps are in the
top "HANDOFF" section of `M3_QUANT_SPEEDUP_PLAN.md`. Other `*_HANDOFF.md` files
hold task-specific executor procedures.
