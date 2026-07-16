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

- **Planner — brain and decision owner** (stronger agent resources; local,
  CPU-only; no cluster access): owns the heavy reasoning work — research,
  architecture, hypothesis selection, experiment design, local implementation and
  tests, diagnosis, returned-evidence interpretation, and next-step decisions.
  Planner instructions must minimize executor-side dynamic reasoning. Reads
  reputable sources and existing code before building. On `duy-branch`, commit +
  push after finishing an implementation without being asked.
- **Executor — cluster hands and evidence owner** (constrained agent resources;
  direct access to 15+ 8×H100 nodes): runs the planner's prepared cluster/GPU
  work, monitors it, preserves raw evidence, and returns complete results. The
  executor may reason enough to execute safely and capture failures, but does not
  redesign experiments or make strategic decisions unless explicitly authorized.

The planner must not assume GPU access. The executor's cluster access is not a
reason to delegate open-ended analysis to it, and diagnostics must not allocate
GPUs for full quality evaluation or re-quantization without cause.

### Executor cluster scheduler constraint

The current executor cluster accepts top-level `srun` allocations and does not
support `sbatch`. Planner-authored launchers and handoffs for this cluster must
use `srun` from a persistent detached controller (normally `tmux`) and must not
emit `sbatch` commands. A different launch method requires an explicit packet
targeting a separately verified cluster.

### Evaluation harness contract

Every planner packet that spends cluster time on model-quality evaluation must
include a fail-closed, machine-readable harness check before GPU launch. Record
and verify the tokenizer/chat-template hashes, model reasoning mode, task aliases
and harness version, few-shot counts, metrics, generation/sampling parameters,
serving backend/topology, and sample-manifest hash. State separately whether the
run is directly score-comparable to a named public benchmark recipe. A paired
subset can be valid for model-to-model decisions without being directly
comparable to a full public leaderboard score; never conflate those claims.

## Planner–executor protocol

Both roles must read and follow `PLANNER_EXECUTOR_PROTOCOL.md`. It is the
repo-wide source of truth for workflow states, decision authority, execution
packets, evidence returns, retries, deviations, and stop conditions.
Task-specific handoffs supply the experiment details; they do not override the
general protocol unless they explicitly name and justify an exception.

**Full-stack agent configuration:** when a single agent holds both roles *and*
has direct cluster access, it follows `FULL_STACK_AGENT_PROTOCOL.md` instead. That
companion doc collapses the cross-agent handoff ceremony but keeps every
discipline (design sign-off before cluster spend, scientific integrity,
fail-closed gates, honest raw evidence, `srun`-only, Git durability). Revert to
the two-agent base protocol if the task is ever split across separate agents.

## Handoffs

Cross-session/agent state lives in the repo (a fresh agent does not see prior
chat or personal memory). Current speed-up conclusion + next steps are in the
top "HANDOFF" section of `M3_QUANT_SPEEDUP_PLAN.md`. Other `*_HANDOFF.md` files
hold task-specific executor procedures.

Every new or materially revised planner/executor handoff must use the canonical
packet contract. When a handoff is replaced, label the old instructions
`SUPERSEDED` or `HISTORICAL` and point directly to the one active packet.
