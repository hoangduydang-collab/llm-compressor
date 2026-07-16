# Full-Stack Agent Protocol

Protocol version: 1
Scope: repository-wide
Companion to: `PLANNER_EXECUTOR_PROTOCOL.md` (the base protocol)

## What this document is

The base protocol (`PLANNER_EXECUTOR_PROTOCOL.md`) assumes two agents with
asymmetric capabilities: a planner that reasons but cannot touch the cluster, and
an executor that runs cluster work but does not make strategic decisions. They
communicate through Git because neither can see the other's session.

The **full-stack agent** is a single agent that holds **both roles at once** and
has **direct cluster access**. There is no planner↔executor boundary to serialize
across, so the handoff ceremony collapses. This document defines what the
full-stack agent keeps, what it may skip, and where it must still stop for the
user.

If a task is ever split back across two separate agents, revert to the base
protocol — this document does not apply to that configuration.

## Role summary

One agent owns the whole loop:

```text
DESIGN (agree with user) -> RUN -> READ EVIDENCE -> DECIDE -> (repeat or close)
```

- **Brain:** research, hypothesis selection, experiment/gate/topology design,
  local implementation and verification, evidence interpretation, next-step
  decisions.
- **Hands:** run the prepared cluster work directly, monitor it, preserve raw
  evidence, capture failures.

## The one gate that remains: design sign-off

The single hard checkpoint is **before spending real cluster time on a new
experiment**. The full-stack agent must communicate and agree with the user on:

- the objective / decision question;
- the experiment design (method, controls, gates, thresholds);
- the resource topology and rough cost;
- what "done" and "pass/fail" mean.

This happens **once, up front**, per experiment. Cheap diagnostics do not need a
full design negotiation, but they still must not silently escalate into a full
quality eval or re-quantization (see the base protocol's cost discipline).

## Autonomy inside an agreed experiment

Once an experiment is agreed and running, the full-stack agent may modify it
**without further permission**, provided the change **does not contradict the
agreed objective or design**. Communicate the design once; execute and self-repair
freely inside it.

**Automatically allowed (just do it, then note what you did):**

- fix a bug in a launcher, harness, or aggregation script;
- fill in a missing detail the design implied but did not spell out;
- correct a launch command, path, environment variable, or flag;
- handle an infrastructure failure (requeue, node swap, retry a crashed step);
- adjust monitoring/aggregation mechanics;
- any mid-run repair that keeps the same objective and same design.

**Still requires coming back to the user (a new agreement, not a fix):**

- a different objective or decision question;
- a different method, model, dataset, or benchmark;
- moving a gate threshold or changing pass/fail criteria;
- a materially different cost or resource topology;
- anything that would change what the result *means* or whether the comparison
  is valid.

Litmus test: **would this change make me report a different kind of conclusion,
or invalidate the comparison we agreed on?** If yes, stop and re-agree. If it just
makes the agreed experiment run correctly, proceed.

## Ceremony that may be skipped

Because there is no cross-agent Git boundary within a session, the full-stack
agent may skip:

- writing copy-ready command packets for a separate executor to paste;
- the round-trip execution-packet / evidence-packet documents whose only purpose
  is to transfer state between two agents;
- `git pull` / commit-and-push done *solely* to hand a packet across the
  planner↔executor gap;
- the formal `READY_FOR_EXECUTOR` -> `EXECUTING` -> `RETURNED_FOR_ANALYSIS` state
  handshake and its state tokens.

Skipping the ceremony does **not** mean skipping the thinking the ceremony
protected. The design still has to be decisive, the gates still fail-closed, the
evidence still raw and honest.

## Discipline that is NOT skipped

Everything below survives the role merge and is non-negotiable:

1. **Prime directive** — search existing code and reputable prior work before
   building bespoke (see `CLAUDE.md`).
2. **Scientific integrity** — a "fix" that would quietly invalidate a comparison
   is a design change, not a fix; it goes back to the user. Gates stay
   fail-closed. Never combine results produced under contradictory designs.
3. **Honest evidence** — failures reported as failures, skips as skips, with the
   raw logs / return codes / measurements, not just prose. Capture ephemeral
   scheduler and log evidence before it disappears.
4. **Cluster constraints** — this cluster is `srun`-only from a persistent
   detached controller (normally `tmux`); **no `sbatch`**. Quality-eval work
   needs a fail-closed, machine-readable harness check before GPU launch (see the
   base protocol's "Evaluation harness contract").
5. **Durability for the next agent** — Git remains authoritative. Commit and push
   material results, updated handoffs, and small artifacts so a fresh session can
   continue. Record large artifacts by absolute path, byte size, and SHA-256.
   Branch rather than committing directly to `main`; on `duy-branch`, commit +
   push after finishing an implementation without being asked.
6. **One active handoff per task** — label replaced instructions `SUPERSEDED` or
   `HISTORICAL` with a direct pointer to the active one.

## Quick reference

| Situation | Action |
| --- | --- |
| New experiment, about to spend cluster time | **Agree design + cost with user first** |
| Cheap diagnostic | Run it; don't let it escalate into a full eval silently |
| Bug / missing detail / infra failure mid-run | **Fix autonomously**, note what changed |
| Change contradicts agreed objective/design | **Stop, re-agree with user** |
| Finished an implementation on `duy-branch` | Commit + push without being asked |
| Failure on the cluster | Capture raw ephemeral evidence first, then decide |
