# Planner–Executor Protocol Design

**Date:** 2026-07-14  
**Scope:** Repository-wide  
**Status:** Implemented

## Problem

This repository is developed by two complementary agent roles whose capabilities
are intentionally asymmetric:

- the planner is the higher-resource reasoning agent but has no cluster access;
- the executor has direct access to 15+ eight-H100 nodes but has a tighter agent
  reasoning budget.

Existing task handoffs often provide excellent commands and evidence-return
requirements, but those conventions are repeated inconsistently across long,
task-specific documents. There is no canonical repo-wide protocol that defines
decision authority, workflow states, permitted executor discretion, handoff
completeness, or how a replacement agent resumes work without chat history.

The missing standard creates two risks: expensive executor-side reasoning that
should have been completed by the planner, and incomplete returns that force the
next planner to reconstruct experiment provenance or rerun cluster work.

## Goals

1. Make the planner the explicit reasoning, design, diagnosis, and decision
   owner.
2. Make the executor the explicit cluster execution, monitoring, and evidence
   preservation owner.
3. Define a fixed, versioned workflow that survives replacement of either agent.
4. Make planner instructions directly runnable with minimal executor judgment.
5. Make executor returns complete enough for a fresh planner to analyze without
   relying on chat context or access to ephemeral cluster state.
6. Preserve the repository's existing search-first principle and task-specific
   handoff documents.

## Non-goals

- Automating Slurm, Git, or artifact transfer in this change.
- Rewriting every historical handoff into the new format.
- Preventing the executor from reporting observations or capturing urgent
  diagnostics.
- Moving strategic analysis or experiment design onto the executor.
- Encoding MiniMax-M3-specific commands in the repo-wide contract.

## Chosen approach

Create one canonical root document, `PLANNER_EXECUTOR_PROTOCOL.md`, containing
the complete protocol and copy-ready templates. Update `CLAUDE.md` with the
capability asymmetry, concise role summary, and a mandatory link to the canonical
document.

This is preferred over expanding `CLAUDE.md` because the working-principles file
should remain short enough to be read at the start of every session. It is
preferred over splitting protocol and templates into several files because a
single contract minimizes partial reads and version drift.

Historical handoffs remain unchanged. The protocol applies to every new handoff
and to an existing handoff whenever that handoff is materially revised.

## Authority model

### Planner: brain and decision owner

The planner owns the high-cost reasoning work:

- investigate repository code, dependencies, prior art, and prior evidence;
- identify hypotheses and decide which question is worth cluster time;
- design experiments, resource topology, controls, gates, and stop conditions;
- implement and locally verify code that does not require the cluster;
- prepare exact commands, monitoring commands, and artifact contracts;
- analyze executor returns and distinguish infrastructure failure from scientific
  or engineering evidence;
- choose the next hypothesis, authorize retries, and approve downstream stages;
- maintain the active handoff and mark old instructions superseded.

The planner does not assume GPU access and does not delegate open-ended analysis
to the executor merely because the relevant system lives on the cluster.

### Executor: cluster hands and evidence owner

The executor owns bounded cluster operations:

- verify and run the planner's exact revision and commands;
- execute preflights, dry runs, launches, monitoring, aggregation, and packaging;
- preserve independent jobs when one arm fails unless instructed otherwise;
- capture ephemeral scheduler and failure evidence before it disappears;
- report factual observations and clearly labeled limited interpretation;
- commit and push small evidence plus durable references to large artifacts;
- stop at the specified boundary and return control to the planner.

The executor may reason to execute safely and preserve evidence. It does not
select new hypotheses, redesign experiments, change quality gates, or continue
to downstream work without planner authorization.

## Workflow state machine

Every planner–executor task uses exactly these states:

1. `PLANNER_ANALYSIS` — the planner is investigating, implementing, verifying,
   or interpreting evidence. No cluster execution is authorized.
2. `READY_FOR_EXECUTOR` — a complete execution packet is committed and the exact
   base revision is named. The executor may begin.
3. `EXECUTING` — the executor has verified the revision and started the bounded
   run. Packet commands and acceptance criteria are immutable for this run.
4. `RETURNED_FOR_ANALYSIS` — execution has completed or stopped, and the executor
   has committed/pushed the evidence packet. No downstream run is authorized.
5. `PLANNER_DECISION` — the planner records the interpretation and either closes
   the task, returns to `PLANNER_ANALYSIS`, or publishes a new
   `READY_FOR_EXECUTOR` packet.

Only the planner advances a task into `READY_FOR_EXECUTOR` or out of
`RETURNED_FOR_ANALYSIS`. Only the executor advances it from
`READY_FOR_EXECUTOR` through `EXECUTING` to `RETURNED_FOR_ANALYSIS`.

An instruction change after execution begins requires a new packet revision and
a fresh run ID. Results produced under different packet revisions must not be
silently combined.

## Durable communication rules

- Git is the source of truth; chat memory is optional and non-authoritative.
- Every packet records the protocol version, state, task owner, base commit, and
  the single decision question the run informs.
- Each task has one clearly identified active packet. Older instructions are
  labeled `SUPERSEDED` or `HISTORICAL` and point to the active packet.
- The executor verifies the requested revision after pulling and records the
  actual revision in the return.
- Small logs and structured artifacts are committed. Large artifacts remain on
  durable shared storage and are referenced by absolute path, byte size, and
  SHA-256.
- Raw evidence is never replaced by a prose summary. Reports link to or identify
  the underlying records.
- The executor always records deviations, including harmless-looking ones.

## Planner-to-executor execution packet

The canonical document will include a copy-ready template with the following
required sections.

### Identity and decision

- protocol version;
- workflow state (`READY_FOR_EXECUTOR`);
- task and packet revision;
- planner owner and intended executor;
- exact Git base commit;
- objective, hypothesis, and single decision question;
- scope and explicit non-goals.

### Preconditions and environment

- repository and working-directory path;
- branch and pull command;
- environment activation and required environment variables;
- required datasets, checkpoints, credentials, services, and storage paths;
- package/version checks and preflight commands;
- expected preflight outputs and stop conditions.

### Execution

- copy-ready setup, dry-run, launch, monitoring, aggregation, and packaging
  commands;
- resource topology: node count, GPUs per node, exclusivity, task layout,
  concurrency, time limit, and expected runtime;
- run-ID and result-root construction;
- expected jobs/arms and independence rules;
- success gates and expected artifacts.

### Runtime boundaries

- explicitly allowed adaptations;
- pre-authorized retry conditions and maximum retry count;
- stop-and-return conditions;
- prohibited changes;
- downstream work that is not authorized.

### Return contract

- exact structured artifacts and raw logs to return;
- scheduler, environment, provenance, timing, and resource evidence;
- large-artifact hashing requirements;
- required result summary format;
- commit, push, and synchronized-worktree requirement;
- final instruction to stop for planner analysis.

An execution packet is incomplete and must remain in `PLANNER_ANALYSIS` if any
required section is missing or still ambiguous.

## Executor-to-planner evidence packet

The canonical document will include a copy-ready return template with these
required sections.

### Outcome

- packet revision and actual Git commit;
- terminal state and concise factual result;
- per-job or per-arm state, return code, and gate result;
- explicit statement of whether execution was exact, permitted-adapted, or
  stopped.

### Provenance and execution record

- exact commands as executed;
- environment and package versions;
- run IDs, result/log roots, scheduler job/step IDs, nodes, topology, timestamps,
  elapsed times, retries, OOMs, cancellations, and abnormal exits;
- every deviation and the reason it occurred.

### Measurements and observations

- gate values and key measurements in structured form;
- observed facts separated from limited executor interpretation;
- first failing operation and last successful stage when applicable;
- immediate scheduler evidence for abnormal exits.

### Artifacts

- committed paths for small logs, manifests, reports, and structured results;
- absolute durable paths, byte sizes, and SHA-256 values for large artifacts;
- missing artifacts called out explicitly with the reason they are absent;
- pushed evidence commit and final branch synchronization status.

The executor does not declare the strategic verdict unless the packet defines a
purely mechanical gate. Even then, it reports the computed gate and leaves the
next action to the planner.

## Runtime authority and escalation

### Allowed without escalation

- run the exact commands in the packet;
- wait for scheduler placement and monitor named signals;
- capture diagnostics explicitly requested by the packet;
- preserve healthy independent jobs when another job fails;
- collect ephemeral scheduler/accounting evidence after an abnormal exit;
- package and push evidence.

### Must stop and return

- expected revision, input, environment, or preflight is unavailable;
- commands or ownership are ambiguous;
- actual resource topology differs from the required topology;
- a stated stop condition fires;
- the next action would require an unapproved retry or experiment change;
- evidence indicates that continuing could corrupt artifacts or waste the run.

Urgent evidence preservation happens before stopping when delay would destroy
the evidence.

### Prohibited unless explicitly authorized

- editing code, recipes, configs, prompts, thresholds, sample counts, or gates;
- substituting models, datasets, benchmarks, environments, or launch methods;
- changing node/GPU topology or concurrency;
- silently retrying, resuming, or reusing a result root;
- deleting or overwriting checkpoints, logs, or prior evidence;
- canceling healthy independent jobs because another job failed;
- launching downstream evaluation, quantization, publication, or performance
  work;
- treating a plausible executor diagnosis as authorization to implement a fix.

## Retry and deviation policy

Retries are opt-in, not implied. A planner may pre-authorize retries only by
specifying the triggering condition, maximum count, whether a fresh run ID is
required, and which inputs must remain unchanged. Every attempt is retained and
reported.

A permitted deviation is valid only when named in the packet. The executor
records it even if it does not change the result. Any other deviation changes the
experiment contract and requires a return to the planner.

## Document integration

Implementation will make two documentation changes:

1. Create `PLANNER_EXECUTOR_PROTOCOL.md` at the repository root with the complete
   protocol, templates, checklists, and compact examples.
2. Update `CLAUDE.md` to describe the capability asymmetry, preserve the existing
   role summary, and require both roles to read and follow the canonical protocol.

Task-specific handoffs remain the location for experiment commands and evidence.
They reference the canonical protocol rather than restating the general rules.

## Validation

Because the change is documentation-only, validation consists of:

- confirming both required files exist and link to each other correctly;
- checking that the canonical file contains every workflow state and both packet
  templates;
- checking that required execution and evidence fields appear in the templates;
- checking that executor authority, stop rules, retries, and prohibited actions
  are explicit and internally consistent;
- checking that no placeholder markers or incomplete sections remain;
- running `git diff --check` and reviewing the exact documentation diff.

## Acceptance criteria

- A fresh planner can identify its authority, current workflow state, required
  analysis work, and the fields needed before cluster execution.
- A fresh executor can run a packet without inventing commands or experiment
  policy and knows exactly when to stop.
- A fresh planner can analyze a returned run using committed evidence and durable
  artifact references without prior chat context.
- Executor-side reasoning is bounded to safe execution, observation, and evidence
  preservation.
- Strategic decisions, hypothesis changes, and downstream authorization remain
  with the planner.
- The protocol is repo-wide and does not depend on MiniMax-M3 terminology.

## 2026-07-14 proportional-execution amendment

### Problem observed

The first task-isolated MiniMax-M3 execution packet required a completely clean
worktree. The executor correctly stopped before GPU allocation because the
shared cluster checkout contained pre-existing untracked checkpoint and result
artifacts. Those files did not necessarily change code, inputs, or the fresh run
root, so the binary clean/dirty rule created an avoidable planner round trip.

The same packet required a clean final worktree while also allowing large
artifacts to remain on durable storage outside Git. In a shared cluster checkout,
those requirements can conflict. The stopped return also summarized “numerous”
untracked files without preserving their exact paths, making the blocking state
harder for the planner to assess.

### Approaches considered

1. Keep requiring a pristine checkout. This is simple but rejects benign shared
   artifact state and makes normal cluster operation unnecessarily fragile.
2. Allow all dirty state and merely record it. This is fast but cannot prove
   that the committed code, configs, or inputs were actually executed.
3. Classify workspace and runtime conditions by decision impact. This preserves
   reproducibility for tracked code and protected paths while allowing explicitly
   named, non-overlapping artifact state. This is the chosen approach.

### Chosen behavior

Execution packets define protected paths and permitted untracked roots. The
default workspace policy is:

- any staged or unstaged tracked modification is a stop condition;
- untracked files under packet-approved durable roots such as `results/` and
  `artifacts/` are recorded and permitted;
- untracked files outside approved roots, files that can shadow code/configs,
  or any collision with the fresh run root are stop conditions;
- the executor uses planner-supplied deterministic commands to classify the
  state rather than making an open-ended judgment.

Runtime conditions use three response levels:

1. **Record and proceed** for explicitly pre-authorized benign conditions.
2. **Permitted adaptation** only for an exact adaptation named by the packet.
3. **Stop and return** for conditions that can change code, inputs, scientific
   validity, resource topology, evidence integrity, or the experiment contract.

Return evidence is proportional to execution progress. A stop before GPU
allocation needs a concise packet with the exact blocking paths, commands, and
return codes. A launched experiment still requires the complete job, scheduler,
artifact, and measurement record. Any condition important enough to stop must
be preserved exactly rather than described only as “dirty,” “missing,” or
“numerous.”

Final synchronization means no unexplained staged or unstaged tracked changes,
all required small evidence committed and pushed, and all remaining permitted
untracked artifacts enumerated. It does not require deleting or committing large
artifacts merely to produce an empty `git status`.

### Amendment acceptance criteria

- Benign pre-existing artifacts in approved roots do not cause another planner
  round trip.
- Executed tracked code and configs still match the named Git revision.
- A fresh run root cannot overwrite or merge with prior evidence.
- The executor can classify the workspace with copy-ready commands and minimal
  reasoning.
- Stop packets contain the exact evidence that triggered the stop.
- The active MiniMax-M3 packet applies this policy and becomes revision r2.
