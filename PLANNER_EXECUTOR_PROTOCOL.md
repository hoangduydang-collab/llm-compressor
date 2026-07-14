# Planner–Executor Protocol

Protocol version: 1  
Scope: repository-wide

## Purpose

This repository uses two complementary agent roles with deliberately asymmetric
capabilities. The planner has the stronger agent resources and owns the heavy
reasoning, but cannot access the compute cluster. The executor has a constrained
reasoning budget, but can directly use a cluster with 15+ nodes of 8×H100 GPUs.

The operating model is therefore simple: **the planner is the brain; the executor
is the hands**. The planner makes expensive cluster work ready to run. The
executor runs it faithfully and returns enough evidence for the planner to make
the next decision.

This document is the repo-wide source of truth for their communication. A fresh
planner or executor must be able to continue from Git without prior chat history.

## Non-negotiable operating model

1. The planner owns strategic reasoning and decision-making.
2. The executor owns bounded cluster execution and evidence preservation.
3. The planner must search existing code and reputable prior work before
   designing bespoke work.
4. Planner instructions must be copy-ready and minimize dynamic executor work.
5. Executor returns must preserve raw evidence, not only prose summaries.
6. The executor does not change the experiment contract unless the active packet
   explicitly permits the change.
7. The planner decides what returned evidence means and what happens next.
8. Git and durable artifact storage are authoritative; chat memory is not.

## Authority and responsibilities

### Planner: brain and decision owner

The planner owns the high-cost thinking:

- inspect repository code, dependencies, prior commits, prior experiments, and
  reputable external work;
- identify the decision that is worth spending cluster time on;
- form hypotheses and design decisive experiments, controls, gates, resource
  topology, and stop conditions;
- implement and verify locally everything that does not require the cluster;
- prepare exact setup, preflight, dry-run, launch, monitoring, aggregation, and
  evidence-packaging commands;
- anticipate likely failure modes and say what evidence to capture for each;
- analyze executor returns, distinguish infrastructure failures from real
  signals, and record the conclusion;
- authorize retries, revised experiments, downstream stages, and task closure;
- keep one active handoff and label replaced instructions clearly.

The planner must not offload open-ended investigation to the executor merely
because the relevant system runs on the cluster. If the executor would need to
invent the experiment, choose thresholds, browse the codebase for the right
command, or decide what to run next, the execution packet is not ready.

### Executor: cluster hands and evidence owner

The executor owns bounded cluster operations:

- pull and verify the exact Git revision named by the planner;
- check prerequisites and run the exact prepared commands;
- launch, monitor, and aggregate cluster jobs within the resource contract;
- preserve healthy independent jobs when another job fails;
- capture logs, manifests, scheduler records, measurements, and failure
  boundaries before ephemeral evidence disappears;
- report factual observations and clearly label any limited interpretation;
- commit and push small artifacts and reports;
- record durable paths, sizes, and hashes for large artifacts;
- stop at the authorized boundary and return control to the planner.

The executor may think enough to execute safely, recognize a stated gate, and
preserve evidence. It must not select a new hypothesis, redesign the experiment,
change policy, or continue to downstream work unless the execution packet
explicitly grants that authority.

## Workflow states

Every planner–executor task follows this state sequence:

```text
PLANNER_ANALYSIS
  -> READY_FOR_EXECUTOR
  -> EXECUTING
  -> RETURNED_FOR_ANALYSIS
  -> PLANNER_DECISION
```

### `PLANNER_ANALYSIS`

The planner is researching, designing, implementing locally, verifying, or
interpreting prior evidence. Cluster execution is not authorized. The task stays
here until every required execution-packet field is complete.

Owner: planner.

### `READY_FOR_EXECUTOR`

The planner has committed a complete execution packet, named its exact base
commit, and authorized only the work in that packet. The executor may pull and
begin prerequisite checks.

Entry authority: planner only.  
Owner after entry: executor.

### `EXECUTING`

The executor verified the requested revision and began the bounded run. The
packet's commands, inputs, topology, gates, and interpretation rules are immutable
for that run.

Owner: executor.

If instructions must change after this point, stop the affected work safely. The
planner must issue a new packet revision, and the next attempt must use a fresh
run ID. Never silently combine results produced under different packet revisions.

### `RETURNED_FOR_ANALYSIS`

The executor completed or stopped the run, committed and pushed its evidence
packet, and returned control. No retry, fix, or downstream stage is authorized.

Entry authority: executor only.  
Owner after entry: planner.

### `PLANNER_DECISION`

The planner analyzes the evidence and records the strategic verdict. The planner
then closes the task, returns it to `PLANNER_ANALYSIS`, or publishes a new
`READY_FOR_EXECUTOR` packet.

Owner: planner.

## Durable communication rules

- Every active packet names protocol version, workflow state, task, packet
  revision, owner, exact base commit, and one decision question.
- Each task has exactly one clearly identified active packet.
- Replaced instructions are labeled `SUPERSEDED` or `HISTORICAL` and point to the
  active packet. A newer date alone is not a sufficient pointer.
- The executor verifies and records the actual Git commit before consuming
  cluster resources.
- Packet commands use exact paths, arguments, environments, and resource values.
- Small structured artifacts and reasonably sized logs are committed to Git.
- Large artifacts stay on durable shared storage and are identified by absolute
  path, byte size, and SHA-256.
- A report never substitutes for raw evidence; it identifies the underlying
  logs, manifests, measurements, and return codes.
- Every deviation is reported, including deviations believed to be harmless.
- Evidence requirements are proportional to execution progress. A pre-GPU stop
  returns exact blocker evidence; a launched run returns the full scheduler,
  artifact, and measurement record.
- Task-specific handoffs provide experiment details. They may add stricter rules,
  but may override this protocol only by explicitly naming and justifying the
  exception.

## Planner-to-executor execution packet

Copy this template into the task-specific handoff. Replace every angle-bracket
field, delete unused optional lines, and make every command directly runnable
before changing the state to `READY_FOR_EXECUTOR`. Phrases such as “run the
evaluation” or “use the usual environment” are not sufficient.

````markdown
# Execution packet: <task name>

- Protocol version: 1
- State: READY_FOR_EXECUTOR
- Packet revision: <monotonic revision or date-revision>
- Planner owner: <agent/session identifier>
- Intended executor: <agent/session identifier or any executor>
- Base Git commit: <full commit SHA>
- Decision question: <one question this run will answer>

## Objective and hypothesis

<What will be measured and why the result is decision-relevant.>

## Scope and non-goals

- In scope: <bounded work>
- Not authorized: <adjacent or downstream work>

## Preconditions and exact environment

- Repository path: <absolute cluster path>
- Branch: <branch>
- Environment activation: `<exact command>`
- Required environment variables: <exact exports or `none`>
- Required package/version checks: `<exact commands>`

## Required inputs

| Input | Exact path or identifier | Required validation |
| --- | --- | --- |
| <dataset/checkpoint/config> | <value> | <command and expected result> |

## Workspace policy

- Protected paths: <tracked code, configs, inputs, and output roots>
- Permitted untracked roots: <exact roots or `None`>
- Record and proceed: <exact benign conditions and classifier commands>
- Stop: <tracked changes, shadowing paths, collisions, or other exact blockers>

## Resource contract

- Nodes: <count>
- GPUs per node: <count>
- Exclusivity: <exclusive/shared and exact flag>
- Task/process layout: <ranks, workers, tensor/pipeline parallelism>
- Concurrency: <which jobs run concurrently or sequentially>
- Time limit: <HH:MM:SS>
- Expected runtime: <range>

## Commands

### Setup and revision verification

```bash
<copy-ready commands including pull and exact revision check>
```

### Preflight

```bash
<copy-ready command>
```

Expected: <files, output markers, and return code>.  
Stop if: <preflight stop condition>.

### Dry run

```bash
<copy-ready command>
```

Expected: <job count, topology, and key command markers>.

### Launch

```bash
<copy-ready command with fresh run ID and durable controller ownership>
```

### Monitoring

```bash
<copy-ready non-owning monitoring commands>
```

### Aggregation and packaging

```bash
<copy-ready commands; aggregation runs even after partial failure when safe>
```

## Expected jobs and independence rules

| Job or arm | Resources | Expected output | Failure effect on other jobs |
| --- | --- | --- | --- |
| <name> | <topology> | <artifact> | <continue/cancel rule> |

## Success gates and expected artifacts

- Gate: <exact field/operator/threshold>
- Expected artifact: <exact relative or absolute path>

## Allowed adaptations

- <An exact adaptation and its boundary, or `None`>

## Pre-authorized record-and-proceed conditions

- <An exact benign condition and required evidence, or `None`>

## Pre-authorized retries

- Trigger: <exact condition or `None`>
- Maximum retry count: <integer>
- Fresh run ID required: <yes/no>
- Inputs that must remain unchanged: <list>

## Stop-and-return conditions

- <specific prerequisite, gate, failure, ambiguity, or safety condition>

## Prohibited actions

- <specific changes and downstream work that are not authorized>

## Return contract

- <exact small artifacts and raw logs to commit>
- <scheduler, command, environment, topology, timing, and return-code evidence>
- <large artifacts requiring absolute path, byte size, and SHA-256>
- <required factual report or table>

## Final instruction

Commit and push the complete evidence packet, set the state to
`RETURNED_FOR_ANALYSIS`, and stop. Do not retry, patch, or launch downstream work
unless this packet explicitly authorizes it.
````

## Executor-to-planner evidence packet

Copy this template into the task result document. Do not omit an empty or missing
field: write `none` or explain why the evidence is absent. Strategic conclusions
and next-step authorization belong to the planner.

````markdown
# Evidence packet: <task name>

- Protocol version: 1
- State: RETURNED_FOR_ANALYSIS
- Packet revision executed: <revision>
- Expected Git commit: <full SHA from execution packet>
- Actual Git commit: <full SHA executed>
- Execution classification: exact | permitted-adapted | stopped
- Evidence commit: <full pushed SHA>

## Factual outcome

<What ran, completed, failed, or stopped. Do not turn this into a strategic
decision.>

## Per-job or per-arm results

| Job or arm | Scheduler ID and node | State | Return code | Gate result | Output path |
| --- | --- | --- | ---: | --- | --- |
| <name> | <ID, node> | <state> | <rc> | <value> | <path> |

## Exact commands executed

```bash
<commands as actually executed, including environment overrides>
```

## Environment and package versions

- Host/control environment: <details>
- Virtual environment/container: <path or image>
- Relevant packages and drivers: <exact versions>

## Scheduler and resource record

- Run ID and result root: <values>
- Log root: <value>
- Job/step IDs, nodes, GPU counts, exclusivity, and layout: <values>
- Queueing or placement events: <values or none>

## Timings, return codes, retries, and abnormal exits

- Start/end/elapsed per job: <values>
- Controller and job return codes: <values>
- Retries: <count, run IDs, and trigger; or none>
- OOMs, cancellations, signals, or scheduler failures: <details or none>

## Gate values and measurements

| Gate or measurement | Value | Required condition | Mechanical result |
| --- | ---: | --- | --- |
| <name> | <value> | <condition> | <pass/fail/not evaluated> |

## Observed facts

- <Directly supported observation with artifact/log reference>

## Limited executor interpretation

- <Clearly labeled local interpretation, or `None; returned for planner analysis`>

## Deviations

- <Every deviation and why it occurred, or `None`>

## Record-and-proceed conditions exercised

- <Every pre-authorized benign condition observed with exact evidence, or `None`>

## Small committed artifacts

- `<repository path>` — <content>

## Large durable artifacts

| Artifact | Absolute path | Byte size | SHA-256 |
| --- | --- | ---: | --- |
| <name> | <path> | <bytes> | <hash> |

## Missing artifacts

- <artifact and reason, or `None`>

## First failure and last successful stage

- Last successful stage: <stage and evidence>
- First failing operation: <operation and evidence>
- Immediate scheduler evidence: <path or none>

## Final repository state

- Evidence commit pushed: <full SHA>
- Branch synchronization: <status output>
- Tracked changes: <none or exact explained paths>
- Permitted untracked artifacts: <none or exact enumerated paths>

## Questions for planner

- <bounded question created by the evidence, or `None`>
````

## Runtime authority

### Proportional condition levels

Every active packet classifies foreseeable runtime conditions into three levels:

1. **Record and proceed** — an explicitly pre-authorized benign condition that
   does not change tracked code, inputs, scientific validity, topology, evidence
   integrity, or the experiment contract. The executor captures the named
   evidence and continues.
2. **Permitted adaptation** — an exact adaptation and boundary named by the
   packet. The executor records its use and continues only within that boundary.
3. **Stop and return** — a condition that can change code, inputs, scientific
   validity, resource topology, evidence integrity, or the experiment contract,
   or any unclassified condition material to the decision.

The executor does not perform open-ended classification. The planner supplies
copy-ready checks, protected paths, permitted untracked roots, and collision
rules. A shared checkout need not be pristine merely for appearance: staged or
unstaged tracked modifications are blocking by default, while untracked files
may proceed only under roots explicitly allowed by the packet and only when they
cannot collide with its fresh result root.

### Allowed without escalation

The executor may:

- run the exact commands in the active packet;
- wait for scheduler placement and monitor the named signals;
- capture diagnostics explicitly requested by the packet;
- preserve healthy independent jobs when another job fails;
- capture ephemeral scheduler/accounting evidence after an abnormal exit;
- aggregate partial results when the packet says aggregation is safe;
- record and proceed through a benign condition explicitly authorized by the
  packet;
- package, commit, and push the required evidence.

### Must stop and return

The executor captures urgent ephemeral evidence first, then stops when:

- the expected revision, input, environment, credential, or service is missing;
- a preflight or stated stop condition fails;
- instructions, paths, ownership, or success criteria contain a material
  ambiguity not resolved by the packet's deterministic checks;
- actual resource topology differs from the required topology;
- continuing would require an unapproved retry or experiment change;
- evidence suggests continuing could corrupt artifacts or invalidate the run;
- the authorized execution boundary has been reached.

### Prohibited unless explicitly authorized

The executor must not:

- edit code, recipes, configs, prompts, thresholds, sample counts, or gates;
- substitute models, datasets, benchmarks, environments, or launch methods;
- change node/GPU topology, process layout, concurrency, or time limits;
- silently retry, resume, or reuse a result root;
- delete, overwrite, or mutate checkpoints, logs, or prior evidence;
- cancel healthy independent jobs merely because another job failed;
- launch downstream evaluation, quantization, publication, or performance work;
- treat a plausible runtime diagnosis as authorization to implement a fix.

## Retry and deviation rules

Retries are opt-in. A valid pre-authorization names all four items:

1. the exact triggering condition;
2. the maximum retry count;
3. whether each retry needs a fresh run ID;
4. the inputs and parameters that must remain unchanged.

Every attempt and its evidence must be retained. If any of the four items is
missing, the executor does not retry.

An adaptation is permitted only when the active packet names it and its boundary.
The executor records every exercised adaptation. Any other deviation changes the
experiment contract and requires a return to the planner.

## Emergency evidence preservation

Stopping does not mean discarding evidence. When scheduler records, process state,
or failure logs may disappear, the executor first captures the smallest safe set
needed to reconstruct the failure, such as:

- `scontrol` and `sacct` output;
- scheduler stdout/stderr and controller logs;
- job/step IDs, nodes, terminal states, signals, and return codes;
- the last successful stage and first failing operation;
- partial manifests, heartbeats, and structured failure records.

After that bounded capture, the executor stops. Evidence preservation does not
authorize a retry or fix.

Evidence depth follows work performed. Before GPU allocation, return the exact
blocking paths or values, commands, return codes, revision, and scheduler state;
do not manufacture empty per-arm records. After any job launches, return the
complete applicable job, scheduler, artifact, timing, and measurement evidence.
Descriptions such as “dirty,” “missing,” or “numerous” never replace the exact
records that caused a stop.

## Planner readiness checklist

A task cannot enter `READY_FOR_EXECUTOR` until every required item is checked:

- [ ] One decision question and bounded hypothesis are stated.
- [ ] Exact base commit, branch, repository path, and environment are named.
- [ ] Required inputs and preflight validations are explicit.
- [ ] Protected paths, permitted untracked roots, deterministic workspace checks,
      and collision rules are explicit.
- [ ] Setup, dry-run, launch, monitoring, aggregation, and packaging commands are
      directly runnable.
- [ ] Node/GPU topology, concurrency, time limit, and expected runtime are named.
- [ ] Success gates, expected artifacts, and job independence rules are explicit.
- [ ] Allowed adaptations and retry policy are explicit, even when both are none.
- [ ] Record-and-proceed conditions are explicit, even when none.
- [ ] Stop conditions and prohibited downstream work are explicit.
- [ ] The return contract names raw logs, small artifacts, and large-artifact
      path/size/hash requirements.
- [ ] Old instructions are labeled and point to this active packet.
- [ ] The packet is committed and its exact commit is recorded.

## Executor return checklist

A task cannot enter `RETURNED_FOR_ANALYSIS` until every applicable item is
checked or its absence is explained:

- [ ] Packet revision, expected commit, and actual commit are recorded.
- [ ] Exact executed commands and every override are recorded.
- [ ] Environment and relevant package/driver versions are recorded.
- [ ] Run IDs, roots, scheduler IDs, nodes, topology, and timings are recorded.
- [ ] Every job has a terminal state, return code, and output path.
- [ ] Gate values and key measurements are preserved in structured form.
- [ ] Deviations, retries, OOMs, cancellations, and abnormal exits are explicit.
- [ ] Raw logs and small structured artifacts are committed.
- [ ] Large artifacts have absolute paths, byte sizes, and SHA-256 values.
- [ ] Missing artifacts, last successful stage, and first failure are explicit.
- [ ] Every stop condition includes the exact paths, values, commands, and return
      codes that triggered it.
- [ ] Facts and limited executor interpretation are separated.
- [ ] Evidence is pushed and final branch/worktree status is recorded.
- [ ] No unexplained staged or unstaged tracked changes remain; permitted
      untracked artifacts are enumerated rather than deleted for cleanliness.
- [ ] The executor has stopped without unauthorized downstream work.

## Compact examples

### Example: bounded planner packet

A planner needs to decide whether a distributed launch reduces wall time without
replicating host memory. The active packet specifies two independent smoke arms,
one node and eight GPUs per arm, exact commands, a 45-minute ceiling, and gates for
peak host RAM and per-layer time. It permits waiting for scheduling but no retry,
topology change, parameter change, or full run. The executor can run the packet
without selecting tools or inventing acceptance criteria.

### Example: stopped executor return

The executor verifies the commit, then a preflight finds that a required
checkpoint path is missing. The packet says this is a stop condition. The executor
does not substitute another checkpoint or launch either arm. It returns an
evidence packet classified `stopped`, including the exact command, nonzero return
code, preflight log, environment, expected path, scheduler state (`no jobs
launched`), and a pushed evidence commit. The planner then decides whether to fix
the path, create the checkpoint, or abandon the experiment.
