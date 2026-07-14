# Planner–Executor Protocol Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Establish one repo-wide planner–executor operating protocol with fixed workflow states, bounded authority, and complete bidirectional handoff templates.

**Architecture:** Add a canonical root-level protocol document that contains the complete role contract and copy-ready packet templates. Keep `CLAUDE.md` concise by recording the capability asymmetry and requiring both roles to follow the canonical document. Existing task-specific handoffs continue to carry experiment details and adopt the protocol when next revised.

**Tech Stack:** Markdown, Git, PowerShell text validation.

## Global Constraints

- The protocol applies repository-wide and must not depend on MiniMax-M3 terminology.
- The planner is the higher-resource reasoning, design, diagnosis, and decision owner.
- The executor is the cluster execution, monitoring, and evidence-preservation owner.
- Executor reasoning is bounded; strategic analysis and experiment redesign remain with the planner.
- Git is the durable source of truth; continuation must not require chat history.
- Every task uses `PLANNER_ANALYSIS`, `READY_FOR_EXECUTOR`, `EXECUTING`, `RETURNED_FOR_ANALYSIS`, and `PLANNER_DECISION`.
- New or materially revised handoffs use the canonical templates; historical handoffs are not rewritten wholesale.
- Planner instructions must be copy-ready and require minimal dynamic executor work.
- Executor returns must include raw evidence, small committed artifacts, and durable path/size/SHA-256 references for large artifacts.
- Retries and runtime deviations are prohibited unless the planner explicitly pre-authorizes their trigger and bounds.

---

### Task 1: Add the canonical planner–executor protocol

**Files:**
- Create: `PLANNER_EXECUTOR_PROTOCOL.md`
- Reference: `docs/superpowers/specs/2026-07-14-planner-executor-protocol-design.md`

**Interfaces:**
- Consumes: the approved authority model, workflow, packet contracts, and runtime boundaries in the design specification.
- Produces: protocol version 1 plus the execution-packet and evidence-packet templates referenced by `CLAUDE.md` and future task handoffs.

- [x] **Step 1: Create the protocol header and authority model**

Add these sections with normative language:

```markdown
# Planner–Executor Protocol

Protocol version: 1
Scope: repository-wide

## Purpose
## Non-negotiable operating model
## Authority and responsibilities
### Planner: brain and decision owner
### Executor: cluster hands and evidence owner
```

The operating model must state that the planner has greater agent resources but no cluster access, while the executor has 15+ eight-H100 nodes and a constrained reasoning budget. It must assign research, hypothesis selection, experiment design, local implementation, diagnosis, result interpretation, and downstream authorization to the planner. It must assign exact command execution, monitoring, evidence preservation, artifact return, and bounded factual reporting to the executor.

- [x] **Step 2: Add the fixed workflow and ownership transitions**

Document the exact state sequence:

```text
PLANNER_ANALYSIS
  -> READY_FOR_EXECUTOR
  -> EXECUTING
  -> RETURNED_FOR_ANALYSIS
  -> PLANNER_DECISION
```

Define entry/exit authority for each state, require a new packet revision and run ID when instructions change after execution starts, and prohibit downstream execution after `RETURNED_FOR_ANALYSIS` until the planner records a decision.

- [x] **Step 3: Add durable communication and packet-revision rules**

Require protocol version, workflow state, task/packet revision, owner, exact base commit, and one decision question in each active packet. Require one clearly marked active packet per task, explicit `SUPERSEDED`/`HISTORICAL` labels for old instructions, exact revision verification before execution, committed small evidence, and durable path/byte-size/SHA-256 references for large evidence.

- [x] **Step 4: Add the copy-ready planner execution-packet template**

The fenced template must contain fields for:

```markdown
# Execution packet: <task>

- Protocol version:
- State: READY_FOR_EXECUTOR
- Packet revision:
- Planner owner:
- Intended executor:
- Base Git commit:
- Decision question:

## Objective and hypothesis
## Scope and non-goals
## Preconditions and exact environment
## Required inputs
## Resource contract
## Commands
### Setup and revision verification
### Preflight
### Dry run
### Launch
### Monitoring
### Aggregation and packaging
## Expected jobs and independence rules
## Success gates and expected artifacts
## Allowed adaptations
## Pre-authorized retries
## Stop-and-return conditions
## Prohibited actions
## Return contract
## Final instruction
```

The template must instruct the planner to replace every angle-bracket field and remove unused optional sections before moving the task to `READY_FOR_EXECUTOR`. It must require exact commands rather than prose such as “run the evaluation.”

- [x] **Step 5: Add the copy-ready executor evidence-packet template**

The fenced template must contain fields for:

```markdown
# Evidence packet: <task>

- Protocol version:
- State: RETURNED_FOR_ANALYSIS
- Packet revision executed:
- Expected Git commit:
- Actual Git commit:
- Execution classification: exact | permitted-adapted | stopped
- Evidence commit:

## Factual outcome
## Per-job or per-arm results
## Exact commands executed
## Environment and package versions
## Scheduler and resource record
## Timings, return codes, retries, and abnormal exits
## Gate values and measurements
## Observed facts
## Limited executor interpretation
## Deviations
## Small committed artifacts
## Large durable artifacts
## Missing artifacts
## First failure and last successful stage
## Final repository state
## Questions for planner
```

Require large-artifact entries to include absolute path, byte size, and SHA-256. Require missing artifacts and deviations to be reported explicitly rather than omitted. State that strategic conclusions and next-step authorization belong to the planner.

- [x] **Step 6: Add runtime authority, retry, and emergency-evidence rules**

Add separate lists for:

- allowed without escalation;
- must stop and return;
- prohibited unless explicitly authorized;
- pre-authorized retry requirements;
- urgent evidence preservation before stopping.

The prohibited list must cover code/config/prompt/gate edits, experiment substitution, topology changes, silent retries, result-root reuse, destructive artifact operations, canceling healthy independent jobs, and unauthorized downstream work.

- [x] **Step 7: Add completeness checklists and compact examples**

Add a planner readiness checklist and executor return checklist whose unchecked required item blocks the corresponding state transition. Add one generic example showing a planner issuing a bounded two-arm smoke packet and one generic example showing an executor returning a stopped packet after preflight failure. Examples must illustrate the schema without MiniMax-specific paths or values.

- [x] **Step 8: Validate the canonical protocol**

Run:

```powershell
rg -n "Protocol version: 1|PLANNER_ANALYSIS|READY_FOR_EXECUTOR|EXECUTING|RETURNED_FOR_ANALYSIS|PLANNER_DECISION|Execution packet:|Evidence packet:|Allowed without escalation|Must stop and return|Prohibited unless explicitly authorized|Planner readiness checklist|Executor return checklist" PLANNER_EXECUTOR_PROTOCOL.md
Select-String -Path PLANNER_EXECUTOR_PROTOCOL.md -Pattern ('TB' + 'D'),('TO' + 'DO')
git diff --check
```

Expected: all required headings and states are present; placeholder search emits no matches; `git diff --check` exits zero.

- [x] **Step 9: Commit the canonical protocol**

```powershell
git add -- PLANNER_EXECUTOR_PROTOCOL.md
git commit -m "docs: add planner executor protocol"
```

Expected: one commit creating the canonical root document.

---

### Task 2: Make the protocol mandatory in repository working principles

**Files:**
- Modify: `CLAUDE.md:39`
- Reference: `PLANNER_EXECUTOR_PROTOCOL.md`

**Interfaces:**
- Consumes: protocol version 1 and its canonical root path.
- Produces: a concise, high-discovery entry point that every new planner and executor reads before task-specific handoffs.

- [x] **Step 1: Strengthen the role descriptions**

Replace the existing compact role bullets with language that preserves their current responsibilities and adds the capability asymmetry:

```markdown
- **Planner — brain and decision owner** (stronger agent resources; local,
  CPU-only; no cluster access): owns the heavy reasoning work — research,
  architecture, hypothesis selection, experiment design, local implementation and
  tests, diagnosis, returned-evidence interpretation, and next-step decisions.
  Planner instructions must minimize executor-side dynamic reasoning. On
  `duy-branch`, commit + push after finishing an implementation without being
  asked.
- **Executor — cluster hands and evidence owner** (constrained agent resources;
  direct access to 15+ 8×H100 nodes): runs the planner's prepared cluster/GPU
  work, monitors it, preserves raw evidence, and returns complete results. The
  executor may reason enough to execute safely and capture failures, but does not
  redesign experiments or make strategic decisions unless explicitly authorized.
```

- [x] **Step 2: Add the mandatory protocol pointer**

Immediately after the role bullets, add:

```markdown
## Planner–executor protocol

Both roles must read and follow `PLANNER_EXECUTOR_PROTOCOL.md`. It is the
repo-wide source of truth for workflow states, decision authority, execution
packets, evidence returns, retries, deviations, and stop conditions.
Task-specific handoffs supply the experiment details; they do not override the
general protocol unless they explicitly name and justify an exception.
```

- [x] **Step 3: Clarify handoff durability**

Retain the existing Git-based cross-session rule and add that every new or
materially revised handoff must use the canonical packet contract, while old
sections must be labeled superseded or historical when replaced.

- [x] **Step 4: Validate discoverability and consistency**

Run:

```powershell
rg -n "brain and decision owner|cluster hands and evidence owner|15\+ 8×H100|PLANNER_EXECUTOR_PROTOCOL.md|source of truth|materially revised" CLAUDE.md
git diff --check
git diff -- CLAUDE.md PLANNER_EXECUTOR_PROTOCOL.md
```

Expected: the capability asymmetry and mandatory link are visible in
`CLAUDE.md`; no whitespace errors; the diff changes only the approved protocol
documentation.

- [x] **Step 5: Commit the working-principles integration**

```powershell
git add -- CLAUDE.md
git commit -m "docs: require planner executor protocol"
```

Expected: one focused commit updating the repository entry point.

---

### Task 3: Final protocol verification and publication

**Files:**
- Verify: `PLANNER_EXECUTOR_PROTOCOL.md`
- Verify: `CLAUDE.md`
- Verify: `docs/superpowers/specs/2026-07-14-planner-executor-protocol-design.md`
- Verify: `docs/superpowers/plans/2026-07-14-planner-executor-protocol.md`

**Interfaces:**
- Consumes: both implementation commits and the approved design.
- Produces: a pushed, synchronized branch containing the complete protocol chain from design to operational entry point.

- [x] **Step 1: Run the full text contract check**

```powershell
$protocol = Get-Content -Raw PLANNER_EXECUTOR_PROTOCOL.md
$claude = Get-Content -Raw CLAUDE.md
$required = @(
  'PLANNER_ANALYSIS', 'READY_FOR_EXECUTOR', 'EXECUTING',
  'RETURNED_FOR_ANALYSIS', 'PLANNER_DECISION',
  'Execution packet:', 'Evidence packet:',
  'Allowed without escalation', 'Must stop and return',
  'Prohibited unless explicitly authorized',
  'Planner readiness checklist', 'Executor return checklist'
)
$missing = $required | Where-Object { -not $protocol.Contains($_) }
if ($missing) { throw "Missing protocol markers: $($missing -join ', ')" }
if (-not $claude.Contains('PLANNER_EXECUTOR_PROTOCOL.md')) {
  throw 'CLAUDE.md does not link the canonical protocol'
}
```

Expected: exit code zero with no missing markers.

- [x] **Step 2: Run final repository documentation checks**

```powershell
Select-String -Path PLANNER_EXECUTOR_PROTOCOL.md,CLAUDE.md -Pattern ('TB' + 'D'),('TO' + 'DO')
git diff --check
git status --short --branch
git log -4 --oneline
```

Expected: no placeholders or whitespace errors; only the implementation plan may remain uncommitted before the final documentation commit.

- [x] **Step 3: Commit the implementation plan if it is not already committed**

```powershell
git add -- docs/superpowers/plans/2026-07-14-planner-executor-protocol.md
git commit -m "docs: plan planner executor protocol"
```

Expected: the design, plan, canonical protocol, and `CLAUDE.md` integration are all committed.

- [x] **Step 4: Push and verify synchronization**

```powershell
git push origin duy-branch
git status --short --branch
```

Expected: push succeeds and status reports `duy-branch...origin/duy-branch` with no ahead/behind count and no working-tree changes.

---

### Task 4: Apply the proportional-execution amendment

**Files:**
- Modify: `PLANNER_EXECUTOR_PROTOCOL.md`
- Modify: `.cursor/rules/planner-executor-protocol.mdc`
- Modify: `M3_PAIRED_GPTQ_AWQ_TASK_ISOLATED_HANDOFF.md`
- Reference: `docs/superpowers/specs/2026-07-14-planner-executor-protocol-design.md`

**Interfaces:**
- Consumes: the approved 2026-07-14 proportional-execution amendment.
- Produces: repo-wide three-level runtime handling and an active r2 packet that
  deterministically permits old untracked files only under `results/` and
  `artifacts/`.

- [x] **Step 1: Amend the canonical protocol and Cursor rule**

Add these exact concepts to the existing runtime-authority sections:

```markdown
1. Record and proceed for explicitly pre-authorized benign conditions.
2. Permitted adaptation only for an exact adaptation named by the packet.
3. Stop and return when code, inputs, scientific validity, topology, evidence,
   or the experiment contract may change.
```

Require packets to name protected paths and permitted untracked roots. Require
exact blocking paths in stopped returns and proportional evidence based on
whether GPU work began. Replace “clean worktree” completion language with no
unexplained tracked changes plus enumeration of permitted untracked artifacts.

- [x] **Step 2: Revise the active packet to r2 in place**

Change the packet revision to `2026-07-14-r2`. Replace the pristine-worktree
stop with copy-ready commands that:

```bash
test -z "$(git diff --name-only)"
test -z "$(git diff --cached --name-only)"
WORKSPACE_RECORD="$(mktemp)"
WORKSPACE_BLOCKERS="$(mktemp)"
git ls-files --others --exclude-standard | tee "$WORKSPACE_RECORD"
awk '!/^(results|artifacts)\//' "$WORKSPACE_RECORD" | tee "$WORKSPACE_BLOCKERS"
test ! -s "$WORKSPACE_BLOCKERS"
```

After creating the fresh run root, copy both records into it. Explicitly permit
record-and-proceed only for pre-existing files under `results/` and `artifacts/`;
retain stops for tracked changes, other untracked paths, collisions, and all
existing hash/topology/environment failures. Require exact paths in any new
stopped return.

- [x] **Step 3: Verify the documentation contract**

Run:

```powershell
$protocol = Get-Content -Raw PLANNER_EXECUTOR_PROTOCOL.md
$cursor = Get-Content -Raw .cursor/rules/planner-executor-protocol.mdc
$packet = Get-Content -Raw M3_PAIRED_GPTQ_AWQ_TASK_ISOLATED_HANDOFF.md
foreach ($marker in @('Record and proceed','Permitted adaptation','protected paths','permitted untracked')) {
  if (-not $protocol.Contains($marker)) { throw "Protocol missing $marker" }
}
if (-not $cursor.Contains('Record and proceed')) { throw 'Cursor rule missing proportional policy' }
if (-not $packet.Contains('2026-07-14-r2')) { throw 'Packet is not r2' }
if (-not $packet.Contains("awk '!/^(results|artifacts)\\//'")) { throw 'Packet lacks deterministic classifier' }
git diff --check
```

Expected: exit code zero and no whitespace errors.

- [x] **Step 4: Commit and push the same branch**

```powershell
git add PLANNER_EXECUTOR_PROTOCOL.md .cursor/rules/planner-executor-protocol.mdc M3_PAIRED_GPTQ_AWQ_TASK_ISOLATED_HANDOFF.md docs/superpowers/plans/2026-07-14-planner-executor-protocol.md
git commit -m "docs: make executor conditions proportional"
git push origin duy-branch
git status --short --branch
```

Expected: `duy-branch` is synchronized with `origin/duy-branch`; no new files
are introduced by this amendment.
