# Evidence packet: task-isolated paired quality rerun

- Protocol version: 1
- State: `RETURNED_FOR_ANALYSIS`
- Packet revision executed: `2026-07-14-r1`
- Expected base commit: `1e2da818a08df09bf1ff0268702bda34bf89ee6e`
- Actual Git commit: `b5d22c7313e36d6fe27f90348c3ca87244f9aa5e`
- Execution classification: `stopped`
- Decision question: Does repaired in-house GPTQ preserve enough quality versus
  cyankiwi AWQ to justify later performance evaluation?

## Factual outcome

No eval jobs were launched. The packet's ancestor check passed, but its
mandatory clean-worktree precondition failed because the workspace contains
pre-existing untracked checkpoint and result artifacts. The packet explicitly
requires stopping when the worktree is not clean.

The scheduler had no executor jobs for this packet:

```text
squeue -u hoangduy
JOBID STATE TIME NAME NODELIST(REASON)
```

## Exact verification commands

```bash
git merge-base --is-ancestor 1e2da818a08df09bf1ff0268702bda34bf89ee6e HEAD
git rev-parse HEAD
git status --short
squeue -u hoangduy -o '%.18i %.12T %.10M %.24j %.14R'
```

Observed:

- Ancestor check: passed (`rc=0`)
- Actual commit: `b5d22c7313e36d6fe27f90348c3ca87244f9aa5e`
- Worktree: not clean; numerous pre-existing untracked artifacts
- Scheduler jobs launched by this packet: none

## Missing artifacts

All run, preflight, dry-run, submission, scheduler, arm, aggregate, and quality
artifacts are absent because execution stopped before input preflight.

## Limited executor interpretation

None; returned for planner analysis. The planner must decide how to provide a
clean execution workspace without deleting or mutating the existing artifacts.

## Deviations and retries

None. No retry, cleanup, result-root reuse, or experiment change was attempted.
