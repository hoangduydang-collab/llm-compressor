# MiniMax-M3 Quality Smoke Tmux Controller Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make the MiniMax-M3 quality smoke survive Cursor tool interruption.

**Architecture:** Preserve the existing four-arm srun sequence in a dedicated controller and start that controller through a verified detached tmux session with durable logs and atomic completion evidence.

**Tech Stack:** Bash, tmux, Slurm srun, pytest.

## Global Constraints

- Do not alter models, prompts, probes, resource sizes, ordering, or timeouts.
- Use tmux, never Cursor background ownership, nohup, setsid, or screen.
- Reject duplicate sessions and stale run evidence.

### Task 1: Controller, wrapper, tests, and handoff

**Files:** Create `pipeline/slurm/run_m3_quality_smoke_srun.sh`, `pipeline/slurm/start_m3_quality_smoke_tmux.sh`, and `pipeline/tests/test_m3_quality_smoke_tmux.py`; modify `M3_QUALITY_THREE_MODEL_SMOKE_RECOVERY_HANDOFF.md`.

- [x] Write failing tests for four-arm parity and verified tmux ownership.
- [x] Implement durable srun and tmux controllers.
- [x] Add duplicate/stale evidence tests and monitoring instructions.
- [x] Run focused tests, Bash syntax, dry run, and diff checks.
- [ ] Commit and push the shared branch.

## Node-allocation invariant

The real tmux launcher and direct controller must run outside any Slurm
allocation. They reject an inherited `SLURM_JOB_ID`; only top-level
`srun --exclusive` provides the required whole-node allocations. Executor
monitoring must confirm disjoint running node lists.

### Task 2: Correct failed Ray gate contract

- [x] Reproduce the placement-output versus `ray_preflight/gate.json` mismatch.
- [x] Replace the placement diagnostic with the proven bounded topology preflight.
- [x] Preserve the four-arm resource plan and BF16 initialization timeout.


### Task 3: Replace blocked TP16/Ray baseline

- [x] Run BF16 on one exclusive node with TP8/mp and a 45-minute bound.
- [x] Remove Ray preflight from the quality controller.
- [x] Add a BF16-only filter and update the executor handoff.
