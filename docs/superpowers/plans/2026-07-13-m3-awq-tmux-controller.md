# MiniMax-M3 AWQ Tmux Controller Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Ensure Cursor tool/session interruption cannot terminate the `srun` controller for the six-arm AWQ diagnostic.

**Architecture:** A short wrapper validates `tmux`, creates a unique detached session, and has the tmux server execute the existing `run_m3_awq_representative_srun.sh` controller with durable controller output. The wrapper returns only after independently verifying the session exists; monitoring uses separate `tmux` and Slurm queries.

**Tech Stack:** Bash, tmux, Slurm `srun`, pytest static/subprocess tests.

## Global Constraints

- Do not change the representative-layer experiment or its six-arm launcher.
- Do not use Cursor background tasks, plain `&`, `nohup`, or `setsid` to own `srun`.
- Fail if tmux is unavailable or the requested session already exists.
- Preserve explicit RUN_ID, SESSION_NAME, LOG_ROOT, RESULT_ROOT, and controller log.
- Verify the detached session after creation and print exact poll/attach commands.

### Task 1: Wrapper, tests, and handoff

**Files:**
- Create: `pipeline/slurm/start_m3_awq_representative_tmux.sh`
- Create: `pipeline/tests/test_m3_awq_representative_tmux.py`
- Modify: `MINIMAX_M3_HANDOFF.md`

- [ ] Write failing tests for required tmux ownership, unique session/run IDs, no unsafe detachment, duplicate-session rejection, command quoting, and printed verify/poll/attach instructions.
- [ ] Run the focused tests and confirm failure because the wrapper is absent.
- [ ] Implement a dry-run-capable wrapper that launches `tmux new-session -d`, writes a durable controller log, and validates with `tmux has-session`.
- [ ] Replace the handoff's foreground-tool launch instruction with exact tmux start, verify, polling, attach, and abnormal-session recovery commands.
- [ ] Run focused tests, Bash syntax, wrapper dry run, `git diff --check`, commit, and push `duy-branch`.
