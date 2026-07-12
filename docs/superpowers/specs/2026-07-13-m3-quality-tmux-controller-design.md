# MiniMax-M3 Quality Smoke Tmux Controller Design

## Goal

Prevent Cursor tool interruption from terminating the repaired-GPTQ quality
smoke `srun` processes while preserving the existing experiment exactly.

## Design

A quality-specific srun controller owns repaired GPTQ, cyankiwi AWQ, Ray
placement, and bounded BF16 arms and records their return codes. A thin launcher
writes a durable controller script under the existing run root, starts it in a
uniquely named detached tmux session, verifies that session, and prints exact
capture, attach, scheduler, log, and completion commands.

The launcher rejects missing tmux, duplicate sessions, and stale controller
evidence. The controller continues to use `srun --exclusive`; no checkpoint,
prompt, probe, timeout, ordering, or quality decision changes. A completed
controller atomically writes `controller.rc`. Missing completion evidence must
be reconciled with durable logs and Slurm state before any retry.

## Verification

CPU tests cover four-arm dry-run parity, verified tmux creation, monitoring
instructions, duplicate/stale rejection, and absence of nohup/setsid/screen.
Bash syntax and the dry-run launcher are checked without allocating GPUs.

## Node-allocation invariant

The real tmux launcher and direct controller must run outside any Slurm
allocation. They reject an inherited `SLURM_JOB_ID`; only top-level
`srun --exclusive` provides the required whole-node allocations. Executor
monitoring must confirm disjoint running node lists.
