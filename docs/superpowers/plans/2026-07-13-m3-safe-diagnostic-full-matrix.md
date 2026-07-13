# MiniMax-M3 Safe and Diagnostic Full Matrix Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Launch three probe-free production quantizations and two fault-localizing diagnostics concurrently on five exclusive nodes.

**Architecture:** Reuse `pipeline.run` unchanged for safe lanes and keep all experimental instrumentation inside explicit guarded-runner diagnostic modes. A durable tmux-owned shell controller performs preflights, launches independent top-level `srun` jobs, statically verifies safe checkpoints, and aggregates evidence without touching worker CUDA state.

**Tech Stack:** Python, PyTorch/llm-compressor, Bash, Slurm `srun`, tmux, pytest.

## Global Constraints

- Continue on `duy-branch`; do not create a worktree.
- Each cluster job uses its own `srun --exclusive --nodes=1 --ntasks=1` allocation.
- Safe lanes use only `python -m pipeline.run --stage quantize` and external static verification.
- No per-layer generation, quality forward, qparam inspection, hook, or callback in a safe lane.
- One lane failure never cancels another lane.
- Use tmux ownership and reject nested Slurm.

---

### Task 1: Specify launcher isolation

**Files:**
- Modify: `pipeline/tests/test_m3_guarded_full_launcher.py`
- Create: `pipeline/slurm/run_m3_safe_diagnostic_full_srun.sh`
- Create: `pipeline/slurm/start_m3_safe_diagnostic_full_tmux.sh`

**Interfaces:**
- Consumes: `pipeline.run`, `pipeline.m3_guarded_full`, existing prequant and trace CLIs.
- Produces: a dry-run with one preflight node plus five distinct exclusive full-lane nodes.

- [ ] Add tests requiring three safe `pipeline.run` commands, two guarded diagnostic commands, six total exclusive `srun` commands including trace, no `sbatch`, nested-Slurm refusal, and tmux ownership.
- [ ] Run the launcher tests and confirm they fail because the new launchers do not exist.
- [ ] Implement the smallest dry-run and durable-controller behavior that satisfies the tests.
- [ ] Re-run launcher tests and confirm they pass.

### Task 2: Add synchronized diagnostic modes

**Files:**
- Modify: `pipeline/tests/test_m3_guarded_full.py`
- Modify: `pipeline/m3_guarded_full.py`

**Interfaces:**
- Consumes: `FullGuardController.note_quant_epoch(modules)` and the guarded modifier lifecycle.
- Produces: `--diagnostic-mode heavy|light`, durable `diagnostic_stage.json`, and mode-specific tensor inspection.

- [ ] Add tests proving light mode omits qparam/weight/fake-quant inspection and both modes persist named synchronization stages.
- [ ] Run the guarded tests and confirm the new expectations fail.
- [ ] Add a stage writer/synchronizer, thread diagnostic mode through the controller and CLI, and split heavy tensor inspection from light candidate setup.
- [ ] Re-run guarded tests and confirm they pass.

### Task 3: Implement safe completion and evidence aggregation

**Files:**
- Modify: `pipeline/tests/test_m3_guarded_full_launcher.py`
- Modify: `pipeline/slurm/run_m3_safe_diagnostic_full_srun.sh`

**Interfaces:**
- Consumes: per-lane fresh output roots and pipeline versioned checkpoint directories.
- Produces: exactly one checkpoint path, `checkpoint.path`, static checker log/return code, lane return code, and aggregate controller status.

- [ ] Add source/dry-run tests requiring fresh-root refusal, exact checkpoint cardinality, `--check-tensors`, atomic return-code files, and independent waits.
- [ ] Confirm the tests fail on missing behavior.
- [ ] Implement safe-lane postprocessing and failure-preserving aggregation.
- [ ] Re-run launcher tests and confirm they pass.

### Task 4: Update the cluster handoff

**Files:**
- Modify: `MINIMAX_M3_HANDOFF.md`

**Interfaces:**
- Produces: exact dry-run, tiny `safe-quant_only` smoke, full launch, monitoring, stop conditions, and evidence-return instructions.

- [ ] Replace the active failed guarded-matrix instructions with the approved safe/diagnostic split.
- [ ] Require the executor to return all provenance, scheduler, checkpoint, static, stage, log, and deviation evidence.
- [ ] Ensure commands use `srun`, separate exclusive nodes, and detached tmux only.

### Task 5: Verify and publish

**Files:** all changed files.

- [ ] Run focused pytest for guarded logic and launchers.
- [ ] Run Ruff on changed Python files, `py_compile`, `bash -n`, dry-run checks, and `git diff --check`.
- [ ] Review the complete diff for safe-lane contamination and unrelated changes.
- [ ] Commit and push `duy-branch` for the executor.
