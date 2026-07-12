# MiniMax-M3 AWQ/GPTQ Repair Matrix Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a parallel evidence matrix that distinguishes AWQ smoothing failure from shared GPTQ/AWQ export failure and tests a narrowly scoped AWQ repair.

**Architecture:** A streaming tensor audit validates AWQ compensation invariants without loading full checkpoints. A MiniMax-M3-only mapping switch produces the repair checkpoint, while a dedicated matrix module and `srun` launcher reuse the existing serving diagnostics and evidence format.

**Tech Stack:** Python 3.12, PyTorch, Safetensors, Bash, Slurm `srun`, existing MiniMax-M3 pipeline diagnostics.

## Global Constraints

- Preserve all source checkpoints and shard payloads.
- Use only `srun`; `sbatch` is unavailable.
- Run independent serving arms concurrently on exclusive eight-H100 nodes.
- Keep TP8, eager execution, deterministic prompts, and CUDA graphs disabled.
- Change only the MiniMax-M3 AWQ MLP-input mapping in the repair quantization.

---

### Task 1: Streaming checkpoint scale audit

**Files:**
- Create: `pipeline/m3_checkpoint_scale_audit.py`
- Test: `pipeline/tests/test_m3_checkpoint_scale_audit.py`

**Interfaces:**
- Produces: `audit_checkpoints(base, reference, awq, gptq, layers) -> dict` and a JSON CLI.

- [ ] Write synthetic-shard tests for suffix resolution, tensor statistics, recovered scales, and compensation error.
- [ ] Run the focused test and confirm the missing module failure.
- [ ] Implement indexed streaming tensor reads and invariant calculations.
- [ ] Run the focused test and confirm it passes.

### Task 2: MiniMax-M3 AWQ mapping switch

**Files:**
- Modify: `pipeline/minimax_m3_config.py`
- Modify: `pipeline/tests/test_minimax_m3_config.py`

**Interfaces:**
- Produces: `get_minimax_m3_awq_mappings(disable_mlp_input_smoothing: bool | None = None)` with environment fallback `M3_AWQ_DISABLE_MLP_INPUT_SMOOTH`.

- [ ] Write a failing test proving only the post-attention mapping is removed.
- [ ] Implement the explicit argument and environment fallback.
- [ ] Run the focused test and confirm attention and up/down mappings remain.

### Task 2b: MiniMax-M3 offset-norm registration

**Files:**
- Modify: src/llmcompressor/modeling/offset_norm.py
- Create: tests/llmcompressor/modeling/test_offset_norm_minimax_m3.py

**Interfaces:**
- Produces: MiniMaxM3VLRMSNorm as an alias of CalibrationOffsetNorm.

- [ ] Test conversion from zero-centered Gemma weights to effective weights.
- [ ] Register the exact Transformers class name.
- [ ] Test restoration after smoothing uses effective_weight - 1.

### Task 3: Integrated evidence matrix and runner

**Files:**
- Create: `pipeline/m3_awq_gptq_repair.py`
- Create: `pipeline/slurm/test_m3_awq_gptq_repair_arm.sh`
- Create: `pipeline/slurm/run_m3_awq_gptq_repair_srun.sh`
- Test: `pipeline/tests/test_m3_awq_gptq_repair.py`
- Test: `pipeline/tests/test_m3_awq_gptq_repair_runner.py`

**Interfaces:**
- Produces: eight serving arms, arm manifests/bundles, matrix comparison, and concurrent `srun` commands.

- [ ] Write failing tests for all arm envelopes and GPTQ/AWQ verdict branches.
- [ ] Implement checkpoint roles, evidence bundling, boundary comparison, and classifier.
- [ ] Implement generic arm execution and concurrent launcher with no `sbatch` strings.
- [ ] Run focused tests, shell syntax checks, and launcher dry-run.

### Task 4: Handoff and verification

**Files:**
- Modify: `MINIMAX_M3_HANDOFF.md`
- Modify: `BUGS_AND_FIXES.md`

**Interfaces:**
- Produces: exact GPTQ re-export, repair quantization, audit, launch, and evidence-return commands.

- [ ] Document current layer-8 evidence and the integrated workflow.
- [ ] Run Python compilation, focused tests, shell syntax, dry-run count, and `git diff --check`.
- [ ] Commit and push to `duy-branch` for cluster execution.
