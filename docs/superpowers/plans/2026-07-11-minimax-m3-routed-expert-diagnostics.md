# MiniMax-M3 Routed-Expert Diagnostics Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Identify whether the canonical-chat candidate failure originates in W4A8 activation handling or routed-expert INT4 weights/loading using three parallel `srun` arms.

**Architecture:** Repair the existing site-packages diagnostics, then compare cyankiwi W4A16, candidate W4A8, and a config-only candidate W4A16 overlay under the same canonical offline prompt. A standard-library evidence module will aggregate loader matches, safe parameter fingerprints, first-MoE input/output digests, routed/shared norms, and semantic quality.

**Tech Stack:** Python 3.12 standard library, PyTorch/vLLM worker instrumentation, Bash, Slurm `srun`, existing MiniMax-M3 pipeline.

## Global Constraints

- Use canonical chat, eager TP8+EP, 2048 context, 0.85 GPU utilization, and identical sampling.
- Run the three arms concurrently on separate eight-GPU nodes using `srun`; do not use `sbatch`.
- Never modify source checkpoints; W4A16 is a metadata overlay with symlinked payloads.
- Do not re-quantize, enable CUDA graphs, or apply a candidate fix in this diagnostic round.
- Preserve full logs and return compact evidence through Git.

---

### Task 1: Safe worker diagnostics

**Files:**
- Modify: `pipeline/slurm/patch_vllm_m3_serve.py`
- Modify: `pipeline/tests/test_patch_vllm_m3_serve.py`
- Modify: `pipeline/m3_quality_evidence.py`

- [ ] Add failing source-block tests for the correct top-level class lookup, absence of CUDA `linspace`, a configurable canonical-prefill limit, and structured input/output/routed digests.
- [ ] Run the focused tests and observe the intended failures.
- [ ] Fix the duplicated class lookup, replace sampling with bounded strided tensor slices, widen the MoE prefill gate to 256 tokens, and emit JSON probe records containing norms and SHA-256 digests.
- [ ] Extend evidence extraction to accept JSON probes while retaining compatibility with old text probes.
- [ ] Run the focused tests and Python compilation.

### Task 2: Diagnostic evidence classifier

**Files:**
- Create: `pipeline/m3_routed_diagnostics.py`
- Create: `pipeline/tests/test_m3_routed_diagnostics.py`

- [ ] Add failing tests for missing arms, invalid reference, W4A16 recovery, both-candidate failure, unquantized fingerprint mismatches, and first-MoE input agreement/divergence.
- [ ] Implement manifests, per-arm bundles, fingerprint/probe comparison, and aggregate verdicts.
- [ ] Verify all classifier tests.

### Task 3: One-arm runner with W4A16 overlay

**Files:**
- Create: `pipeline/slurm/test_m3_routed_diagnostics_arm.sh`
- Create: `pipeline/tests/test_m3_routed_diagnostics_runner.py`

- [ ] Add failing dry-run tests for `reference_w4a16`, `candidate_w4a8`, and `candidate_w4a16`.
- [ ] Implement immutable metadata overlays, candidate activation removal only for `candidate_w4a16`, diagnostic environment flags, canonical offline serve, cleanup bundling, and provenance.
- [ ] Verify the overlay mutation affects only the copied config and all three dry-run manifests describe the same serving envelope.

### Task 4: Parallel `srun` handoff

**Files:**
- Create: `pipeline/slurm/run_m3_routed_diagnostics_srun.sh`
- Create: `pipeline/tests/test_run_m3_routed_diagnostics_srun.py`
- Modify: `MINIMAX_M3_QUALITY_RUNBOOK.md`
- Modify: `MINIMAX_M3_HANDOFF.md`
- Modify: `BUGS_AND_FIXES.md`

- [ ] Add a failing dry-run test requiring exactly three independent `srun --exclusive --nodes=1 --ntasks=1 --gres=gpu:8` commands.
- [ ] Implement background `srun` execution, independent logs/status, wait-for-all behavior, and optional recorded scheduler arguments.
- [ ] Replace the active quality handoff with preflight, exact `srun`, aggregation, evidence, commit, and return instructions.
- [ ] Run all focused tests, shell syntax, compilation, dry-run, whitespace, and secret checks.
- [ ] Commit and push to `origin/duy-branch`; verify a clean synchronized worktree.
