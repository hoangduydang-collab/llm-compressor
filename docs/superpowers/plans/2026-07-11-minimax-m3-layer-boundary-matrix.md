# MiniMax-M3 Layer-Boundary Matrix Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build and hand off an eleven-node diagnostic matrix that identifies the first corrupt MiniMax-M3 runtime boundary while testing router, expert-parallel, and KV-cache hypotheses concurrently.

**Architecture:** Extend the existing persistent vLLM diagnostic injection with layer-aware boundary and router evidence. A focused evidence module owns arm definitions, manifests, bundling, comparisons, and verdicts; one generic arm runner and one parallel `srun` launcher execute the matrix.

**Tech Stack:** Python 3.12 standard library, Bash, Slurm `srun`, existing vLLM/Compressed Tensors diagnostics and quality normalization.

## Global Constraints

- Continue only the MiniMax-M3 quality issue; do not enable or diagnose CUDA graphs.
- Preserve source checkpoints and tensor shards; metadata experiments use immutable symlink overlays.
- Each hypothesis arm changes one variable relative to a named control.
- Launch all eleven arms concurrently on exclusive eight-GPU nodes with `srun`; do not use `sbatch`.
- Retain TP8, eager mode, block size 128, context 2048, 0.85 utilization, deterministic prompts, disabled custom all-reduce, and disabled shared-expert stream.
- Boundary diagnostics must be bounded, capture-safe, layer-resolved, and enabled only for explicit offline diagnostic arms.

---

### Task 1: Boundary and router instrumentation

**Files:**
- Modify: `pipeline/slurm/patch_vllm_m3_serve.py`
- Modify: `pipeline/tests/test_patch_vllm_m3_serve.py`

**Interfaces:**
- Produces: `M3_LAYER_BOUNDARY=1`, `M3_LAYER_BOUNDARY_LAYERS=3,4,5,6,7,8,9`, `M3_LAYER_BOUNDARY# {json}` records, and `moe_router` fingerprints/load matches.

- [ ] Write tests requiring runtime layer IDs, decoder/attention/MoE boundary names, bounded statistics/digests, capture guards, router tracking, and compilable injected code.
- [ ] Run the focused test and confirm it fails because the boundary block/router category are absent.
- [ ] Implement the minimal class wrappers and router tracking.
- [ ] Run the focused test and confirm it passes.

### Task 2: Evidence model and classifier

**Files:**
- Create: `pipeline/m3_layer_boundary_diagnostics.py`
- Create: `pipeline/tests/test_m3_layer_boundary_diagnostics.py`
- Modify: `pipeline/m3_quality_evidence.py`
- Modify: `pipeline/tests/test_m3_quality_evidence.py`

**Interfaces:**
- Produces: eleven `EXPECTED_ARMS`; manifest, bundle, aggregate CLI commands; parsed `layer_boundary_records`; first-explosion and hypothesis verdict summaries.

- [ ] Write failing parsing and classifier tests for missing arms, invalid reference, router recovery, EP recovery, KV recovery, and attention/MoE/residual localization.
- [ ] Run tests and confirm expected missing-interface failures.
- [ ] Implement structured parsing, manifests, bundling, boundary comparison, and ordered verdicts.
- [ ] Run the focused tests and confirm they pass.

### Task 3: Immutable router overlay and eleven-arm execution

**Files:**
- Modify: `pipeline/m3_routed_diagnostics.py`
- Modify: `pipeline/tests/test_m3_routed_diagnostics_runner.py`
- Create: `pipeline/slurm/test_m3_layer_boundary_arm.sh`
- Create: `pipeline/slurm/run_m3_layer_boundary_srun.sh`
- Create: `pipeline/tests/test_m3_layer_boundary_runner.py`
- Create: `pipeline/tests/test_run_m3_layer_boundary_srun.py`

**Interfaces:**
- Produces: `--add-vllm-router-ignore`, generic arm execution, and exactly eleven concurrent exclusive node allocations.

- [ ] Write failing tests for idempotent router alias metadata, every arm's checkpoint/interface/activation/EP/KV envelope, and eleven `srun` dry-run lines with no `sbatch`.
- [ ] Run the focused tests and confirm they fail on missing behavior/scripts.
- [ ] Implement overlay flag, arm runner, and launcher with independent sibling failure handling and aggregate-on-completion.
- [ ] Run focused tests and shell syntax checks.

### Task 4: Executor runbook and verification

**Files:**
- Modify: `MINIMAX_M3_HANDOFF.md`
- Modify: `MINIMAX_M3_QUALITY_RUNBOOK.md`
- Modify: `BUGS_AND_FIXES.md`

**Interfaces:**
- Produces: exact `srun` preflight/live/return workflow and complete evidence-return contract.

- [ ] Document the latest boundary, exact matrix, invariants, decision rules, evidence tree, cluster adaptations, and required return information.
- [ ] Run focused pytest, all touched shell syntax checks, Python compilation, dry-run count checks, and `git diff --check`.
- [ ] Review the diff for scope discipline and secrets/large files.
- [ ] Commit and push the implementation to `duy-branch`, then stop for GPU execution.
