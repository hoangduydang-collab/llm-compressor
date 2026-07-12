# MiniMax-M3 Three-Model Smoke Recovery Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Restore a trustworthy smoke gate for BF16, in-house GPTQ, and cyankiwi AWQ while deferring the incompatible AutoRound checkpoint.

**Architecture:** Keep the existing matrix controller and arm runner, but make active/deferred model scope explicit. Validate reasoning semantics before GPU launch, make smoke-gate evaluation total over incomplete evidence, and insert a rank-observable Ray topology gate before BF16 model loading.

**Tech Stack:** Python 3.12, PyYAML, lm-eval 0.4.12, vLLM 0.24, Ray, Bash, Slurm `srun`, pytest.

## Global Constraints

- Model quality remains primary; serving performance is out of scope.
- Active models are BF16, in-house GPTQ, and cyankiwi AWQ.
- AutoRound is deferred, not silently replaced or treated as a failed quality arm.
- Smoke uses four nodes; production uses eight nodes.
- `srun` is required; `sbatch` is unavailable.
- Production remains locked behind `ready_for_production: true`.
- Follow TDD and preserve all returned smoke evidence.

---

### Task 1: Active/deferred matrix and MiniMax reasoning validation

**Files:**
- Modify: `pipeline/configs/minimax_m3_quality_matrix.yaml`
- Modify: `pipeline/configs/eval_minimax_m3_quality.yaml`
- Modify: `pipeline/m3_quality_eval.py`
- Modify: `pipeline/m3_quality_preflight.py`
- Test: `pipeline/tests/test_m3_quality_eval.py`

**Interfaces:**
- Produces: `MatrixSpec.deferred_models: tuple[DeferredModelSpec, ...]`
- Produces: `validate_reasoning_config(eval_config: dict[str, Any]) -> None`
- Changes: active launch topology to three smoke arms/four nodes and six production arms/eight nodes.

- [ ] Add failing tests asserting three active models, AutoRound deferred metadata, node counts 4/8, and rejection of `enable_thinking=True` with mixed loglikelihood/generative tasks.
- [ ] Run `.venv/bin/python -m pytest pipeline/tests/test_m3_quality_eval.py -q` and confirm failures describe the old four-model topology and missing validator.
- [ ] Add `DeferredModelSpec`, parse `deferred_models`, move AutoRound there, set `enable_thinking: null` and `think_end_token: </mm:think>`, and call `validate_reasoning_config` before checkpoint diagnostics/corpus work.
- [ ] Re-run the focused tests and confirm PASS.
- [ ] Commit with `fix(eval): scope MiniMax smoke to compatible models`.

### Task 2: Failure-safe smoke gate

**Files:**
- Modify: `pipeline/m3_quality_eval.py`
- Test: `pipeline/tests/test_m3_quality_eval.py`

**Interfaces:**
- Changes: `validate_smoke_gate(spec, report)` never calls `project_probe_overhead` with nonpositive inputs.
- Produces: per-model `probe_projection` with `within_budget: false` and a concrete `reason` when timing evidence is absent.

- [ ] Add failing tests for zero probe tokens/time, a missing model, and mixed valid/invalid arms; assert a returned failed gate rather than an exception.
- [ ] Run the focused tests and confirm the zero-evidence case raises under the old implementation.
- [ ] Implement guarded probe projection and preserve every other failed check in the returned JSON.
- [ ] Re-run focused tests and confirm PASS.
- [ ] Commit with `fix(eval): report incomplete MiniMax smoke evidence`.

### Task 3: Observable two-node Ray topology gate

**Files:**
- Create: `pipeline/slurm/test_m3_ray_topology.sh`
- Modify: `pipeline/slurm/test_m3_quality_eval_arm.sh`
- Modify: `pipeline/slurm/run_m3_quality_eval_srun.sh`
- Modify: `pipeline/tests/test_m3_quality_eval_runner.py`

**Interfaces:**
- Produces: `<run-root>/ray_preflight/rank-<rank>.json`, `ray_status.txt`, `ray_nodes.json`, and `gate.json`.
- Consumes: Slurm rank/node variables and the executor Python/Ray environment.
- Changes: BF16 arm requires `ray_preflight/gate.json` with `ready: true` before model loading.

- [ ] Add failing shell-contract tests asserting a Ray-only command is emitted before smoke arms, rank-local diagnostics are retained, and BF16 refuses a missing/failed topology gate.
- [ ] Run `pytest pipeline/tests/test_m3_quality_eval_runner.py -q` and confirm failure because the topology script/gate does not exist.
- [ ] Implement rank-local environment/IP/version logging, Ray head/worker startup without `exec`, two-node/16-GPU visibility checks, cleanup, and explicit return artifacts. Avoid pipelines that can fail under `pipefail` while selecting the head node.
- [ ] Re-run shell tests plus `bash -n` for all three scripts and confirm PASS.
- [ ] Commit with `fix(eval): gate BF16 smoke on Ray topology`.

### Task 4: Handoff, regression verification, and push

**Files:**
- Modify: `MINIMAX_M3_HANDOFF.md`
- Modify: `M3_QUALITY_SMOKE_REPORT.md`

**Interfaces:**
- Documents: AutoRound deferral, exact Ray-only command, three-arm parallel smoke, 4/8-node totals, and complete returned evidence.

- [ ] Update the handoff and report without rewriting the historical failed-run evidence.
- [ ] Run the affected evaluation tests, Python compile checks, `bash -n`, launcher dry runs, and `git diff --check`.
- [ ] Confirm the worktree contains only intended changes and commit with `docs: hand off three-model MiniMax smoke retry`.
- [ ] Push `duy-branch` for the executor.
