# MiniMax-M3 Static Serving ABI Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a CPU-only hard gate that rejects Transformers-to-vLLM quantization metadata mismatches before GPU allocation.

**Architecture:** A pure metadata analyzer compares compressed-tensors decisions with packed/plain Safetensors module inventories in source and vLLM namespaces. MiniMax quality preflight persists the report and refuses invalid quantized checkpoints.

**Tech Stack:** Python, JSON/Safetensors index metadata, pytest, Bash, lm-eval 0.4.12, vLLM.

## Global Constraints

- Continue on the current shared branch.
- No model tensor loading is required for the default ABI gate.
- Runtime probes confirm a statically valid contract; they never waive failure.
- Keep model quality work in scope and serving performance deferred.

---

### Task 1: Static ABI analyzer

**Files:** Create `pipeline/m3_serve_abi.py`; create `pipeline/tests/test_m3_serve_abi.py`.

- [ ] Add failing tests for source-only shared-expert rules, valid dual-namespace rules, and ignored packed modules.
- [ ] Implement inventory extraction, alias translation, ignore matching, precision conflicts, and JSON-serializable reports.
- [ ] Run the focused tests.

### Task 2: Hard preflight integration

**Files:** Modify `pipeline/m3_quality_preflight.py`; modify `pipeline/tests/test_m3_quality_eval.py`.

- [ ] Add a failing test proving invalid quantized ABI reports abort preflight validation.
- [ ] Persist reports under `preflight/checkpoint_diagnostics` and raise with actionable reasons before GPU launch.
- [ ] Run MiniMax quality tests.

### Task 3: Deterministic runner corrections

**Files:** Modify `pipeline/m3_quality_eval.py`, `pipeline/configs/eval_minimax_m3_quality.yaml`, `pipeline/m3_distributional_probe.py`, `pipeline/slurm/test_m3_quality_eval_arm.sh`, and relevant tests.

- [ ] Reproduce empty grouped-leaf expansion, wrong MMLU metric, and missing Ray probe backend in tests.
- [ ] Allocate at least one smoke sample per leaf, use the installed MMLU metric, and forward distributed backend to vLLM probe construction.
- [ ] Run focused and complete affected suites.

### Task 4: Handoff and verification

**Files:** Modify `M3_QUALITY_THREE_MODEL_SMOKE_RECOVERY_HANDOFF.md`.

- [ ] Make the ABI report a mandatory executor-side CPU gate and return artifact.
- [ ] Run all affected tests, shell syntax, diff checks, commit, and push.
