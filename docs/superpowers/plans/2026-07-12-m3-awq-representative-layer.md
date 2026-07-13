# MiniMax-M3 AWQ Representative-Layer Diagnostic Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a six-arm, in-memory AWQ diagnostic for MiniMax-M3 layers 8, 31, and 59 without exporting checkpoints.

**Architecture:** A CPU-testable module owns layer selection, tensor fidelity metrics, verdicts, evidence aggregation, and a CLI. Its GPU arm path reuses the production model loader, calibration builder, recipe, and `oneshot`, but augments the ignore list and sequential target so only one decoder layer is modified. A shell launcher runs the six independent arms through concurrent `srun` steps and invokes the aggregator after all return.

**Tech Stack:** Python 3.11, PyTorch, Transformers, llm-compressor `oneshot`, pytest, Bash, Slurm `srun`.

## Global Constraints

- Quantize layers 8, 31, and 59 for variants `offsetfix` and `nosmooth`.
- Preserve the production 512-sample, 2048-token W4AFP8 calibration recipe.
- Do not save, export, or serve a checkpoint.
- Capture layer input, MoE input, MoE output, and decoder-layer output.
- Fail before calibration unless targeting and AWQ mapping isolation are exact.
- Run all six arms independently; one failure must not cancel another.
- Keep CUDA-graph and throughput diagnosis out of scope.

---

### Task 1: CPU selection, metric, and verdict core

**Files:**
- Create: `pipeline/m3_awq_representative.py`
- Create: `pipeline/tests/test_m3_awq_representative.py`

**Interfaces:**
- Produces: `layer_exclusion_pattern(layer: int) -> str`, `sequential_target_pattern(layer: int) -> str`, `tensor_fidelity(reference, candidate) -> dict`, `classify_boundaries(boundaries: dict) -> dict`, and `aggregate_matrix(root: Path) -> dict`.

- [ ] **Step 1: Write failing tests** covering allowed layers, negative-lookahead exclusion behavior, exact sequential matching, identical/exploded/non-finite tensors, verdict thresholds, and aggregation with missing/failed arms.
- [ ] **Step 2: Run** `pytest -q pipeline/tests/test_m3_awq_representative.py` and verify import/behavior failures.
- [ ] **Step 3: Implement** pure selection, metric, classification, and aggregation functions with constants `LAYERS=(8,31,59)`, `VARIANTS=("offsetfix","nosmooth")`, and `BOUNDARIES=("layer_input","moe_input","moe_output","layer_output")`.
- [ ] **Step 4: Run the focused test and verify it passes.**

### Task 2: In-memory GPU arm

**Files:**
- Modify: `pipeline/m3_awq_representative.py`
- Modify: `pipeline/tests/test_m3_awq_representative.py`

**Interfaces:**
- Consumes: Task 1 selection and metric functions.
- Produces: `prepare_arm_config`, `validate_target_isolation`, `capture_boundaries`, `run_arm`, and CLI command `python -m pipeline.m3_awq_representative arm`.

- [ ] **Step 1: Write failing tests** proving only the requested layer is targeted, `nosmooth` changes only the MLP-input mapping, boundary hooks unwrap tuple outputs, evidence includes provenance, and no code path calls `save_pretrained` or the MiniMax re-export module.
- [ ] **Step 2: Run the focused tests and verify the new failures.**
- [ ] **Step 3: Implement the arm:** load the production YAML, deep-copy it, append the non-selected-layer exclusion, set the exact sequential target, set the existing variant environment switch, load model/tokenizer and deterministic probes, capture BF16 boundaries, call `oneshot` with the existing recipe and full calibration dataset, capture candidate boundaries, validate isolation, calculate metrics/verdict, and atomically write `arm.json`. Never call `model.save_pretrained`.
- [ ] **Step 4: Run the focused tests and existing MiniMax mapping/config tests.**

### Task 3: Six-way srun launcher and aggregation CLI

**Files:**
- Create: `pipeline/slurm/run_m3_awq_representative_srun.sh`
- Modify: `pipeline/m3_awq_representative.py`
- Modify: `pipeline/tests/test_m3_awq_representative.py`

**Interfaces:**
- Consumes: CLI arm and aggregate commands.
- Produces: six logs, six return-code files, `matrix.json`, `report.md`, and a dry-run command matrix.

- [ ] **Step 1: Write failing tests** for the expected six arm names, matrix verdict combinations, and static launcher requirements (`srun`, concurrent background PIDs, wait-all behavior, unique logs, no `sbatch`, and aggregation after waits).
- [ ] **Step 2: Run the focused tests and verify failures.**
- [ ] **Step 3: Implement** `aggregate` CLI/report rendering and the shell launcher with `TIME_LIMIT`, `LOG_ROOT`, `RESULT_ROOT`, `ENV_FILE`, `VENV_ACTIVATE`, `DRY_RUN`, and `SRUN_ARGS` overrides. Each step gets one GPU and writes its own rc file even on failure.
- [ ] **Step 4: Run focused tests, `bash -n`, ShellCheck if available, and a dry run.**

### Task 4: Handoff and final verification

**Files:**
- Modify: `MINIMAX_M3_HANDOFF.md`
- Modify: `M3_AWQ_REQUANTIZATION_REPORT.md`

**Interfaces:**
- Consumes: Task 3 launcher and evidence contract.
- Produces: exact cluster commands and return requirements.

- [ ] **Step 1: Update the handoff** to supersede another full AWQ retry with the representative-layer matrix, including dry-run/run commands, stop conditions, scheduler evidence, and interpretation rules.
- [ ] **Step 2: Run** `pytest -q pipeline/tests/test_m3_awq_representative.py pipeline/tests/test_minimax_m3_config.py tests/llmcompressor/modeling/test_offset_norm_minimax_m3.py`, `bash -n pipeline/slurm/run_m3_awq_representative_srun.sh`, launcher dry run, and `git diff --check`.
- [ ] **Step 3: Review the diff against every global constraint, commit, and push `duy-branch` for the cluster agent.**


### Task 5: Empty-metric lifecycle recovery

- [x] Reproduce the zero-completed-metric finalizer failure from executor logs.
- [x] Guard AWQ summary statistics and record explicit mapping skip reasons.
- [x] Persist resolved/completed/skipped/unprocessed lifecycle evidence.
- [x] Add an `ARM_FILTER` one-arm smoke gate through srun and tmux.
- [x] Update the handoff to expand to six arms only after a passing smoke.
