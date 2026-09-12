# Native AWQ and MTP Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development or superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Both GLM AWQ and GPTQ produce native W4AFP8 directly, with optional automatically assembled MTP from the same BF16 source.

**Architecture:** Reuse the collective CT/Transformers native writer for both methods. Assemble only the missing MTP layer after collective saving using existing RTN and block-FP8 kernels, transactionally updating the native inventory.

**Tech Stack:** Python, PyTorch, compressed-tensors, Transformers, safetensors, pytest.

## Global Constraints

- Work on the owner-authorized `duy-branch`; preserve unrelated `.gitignore`, `predictive_ptq/`, and `results/predictive-ptq/`.
- No second main-model conversion or rewritten main-model shards; reuse existing quantization kernels and native writer.
- Source MTP from the main job's same BF16 source snapshot; record draft experts as RTN, not GPTQ or AWQ calibrated.
- Preserve fixed-unit expert input scales and record policy; no new activation optimization.
- No GPU allocation or full checkpoint conversion in local implementation. Runtime qualification remains executor work.
- Default optional MTP policy is absent; provided full GLM-5.3 recipes enable source assembly, representative recipes remain absent.
- CPU tests use `OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 PYTHONPATH=src:. .venv-prequant/bin/python -m pytest`.

### Task 1: Direct native AWQ output

**Files:** `pipeline/config.py`, `pipeline/quantize.py`, `pipeline/configs/glm53_distributed_w4afp8_awq_full.yaml`, `pipeline/tests/test_native_sglang_save.py`, config/recipe tests as needed.

**Interfaces:** Existing `native_sglang_save(model)` and `assert_native_sglang_preflight(model, schemes)` remain method-independent. Config `checkpoint_format: sglang-w4afp8` supports plain AWQ and GPTQ.

- [ ] Write failing config and actual tiny GLM AWQ lifecycle tests. The output must pass `verify_native_sglang_checkpoint` on first save, preserve folds/FP8 payload and restore CT state. Exercise disk-offload saving as feasible with the existing fixture; no converter invocation.
- [ ] Accept AWQ W4AFP8 with disjoint FP8_BLOCK targets, keeping GPTQ early preparation mandatory and rejecting unsupported methods/layouts.
- [ ] Resolve schemes only from quantization modifiers; AWQ is transform-only and must not receive `resolve_quantization_config()`.
- [ ] Enable direct native output in the full GLM-5.3 AWQ recipe; preserve calibration behavior.
- [ ] Run relevant native/config/recipe tests and Ruff; commit only task files and report exact commands/results.

### Task 2: Optional native MTP assembly

**Files:** Create `pipeline/native_mtp.py`, `pipeline/tests/test_native_mtp.py`; modify config/recipe provenance, quantize lifecycle, native verifier, both full GLM-5.3 recipes.

**Interfaces:** Add `quantization.mtp_policy` values `absent` (default) and `source-rtn`; permit source-rtn only for native W4AFP8. Export `preflight_native_mtp(source, model_config)` returning a validated source plan, and `assemble_native_mtp(checkpoint, plan)` returning assembly evidence. Resolve Hub source IDs to the already-cached snapshot using existing Hub helpers, never choose a different source revision.

- [ ] Test full tiny source inventory, missing/mismatched source pre-calibration, correct expert `.w1/w2/w3.input_scale` names, exact reused INT4/FP8 output, complete hashes including copied BF16 tensors, duplicate assembly and interrupted/failed publication.
- [ ] Validate source config/index/header metadata, main/source depth and architecture, expert IDs/projections, draft tensors, dtypes/geometries before oneshot. Use bounded reads and explicit required inventory; a partial source must fail.
- [ ] Reuse `graft_mtp_w4afp8` quantizers/classification and `graft_mtp_head` inventory helpers. Stage only new MTP shards; reject file collisions; verify before publishing. On ordinary failure restore metadata and remove only created files. An interruption marker invalidates incomplete artifacts.
- [ ] Extend manifest/verifier for complete MTP assembly, source provenance, layer and RTN policy. Main shard bytes must remain unchanged. Handle both single-shard and indexed native main checkpoints.
- [ ] Invoke preflight before oneshot; assembly source-only after final existing barrier and before offline verification. No non-source collective follows assembly.
- [ ] Enable source-rtn in the AWQ and GPTQ full recipes, keep representative absent, record policy in recipe provenance.
- [ ] Run MTP/native/config/graft regression tests and Ruff; commit only task files and report commands/results.

### Task 3: Review and executor documentation

**Files:** `GLM53_QUANTIZATION_OBJECTIVES.md`, implementation docs and `docs/glm53-native-w4afp8-executor-handoff.md`, evidence JSON.

- [ ] Review task diffs for spec compliance and quality; fix important findings with covering tests.
- [ ] Document final recipe controls, same-source resolution, RTN MTP semantics, failure behavior and direct output for both methods.
- [ ] Update executor handoff with CPU evidence and separate AWQ/MTP runtime load/forward checks; do not imply GPU qualification from serialization tests.
- [ ] Run combined targeted CPU suite once after integration, required lint and diff checks; record results.
- [ ] Commit and push completed work using the dedicated hoangduydang-collab credential helper, verify remote branch SHA.
