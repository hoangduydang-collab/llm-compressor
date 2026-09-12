# GLM-5.3 FP8-before-GPTQ Implementation Plan

> **For agentic workers:** Use executing-plans or subagent-driven-development to implement task by task.

**Goal:** Calibrate routed INT4 experts with upstream FP8 weight error present and enable direct W4AFP8 output.

**Architecture:** Reuse CT FP8 primitives and the existing resident-block loop. A bounded preparation helper freezes scales and publishes dequantized weights before calibration. A native-save adapter reuses distributed checkpoint saving and existing format helpers.

**Tech Stack:** Python, PyTorch, compressed-tensors, Transformers, pytest CPU/Gloo.

## Global Constraints

- Preserve GPTQ math, expert ownership, calibration coverage and active W4A16 packet.
- No new FP8 arithmetic, custom distributed cache or checkpoint writer.
- No full-model/GPU launch; distinguish CPU qualification from runtime quality.
- Preserve unrelated `.gitignore`, `predictive_ptq/`, and `results/predictive-ptq/` changes.

## Task 1: FP8-before-GPTQ calibration

- [x] Add opt-in `fp8_weights_before_gptq` configuration and modifier flag `quantize_weights_before_calibration`.
- [x] Add a resident-subgraph preparation helper; validate GPTQ-only composition, target disjointness and symmetric 128x128 FP8 weights.
- [x] Reuse `observe`/observer qparams, CT `fake_quantize`, and offload publication; preserve prepared scales at epoch end.
- [x] Add explicit-reference input/Hessian/propagation and save/reload tests; run CPU/Gloo mixed EP cases.
- [x] Add separate W4AFP8 representative/full YAMLs and recipe validation tests.

Files: `pipeline/config.py`, `pipeline/recipe.py`, `src/llmcompressor/pipelines/sequential/pipeline.py`, `src/llmcompressor/modifiers/quantization/quantization/base.py`, a focused helper beside that modifier, tests beside existing sequential/EP fixtures.

## Task 2: Direct native W4AFP8 save

The [approved design](../specs/2026-09-12-glm53-fp8-before-gptq-design.md) defines the native contract. The adapter and its tests live in `pipeline/native_sglang_save.py` and `pipeline/tests/test_native_sglang_save.py`.

- [x] Inspect actual CT save/compressor APIs and existing converter helpers.
- [x] Adapt compressed tensors before their first shard write using existing save machinery.
- [x] Emit explicit SGLang layout/metadata; require supported full module scope and reject unsupported act-order/group/scale layouts.
- [x] Verify exact numerical preservation and collective save on tiny fixtures.

## Task 3: Integration and review

- [x] Connect config, calibration, native save and appropriate validation; document runnable commands and remaining GPU gates.
- [x] Run focused recipe, lifecycle, EP, native-save and converter conformance suites.
- [x] Review the complete diff; fix correctness findings and record tested versions/results.

Implementation and review complete. See [qualification record](../../glm53-fp8-before-gptq-implementation.md) for CPU evidence and remaining GPU/runtime gates. No GPU/full-model execution was performed.
