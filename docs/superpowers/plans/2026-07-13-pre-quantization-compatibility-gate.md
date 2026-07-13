# Pre-Quantization Compatibility Gate Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a fail-fast structural compatibility report for original models and AWQ/GPTQ recipes before calibration.

**Architecture:** Analyze a disposable meta model with llm-compressor's real target/configuration and AWQ mapping resolvers, stopping before calibration hooks or forwards. Expose a library report and a pipeline-config CLI that writes versioned JSON and exits nonzero on hard failures.

**Tech Stack:** Python, PyTorch meta tensors, compressed-tensors, llm-compressor modifiers, pytest, argparse/JSON.

## Global Constraints

- Support AWQ and GPTQ first; report other methods as unsupported.
- Invoke real planner/mapping logic and do not duplicate target matching.
- Never start calibration, attach hooks, run forwards, or load checkpoint tensors.
- Static success does not claim numerical quality or runtime compatibility.

---

### Task 1: Library analyzer and report

**Files:**
- Create: `src/llmcompressor/preflight/__init__.py`
- Create: `src/llmcompressor/preflight/quantization.py`
- Create: `tests/llmcompressor/preflight/test_quantization.py`

**Interfaces:**
- Consumes: `torch.nn.Module`, ordered `list[Modifier]`.
- Produces: `analyze_quantization_compatibility(model, modifiers) -> QuantizationCompatibilityReport` and JSON-safe `to_dict()`.

- [ ] Write failing synthetic-model tests for GPTQ/AWQ success and each hard failure.
- [ ] Run focused tests and confirm failures are caused by the missing analyzer.
- [ ] Implement immutable findings/report types and shared recipe/target inventory.
- [ ] Apply quantization metadata through the actual `QuantizationMixin` initializer.
- [ ] Resolve AWQ mappings through `AWQModifier.on_initialize` and `_set_resolved_mappings`, recording norm adapter coverage.
- [ ] Run focused tests and refactor only while green.

### Task 2: Pipeline CLI and MiniMax regression

**Files:**
- Create: `pipeline/prequant_compatibility.py`
- Create: `pipeline/tests/test_prequant_compatibility.py`
- Modify: `BUGS_AND_FIXES.md`
- Modify: `docs/quantization-static-serving-preflight-status-and-roadmap.md`

**Interfaces:**
- Consumes: existing pipeline YAML config and optional model-id override.
- Produces: versioned JSON report at `--output`; exit 0 only when `compatible` is true.

- [ ] Write failing CLI serialization/exit-code tests using injected synthetic builders.
- [ ] Confirm tests fail because CLI orchestration is absent.
- [ ] Implement meta-model construction, exact recipe construction, report writing, and concise terminal summary.
- [ ] Add a MiniMax-named synthetic test proving its offset norm resolves to `CalibrationOffsetNorm` and fails if adapter coverage is removed.
- [ ] Document how this gate precedes representative canaries and the post-quantization ABI gate.
- [ ] Run focused tests, format checks, and compile checks.

### Task 3: Verification and handoff readiness

**Files:**
- Modify only files above if verification exposes defects.

- [ ] Run all new tests plus existing AWQ mapping, offset-norm, group-size, and recipe tests.
- [ ] Run Ruff on changed Python files and compile the CLI.
- [ ] Inspect the final diff for scope, planner reuse, and accidental calibration paths.
- [ ] Commit the implementation and record exact local verification evidence.
