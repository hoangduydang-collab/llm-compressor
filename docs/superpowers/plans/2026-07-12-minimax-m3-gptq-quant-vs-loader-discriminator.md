# MiniMax-M3 GPTQ Discriminator Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Repair the MiniMax-M3 smoke evaluator and preserve quantization-fidelity evidence early enough to distinguish GPTQ checkpoint error from loader/runtime error.

**Architecture:** Keep pure validation and metric logic in Python, keep cluster orchestration in the existing `srun` shell runners, and reuse the existing teacher-forced and layer-boundary probes. Smoke arms run the distributional probe before lm-eval; exact-sample preflight uses the same filtered `eval_docs` collection as lm-eval.

**Tech Stack:** Python 3.11, pytest, lm-eval 0.4.12, vLLM 0.24, Bash, Slurm `srun`, Ray.

## Global Constraints

- Continue on the current shared branch; the user explicitly rejected an isolated worktree.
- Model quality and quantization fidelity are primary; performance remains deferred.
- Do not launch a 7-to-15-hour re-quantization run without direct offline-dequant evidence.
- AutoRound remains deferred.
- Capable-cluster work starts with bounded smoke runs and returns raw evidence through Git.

---

### Task 1: Exact filtered sample manifests

**Files:**
- Modify: `pipeline/m3_quality_preflight.py`
- Modify: `pipeline/m3_quality_eval.py`
- Test: `pipeline/tests/test_m3_quality_eval.py`

**Interfaces:**
- Consumes: loaded lm-eval leaf tasks exposing `eval_docs`.
- Produces: `inspect_leaf_sizes(manager, installed_task) -> dict[str, int]` and `validate_sample_indices(tasks, leaf_sizes) -> None`.

- [ ] Write tests proving raw dataset size can differ from `eval_docs`, and proving invalid indices report task, leaf, size, and maximum index.
- [ ] Run the focused tests and confirm they fail for the expected old behavior/missing function.
- [ ] Size leaves with `len(task.eval_docs)` and validate both smoke and production manifest data before writing successful preflight artifacts.
- [ ] Run the focused tests and the complete MiniMax quality test module.
- [ ] Commit the task.

### Task 2: Nested generation-health normalization

**Files:**
- Modify: `pipeline/evalsuite/static.py`
- Modify: `pipeline/evalsuite/health.py`
- Test: `pipeline/tests/test_eval_health.py`
- Test: `pipeline/tests/test_eval_static.py`

**Interfaces:**
- Consumes: lm-eval response shapes such as `['text']` and `[[score, false]]`.
- Produces: `_first_response(sample)` returning text only for singleton textual containers, while retaining structured likelihood results as non-generative evidence.

- [ ] Add regression tests using the exact observed nested generation and likelihood response shapes.
- [ ] Run them and confirm the nested-text test fails because health is currently not applicable.
- [ ] Add minimal singleton-text unwrapping shared by extraction and enrichment without flattening numeric/boolean likelihood structures.
- [ ] Run focused health/static tests and their full modules.
- [ ] Commit the task.

### Task 3: Quantization-oriented distribution metrics

**Files:**
- Modify: `pipeline/evalsuite/distributional.py`
- Test: `pipeline/tests/test_eval_distributional.py`

**Interfaces:**
- Consumes: paired teacher-forced top-k prompt-logprob records.
- Produces: explicit `argmax_flip_ratio`, observed-logprob absolute-error quantiles, and reference-argmax candidate-rank summaries at aggregate, length-bucket, and position-quartile levels.

- [ ] Add failing assertions for an argmax flip and candidate rank displacement.
- [ ] Run the focused test and confirm missing metric keys cause failure.
- [ ] Implement the metrics using only paired top-k support; do not claim full-vocabulary KL divergence.
- [ ] Run the full distributional test module.
- [ ] Commit the task.

### Task 4: Probe-first smoke execution

**Files:**
- Modify: `pipeline/slurm/test_m3_quality_eval_arm.sh`
- Test: `pipeline/tests/test_m3_quality_eval_runner.py`

**Interfaces:**
- Consumes: `--run-probe`, profile, corpus, model, and tensor-parallel arguments.
- Produces: probe JSONL/summary before task artifacts; a failed probe writes return evidence and skips lm-eval.

- [ ] Add shell-runner contract tests proving probe invocation precedes lm-eval in smoke and a probe failure prevents lm-eval.
- [ ] Run tests and confirm the old ordering fails.
- [ ] Reorder only probe-enabled smoke arms, keep return-code and smoke-evidence generation total, and preserve production behavior.
- [ ] Run focused runner tests and shell syntax validation.
- [ ] Commit the task.

### Task 5: Parallel executor diagnostic handoff

**Files:**
- Modify: `M3_QUALITY_THREE_MODEL_SMOKE_RECOVERY_HANDOFF.md`
- Create: `pipeline/slurm/test_m3_ray_placement_group.sh`
- Test: `pipeline/tests/test_m3_quality_eval_runner.py`

**Interfaces:**
- Consumes: two Slurm nodes with eight GPUs each and the existing Ray topology gate.
- Produces: bounded 16-bundle placement-group evidence, parallel GPTQ/AWQ probe-first smoke commands, a bounded BF16 vLLM initialization diagnostic, and a complete artifact-return contract.

- [ ] Add a contract test for placement-group bundle count, timeout, status capture, and cleanup.
- [ ] Run it and confirm the helper is absent.
- [ ] Implement the bounded helper and update the handoff with exact `srun` commands, checkpoint paths, stop/go rules, parallel resource assignments, required raw artifacts, and analysis questions.
- [ ] Run shell syntax, focused tests, documentation placeholder scan, and `git diff --check`.
- [ ] Commit and push the shared branch.

### Task 6: Final verification

**Files:**
- Verify all modified files.

**Interfaces:**
- Consumes: all prior task outputs.
- Produces: fresh test evidence and a clean pushed handoff commit.

- [ ] Run all evaluator health, static, distributional, MiniMax quality, and runner tests.
- [ ] Run `bash -n` on every modified shell script.
- [ ] Inspect the complete diff against the approved design and confirm no unrelated serving/performance work entered scope.
- [ ] Push the branch and report the commit and executor's next command.
