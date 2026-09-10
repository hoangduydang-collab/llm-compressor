# Rancher GLM-5.3 EP GPTQ Chain Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build and queue one reproducible eight-H100 Rancher Job that gates a full GLM-5.3 W4A16 GPTQ EP8 run behind tiny NCCL and real-width representative checks, holds 24 hours only on failure, and releases immediately on success.

**Architecture:** Add deterministic local calibration-file support, explicit representative and full expert-only configs, a static checkpoint/evidence validator, and a rendered Kubernetes Job driven by a small launcher. The Job checks out one pushed commit, runs every lane sequentially in one container, writes all evidence to one CephFS run root, and implements failure holding in a single tested shell function.

**Tech Stack:** Python 3.12, pytest, PyYAML, Hugging Face Datasets/Transformers, llm-compressor, compressed-tensors, torchrun/NCCL, Bash, Kubernetes.

## Global Constraints

- Quantize only routed expert gate/up/down projections; keep attention, indexer, router, shared experts, dense layers, embeddings, norms and `lm_head` BF16.
- Use W4A16 symmetric group-128 GPTQ; do not add activation quantization.
- Representative source is the immutable five-layer copy of BF16 revision `304b8051cfb2b260b61ce0cbe330e02a98e73639`.
- Representative order is tiny NCCL, EP4, EP8, offloaded DDP4 control, validation.
- Full EP8 starts only if every representative gate passes.
- Any failure holds 86,400 seconds and then exits nonzero; full success exits immediately.
- Job retries are disabled.
- Do not touch or stage unrelated untracked files.

---

### Task 1: Deterministic calibration fixture support

**Files:**
- Modify: `pipeline/config.py`
- Modify: `pipeline/calibration.py`
- Create: `pipeline/fixtures/glm53_ep_gptq_representative.jsonl`
- Test: `pipeline/tests/test_calibration.py`

**Interfaces:**
- Consumes: `CalibrationConfig.dataset_id`, `dataset_split`, and tokenizer.
- Produces: optional `CalibrationConfig.dataset_data_files: str | None`; passes it to `datasets.load_dataset` without changing existing remote-dataset behavior.

- [ ] Add failing tests proving a configured local JSONL path reaches `load_dataset(..., data_files=...)` and the default call omits `data_files`.
- [ ] Run the focused tests and verify the new-path test fails because the field/argument is absent.
- [ ] Implement the optional field and conditional loader keyword.
- [ ] Add eight deterministic, non-sensitive text rows of varied length.
- [ ] Run `python -m pytest -q pipeline/tests/test_calibration.py`.

### Task 2: Expert-only representative and full configurations

**Files:**
- Create: `pipeline/configs/glm53_ep_gptq_representative.yaml`
- Create: `pipeline/configs/glm53_ep_gptq_full.yaml`
- Create: `pipeline/validate_glm53_ep_gptq.py`
- Create: `pipeline/tests/test_glm53_ep_gptq_configs.py`
- Create: `pipeline/tests/test_validate_glm53_ep_gptq.py`

**Interfaces:**
- Consumes: pipeline configs, run directories, checkpoint indices, safetensors headers, per-rank `quant_metrics` JSONL.
- Produces: `validate_config_scope(config, module_names, expected_layers)`, `validate_checkpoint(path, expected_layers)`, `validate_phase_evidence(run_dir, world_size)`, and a JSON report with a nonzero process exit on any failed invariant.

- [ ] Write failing config tests for W4A16, EP enablement, layer scope, local fixed inputs, disabled activation targets, and production calibration values.
- [ ] Write failing validator tests using synthetic checkpoint indexes/headers and phase records for missing experts, forbidden INT4 modules, invalid scales, incomplete ranks, peak memory and EP8-to-EP4 ratio.
- [ ] Run both new test files and confirm they fail because the configs and validator do not exist.
- [ ] Implement only the validation needed by the tests, reusing `pipeline.metrics.summarize_phases`.
- [ ] Run both new test files to green.

### Task 3: Immutable chain renderer and shell semantics

**Files:**
- Create: `pipeline/k8s/glm53-ep-gptq-chain.yaml.tmpl`
- Create: `pipeline/k8s/render_glm53_ep_gptq_chain.py`
- Create: `pipeline/k8s/launch-glm53-ep-gptq-chain.sh`
- Create: `pipeline/tests/test_render_glm53_ep_gptq_chain.py`

**Interfaces:**
- Renderer consumes `--run-tag`, `--ref`, `--out`, and optional `--node`.
- Renderer produces one parsed manifest with no unresolved placeholders and prints persisted key values.
- Launcher consumes the same values plus `--dry-run`; it requires a pushed commit, runs `scripts/gpu-free.sh --verify`, requires largest schedulable pod size eight, validates the extracted shell, and only then applies.

- [ ] Write failing renderer tests asserting one eight-GPU Job, no retries, exact BF16 revision, stage order, full-run pass dependency, failure-only 24-hour sleep, no success sleep, durable run root, and no placeholders.
- [ ] Run the renderer tests and confirm failure because the renderer is absent.
- [ ] Implement the renderer and manifest with an isolated venv, pinned versions, physical five-layer subset, unique lane paths, `torchrun` commands, validator calls, and status JSON.
- [ ] Add the launcher's dry-run and capacity gates.
- [ ] Run renderer tests, parse the rendered YAML, extract the command, and run `bash -n`.

### Task 4: Complete execution packet

**Files:**
- Create: `docs/glm53-ep-gptq-execution-packet.md`

**Interfaces:**
- Consumes: exact committed configs, renderer, launcher, thresholds and paths.
- Produces: a `READY_FOR_EXECUTOR` protocol-v1 packet naming the final commit, resource contract, commands, expected markers, stop conditions, evidence schema, and cleanup behavior.

- [ ] Write the packet with no dynamic executor decisions.
- [ ] Include the prior-art URLs, W4A16 module table, calibration deviation, MTP limitation, representative thresholds and failure-hold semantics.
- [ ] Verify every command/path matches the implemented artifacts and scan for placeholders.

### Task 5: Local qualification and commit

**Files:**
- All files above.

- [ ] Run all focused new and modified tests.
- [ ] Run existing EP recipe, pipeline metrics, calibration, subset-checkpoint and renderer regressions.
- [ ] Run `python -m compileall` on new Python modules.
- [ ] Render a dry-run manifest, parse it with PyYAML, extract the container script, and run `bash -n`.
- [ ] Run `git diff --check` and inspect the complete scoped diff.
- [ ] Commit the implementation without staging unrelated files.

### Task 6: Push and GPU launch gate

- [ ] Ask the user before pushing the commits.
- [ ] After push approval, push `duy-branch` and verify the final commit exists on the remote.
- [ ] Run `scripts/gpu-free.sh --verify`.
- [ ] Show total/per-node free GPUs, fully free nodes, largest schedulable pod, exact Job name, commit, image, resources and durable paths.
- [ ] Ask for explicit approval to create the eight-GPU Job.
- [ ] Apply only after approval, verify Pending or Running state, and start stage-aware monitoring.
