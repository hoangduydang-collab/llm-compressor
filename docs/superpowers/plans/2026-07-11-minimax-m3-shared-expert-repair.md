# MiniMax-M3 Shared-Expert Repair Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [x]`) syntax for tracking.

**Goal:** Repair vLLM shared-expert construction with a config-only alias and hand off one parallel GPU matrix that can close MiniMax-M3 quality.

**Architecture:** Extend the existing immutable checkpoint overlay to add the vLLM-native shared-expert ignore regex while retaining the Transformers regex. A focused repair evidence module owns three arm definitions, manifests, bundling, and verdicts; one arm runner handles offline/HTTP execution and one `srun` launcher starts all arms concurrently.

**Tech Stack:** Python 3.12 standard library, PyYAML configuration, Bash, Slurm `srun`, existing vLLM/Compressed Tensors instrumentation.

## Global Constraints

- Preserve every source checkpoint file; symlink payloads and copy only `config.json`.
- Add `re:.*block_sparse_moe[.]shared_experts[.].*` without removing `re:.*mlp[.]shared_experts[.].*`.
- Do not patch vLLM matching logic, repack tensors, rewrite shards, or re-quantize.
- Use three concurrent exclusive eight-GPU `srun` allocations; do not use `sbatch`.
- Keep TP8, EP, eager mode, block size 128, FP8 KV cache, 2048 context, 0.85 utilization, disabled custom all-reduce, deterministic canonical chat, and disabled shared-expert auxiliary stream.
- Do not resume CUDA-graph diagnosis until repaired W4A8 passes offline and canonical HTTP.

---

### Task 1: Immutable repair overlay and permanent recipe alias

**Files:**
- Modify: `pipeline/m3_routed_diagnostics.py`
- Modify: `pipeline/configs/minimax_m3.yaml`
- Modify: `pipeline/configs/minimax_m3_full_calib.yaml`
- Modify: `pipeline/tests/test_m3_routed_diagnostics_runner.py`

**Interfaces:**
- Consumes: source checkpoint directory and `disable_activations: bool`.
- Produces: `prepare_checkpoint_overlay(..., add_vllm_shared_expert_ignore: bool = False)` and CLI flag `--add-vllm-shared-expert-ignore`.

- [x] **Step 1: Write failing overlay and recipe tests**

```python
def test_repair_overlay_adds_vllm_shared_ignore_once_without_mutating_source():
    prepare_checkpoint_overlay(source, overlay, disable_activations=False,
                               add_vllm_shared_expert_ignore=True)
    assert source_config_is_unchanged
    assert repaired_ignore.count(VLLM_SHARED_EXPERT_IGNORE) == 1
    assert TRANSFORMERS_SHARED_EXPERT_IGNORE in repaired_ignore

def test_minimax_recipes_persist_both_shared_expert_names():
    for path in MINIMAX_CONFIGS:
        ignore = yaml.safe_load(path.read_text())["quantization"]["ignore"]
        assert TRANSFORMERS_SHARED_EXPERT_IGNORE in ignore
        assert VLLM_SHARED_EXPERT_IGNORE in ignore
```

- [x] **Step 2: Run the focused tests and verify RED**

Run: `pytest -q pipeline/tests/test_m3_routed_diagnostics_runner.py`
Expected: failure because the constant/keyword and recipe alias do not exist.

- [x] **Step 3: Implement the minimal overlay and CLI change**

```python
VLLM_SHARED_EXPERT_IGNORE = "re:.*block_sparse_moe[.]shared_experts[.].*"

if add_vllm_shared_expert_ignore:
    qc = config.get("quantization_config")
    if not isinstance(qc, dict):
        raise ValueError("candidate config has no quantization_config")
    ignore = qc.setdefault("ignore", [])
    if VLLM_SHARED_EXPERT_IGNORE not in ignore:
        ignore.append(VLLM_SHARED_EXPERT_IGNORE)
```

Add the same exact regex to both MiniMax recipe ignore lists and wire the CLI flag to the function.

- [x] **Step 4: Run the focused tests and verify GREEN**

Run: `pytest -q pipeline/tests/test_m3_routed_diagnostics_runner.py`
Expected: all tests pass.

---

### Task 2: Repair evidence model and classifier

**Files:**
- Create: `pipeline/m3_shared_expert_repair.py`
- Create: `pipeline/tests/test_m3_shared_expert_repair.py`

**Interfaces:**
- Produces: `EXPECTED_ARMS`, `write_arm_manifest`, `bundle_arm`, `classify_repair`, `aggregate_matrix`, and CLI commands `manifest`, `bundle-arm`, `aggregate`.
- Consumes: existing canonical quality normalization and structured loader/fingerprint/probe extraction.

- [x] **Step 1: Write failing classifier tests**

```python
def test_all_repaired_arms_pass():
    assert classify_repair(healthy_arms())['verdict'] == 'quality_repair_pass'

def test_zero_or_unmatched_shared_experts_fail_repair():
    arms = healthy_arms(); arms['repaired_w4a8_offline']['shared_load_ok'] = False
    assert classify_repair(arms)['verdict'] == 'shared_ignore_repair_failed'

def test_w4a16_only_recovery_selects_activation_boundary():
    arms = healthy_arms(); arms['repaired_w4a8_offline']['quality_ok'] = False
    arms['repaired_w4a8_http']['quality_ok'] = False
    assert classify_repair(arms)['verdict'] == 'activation_boundary_after_shared_repair'

def test_offline_http_disagreement_is_explicit():
    arms = healthy_arms(); arms['repaired_w4a8_http']['quality_ok'] = False
    assert classify_repair(arms)['verdict'] == 'candidate_interface_disagreement'
```

- [x] **Step 2: Run tests and verify RED**

Run: `pytest -q pipeline/tests/test_m3_shared_expert_repair.py`
Expected: import failure because the repair module does not exist.

- [x] **Step 3: Implement evidence health checks and ordered verdicts**

Require eight rank summaries with `shared_seen=171` and `unmatched_this_scope=0`; BF16 `.weight` shared fingerprints with positive sampled magnitude; nonzero, present, non-dropped shared output on all structured probes; valid infrastructure; and canonical quality. Return explicit missing-evidence, repair-failed, interface-disagreement, activation-boundary, W4A16-regression, post-shared-routed-boundary, and pass verdicts in that order.

- [x] **Step 4: Implement manifest/bundling CLI**

Record source and overlay hashes/aliases, scheduler/job/node, immutable envelope, diagnostics, raw responses, software/GPU provenance, compact extracted evidence, full-log hashes, deviations, and retries. Normalize offline `serve_report.json` and HTTP response JSON through existing quality helpers.

- [x] **Step 5: Run tests and verify GREEN**

Run: `pytest -q pipeline/tests/test_m3_shared_expert_repair.py pipeline/tests/test_m3_quality_evidence.py`
Expected: all tests pass.

---

### Task 3: Three-arm execution and parallel srun launcher

**Files:**
- Create: `pipeline/slurm/test_m3_shared_expert_repair_arm.sh`
- Create: `pipeline/slurm/run_m3_shared_expert_repair_srun.sh`
- Create: `pipeline/tests/test_m3_shared_expert_repair_runner.py`
- Create: `pipeline/tests/test_run_m3_shared_expert_repair_srun.py`

**Interfaces:**
- Arms: `repaired_w4a8_offline`, `repaired_w4a16_offline`, `repaired_w4a8_http`.
- Launcher inputs: `MATRIX_ID`, `SRUN_ARGS`, `TIME_LIMIT`, checkpoint/environment/root overrides.

- [x] **Step 1: Write failing dry-run tests**

```python
def test_all_repair_arms_prepare_alias_overlay_in_dry_run():
    # Run each arm with DRY_RUN=1; assert source hash unchanged, overlay contains
    # the vLLM alias, W4A16 alone has input_activations=None, and manifests record it.

def test_launcher_emits_exactly_three_exclusive_srun_commands():
    assert output.count('srun ') == 3
    assert output.count('--exclusive') == 3
    assert 'sbatch' not in output.lower()
```

- [x] **Step 2: Run tests and verify RED**

Run: `pytest -q pipeline/tests/test_m3_shared_expert_repair_runner.py pipeline/tests/test_run_m3_shared_expert_repair_srun.py`
Expected: failures because both scripts are absent.

- [x] **Step 3: Implement the arm runner**

Create the alias overlay before manifest emission. Offline arms enable the existing loader/fingerprint/MoE diagnostics and call `pipeline.run`; HTTP disables diagnostics, launches the proven canonical server, waits for health, sends both fixed chat requests, preserves raw request/response bodies, and always bundles on exit.

- [x] **Step 4: Implement the srun launcher**

Patch/check the shared vLLM environment once, launch all three exclusive node commands in parallel, wait without cancelling successful siblings, rebundle closed logs, aggregate even after an arm failure, and return nonzero if any `srun` fails.

- [x] **Step 5: Run tests and verify GREEN**

Run: `pytest -q pipeline/tests/test_m3_shared_expert_repair_runner.py pipeline/tests/test_run_m3_shared_expert_repair_srun.py`
Expected: all tests pass and dry-run prints exactly three `srun` commands.

---

### Task 4: Executor handoff, documentation, and verification

**Files:**
- Modify: `BUGS_AND_FIXES.md`
- Modify: `MINIMAX_M3_HANDOFF.md`
- Modify: `MINIMAX_M3_QUALITY_RUNBOOK.md`

**Interfaces:**
- Produces: exact preflight/live/return instructions for the GPU executor using only `srun`.

- [x] **Step 1: Update the active boundary and runbook**

Document the 171/171 mismatch, 48/48 zero probes, config alias root cause, exact three-arm command, fixed invariants, allowed scheduler adaptations, evidence tree, verdict meanings, and stop condition. Require the executor to return commit, matrix ID, all job IDs/nodes, arm/aggregate outcomes, retries/deviations, missing signals, and full-log paths/hashes/retention.

- [x] **Step 2: Run full local verification**

Run:

```bash
pytest -q pipeline/tests/test_m3_shared_expert_repair.py \
  pipeline/tests/test_m3_shared_expert_repair_runner.py \
  pipeline/tests/test_run_m3_shared_expert_repair_srun.py \
  pipeline/tests/test_m3_routed_diagnostics_runner.py \
  pipeline/tests/test_m3_quality_evidence.py \
  pipeline/tests/test_patch_vllm_m3_serve.py
bash -n pipeline/slurm/test_m3_shared_expert_repair_arm.sh \
  pipeline/slurm/run_m3_shared_expert_repair_srun.sh
python -m compileall -q pipeline
git diff --check
```

Expected: all available tests pass, scripts parse, Python compiles, and diff check is clean. If local pytest is unavailable, run every zero-fixture test directly and make full pytest mandatory in executor preflight.

- [x] **Step 3: Commit and push the implementation**

```bash
git add BUGS_AND_FIXES.md MINIMAX_M3_HANDOFF.md MINIMAX_M3_QUALITY_RUNBOOK.md \
  pipeline docs/superpowers/plans/2026-07-11-minimax-m3-shared-expert-repair.md
git commit -m "fix: repair MiniMax-M3 shared expert loading"
git push origin duy-branch
```

- [x] **Step 4: Stop for GPU execution**

Do not claim quality fixed or resume CUDA graphs until returned runtime evidence satisfies `quality_repair_pass`.
