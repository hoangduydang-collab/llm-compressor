# MiniMax-M3 BF16 r4 Companion Evaluation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use
> superpowers:subagent-driven-development (recommended) or
> superpowers:executing-plans to implement this plan task-by-task. Steps use
> checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a BF16-only TP8xPP2/Ray smoke and quick evaluation that is
sample-for-sample comparable with the active GPTQ/AWQ r4.5 reasoning run.

**Architecture:** A committed single-model matrix reuses the existing r4 eval
configuration and two-shard launcher. A small run-contract gate compares the
BF16 preflight manifest with the active GPTQ/AWQ run before production. The
existing production-eval handoff becomes the single copy-ready executor packet.

**Tech Stack:** Python, pytest, YAML, Bash, Slurm `srun`, Ray, vLLM, lm-eval
0.4.12.

## Global Constraints

- Work and push only on `duy-branch`; do not create a worktree.
- Use top-level `srun`; `sbatch` is unavailable.
- BF16 arms use exactly two 8xH100 nodes, TP=8, PP=2, and Ray.
- Reuse `pipeline/configs/eval_minimax_m3_reasoning_r4.yaml` unchanged.
- Production uses 100 GPQA, 100 MMLU-Pro, 100 GSM8K, and all 30 AIME
  questions, each with seeds 42, 1234, and 4158.
- Smoke uses two questions per task and is setup validation, not quality
  evidence.
- Do not launch, cancel, modify, or reuse the active GPTQ/AWQ run.
- Production must fail closed when its contract differs from the reference
  GPTQ/AWQ run.

---

### Task 1: BF16-only matrix and launch contract

**Files:**
- Create: `pipeline/configs/minimax_m3_bf16_reasoning_r4.yaml`
- Modify: `pipeline/tests/test_m3_quality_eval_runner.py`

**Interfaces:**
- Consumes: `load_matrix()` and `build_launch_plan()` from
  `pipeline.m3_quality_eval`.
- Produces: a one-model matrix whose smoke plan has one two-node arm and whose
  production plan has two two-node arms.

- [ ] **Step 1: Write the failing matrix/launch-plan test**

Load the new matrix, assert its sole model is BF16 with `nodes=2`, TP8, PP2,
and Ray, then assert smoke has one arm and production has two arms with the
same topology, r4 task grouping, 24-hour time limit, and four total nodes.

- [ ] **Step 2: Run the focused test and verify it fails because the matrix is missing**

Run:

```bash
python -m pytest -q \
  pipeline/tests/test_m3_quality_eval_runner.py \
  -k bf16_reasoning_r4
```

Expected: FAIL with `FileNotFoundError` for the new matrix.

- [ ] **Step 3: Add the minimal BF16 matrix**

Create the complete matrix:

```yaml
schema_version: 1
name: minimax-m3-bf16-reasoning-r4
backend: vllm
baseline_label: bf16
model_source: /mnt/nfs/hoangduy/hf_assets/MiniMaxAI/MiniMax-M3
eval_config: pipeline/configs/eval_minimax_m3_reasoning_r4.yaml
models:
  - label: bf16
    path: /mnt/nfs/hoangduy/hf_assets/MiniMaxAI/MiniMax-M3
    kind: bf16
    nodes: 2
    tensor_parallel_size: 8
    pipeline_parallel_size: 2
    distributed_executor_backend: ray
shards:
  - name: gpqa
    tasks: [gpqa_diamond]
    distributional_probe: false
  - name: reasoning_suite
    tasks: [mmlu_pro, gsm8k, aime_2025]
    distributional_probe: false
task_aliases:
  gpqa_diamond: [gpqa_diamond_cot_zeroshot]
  mmlu_pro: [mmlu_pro]
  gsm8k: [gsm8k_cot]
  aime_2025: [aime25, aime_2025]
sampling:
  seed: 42
  mmlu_pro_samples: 100
  production_samples_per_task: 100
scheduling:
  max_parallel_arms: 2
  arm_time_limit: "24:00:00"
probe:
  enabled: false
  total_tokens: 8192
  top_k: 20
  max_overhead_seconds: 900
gates:
  max_task_drop: 0.02
  min_macro_recovery: 0.98
  max_conditional_regression: 0.05
  max_perplexity_increase: null
  max_degeneration_failures: 0
```

- [ ] **Step 4: Run the focused test and verify it passes**

Run the Step 2 command. Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add pipeline/configs/minimax_m3_bf16_reasoning_r4.yaml \
  pipeline/tests/test_m3_quality_eval_runner.py
git commit -m "feat(eval): add BF16 r4 companion matrix"
```

### Task 2: Cross-run comparability gate

**Files:**
- Modify: `pipeline/m3_quality_eval.py`
- Modify: `pipeline/tests/test_m3_quality_eval.py`

**Interfaces:**
- Produces: `compare_run_contracts(reference_root: Path, candidate_root: Path)
  -> dict[str, Any]`.
- Produces CLI: `python -m pipeline.m3_quality_eval contract-gate
  --reference-root REF --candidate-root CANDIDATE --out REPORT`.

- [ ] **Step 1: Write failing unit and CLI tests**

Create reference/candidate `run_manifest.json` fixtures. Assert identical
`lm_eval_version`, `harness_contract_sha256`, `sample_manifest_sha256`,
`eval_config_sha256`, `tokenizer_sha256`, `chat_template_sha256`,
`rendered_prompt_sha256`, `generation_seeds`, `expected_question_counts`, and
`resolved_tasks` return `valid: true`. Mutate one field and assert `valid:
false`, the field appears in `mismatches`, the report is written, and CLI exits
1.

- [ ] **Step 2: Run the focused tests and verify they fail**

```bash
python -m pytest -q pipeline/tests/test_m3_quality_eval.py -k run_contract
```

Expected: FAIL because `compare_run_contracts` and `contract-gate` do not exist.

- [ ] **Step 3: Implement the minimal fail-closed gate**

Read both `run_manifest.json` files, compare only the pinned scientific
contract fields above, and return a deterministic report containing roots,
matched fields, mismatches with both values, and `valid`. The CLI writes JSON
and returns 0 only when valid.

```python
RUN_CONTRACT_FIELDS = (
    "lm_eval_version",
    "harness_contract_sha256",
    "sample_manifest_sha256",
    "eval_config_sha256",
    "tokenizer_sha256",
    "chat_template_sha256",
    "rendered_prompt_sha256",
    "generation_seeds",
    "expected_question_counts",
    "resolved_tasks",
)


def compare_run_contracts(reference_root: Path, candidate_root: Path) -> dict:
    reference = _read_json(reference_root / "run_manifest.json")
    candidate = _read_json(candidate_root / "run_manifest.json")
    matched = []
    mismatches = {}
    for field in RUN_CONTRACT_FIELDS:
        if reference.get(field) == candidate.get(field):
            matched.append(field)
        else:
            mismatches[field] = {
                "reference": reference.get(field),
                "candidate": candidate.get(field),
            }
    return {
        "schema_version": 1,
        "valid": not mismatches,
        "reference_root": str(reference_root),
        "candidate_root": str(candidate_root),
        "matched_fields": matched,
        "mismatches": mismatches,
    }
```

- [ ] **Step 4: Run the focused tests and verify they pass**

Run the Step 2 command. Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add pipeline/m3_quality_eval.py pipeline/tests/test_m3_quality_eval.py
git commit -m "feat(eval): gate BF16 cross-run comparability"
```

### Task 3: Executor handoff

**Files:**
- Modify: `M3_PRODUCTION_EVAL_HANDOFF.md`

**Interfaces:**
- Consumes: the BF16 matrix and `contract-gate` CLI from Tasks 1-2.
- Produces: one active `READY_FOR_EXECUTOR` BF16 packet with exact commands.

- [ ] **Step 1: Update the active packet metadata and supersession notice**

Point executors to the BF16 r4 companion section and mark older full-suite
commands historical. Require one dynamic input only:
`REFERENCE_RUN_ROOT`, the active GPTQ/AWQ r4.5 root.

- [ ] **Step 2: Add copy-ready preflight, smoke, and production commands**

Commands must pull `duy-branch`, verify the required ancestor, run focused CPU
tests and Bash syntax checks, create a fresh BF16 root, run preflight and smoke,
assert the smoke gate, run `contract-gate`, inspect a dry run for one smoke arm
and two TP8xPP2 production arms, then start production under detached `tmux`.

- [ ] **Step 3: Add monitoring and return requirements**

Record `squeue`, `sacct`, `scontrol`, controller/arm return codes, task/seed
progress, manifests, raw-log paths/hashes, contract-gate report, and partial
artifacts. Stop on mismatch or failure; no topology change or retry.

- [ ] **Step 4: Self-review the handoff**

Verify there are no placeholders, `sbatch` commands, stale TP16xPP1 commands,
or instructions to rerun GPTQ/AWQ. Verify AIME is explicitly 30 questions and
all other tasks are 100.

- [ ] **Step 5: Commit**

```bash
git add M3_PRODUCTION_EVAL_HANDOFF.md
git commit -m "docs(handoff): authorize BF16 r4 companion eval"
```

### Task 4: Final verification and push

**Files:**
- Verify only; no planned source changes.

- [ ] **Step 1: Run focused evaluation tests**

```bash
python -m pytest -q \
  pipeline/tests/test_m3_quality_eval.py \
  pipeline/tests/test_m3_quality_eval_runner.py \
  pipeline/tests/test_m3_quality_evidence.py \
  pipeline/tests/test_static_checkpoint.py \
  pipeline/tests/test_lmeval_runner.py
```

Expected: all tests pass.

- [ ] **Step 2: Validate Bash and repository state**

```bash
bash -n pipeline/slurm/run_m3_quality_eval_srun.sh
bash -n pipeline/slurm/test_m3_quality_eval_arm.sh
git diff --check origin/duy-branch..HEAD
git status --short --branch
```

Expected: syntax checks and diff check exit 0; only intended commits are ahead.

- [ ] **Step 3: Push and verify synchronization**

```bash
git push origin duy-branch
git rev-parse HEAD
git rev-parse origin/duy-branch
```

Expected: local and remote revisions are identical.
