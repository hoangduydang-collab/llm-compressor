# MiniMax-M3 r4.8 Replay Repair and BF16 TP16 Qualification Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use
> superpowers:subagent-driven-development (recommended) or
> superpowers:executing-plans to implement this plan task-by-task. Steps use
> checkbox (`- [ ]`) syntax for tracking.

**Goal:** Repair the exact GPTQ raw replay's missing serving envelope and
qualify BF16 TP16xPP1/Ray without relaunching paired production or authorizing
BF16 production.

**Architecture:** The shared lm-eval vLLM argument builder forwards the existing
typed MiniMax serving fields so standalone replay matches the successful smoke.
The existing BF16 matrix moves from unsupported PP2 to vLLM's documented
two-node TP16/Ray alternative, but only a bounded, evidence-rich smoke is
authorized. The active handoff is replaced in place and never launches Wave B.

**Tech Stack:** Python, pytest, YAML, Bash, Slurm `srun`, Ray, vLLM, lm-eval
0.4.12.

## Global Constraints

- Work and push only on `duy-branch`; do not create a worktree.
- Use top-level `srun`; `sbatch` is unavailable.
- BF16 qualification uses exactly two 8xH100 nodes, TP=16, PP=1, and Ray.
- BF16 qualification enforces a 15-minute Ray placement boundary, a 3-hour
  model-constructor boundary, and a 12-hour allocation boundary.
- BF16 production and any topology fallback remain unauthorized.
- Existing paired GPTQ/AWQ production is not relaunched or modified.
- Reuse `pipeline/configs/eval_minimax_m3_reasoning_r4.yaml` unchanged.
- Production uses 100 GPQA, 100 MMLU-Pro, 100 GSM8K, and all 30 AIME
  questions, each with seeds 42, 1234, and 4158.
- Smoke uses two GPQA/GSM8K/AIME questions plus one question per MMLU-Pro leaf
  and is setup validation, not quality evidence.
- A completed empty response is scored incorrect and reported as health; it is
  not a missing attempt or an automatic retry.
- Replay exactly attempt UID
  `8e98c89a40db606e115a1d388e89a58518582d44f2f48dafcf389a1e1e146878`
  with fixed token caps 256 and 16,384; do not add `min_tokens`.
- Never write replay output into benchmark sample or aggregate files.
- Use at most seven packet-owned nodes: four existing paired nodes plus one
  replay node and two BF16 qualification nodes. There is no Wave B.
- Production must fail closed when its contract differs from the reference
  GPTQ/AWQ run.

---

## Completed foundation

Tasks 1-9 below are completed r4.7 history and must not be re-executed. Active
r4.8 work begins at Task 10.

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

---

### Task 5: Separate score quality from generation-health advisory

**Files:**
- Modify: `pipeline/m3_quality_eval.py:818-880`
- Modify: `pipeline/tests/test_m3_quality_eval.py:1030-1080`

**Interfaces:**
- Consumes: each comparison's existing `generation_health` object.
- Preserves: `GateThresholds.max_degeneration_failures` for historical matrix
  compatibility; it no longer contributes to `quality_ok`.
- Produces: top-level `health_advisory = {"has_findings": bool, "models":
  {...}}` in `gates.json`.

- [ ] **Step 1: Replace the old zero-degeneration gate test with the r4.7 contract**

Write this focused test using a passing score comparison with one completed
empty-response health finding:

```python
def test_r4_complete_empty_response_is_health_advisory_not_quality_failure():
    matrix = {
        "infrastructure_ok": True,
        "comparisons": {
            "quant": {
                "tasks": {
                    "gpqa": {
                        "n_paired": 300,
                        "delta": 0.0,
                        "score_recovery_ratio": 1.0,
                        "regressions_a_correct_b_wrong": 0,
                        "both_correct": 100,
                    }
                },
                "generation_health": {
                    "baseline": {
                        "tasks": {
                            "gpqa": {
                                "samples": 300,
                                "empty_count": 0,
                                "empty_rate": 0.0,
                            }
                        },
                        "degeneration_failures": 0,
                    },
                    "candidate": {
                        "tasks": {
                            "gpqa": {
                                "samples": 300,
                                "empty_count": 1,
                                "empty_rate": 1 / 300,
                            }
                        },
                        "degeneration_failures": 1,
                    },
                    "degeneration_failures": 1,
                },
            }
        },
    }

    gates = evaluate_gates(matrix, GateThresholds(0.02, 0.98, 0.05, None, 0))

    assert gates["infrastructure_ok"] is True
    assert gates["quality_ok"] is True
    assert "degeneration_failures" not in gates["models"]["quant"]
    advisory = gates["health_advisory"]
    assert advisory["has_findings"] is True
    assert advisory["models"]["quant"]["combined_degeneration_failures"] == 1
    assert advisory["models"]["quant"]["candidate"]["tasks"]["gpqa"][
        "empty_count"
    ] == 1
```

Also extend `test_matrix_report_surfaces_quantization_metrics_and_gates` so a
report with findings contains `Health advisory: findings present` while still
showing the score verdict as `PASS`.

- [ ] **Step 2: Run the focused tests and verify RED**

```bash
python -m pytest -q pipeline/tests/test_m3_quality_eval.py \
  -k "complete_empty_response or matrix_report_surfaces"
```

Expected: FAIL because one degeneration currently sets `quality_ok=false` and
`health_advisory` does not exist.

- [ ] **Step 3: Implement the minimal advisory split**

Add this helper beside `evaluate_gates`:

```python
def _build_health_advisory(comparisons: dict[str, dict]) -> dict[str, Any]:
    models = {}
    for model, comparison in comparisons.items():
        health = dict(comparison.get("generation_health") or {})
        combined = int(health.get("degeneration_failures", 0))
        models[model] = {
            "has_findings": combined > 0,
            "combined_degeneration_failures": combined,
            "baseline": health.get("baseline") or {"tasks": {}},
            "candidate": health.get("candidate") or {"tasks": {}},
        }
    return {
        "has_findings": any(model["has_findings"] for model in models.values()),
        "models": models,
    }
```

In `evaluate_gates`, delete the `degeneration_failures` entry from the model's
score checks, compute `quality_ok` from the remaining score checks, and attach:

```python
comparisons = matrix.get("comparisons") or {}
health_advisory = _build_health_advisory(comparisons)
```

Return `health_advisory` beside `infrastructure_ok`, `quality_ok`, and `models`.
Update `render_matrix_report` with exactly one summary line derived from
`health_advisory.has_findings`; do not suppress the existing degeneration
count column.

- [ ] **Step 4: Verify GREEN and regression coverage**

```bash
python -m pytest -q pipeline/tests/test_m3_quality_eval.py
```

Expected: all tests pass; a complete empty row remains correctness zero through
the existing checkpoint tests.

- [ ] **Step 5: Commit**

```bash
git add pipeline/m3_quality_eval.py pipeline/tests/test_m3_quality_eval.py
git commit -m "fix(eval): make generation health advisory non-gating"
```

### Task 6: Pure replay contract and postprocessing stages

**Files:**
- Create: `pipeline/m3_empty_output_replay.py`
- Create: `pipeline/tests/test_m3_empty_output_replay.py`

**Interfaces:**
- Produces: `ReplayAttempt`, `load_replay_attempt()`,
  `postprocess_stages()`, and `run_controls()`.
- Does not import `vllm` or `lm_eval` at module import time; CPU tests remain
  runnable without the cluster environment.

- [ ] **Step 1: Write failing pure-contract tests**

Create a temporary JSONL row with the exact normalized shape:

```python
ROW = {
    "attempt_uid": "8e98c89a40db606e115a1d388e89a58518582d44f2f48dafcf389a1e1e146878",
    "task": "mmlu_pro",
    "subtask": "mmlu_pro_economics",
    "doc_id": 45,
    "generation_seed": 1234,
    "response": "",
    "generation_arguments": [[
        "rendered prompt",
        {
            "until": ["Question:"],
            "max_gen_toks": 256,
            "do_sample": True,
            "temperature": 1.0,
            "top_p": 0.95,
            "seed": 1234,
        },
    ]],
}
```

Tests must assert:

```python
attempt = load_replay_attempt(path, ROW["attempt_uid"])
assert attempt.prompt == "rendered prompt"
assert attempt.prompt_sha256 == hashlib.sha256(b"rendered prompt").hexdigest()
assert attempt.generation_kwargs["max_gen_toks"] == 256

stages = postprocess_stages(
    "<mm:think>reasoning</mm:think>Question: hidden",
    think_end_token="</mm:think>",
    until=["Question:"],
)
assert stages == {
    "raw_text": "<mm:think>reasoning</mm:think>Question: hidden",
    "after_thinking": "Question: hidden",
    "after_task_stops": "",
    "thinking_marker_present": True,
    "matched_stop": "Question:",
}
```

Add rejection tests for a non-unique UID, wrong task/subtask/doc/seed, malformed
`generation_arguments`, and any original generation setting other than the
pinned r4.5 smoke values.

- [ ] **Step 2: Run the new test and verify RED**

```bash
python -m pytest -q pipeline/tests/test_m3_empty_output_replay.py
```

Expected: collection fails because `pipeline.m3_empty_output_replay` does not
exist.

- [ ] **Step 3: Implement the pure data contract**

Create these fixed definitions:

```python
REPLAY_CAPS = (256, 16384)
EXPECTED_ATTEMPT = {
    "task": "mmlu_pro",
    "subtask": "mmlu_pro_economics",
    "doc_id": 45,
    "generation_seed": 1234,
}
EXPECTED_GENERATION = {
    "until": ["Question:"],
    "max_gen_toks": 256,
    "do_sample": True,
    "temperature": 1.0,
    "top_p": 0.95,
    "seed": 1234,
}

@dataclass(frozen=True)
class ReplayAttempt:
    attempt_uid: str
    prompt: str
    prompt_sha256: str
    generation_kwargs: dict[str, Any]
    source_row: dict[str, Any]
```

`load_replay_attempt(path, attempt_uid)` reads exactly one matching JSONL row,
validates `EXPECTED_ATTEMPT`, requires `response == ""`, unwraps the sole
`[prompt, kwargs]` request, compares `kwargs` to `EXPECTED_GENERATION`, and
returns `ReplayAttempt`.

Implement `postprocess_stages(raw_text, *, think_end_token, until)` to mirror
lm-eval 0.4.12 exactly: split on the thinking marker and take the last segment,
`lstrip()`, then split sequentially on each non-empty task stop. Return the
five-field dictionary asserted above.

Implement:

```python
def run_controls(attempt: ReplayAttempt, generate) -> list[dict[str, Any]]:
    controls = []
    for cap in REPLAY_CAPS:
        completion = generate(attempt, cap)
        controls.append({
            "max_gen_toks": cap,
            **completion,
            "postprocessing": postprocess_stages(
                completion["raw_text"],
                think_end_token="</mm:think>",
                until=attempt.generation_kwargs["until"],
            ),
        })
    return controls
```

The injected `generate` callable allows CPU tests to prove cap order and report
shape without mocking vLLM internals.

- [ ] **Step 4: Run the new test and verify GREEN**

Run the Step 2 command. Expected: all replay-contract tests pass.

- [ ] **Step 5: Commit**

```bash
git add pipeline/m3_empty_output_replay.py \
  pipeline/tests/test_m3_empty_output_replay.py
git commit -m "feat(eval): add exact empty-output replay contract"
```

### Task 7: Raw vLLM replay runtime and CLI

**Files:**
- Modify: `pipeline/m3_empty_output_replay.py`
- Modify: `pipeline/tests/test_m3_empty_output_replay.py`

**Interfaces:**
- Consumes: `load_config()` and `_load_lm_model()` from the existing pipeline,
  plus pinned lm-eval v0.4.12 VLLM methods `tok_encode`, `modify_gen_kwargs`,
  `_model_generate`, and the existing `clean`/`cleanup` lifecycle.
- Produces: `classify_controls(controls: list[dict]) -> dict`,
  `build_replay_report(attempt: ReplayAttempt, *, config_path: Path,
  model_path: Path, generate: Callable, versions: dict[str, str]) -> dict`, and
  `write_replay_report(path: Path, report: dict) -> None`.
- Produces CLI: `python -m pipeline.m3_empty_output_replay --config CONFIG
  --model MODEL --samples JSONL --attempt-uid UID --out REPORT`.

- [ ] **Step 1: Write failing orchestration and CLI tests**

Use a fake `generate` callable returning:

```python
{
    "raw_text": "<mm:think>reasoning</mm:think>",
    "token_ids": [1, 2, 3],
    "token_count": 3,
    "finish_reason": "stop",
    "stop_reason": 2,
}
```

Assert `build_replay_report()` calls caps `[256, 16384]`, preserves source
identity and prompt SHA, records Python/lm-eval/vLLM versions, classifies the
processed empty stage, and writes atomically. Assert validation happens before
the injected model loader is called. Assert CLI argument parsing exposes only
the five approved inputs and has no `--min-tokens` or user-selected cap option.

Use this exact classification fixture:

```python
controls = [
    {
        "max_gen_toks": 256,
        "raw_text": "<mm:think>reasoning</mm:think>",
        "token_ids": [1, 2, 3],
        "token_count": 3,
        "finish_reason": "stop",
        "stop_reason": 2,
        "postprocessing": {
            "raw_text": "<mm:think>reasoning</mm:think>",
            "after_thinking": "",
            "after_task_stops": "",
            "thinking_marker_present": True,
            "matched_stop": None,
        },
    },
    {
        "max_gen_toks": 16384,
        "raw_text": "<mm:think>reasoning</mm:think>The answer is (C).",
        "token_ids": [1, 2, 3, 4],
        "token_count": 4,
        "finish_reason": "stop",
        "stop_reason": 2,
        "postprocessing": {
            "raw_text": "<mm:think>reasoning</mm:think>The answer is (C).",
            "after_thinking": "The answer is (C).",
            "after_task_stops": "The answer is (C).",
            "thinking_marker_present": True,
            "matched_stop": None,
        },
    },
]
assert classify_controls(controls) == {
    "kind": "thinking_only_at_smoke_cap",
    "smoke_processed_empty": True,
    "production_processed_empty": False,
}
```

- [ ] **Step 2: Run focused tests and verify RED**

```bash
python -m pytest -q pipeline/tests/test_m3_empty_output_replay.py \
  -k "report or cli or validation_before_model_load"
```

Expected: FAIL because runtime/report/CLI functions are absent.

- [ ] **Step 3: Implement the pinned-adapter raw generator**

Keep all heavy imports inside the runtime path. Resolve the distribution version
by trying `importlib.metadata.version("lm_eval")` and then `version("lm-eval")`;
require the result to equal `0.4.12`. Load the committed config, require backend
`vllm`, thinking enabled, marker `</mm:think>`, and the pinned sampling
parameters, then call `_load_lm_model(cfg, model_path)` once.

For each fixed cap, follow the installed VLLM adapter's generation path:

```python
eos = lm.tok_decode(lm.eot_token_id)
kwargs = dict(attempt.generation_kwargs)
kwargs["max_gen_toks"] = cap
normalized, until, max_gen_toks = lm.modify_gen_kwargs(
    kwargs, eos=eos, default_max_gen_toks=lm.max_gen_toks
)
token_ids = lm.tok_encode(attempt.prompt)
from lm_eval.models.utils import maybe_truncate
token_ids, max_gen_toks = maybe_truncate(
    token_ids,
    max_gen_toks=max_gen_toks,
    max_model_len=lm.max_length,
    side=lm.truncation_side,
    verbose=True,
)
stop = [value for value in until if value == eos]
from vllm import SamplingParams
params = SamplingParams(max_tokens=max_gen_toks, stop=stop, **normalized)
request = lm._model_generate(
    requests=[token_ids], generate=True, sampling_params=[params]
)[0]
completion = request.outputs[0]
```

Return only JSON-safe copies of `completion.text`, `completion.token_ids`,
`finish_reason`, and `stop_reason`. In `finally`, call `clean` or `cleanup` if
available. Do not call lm-eval task evaluation and do not write to the source
run's samples, aggregates, health summaries, manifests, or gates.

Implement classification in this order:

```python
def classify_controls(controls):
    by_cap = {control["max_gen_toks"]: control for control in controls}
    smoke = by_cap[256]
    production = by_cap[16384]
    smoke_stages = smoke["postprocessing"]
    production_stages = production["postprocessing"]
    smoke_empty = not smoke_stages["after_task_stops"].strip()
    production_empty = not production_stages["after_task_stops"].strip()
    if smoke["token_count"] == 0:
        kind = "zero_raw_tokens"
    elif smoke_empty and not production_empty and not smoke_stages["after_thinking"]:
        kind = "thinking_only_at_smoke_cap"
    elif smoke_empty and not production_empty and smoke_stages["matched_stop"]:
        kind = "task_stop_at_smoke_cap"
    elif smoke_empty and smoke.get("finish_reason") == "length" and not production_empty:
        kind = "length_cap_interaction"
    elif smoke_empty:
        kind = "processed_empty_unclassified"
    else:
        kind = "not_reproduced"
    return {
        "kind": kind,
        "smoke_processed_empty": smoke_empty,
        "production_processed_empty": production_empty,
    }
```

`build_replay_report()` returns schema version 1, attempt UID/task/subtask/doc
ID/seed, prompt SHA, checkpoint path, config SHA, fixed caps, environment
versions, controls, and `classify_controls(controls)`. `write_replay_report()`
writes through `OUT.tmp` followed by `replace()`.

- [ ] **Step 4: Verify GREEN and all CPU regressions**

```bash
python -m pytest -q \
  pipeline/tests/test_m3_empty_output_replay.py \
  pipeline/tests/test_m3_quality_eval.py \
  pipeline/tests/test_lmeval_runner.py
```

Expected: all tests pass without importing vLLM during collection.

- [ ] **Step 5: Commit**

```bash
git add pipeline/m3_empty_output_replay.py \
  pipeline/tests/test_m3_empty_output_replay.py
git commit -m "feat(eval): capture raw vLLM replay evidence"
```

### Task 8: Replace the active executor packet with the r4.7 schedule

**Files:**
- Modify: `M3_PRODUCTION_EVAL_HANDOFF.md`
- Modify: `docs/superpowers/specs/2026-07-14-m3-task-isolated-paired-quality-rerun-design.md`

**Interfaces:**
- Consumes: Tasks 5-7, existing paired/BF16 matrices, smoke gate, and cross-run
  contract gate.
- Produces: one `READY_FOR_EXECUTOR` packet with exact three-session Wave A and
  bounded Wave B commands.

- [ ] **Step 1: Update packet metadata and pre-allocation checks**

Set revision `2026-07-15-r4.7`, required ancestor to the Task 7 commit, and
maximum allocation to eight nodes. Add the replay tests to the required CPU
suite. Keep the existing r4.5 smoke promotion and BF16 cross-run contract gate.

- [ ] **Step 2: Write the seven-node Wave A commands**

The handoff must create three detached controllers outside Slurm:

```bash
# Four nodes: paired 100-sample production.
bash pipeline/slurm/run_m3_quality_eval_srun.sh \
  --profile production --matrix "$PAIR_MATRIX" --run-root "$PAIR_ROOT" \
  --smoke-gate "$PAIR_ROOT/smoke_gate.json"

# One node: exact GPTQ raw replay, one model load and two controls.
srun --exclusive --nodes=1 --ntasks=1 --gpus-per-node=8 \
  --kill-on-bad-exit=1 --time=12:00:00 \
  python -m pipeline.m3_empty_output_replay \
  --config pipeline/configs/eval_minimax_m3_reasoning_r4.yaml \
  --model /mnt/nfs/hoangduy/projects/llm-compressor/artifacts/m3-awq-gptq-prepared/gptq-checkpoint-vllm-w123-abi-overlay \
  --samples "$PAIR_ROOT/models/inhouse_gptq/shards/smoke/samples/mmlu_pro.jsonl" \
  --attempt-uid 8e98c89a40db606e115a1d388e89a58518582d44f2f48dafcf389a1e1e146878 \
  --out "$PAIR_ROOT/diagnostics/empty-output-replay.json"

# Two nodes: BF16 TP8xPP2/Ray smoke.
bash pipeline/slurm/run_m3_quality_eval_srun.sh \
  --profile smoke --matrix "$BF16_MATRIX" --run-root "$BF16_ROOT"
```

Wrap each command in its own named tmux session with stdout, stderr, and an rc
file. State and assert `4 + 1 + 2 = 7` nodes. Do not put `srun` inside an
existing allocation and do not use `sbatch`.

- [ ] **Step 3: Write the eight-node Wave B gate**

Require both replay and BF16-smoke controller rc files before Wave B. Replay rc
must be zero and its JSON must contain exactly caps `[256, 16384]`; BF16 smoke
must be ready and the cross-run contract must still be valid. Then allow BF16
production's four nodes to overlap paired production's four nodes. Explicitly
assert that the replay allocation has ended first, so the packet never reaches
nine nodes.

- [ ] **Step 4: Update evidence return and design state**

Require the replay JSON, logs, scheduler identity, raw/postprocessed stage
classification, both eval roots, all controller/arm rc files, gates, health
advisory, and reports. Empty responses remain score-zero observations; the
executor does not retry or interpret them. Set the design workflow state to
`READY_FOR_EXECUTOR` after verification.

- [ ] **Step 5: Self-review and commit**

```bash
rg -n "sbatch|six nodes|Maximum allocation|REPLAY_CAPS|min_tokens" \
  M3_PRODUCTION_EVAL_HANDOFF.md \
  docs/superpowers/specs/2026-07-14-m3-task-isolated-paired-quality-rerun-design.md
git diff --check
git add M3_PRODUCTION_EVAL_HANDOFF.md \
  docs/superpowers/specs/2026-07-14-m3-task-isolated-paired-quality-rerun-design.md
git commit -m "docs(handoff): authorize r4.7 eval and raw replay"
```

Expected: `sbatch` appears only in prohibition text; no stale six-node schedule
or runtime-selected diagnostic parameter remains.

### Task 9: Final verification and push

**Files:**
- Verify only; no planned source changes.

- [ ] **Step 1: Run the complete focused suite**

```bash
python -m pytest -q \
  pipeline/tests/test_m3_empty_output_replay.py \
  pipeline/tests/test_m3_quality_eval.py \
  pipeline/tests/test_m3_quality_eval_runner.py \
  pipeline/tests/test_m3_quality_evidence.py \
  pipeline/tests/test_static_checkpoint.py \
  pipeline/tests/test_lmeval_runner.py
```

Expected: all tests pass on Linux. On the planner's Windows host, separately
record any runner tests skipped solely because `bash` is unavailable; the
executor must run them before allocation.

- [ ] **Step 2: Verify scripts, evidence promotion, and repository state**

```bash
bash -n pipeline/slurm/run_m3_quality_eval_srun.sh
bash -n pipeline/slurm/test_m3_quality_eval_arm.sh
python -m pipeline.m3_quality_eval smoke-gate \
  --matrix pipeline/configs/minimax_m3_paired_gptq_awq_reasoning_r4.yaml \
  --report results/m3-quality/20260715T075800Z-m3-paired-reasoning-r4/smoke_report.json \
  --out /tmp/m3-r47-smoke-gate.json
git diff --check origin/duy-branch..HEAD
git status --short --branch
```

Expected: Bash syntax passes, the real smoke report promotes with its recorded
warning, diff check passes, and only intended commits are ahead.

- [ ] **Step 3: Push and confirm synchronization**

```bash
git push origin duy-branch
test "$(git rev-parse HEAD)" = "$(git rev-parse origin/duy-branch)"
```

Expected: exit 0 and identical local/remote revisions.

---

## r4.8 correction tasks

Tasks 1-9 above are completed r4.7 foundation and must not be rerun. The
following tasks implement the r4.8 correction approved on 2026-07-16 from the
partial Wave A evidence.

### Task 10: Forward the typed vLLM serving envelope

**Files:**
- Modify: `pipeline/lmeval_runner.py:26-43`
- Modify: `pipeline/tests/test_lmeval_runner.py:145-180`

**Interfaces:**
- Consumes: `ServeConfig.enable_expert_parallel`, `ServeConfig.block_size`,
  `ServeConfig.kv_cache_dtype`, and `ServeConfig.vllm_kwargs`.
- Produces: `vllm_model_args(cfg, model_path)` with one effective value for
  each runtime key. Explicit `serve.vllm_kwargs` values override typed fields.
- Preserves: the existing lm-eval/vLLM loader, MiniMax runtime preparation,
  replay CLI, and quality-arm wrapper.

- [ ] **Step 1: Add failing model-argument regressions**

Add these focused tests:

```python
def test_vllm_model_args_forwards_typed_serve_runtime_envelope():
    cfg = PipelineConfig()
    cfg.serve.enable_expert_parallel = True
    cfg.serve.block_size = 128
    cfg.serve.kv_cache_dtype = "fp8"

    args = vllm_model_args(cfg, "/models/minimax-m3")

    assert "enable_expert_parallel=True" in args
    assert "block_size=128" in args
    assert "kv_cache_dtype=fp8" in args


def test_vllm_model_args_explicit_kwargs_override_typed_envelope_once():
    cfg = PipelineConfig()
    cfg.serve.enable_expert_parallel = True
    cfg.serve.block_size = 128
    cfg.serve.kv_cache_dtype = "fp8"
    cfg.serve.vllm_kwargs = {
        "enable_expert_parallel": False,
        "block_size": 64,
        "kv_cache_dtype": "auto",
    }

    args = vllm_model_args(cfg, "/models/minimax-m3")

    assert args.count("enable_expert_parallel=") == 1
    assert args.count("block_size=") == 1
    assert args.count("kv_cache_dtype=") == 1
    assert "enable_expert_parallel=False" in args
    assert "block_size=64" in args
    assert "kv_cache_dtype=auto" in args
```

- [ ] **Step 2: Verify RED**

```powershell
python -m pytest -q pipeline/tests/test_lmeval_runner.py -k "typed_serve_runtime_envelope or explicit_kwargs_override"
```

Expected: the typed-envelope test fails because the three values are currently
omitted; the override test exposes duplicate-key behavior after the minimal
typed forwarding is introduced if precedence is not handled.

- [ ] **Step 3: Implement one merged runtime-kwargs mapping**

Immediately before constructing the vLLM argument list, add:

```python
runtime_kwargs: dict[str, object] = {}
if s.enable_expert_parallel:
    runtime_kwargs["enable_expert_parallel"] = True
if s.block_size is not None:
    runtime_kwargs["block_size"] = s.block_size
if s.kv_cache_dtype is not None:
    runtime_kwargs["kv_cache_dtype"] = s.kv_cache_dtype
runtime_kwargs.update(s.vllm_kwargs)
```

Replace iteration over `s.vllm_kwargs.items()` with
`runtime_kwargs.items()`. Do not add replay-specific model loading or copy the
quality wrapper's three overrides into the replay module.

- [ ] **Step 4: Verify the loader and replay regressions**

```powershell
python -m pytest -q pipeline/tests/test_lmeval_runner.py pipeline/tests/test_m3_empty_output_replay.py
python -m ruff check pipeline/lmeval_runner.py pipeline/tests/test_lmeval_runner.py
```

Expected: all tests and Ruff pass. The resolved replay config now produces the
same `enable_expert_parallel=True`, `block_size=128`, and `kv_cache_dtype=fp8`
arguments observed in the successful GPTQ smoke.

- [ ] **Step 5: Commit**

```bash
git add pipeline/lmeval_runner.py pipeline/tests/test_lmeval_runner.py
git commit -m "fix(eval): preserve MiniMax serving envelope"
```

### Task 11: Replace unsupported BF16 PP with qualification-only TP16

**Files:**
- Modify: `pipeline/configs/minimax_m3_bf16_reasoning_r4.yaml:9-16`
- Modify: `pipeline/tests/test_m3_quality_eval.py:50-75`
- Modify: `pipeline/tests/test_m3_quality_eval_runner.py:210-265`

**Interfaces:**
- Consumes: the existing generic `MatrixModel` topology fields and Ray-backed
  two-node launcher.
- Produces: one BF16 smoke arm with `nodes=2`, TP16, PP1, Ray, and expected
  distributed world size 16.
- Does not authorize: BF16 production. The matrix remains reusable for a later
  planner packet, but r4.8 exposes only its smoke dry-run/launch command.

- [ ] **Step 1: Change tests first to the supported topology**

Update the BF16 model contract test to require:

```python
model = load_matrix_spec(
    Path("pipeline/configs/minimax_m3_bf16_reasoning_r4.yaml")
).models[0]
assert model.nodes == 2
assert model.tensor_parallel_size == 16
assert model.pipeline_parallel_size == 1
assert model.distributed_executor_backend == "ray"
```

Update the BF16 smoke dry-run test to assert exactly one arm with:

```python
assert arm["profile"] == "smoke"
assert arm["nodes"] == 2
assert arm["tensor_parallel_size"] == 16
assert arm["pipeline_parallel_size"] == 1
assert arm["distributed_executor_backend"] == "ray"
```

Keep the existing production-plan parser tests topology-consistent, but do not
add a production command to the r4.8 handoff.

- [ ] **Step 2: Verify RED**

```powershell
python -m pytest -q pipeline/tests/test_m3_quality_eval.py pipeline/tests/test_m3_quality_eval_runner.py -k "bf16"
```

Expected: failures show the committed matrix still resolves TP8xPP2.

- [ ] **Step 3: Make the minimal matrix change**

Change only the BF16 model topology:

```yaml
    nodes: 2
    tensor_parallel_size: 16
    pipeline_parallel_size: 1
    distributed_executor_backend: ray
```

Do not change the checkpoint, tasks, aliases, sample counts, seeds, generation
parameters, gates, or disabled distributional probe.

- [ ] **Step 4: Verify topology and dry-run regressions**

```powershell
$env:PATH='C:\Program Files\Git\bin;' + $env:PATH
python -m pytest -q pipeline/tests/test_m3_quality_eval.py pipeline/tests/test_m3_quality_eval_runner.py pipeline/tests/test_m3_quality_evidence.py
```

Expected on the planner host: all tests pass with Git Bash available. The dry
run contains TP16xPP1/Ray and no `sbatch` command.

- [ ] **Step 5: Commit**

```bash
git add pipeline/configs/minimax_m3_bf16_reasoning_r4.yaml \
  pipeline/tests/test_m3_quality_eval.py \
  pipeline/tests/test_m3_quality_eval_runner.py
git commit -m "fix(eval): qualify BF16 with TP16 Ray"
```

### Task 12: Preserve multi-node placement, worker, and GPU evidence

**Files:**
- Modify: `pipeline/lmeval_runner.py:118-132`
- Modify: `pipeline/slurm/test_m3_quality_eval_arm.sh:18-55`
- Modify: `pipeline/tests/test_lmeval_runner.py`
- Modify: `pipeline/tests/test_m3_quality_eval_runner.py`

**Interfaces:**
- Produces for every multi-node arm:
  - `ray_runtime/placement-monitor.log`;
  - `ray_runtime/gpu-monitor-rank-<rank>.log`;
  - `ray_runtime/ray-logs-rank-<rank>.tar.gz`, or a same-name `.missing`
    marker when Ray creates no session logs.
- Produces: `models/bf16/shards/smoke/model-ready.json` only after the backend
  constructor returns, plus `placement-timeout.json` or
  `model-init-timeout.json` when the corresponding opt-in watchdog terminates
  a stalled launch.
- Preserves: the existing rank lifecycle, `driver-done` synchronization, Ray
  topology gate, return codes, and cleanup on success/failure. The watchdog is
  disabled unless `M3_PLACEMENT_TIMEOUT_SECONDS` or
  `M3_MODEL_INIT_TIMEOUT_SECONDS` is a positive integer.

- [ ] **Step 1: Add failing shell-contract assertions**

Extend the runner script contract test:

```python
arm_script = Path("pipeline/slurm/test_m3_quality_eval_arm.sh").read_text()
assert "gpu-monitor-rank-$rank.log" in arm_script
assert "nvidia-smi --query-gpu=" in arm_script
assert "ray-logs-rank-$rank.tar.gz" in arm_script
assert "session_latest/logs" in arm_script
assert "placement-monitor.log" in arm_script
assert "M3_PLACEMENT_TIMEOUT_SECONDS" in arm_script
assert "placement-timeout.json" in arm_script
assert "M3_MODEL_INIT_TIMEOUT_SECONDS" in arm_script
assert "model-init-timeout.json" in arm_script
```

Import `_load_lm_model` in `test_lmeval_runner.py` and add:

```python
def test_load_lm_model_writes_ready_marker_only_after_constructor(
    monkeypatch, tmp_path
):
    ready = tmp_path / "model-ready.json"
    constructor_observations = []

    class FakeModel:
        @classmethod
        def create_from_arg_string(cls, *_args, **_kwargs):
            constructor_observations.append(ready.exists())
            return object()

    monkeypatch.setenv("M3_MODEL_READY_FILE", str(ready))
    monkeypatch.setattr(
        "pipeline.lmeval_runner._prepare_vllm_runtime", lambda *_: {}
    )
    monkeypatch.setattr("lm_eval.api.registry.get_model", lambda _: FakeModel)

    _load_lm_model(PipelineConfig(), "/models/minimax-m3")

    assert constructor_observations == [False]
    assert json.loads(ready.read_text())["status"] == "ready"
```

- [ ] **Step 2: Verify RED**

```powershell
python -m pytest -q pipeline/tests/test_m3_quality_eval_runner.py -k "runner_scripts_are_valid_bash"
```

Expected: fail because GPU snapshots, Ray-log archival, model-ready evidence,
and the opt-in initialization watchdog are absent.

- [ ] **Step 3: Add bounded evidence monitors**

After the Ray topology helper returns on each rank, start a background monitor
whose output is unique per rank:

```bash
gpu_monitor_pid=""
(
  while [[ ! -f "$ARM/ray_runtime/driver-done" ]]; do
    echo "timestamp=$(date -Is)"
    nvidia-smi --query-gpu=index,uuid,memory.used,memory.total,utilization.gpu \
      --format=csv,noheader,nounits || true
    sleep 60
  done
) >"$ARM/ray_runtime/gpu-monitor-rank-$rank.log" 2>&1 &
gpu_monitor_pid=$!
```

Extend `cleanup_ray()` to stop/wait for this monitor, then preserve local Ray
worker logs before `ray stop --force`:

```bash
if [[ -n "$gpu_monitor_pid" ]]; then
  kill "$gpu_monitor_pid" 2>/dev/null || true
  wait "$gpu_monitor_pid" 2>/dev/null || true
fi
ray_logs="/tmp/ray/session_latest/logs"
archive="$ARM/ray_runtime/ray-logs-rank-$rank.tar.gz"
if [[ -d "$ray_logs" ]]; then
  tar -czf "$archive" -C "$ray_logs" . || touch "$archive.missing"
else
  touch "$archive.missing"
fi
```

Initialize both monitor PID variables before defining cleanup. All diagnostic
commands are best-effort; they must not replace the arm's scientific return
code or prevent Ray cleanup.

After `create_from_arg_string()` returns successfully, `_load_lm_model()` must
atomically write the optional marker named by `M3_MODEL_READY_FILE`:

```python
lm = model_cls.create_from_arg_string(
    model_args(cfg, model_path),
    {"batch_size": batch_size},
)
ready_file = os.environ.get("M3_MODEL_READY_FILE")
if ready_file:
    ready = Path(ready_file)
    ready.parent.mkdir(parents=True, exist_ok=True)
    temp = ready.with_suffix(ready.suffix + ".tmp")
    temp.write_text(json.dumps({"status": "ready"}) + "\n")
    temp.replace(ready)
return lm
```

Add the required standard-library imports inside the function or module. In the
arm script, export the ready-marker path and clear stale markers before launch:

```bash
export M3_MODEL_READY_FILE="$ARM/model-ready.json"
rm -f "$M3_MODEL_READY_FILE" \
  "$ARM/placement-timeout.json" \
  "$ARM/model-init-timeout.json"
```

Start the eval command in the background. When
`M3_PLACEMENT_TIMEOUT_SECONDS` is positive, capture placement state at the
deadline and terminate only if no placement group reached `CREATED`:

```bash
placement_timeout=${M3_PLACEMENT_TIMEOUT_SECONDS:-0}
"${eval_cmd[@]}" &
eval_pid=$!
placement_watchdog_pid=""
if ((placement_timeout > 0)); then
  (
    sleep "$placement_timeout"
    if kill -0 "$eval_pid" 2>/dev/null; then
      placement_deadline="$ARM/ray_runtime/placement-at-deadline.log"
      ray list placement-groups --detail >"$placement_deadline" 2>&1 || true
      if ! grep -q "CREATED" "$placement_deadline"; then
        printf '{"status":"stalled","timeout_seconds":%s}\n' \
          "$placement_timeout" >"$ARM/placement-timeout.json"
        kill -TERM "$eval_pid" 2>/dev/null || true
      fi
    fi
  ) &
  placement_watchdog_pid=$!
fi
```

Independently, only when the model-init timeout is positive, start this
constructor watchdog:

```bash
init_timeout=${M3_MODEL_INIT_TIMEOUT_SECONDS:-0}
init_watchdog_pid=""
if ((init_timeout > 0)); then
  (
    sleep "$init_timeout"
    if kill -0 "$eval_pid" 2>/dev/null \
       && [[ ! -f "$M3_MODEL_READY_FILE" ]]; then
      printf '{"status":"stalled","timeout_seconds":%s}\n' \
        "$init_timeout" >"$ARM/model-init-timeout.json"
      kill -TERM "$eval_pid" 2>/dev/null || true
    fi
  ) &
  init_watchdog_pid=$!
fi
wait "$eval_pid"
rc=$?
if [[ -n "$placement_watchdog_pid" ]]; then
  kill "$placement_watchdog_pid" 2>/dev/null || true
  wait "$placement_watchdog_pid" 2>/dev/null || true
fi
if [[ -n "$init_watchdog_pid" ]]; then
  kill "$init_watchdog_pid" 2>/dev/null || true
  wait "$init_watchdog_pid" 2>/dev/null || true
fi
```

Use this path only where the current script directly executes `eval_cmd`; keep
the existing probe ordering and return-code behavior unchanged.

- [ ] **Step 4: Verify shell and runner behavior**

```powershell
& 'C:\Program Files\Git\bin\bash.exe' -n pipeline/slurm/test_m3_quality_eval_arm.sh
$env:PATH='C:\Program Files\Git\bin;' + $env:PATH
python -m pytest -q pipeline/tests/test_lmeval_runner.py pipeline/tests/test_m3_quality_eval_runner.py
```

Expected: Bash syntax and all runner tests pass.

- [ ] **Step 5: Commit**

```bash
git add pipeline/slurm/test_m3_quality_eval_arm.sh \
  pipeline/lmeval_runner.py \
  pipeline/tests/test_lmeval_runner.py \
  pipeline/tests/test_m3_quality_eval_runner.py
git commit -m "feat(eval): retain multi-node startup evidence"
```

### Task 13: Replace r4.7 with the bounded r4.8 executor packet

**Files:**
- Modify: `M3_PRODUCTION_EVAL_HANDOFF.md`
- Modify: `docs/superpowers/specs/2026-07-14-m3-task-isolated-paired-quality-rerun-design.md`

**Interfaces:**
- Consumes: Tasks 10-12, the immutable r4.7 failure evidence, the active paired
  run root, and the existing BF16 preflight/contract machinery.
- Produces: one canonical `READY_FOR_EXECUTOR` r4.8 correction packet.
- Authorizes only: repaired exact GPTQ replay and BF16 TP16 qualification.
- Does not authorize: paired-production relaunch, BF16 production, topology
  fallback, automatic retry, or local vLLM model patches.

- [ ] **Step 1: Replace packet metadata and scope**

Set packet revision `2026-07-16-r4.8`, state `READY_FOR_EXECUTOR`, and base Git
commit to the reviewed Task 12 tip. Mark the r4.7 launch instructions in the
same file `SUPERSEDED`; retain the factual note that paired production already
runs independently on four nodes.

The pre-allocation suite is:

```bash
python -m pytest -q \
  pipeline/tests/test_m3_empty_output_replay.py \
  pipeline/tests/test_lmeval_runner.py \
  pipeline/tests/test_m3_quality_eval.py \
  pipeline/tests/test_m3_quality_eval_runner.py \
  pipeline/tests/test_m3_quality_evidence.py
```

- [ ] **Step 2: Use fresh correction artifact names and BF16 root**

Keep the paired root immutable and define:

```bash
PAIR_ROOT=results/m3-quality/20260715T075800Z-m3-paired-reasoning-r4
BF16_RUN_ID="$(date -u +%Y%m%dT%H%M%SZ)-m3-bf16-tp16-qualification-r48"
BF16_ROOT="results/m3-quality/$BF16_RUN_ID"
REPLAY_JSON="$PAIR_ROOT/diagnostics/empty-output-replay-r48.json"
REPLAY_OUT="$PAIR_ROOT/logs/empty-output-replay-r48.out"
REPLAY_ERR="$PAIR_ROOT/logs/empty-output-replay-r48.err"
REPLAY_RC="$PAIR_ROOT/diagnostics/empty-output-replay-r48.rc"
```

Reject any collision with the r4.7 replay stdout, stderr, or rc. Run the
existing BF16 CPU preflight and cross-run contract gate into the fresh root,
then require a smoke dry-run showing one two-node TP16xPP1/Ray arm.

- [ ] **Step 3: Launch exactly two detached correction controllers**

The replay controller uses the unchanged exact UID and fixed caps:

```bash
srun --exclusive --nodes=1 --ntasks=1 --gpus-per-node=8 \
  --kill-on-bad-exit=1 --time=12:00:00 \
  python -m pipeline.m3_empty_output_replay \
  --config pipeline/configs/eval_minimax_m3_reasoning_r4.yaml \
  --model /mnt/nfs/hoangduy/projects/llm-compressor/artifacts/m3-awq-gptq-prepared/gptq-checkpoint-vllm-w123-abi-overlay \
  --samples "$PAIR_ROOT/models/inhouse_gptq/shards/smoke/samples/mmlu_pro.jsonl" \
  --attempt-uid 8e98c89a40db606e115a1d388e89a58518582d44f2f48dafcf389a1e1e146878 \
  --out "$REPLAY_JSON"
```

The BF16 controller uses only the smoke profile and a fresh root:

```bash
M3_PLACEMENT_TIMEOUT_SECONDS=900 \
M3_MODEL_INIT_TIMEOUT_SECONDS=10800 \
bash pipeline/slurm/run_m3_quality_eval_srun.sh \
  --profile smoke \
  --matrix pipeline/configs/minimax_m3_bf16_reasoning_r4.yaml \
  --run-root "$BF16_ROOT"
```

Wrap each in a distinct detached `tmux` session outside Slurm, with the fresh
stdout/stderr/rc paths above. State that the two correction allocations use
three nodes; if paired production is still active, packet-owned concurrency is
`4 + 1 + 2 = 7`. Use no `sbatch` and no nested allocation.

- [ ] **Step 4: Write exact stop and return rules**

Require no automatic retry. Return:

- replay JSON with caps `[256, 16384]`, raw/effective arguments, token IDs,
  finish/stop reasons, postprocessing stages, and classification, or the first
  model-load failure with the effective vLLM arguments;
- BF16 topology gate, placement monitor, both rank GPU monitors, both Ray-log
  archives/markers, placement-at-deadline evidence when reached, timeout
  markers, arm/controller rc, smoke report/gate, scheduler identity, and
  first/last load markers;
- the full fresh BF16 root and immutable r4.7 partial reports;
- a factual boundary classification: placement, worker start, weight load,
  collective communication, request generation, or evaluation.

State that absence of a `CREATED` placement group at 15 minutes is a placement
stall enforced by the 900-second watchdog. Absence of model-ready progress at
three hours is a stalled-load finding enforced by the 10,800-second watchdog,
while the top-level allocation remains bounded at 12 hours. There is no BF16
production command anywhere in the active packet.

- [ ] **Step 5: Self-review and commit**

```bash
rg -n "2026-07-16-r4.8|SUPERSEDED|TP16xPP1|empty-output-replay-r48|12:00:00|three hours|BF16 production" \
  M3_PRODUCTION_EVAL_HANDOFF.md \
  docs/superpowers/specs/2026-07-14-m3-task-isolated-paired-quality-rerun-design.md
rg -n "sbatch|TP8xPP2|--profile production" M3_PRODUCTION_EVAL_HANDOFF.md
git diff --check
```

Expected: `sbatch`, TP8xPP2, and production appear only in prohibition or
superseded/historical prose; there is no executable production command.

```bash
git add M3_PRODUCTION_EVAL_HANDOFF.md \
  docs/superpowers/specs/2026-07-14-m3-task-isolated-paired-quality-rerun-design.md
git commit -m "docs(handoff): authorize r4.8 recovery qualification"
```

### Task 14: Final verification and synchronization

**Files:**
- Verify only; no planned source changes.

- [ ] **Step 1: Run the complete focused suite**

```powershell
$env:PATH='C:\Program Files\Git\bin;' + $env:PATH
python -m pytest -q pipeline/tests/test_m3_empty_output_replay.py pipeline/tests/test_lmeval_runner.py pipeline/tests/test_m3_quality_eval.py pipeline/tests/test_m3_quality_eval_runner.py pipeline/tests/test_m3_quality_evidence.py pipeline/tests/test_static_checkpoint.py
```

Expected: all tests pass. Any symlink test skipped solely for Windows privilege
is recorded; all non-platform tests must pass.

- [ ] **Step 2: Verify shell, packet, and repository state**

```powershell
& 'C:\Program Files\Git\bin\bash.exe' -n pipeline/slurm/run_m3_quality_eval_srun.sh
& 'C:\Program Files\Git\bin\bash.exe' -n pipeline/slurm/test_m3_quality_eval_arm.sh
python -m ruff check pipeline/lmeval_runner.py pipeline/tests/test_lmeval_runner.py pipeline/tests/test_m3_quality_eval.py pipeline/tests/test_m3_quality_eval_runner.py
git diff --check origin/duy-branch..HEAD
git status --short --branch
```

Expected: Bash, Ruff, and diff checks pass; only the user's untracked
`AGENTS.md` remains outside the committed task files.

- [ ] **Step 3: Push and verify exact synchronization**

```bash
git push origin duy-branch
test "$(git rev-parse HEAD)" = "$(git rev-parse origin/duy-branch)"
```

Expected: exit 0 with identical revisions and r4.8 ready for the executor.
