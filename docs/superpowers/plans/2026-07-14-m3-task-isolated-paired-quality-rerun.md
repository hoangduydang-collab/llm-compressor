# MiniMax-M3 Generated Reasoning Quality Rerun Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the obsolete likelihood/greedy MiniMax-M3 quick comparison with a reproducible, three-seed generated-reasoning evaluation of GPTQ versus AWQ.

**Architecture:** Keep lm-eval and its vLLM backend as the task and execution engine. Reuse its stock generated-reasoning tasks for GPQA, MMLU-Pro, GSM8K, and AIME, then extend the local runner, checkpoint normalization, comparison statistics, preflight, and `srun` launcher to understand repeated paired generation seeds. Preserve historical configs and results by adding explicit r4 configs and revising the existing task-specific executor packet.

**Tech Stack:** Python 3.10+, lm-eval 0.4.12, vLLM 0.11+, PyYAML, pytest, Bash, Slurm `srun`

## Global Constraints

- Use exactly 100 unique questions for GPQA Diamond, MMLU-Pro, and GSM8K; use all 30 AIME 2025 questions.
- Use generation seeds `42`, `1234`, and `4158` for both models.
- Use `temperature=1.0`, `top_p=0.95`, `do_sample=true`, explicit thinking, and `max_gen_toks=16384`.
- Use stock lm-eval tasks `gpqa_diamond_cot_zeroshot`, `mmlu_pro`, `gsm8k_cot`, and `aime25`.
- GPQA must resolve to output type `generate_until`, zero shots, task version `2.2`, and metric `exact_match,flexible-extract`.
- Pin and fail closed on lm-eval `0.4.12`, resolved task versions, output type, prompt/extractor configuration, tokenizer, and chat template.
- Do not reuse old reasoning checkpoints, rerun IFEval, or run distributional probes.
- Production uses four independent one-node 8xH100 arms, at most four concurrent nodes, and a `24:00:00` limit per arm.
- Use top-level `srun` only. Never emit `sbatch` or start a nested allocation from an arm.
- Preserve full raw responses and distinguish `sample_uid` (question) from `attempt_uid` (`sample_uid` plus generation seed).
- Bootstrap by question UID while retaining all three paired seed outcomes.
- Keep historical configs and evidence immutable; add r4-specific configs and revise the existing active handoff.

---

## File structure

**Create**

- `pipeline/configs/eval_minimax_m3_reasoning_r4.yaml` — exact generated-reasoning task and sampling contract.
- `pipeline/configs/minimax_m3_paired_gptq_awq_reasoning_r4.yaml` — two-model, two-shard, four-arm resource matrix.

**Modify**

- `pipeline/config.py` — represent repeated generation seeds.
- `pipeline/lmeval_runner.py` — reuse one loaded model while evaluating each task/seed pair.
- `pipeline/evalsuite/static.py` — normalize and checkpoint repeated attempts without losing raw evidence.
- `pipeline/evalsuite/compare.py` — question-grouped paired reasoning statistics.
- `pipeline/evalsuite/health.py` — expose parse, truncation, empty, and degeneration counts used by the verdict.
- `pipeline/m3_quality_eval.py` — r4 matrix validation, attempt-aware merge, optional probes, health-aware gates, and four-arm plan.
- `pipeline/m3_quality_preflight.py` — fail-closed task/harness contract and representative rendered prompts.
- `pipeline/slurm/test_m3_quality_eval_arm.sh` — record the r4 harness contract and run the repeated-seed evaluator.
- `pipeline/slurm/run_m3_quality_eval_srun.sh` — dry-run and launch exactly the matrix-produced top-level `srun` arms.
- `pipeline/requirements.txt` — pin lm-eval to the validated version.
- Existing focused tests under `pipeline/tests/` — contract, resume, statistics, preflight, merge, gate, and launcher coverage.
- `M3_PAIRED_GPTQ_AWQ_TASK_ISOLATED_HANDOFF.md` — mark prior packets historical and publish the r4 copy-ready executor packet.

lm-eval owns datasets, task loading, prompts, choice preprocessing, few-shot construction, inference, extraction, and metric aggregation. Local code records and verifies the resolved contract.

---

### Task 1: Add explicit r4 configuration contracts

**Files:**

- Create: `pipeline/configs/eval_minimax_m3_reasoning_r4.yaml`
- Create: `pipeline/configs/minimax_m3_paired_gptq_awq_reasoning_r4.yaml`
- Modify: `pipeline/config.py`
- Modify: `pipeline/requirements.txt`
- Test: `pipeline/tests/test_lmeval_runner.py`
- Test: `pipeline/tests/test_m3_quality_eval.py`

**Interfaces:**

- Produces: `EvalConfig.generation_seeds: list[int]`, defaulting to an empty list for legacy configs.
- Produces: an r4 matrix with shards `gpqa` and `reasoning_suite` and `probe.enabled: false`.
- Consumes later: `cfg.eval.generation_seeds`, r4 aliases, sampling settings, and scheduling settings.

- [ ] **Step 1: Write failing configuration tests**

```python
def test_eval_config_parses_generation_seeds(tmp_path):
    cfg = load_config(Path("pipeline/configs/eval_minimax_m3_reasoning_r4.yaml"))
    assert cfg.eval.generation_seeds == [42, 1234, 4158]
    assert cfg.eval.enable_thinking is True
    assert cfg.eval.gen_kwargs == {
        "temperature": 1.0,
        "top_p": 0.95,
        "do_sample": True,
        "max_gen_toks": 16384,
    }


def test_r4_matrix_has_two_shards_four_production_arms_and_no_probe():
    spec = load_matrix(
        "pipeline/configs/minimax_m3_paired_gptq_awq_reasoning_r4.yaml"
    )
    assert [(s.name, s.tasks) for s in spec.shards] == [
        ("gpqa", ("gpqa_diamond",)),
        ("reasoning_suite", ("mmlu_pro", "gsm8k", "aime_2025")),
    ]
    assert spec.probe.enabled is False
    assert spec.scheduling.max_parallel_arms == 4
    assert spec.scheduling.arm_time_limit == "24:00:00"
    assert len(spec.expected_arms) == 4
```

- [ ] **Step 2: Run tests and verify the new fields/files are absent**

Run:

```bash
pytest -q pipeline/tests/test_lmeval_runner.py pipeline/tests/test_m3_quality_eval.py -k "generation_seeds or r4_matrix"
```

Expected: failure because `EvalConfig.generation_seeds`, `ProbeSpec.enabled`, and both r4 YAML files do not exist.

- [ ] **Step 3: Add the minimal configuration model and r4 YAMLs**

Add to `EvalConfig`:

```python
generation_seeds: list[int] = field(default_factory=list)
```

Pin the dependency:

```text
lm-eval==0.4.12
```

The r4 eval task entries must be:

```yaml
eval:
  backend: vllm
  apply_chat_template: true
  fewshot_as_multiturn: true
  enable_thinking: true
  think_end_token: "</mm:think>"
  generation_seeds: [42, 1234, 4158]
  gen_kwargs:
    temperature: 1.0
    top_p: 0.95
    do_sample: true
    max_gen_toks: 16384
  tasks:
    - {name: gpqa_diamond, metric: "exact_match,flexible-extract", num_fewshot: 0, limit: null}
    - {name: mmlu_pro, metric: "exact_match,custom-extract", num_fewshot: 5, limit: null}
    - {name: gsm8k, metric: "exact_match,strict-match", num_fewshot: 8, limit: null}
    - {name: aime_2025, metric: "exact_match,none", num_fewshot: 0, limit: null}
```

The r4 matrix aliases and scheduling must be:

```yaml
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
  max_parallel_arms: 4
  arm_time_limit: "24:00:00"
probe:
  enabled: false
  total_tokens: 8192
  top_k: 20
  max_overhead_seconds: 900
```

Keep the existing AWQ and GPTQ checkpoint paths unchanged.

- [ ] **Step 4: Run focused tests**

Run:

```bash
pytest -q pipeline/tests/test_lmeval_runner.py pipeline/tests/test_m3_quality_eval.py -k "generation_seeds or r4_matrix"
```

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add pipeline/config.py pipeline/requirements.txt pipeline/configs/eval_minimax_m3_reasoning_r4.yaml pipeline/configs/minimax_m3_paired_gptq_awq_reasoning_r4.yaml pipeline/tests/test_lmeval_runner.py pipeline/tests/test_m3_quality_eval.py
git commit -m "feat(eval): configure M3 generated reasoning rerun"
```

---

### Task 2: Run and checkpoint each task under three paired seeds

**Files:**

- Modify: `pipeline/lmeval_runner.py`
- Modify: `pipeline/evalsuite/static.py`
- Test: `pipeline/tests/test_lmeval_runner.py`
- Test: `pipeline/tests/test_static_checkpoint.py`

**Interfaces:**

- Produces: `evaluate_tasks(..., completed_task_seeds: set[tuple[str, int]] | None = None, on_task_complete: Callable[[EvalTask, int | None, dict], None] | None = None) -> dict`.
- Produces: attempt rows containing `sample_uid`, `attempt_uid`, `generation_seed`, raw `response`, extracted answer, source document, generation arguments, correctness, and health.
- Produces: `seed_progress.json` and aggregate metrics named `pass_at_1_seed_<seed>` and `mean_pass_at_1`.
- Consumes: `EvalConfig.generation_seeds` from Task 1.

- [ ] **Step 1: Write failing runner tests**

```python
def test_evaluate_tasks_reuses_model_and_runs_each_task_seed(monkeypatch):
    cfg = PipelineConfig()
    cfg.eval.generation_seeds = [42, 1234, 4158]
    cfg.eval.gen_kwargs = {"temperature": 1.0, "top_p": 0.95}
    tasks = [EvalTask("gpqa"), EvalTask("aime")]
    calls = []
    completed = []
    model = SimpleNamespace(clean=lambda: None)
    monkeypatch.setattr("pipeline.lmeval_runner._load_lm_model", lambda *_: model)
    monkeypatch.setattr(
        lm_eval,
        "simple_evaluate",
        lambda **kwargs: calls.append(kwargs) or {
            "results": {kwargs["tasks"][0]: {"acc,none": 1.0}},
            "samples": {},
        },
    )

    evaluate_tasks(
        "/model",
        cfg,
        tasks,
        on_task_complete=lambda task, seed, batch: completed.append((task.name, seed)),
    )

    assert [(c["tasks"][0], c["gen_kwargs"]["seed"]) for c in calls] == [
        ("gpqa", 42), ("gpqa", 1234), ("gpqa", 4158),
        ("aime", 42), ("aime", 1234), ("aime", 4158),
    ]
    assert completed == [(c["tasks"][0], c["gen_kwargs"]["seed"]) for c in calls]
```

Also add a test where `completed_task_seeds={('gpqa', 42)}` and assert only the other five calls occur.

- [ ] **Step 2: Write failing checkpoint tests**

```python
def test_repeated_checkpoint_preserves_question_and_attempt_identity(tmp_path):
    aggregate = {}
    task = EvalTask("gpqa", metric="exact_match,flexible-extract")
    for seed, correct in ((42, 1), (1234, 0), (4158, 1)):
        batch = {
            "results": {"gpqa": {"exact_match,flexible-extract": correct}},
            "samples": {"gpqa": [{
                "doc_id": 7,
                "doc": {"Question": "Q", "answer": "A"},
                "arguments": [["Q\nAnswer:", {"max_gen_toks": 16384}]],
                "resps": [["Answer: A"]],
                "filtered_resps": ["A"],
                "exact_match,flexible-extract": correct,
            }]},
        }
        checkpoint_task_result(
            task=task,
            generation_seed=seed,
            batch=batch,
            aggregate=aggregate,
            aggregate_path=tmp_path / "aggregate.json",
            samples_dir=tmp_path / "samples",
            progress_path=tmp_path / "seed_progress.json",
            log_samples=True,
        )

    rows = [
        json.loads(line)
        for line in (tmp_path / "samples/gpqa.jsonl").read_text().splitlines()
    ]
    assert len({row["sample_uid"] for row in rows}) == 1
    assert len({row["attempt_uid"] for row in rows}) == 3
    assert {row["generation_seed"] for row in rows} == {42, 1234, 4158}
    assert aggregate["gpqa"]["mean_pass_at_1"] == pytest.approx(2 / 3)
```

Add tests that an identical attempt collapses, a conflicting `(sample_uid, seed)` row fails, and a progress file with an unexpected seed fails closed.

- [ ] **Step 3: Run focused tests and confirm failures**

Run:

```bash
pytest -q pipeline/tests/test_lmeval_runner.py pipeline/tests/test_static_checkpoint.py -k "task_seed or repeated_checkpoint or attempt_identity or progress"
```

Expected: failure because callbacks have no seed and checkpoint rows have no attempt identity.

- [ ] **Step 4: Implement repeated evaluation while preserving legacy behavior**

Use this seed selection in `evaluate_tasks`:

```python
seeds: tuple[int | None, ...] = (
    tuple(int(seed) for seed in ev.generation_seeds)
    if ev.generation_seeds
    else (None,)
)
for task in tasks:
    for generation_seed in seeds:
        if generation_seed is not None and (
            task.name,
            generation_seed,
        ) in completed_task_seeds:
            continue
        gen_kwargs = dict(ev.gen_kwargs or {})
        if generation_seed is not None:
            gen_kwargs["seed"] = generation_seed
        batch = lm_eval.simple_evaluate(
            model=lm,
            tasks=[task.name],
            task_manager=task_manager,
            num_fewshot=task.num_fewshot,
            apply_chat_template=ev.apply_chat_template,
            fewshot_as_multiturn=ev.fewshot_as_multiturn,
            gen_kwargs=gen_kwargs,
            samples=sample_map,
            log_samples=log_samples,
            random_seed=42,
            numpy_random_seed=42,
            torch_random_seed=42,
            fewshot_random_seed=42,
        )
        if on_task_complete is not None:
            on_task_complete(task, generation_seed, batch)
```

Create one `TaskManager` before the loops and reuse it with the already loaded model. Omit keyword arguments that are inapplicable (`samples`, `fewshot_as_multiturn`, or `gen_kwargs`) using the current conditional style.

- [ ] **Step 5: Implement attempt-aware normalization and atomic progress**

Derive attempt identity without changing question identity:

```python
def stable_attempt_uid(sample_uid: str, generation_seed: int) -> str:
    payload = f"{sample_uid}\0{generation_seed}".encode("utf-8")
    return hashlib.sha256(payload).hexdigest()
```

For repeated runs, `_extract_sample_row` must add:

```python
row.update(
    generation_seed=generation_seed,
    attempt_uid=stable_attempt_uid(row["sample_uid"], generation_seed),
    source_doc=sample.get("doc"),
    generation_arguments=sample.get("arguments"),
    extracted_answer=_first_filtered_response(sample),
)
```

Deduplicate by `attempt_uid` when present and by `sample_uid` for legacy rows. Merge new seed rows with the existing task JSONL atomically. Write `seed_progress.json` only after the sample file, health summary, and aggregate are durable. Compute every seed score from its stored binary rows and `mean_pass_at_1` from all stored attempts; never vote across seeds.

- [ ] **Step 6: Run focused tests**

Run:

```bash
pytest -q pipeline/tests/test_lmeval_runner.py pipeline/tests/test_static_checkpoint.py
```

Expected: PASS, including all legacy single-generation tests.

- [ ] **Step 7: Commit**

```bash
git add pipeline/lmeval_runner.py pipeline/evalsuite/static.py pipeline/tests/test_lmeval_runner.py pipeline/tests/test_static_checkpoint.py
git commit -m "feat(eval): checkpoint paired reasoning generations"
```

---

### Task 3: Add grouped repeated-measure statistics and health comparison

**Files:**

- Modify: `pipeline/evalsuite/compare.py`
- Modify: `pipeline/evalsuite/health.py`
- Test: `pipeline/tests/test_compare.py`
- Test: `pipeline/tests/test_eval_health.py`

**Interfaces:**

- Produces: `_pair_repeated_binary(rows_a, rows_b, *, seed, iterations) -> dict`.
- Produces: per-seed pass@1, aggregate pass@1, paired delta, question-grouped bootstrap interval, and question win/tie/loss.
- Produces: health summaries with parser failure, length-cap hit, empty output, periodic loop, and token-count distributions.
- Consumes: repeated attempt rows from Task 2.

- [ ] **Step 1: Write failing grouped-statistics tests**

```python
def repeated_rows(values_by_question):
    seeds = (42, 1234, 4158)
    return [
        {
            "sample_uid": question,
            "attempt_uid": f"{question}-{seed}",
            "generation_seed": seed,
            "correct": correct,
        }
        for question, values in values_by_question.items()
        for seed, correct in zip(seeds, values, strict=True)
    ]


def test_repeated_pairing_bootstraps_questions_not_attempts():
    awq = repeated_rows({"q1": [1, 1, 0], "q2": [0, 0, 0]})
    gptq = repeated_rows({"q1": [1, 0, 0], "q2": [1, 0, 0]})
    result = _pair_repeated_binary(awq, gptq, seed=9, iterations=200)

    assert result["n_questions"] == 2
    assert result["n_paired"] == 6
    assert result["acc_a"] == pytest.approx(2 / 6)
    assert result["acc_b"] == pytest.approx(2 / 6)
    assert result["per_seed"]["42"]["acc_a"] == 0.5
    assert result["question_wins"] == 1
    assert result["question_ties"] == 0
    assert result["question_losses"] == 1
    assert result["bootstrap"]["accuracy_delta"]["resampling_unit"] == "question"
```

Add fail-closed cases for missing seeds, duplicate attempt UIDs, mismatched seed sets, and repeated rows on only one side.

- [ ] **Step 2: Run the test and verify failure**

Run:

```bash
pytest -q pipeline/tests/test_compare.py -k repeated
```

Expected: failure because `_pair_repeated_binary` does not exist.

- [ ] **Step 3: Implement question-grouped pairing**

Group rows as `dict[sample_uid, dict[generation_seed, row]]`, require identical question and seed grids, and compute:

```python
question_means_a = [mean(correctness for each seed) for each question]
question_means_b = [mean(correctness for each seed) for each question]
bootstrap = paired_bootstrap(
    question_means_a,
    question_means_b,
    statistic=_delta_mean,
    seed=seed,
    iterations=iterations,
)
bootstrap["resampling_unit"] = "question"
```

Keep attempt-level flip and conditional-regression counts for compatibility with existing gates, but set `inference_unit: "question"` and do not report an attempt-level McNemar p-value as inferential evidence. Dispatch to this function in `_compare_task` whenever either side contains `generation_seed`; reject mixed repeated/legacy inputs.

- [ ] **Step 4: Extend health tests and summaries**

```python
def test_health_summary_reports_reasoning_failure_modes():
    rows = [
        {"health": {"applicable": True, "answer_extraction_failed": True, "token_count": 4}},
        {"health": {"applicable": True, "length_cap_hit": True, "token_count": 16384}},
        {"health": {"applicable": True, "empty": True, "token_count": 0}},
        {"health": {"applicable": True, "periodic_loop": True, "token_count": 32}},
    ]
    summary = summarize_generation_health(rows)
    assert summary["answer_extraction_failure_count"] == 1
    assert summary["length_cap_hit_count"] == 1
    assert summary["empty_count"] == 1
    assert summary["periodic_loop_count"] == 1
    assert summary["output_tokens"]["count"] == 4
```

Retain raw counts and rates already produced by `summarize_generation_health`; add `reasoning_failure_count` as the union-by-row of extraction failure, cap hit, empty output, and periodic loop so one row is not counted multiple times.

- [ ] **Step 5: Run focused comparison and health tests**

Run:

```bash
pytest -q pipeline/tests/test_compare.py pipeline/tests/test_eval_health.py
```

Expected: PASS.

- [ ] **Step 6: Commit**

```bash
git add pipeline/evalsuite/compare.py pipeline/evalsuite/health.py pipeline/tests/test_compare.py pipeline/tests/test_eval_health.py
git commit -m "feat(eval): compare repeated reasoning attempts by question"
```

---

### Task 4: Make preflight pin the actual paper-grade harness contract

**Files:**

- Modify: `pipeline/m3_quality_preflight.py`
- Modify: `pipeline/m3_quality_eval.py`
- Test: `pipeline/tests/test_m3_quality_eval.py`

**Interfaces:**

- Produces: `preflight/harness_contract.json` with lm-eval version, task alias, task version, output type, metric/filter, few-shot count, generation contract, representative prompt hashes, and sample counts.
- Produces: `run_manifest.json.harness_contract_sha256`.
- Consumes: the r4 configs from Task 1 and lm-eval task metadata.

- [ ] **Step 1: Write failing harness-contract tests**

```python
def test_reasoning_contract_requires_only_generate_until_tasks():
    task = lambda version: {
        "output_type": "generate_until",
        "task_version": version,
    }
    contract = build_reasoning_harness_contract(
        revision="0.4.12",
        task_records={
            "gpqa_diamond": task("2.2"),
            "mmlu_pro": task("3.1"),
            "gsm8k": task("3.0"),
            "aime_2025": task("1.0"),
        },
        generation_seeds=[42, 1234, 4158],
        gen_kwargs={
            "temperature": 1.0,
            "top_p": 0.95,
            "do_sample": True,
            "max_gen_toks": 16384,
        },
    )
    assert contract["valid"] is True
    assert contract["tasks"]["gpqa_diamond"]["output_type"] == "generate_until"


def test_reasoning_contract_rejects_wrong_revision(valid_contract_inputs):
    valid_contract_inputs["revision"] = "0.4.11"
    with pytest.raises(ValueError, match="revision"):
        build_reasoning_harness_contract(**valid_contract_inputs)


def test_reasoning_contract_rejects_non_generation_task(valid_contract_inputs):
    valid_contract_inputs["task_records"]["gpqa_diamond"]["output_type"] = "multiple_choice"
    with pytest.raises(ValueError, match="output_type"):
        build_reasoning_harness_contract(**valid_contract_inputs)


def test_reasoning_contract_rejects_wrong_seeds(valid_contract_inputs):
    valid_contract_inputs["generation_seeds"] = [42]
    with pytest.raises(ValueError, match="generation_seeds"):
        build_reasoning_harness_contract(**valid_contract_inputs)
```

- [ ] **Step 2: Run tests and verify failure**

Run:

```bash
pytest -q pipeline/tests/test_m3_quality_eval.py -k reasoning_contract
```

Expected: failure because no task-level r4 contract exists.

- [ ] **Step 3: Implement contract inspection using lm-eval task objects**

For every resolved task, record:

```python
record = {
    "canonical_name": canonical,
    "installed_name": installed,
    "output_type": str(task.OUTPUT_TYPE),
    "task_version": str(task.VERSION),
    "num_fewshot": configured_task.num_fewshot,
    "metric": configured_task.metric,
    "dataset_path": str(task.config.dataset_path),
    "dataset_name": str(task.config.dataset_name),
    "representative_doc_id": selected_doc_id,
    "representative_prompt_sha256": sha256(rendered_prompt.encode()).hexdigest(),
}
```

Use the installed task's own `fewshot_context`/document formatter and the official MiniMax tokenizer's `apply_chat_template` to render one selected prompt per task. For GPQA, also record the displayed A-D choices and correct displayed label from the processed document. Invoke the formatter twice from the same seeded state and fail if the prompt or displayed choices differ.

Require exact lm-eval version `0.4.12`, all four output types `generate_until`, the four configured metric keys, shot counts `0/5/8/0`, three exact generation seeds, and exact shared generation kwargs. Require GPQA task version `2.2` and `exact_match,flexible-extract`. Resolve GPQA twice from the same fixed harness seed and fail if its representative prompt or displayed-choice mapping differs. Write the contract atomically and add its hash to the run and arm manifests.

- [ ] **Step 4: Remove the obsolete mixed-task thinking rejection**

Replace `validate_reasoning_config` with explicit r4 validation:

```python
if eval_raw.get("enable_thinking") is not True:
    raise ValueError("r4 reasoning evaluation requires enable_thinking=true")
if eval_raw.get("generation_seeds") != [42, 1234, 4158]:
    raise ValueError("r4 reasoning evaluation requires seeds 42, 1234, 4158")
if eval_raw.get("think_end_token") != "</mm:think>":
    raise ValueError("MiniMax-M3 reasoning requires think_end_token='</mm:think>'")
```

Legacy configs with no `generation_seeds` retain the existing adaptive-mode validation path.

- [ ] **Step 5: Run preflight-focused tests**

Run:

```bash
pytest -q pipeline/tests/test_m3_quality_eval.py -k "reasoning_config or reasoning_contract or preflight or tokenizer"
```

Expected: PASS.

- [ ] **Step 6: Commit**

```bash
git add pipeline/m3_quality_preflight.py pipeline/m3_quality_eval.py pipeline/tests/test_m3_quality_eval.py
git commit -m "feat(eval): pin M3 reasoning harness provenance"
```

---

### Task 5: Merge repeated attempts, gate health, and launch four independent arms

**Files:**

- Modify: `pipeline/m3_quality_eval.py`
- Modify: `pipeline/slurm/test_m3_quality_eval_arm.sh`
- Modify: `pipeline/slurm/run_m3_quality_eval_srun.sh`
- Test: `pipeline/tests/test_m3_quality_eval.py`
- Test: `pipeline/tests/test_m3_quality_eval_runner.py`

**Interfaces:**

- Produces: merged JSONL keyed by `attempt_uid`, while preserving `sample_uid` for grouped statistics.
- Produces: gates that omit the distributional check when `probe.enabled=false` and fail on incomplete grids or reasoning-health failures.
- Produces: exactly four production `srun` commands with two task groupings and a 24-hour limit.
- Consumes: repeated rows/statistics from Tasks 2-3 and harness hash from Task 4.

- [ ] **Step 1: Write failing merge and gate tests**

```python
def test_merge_preserves_three_attempts_per_question(tmp_path):
    write_complete_r4_run(tmp_path, questions=2, seeds=(42, 1234, 4158))
    result = validate_and_merge(tmp_path)
    rows = [
        json.loads(line)
        for line in (
            tmp_path / "merged/inhouse_gptq/samples/gpqa_diamond.jsonl"
        ).read_text().splitlines()
    ]
    assert result["infrastructure_ok"] is True
    assert len(rows) == 6
    assert len({row["sample_uid"] for row in rows}) == 2
    assert len({row["attempt_uid"] for row in rows}) == 6


def test_r4_gate_omits_distributional_and_rejects_health_failure():
    thresholds = GateThresholds(
        max_task_drop=0.02,
        min_macro_recovery=0.98,
        max_conditional_regression=0.05,
        max_perplexity_increase=None,
        max_degeneration_failures=0,
    )
    passing = evaluate_gates(repeated_matrix(reasoning_failure_count=0), thresholds)
    failing = evaluate_gates(repeated_matrix(reasoning_failure_count=1), thresholds)
    assert "perplexity_increase" not in passing["models"]["inhouse_gptq"]
    assert passing["quality_ok"] is True
    assert failing["quality_ok"] is False
```

- [ ] **Step 2: Write failing launcher tests**

```python
def test_r4_dry_run_emits_four_top_level_srun_arms(tmp_path, request):
    result = run_r4_production_dry_run(tmp_path, request)
    commands = [line for line in result.stdout.splitlines() if line.startswith("srun ")]
    assert len(commands) == 4
    assert all("--nodes=1" in line and "--gpus-per-node=8" in line for line in commands)
    assert all("--time 24:00:00" in line for line in commands)
    assert all("--run-probe 0" in line for line in commands)
    assert sum("--tasks gpqa_diamond" in line for line in commands) == 2
    assert sum("--tasks mmlu_pro,gsm8k,aime25" in line for line in commands) == 2
    assert "sbatch" not in result.stdout
    assert "total_nodes=4" in result.stdout
```

- [ ] **Step 3: Run focused tests and verify failure**

Run:

```bash
pytest -q pipeline/tests/test_m3_quality_eval.py pipeline/tests/test_m3_quality_eval_runner.py -k "r4 or three_attempts or health_failure"
```

Expected: failure because merge keys by question UID, probes are mandatory in smoke/gates, and the old matrix has six arms.

- [ ] **Step 4: Implement optional probes and attempt-aware merge**

Add `ProbeSpec.enabled: bool = True`; parse omitted `enabled` as true for legacy matrices. Make `GateThresholds.max_perplexity_increase` optional and set it to `None` in r4. In `_merge_model_arms`, use:

```python
identity = row.get("attempt_uid") or row.get("sample_uid")
if not identity:
    raise ValueError(f"sample without stable identity in {sample_path}")
```

Before a successful merge, validate each repeated task contains the exact configured seed set for every selected question and the expected question count from the production manifest. Add `harness_contract_sha256` to arm-manifest equality checks.

When probes are disabled, smoke arms set `distributional_probe=false`, smoke validation omits `probe_budget`, and quality gates omit `perplexity_increase`. Health evidence must include both AWQ and GPTQ summaries and fail when either side has missing, empty, extraction-failed, capped, periodic-loop, or nonfinite rows.

- [ ] **Step 5: Update arm and launcher contracts**

The arm script continues to call `pipeline.evalsuite.cli run`; Task 2 makes that path repeated-seed aware. Record `harness_contract_sha256` and the selected task/seed contract in `arm_manifest.json`. The launcher must rely only on the r4 matrix-produced plan and emit four concurrent top-level allocations. Do not add an internal `srun` call to the arm.

- [ ] **Step 6: Run focused tests and Bash syntax checks**

Run:

```bash
pytest -q pipeline/tests/test_m3_quality_eval.py pipeline/tests/test_m3_quality_eval_runner.py
bash -n pipeline/slurm/test_m3_quality_eval_arm.sh
bash -n pipeline/slurm/run_m3_quality_eval_srun.sh
```

Expected: all PASS.

- [ ] **Step 7: Commit**

```bash
git add pipeline/m3_quality_eval.py pipeline/slurm/test_m3_quality_eval_arm.sh pipeline/slurm/run_m3_quality_eval_srun.sh pipeline/tests/test_m3_quality_eval.py pipeline/tests/test_m3_quality_eval_runner.py
git commit -m "feat(eval): launch and gate four-arm M3 reasoning run"
```

---

### Task 6: Publish the copy-ready executor packet and verify the complete change

**Files:**

- Modify: `M3_PAIRED_GPTQ_AWQ_TASK_ISOLATED_HANDOFF.md`
- Modify: `docs/superpowers/specs/2026-07-14-m3-task-isolated-paired-quality-rerun-design.md` only if implementation reveals a factual interface correction; do not change approved scientific semantics.

**Interfaces:**

- Produces: the sole active `READY_FOR_EXECUTOR` r4 packet with exact commit, preflight, smoke, dry-run, production, monitoring, aggregation, evidence, and push commands.
- Consumes: all commands and artifact names implemented in Tasks 1-5.

- [ ] **Step 1: Run the complete CPU verification suite**

Run:

```bash
pytest -q pipeline/tests/test_lmeval_runner.py pipeline/tests/test_static_checkpoint.py pipeline/tests/test_compare.py pipeline/tests/test_eval_health.py pipeline/tests/test_m3_quality_eval.py pipeline/tests/test_m3_quality_eval_runner.py pipeline/tests/test_m3_quality_evidence.py
ruff check pipeline/config.py pipeline/lmeval_runner.py pipeline/evalsuite/static.py pipeline/evalsuite/compare.py pipeline/evalsuite/health.py pipeline/m3_quality_eval.py pipeline/m3_quality_preflight.py pipeline/tests/test_lmeval_runner.py pipeline/tests/test_static_checkpoint.py pipeline/tests/test_compare.py pipeline/tests/test_eval_health.py pipeline/tests/test_m3_quality_eval.py pipeline/tests/test_m3_quality_eval_runner.py
ruff format --check pipeline/config.py pipeline/lmeval_runner.py pipeline/evalsuite/static.py pipeline/evalsuite/compare.py pipeline/evalsuite/health.py pipeline/m3_quality_eval.py pipeline/m3_quality_preflight.py pipeline/tests/test_lmeval_runner.py pipeline/tests/test_static_checkpoint.py pipeline/tests/test_compare.py pipeline/tests/test_eval_health.py pipeline/tests/test_m3_quality_eval.py pipeline/tests/test_m3_quality_eval_runner.py
bash -n pipeline/slurm/test_m3_quality_eval_arm.sh
bash -n pipeline/slurm/run_m3_quality_eval_srun.sh
git diff --check
```

Expected: all tests and checks pass. If the local environment lacks a cluster-only dependency, run every import-independent test, record the exact missing dependency, and make the executor preflight run the deferred test before GPU allocation.

- [ ] **Step 2: Generate and inspect a dry-run launch plan**

Create a temporary passing smoke-gate JSON and run:

```bash
bash pipeline/slurm/run_m3_quality_eval_srun.sh \
  --profile production \
  --matrix pipeline/configs/minimax_m3_paired_gptq_awq_reasoning_r4.yaml \
  --run-root /tmp/m3-reasoning-r4-dry-run \
  --smoke-gate /tmp/m3-reasoning-r4-dry-run/smoke_gate.json \
  --dry-run
```

Expected: four `srun` commands, four total nodes, no probe, no `sbatch`, and `--time 24:00:00` on every arm.

- [ ] **Step 3: Rewrite the existing task handoff as r4**

At the top of `M3_PAIRED_GPTQ_AWQ_TASK_ISOLATED_HANDOFF.md`, set:

```markdown
- Protocol version: 1
- State: READY_FOR_EXECUTOR
- Packet revision: r4-generated-reasoning
- Branch: duy-branch
- Base Git commit: the exact 40-character output recorded by Step 5
- Decision question: Does the repaired in-house GPTQ checkpoint preserve generated reasoning pass@1 relative to cyankiwi AWQ on the paired r4 sample?
```

Label all r1-r3 instructions `HISTORICAL — SUPERSEDED BY r4`. Include exact commands for environment activation, version checks, clean tracked-worktree classification, preflight, smoke, gate, dry-run, detached `tmux` production launch, non-owning monitoring, partial-safe aggregation, artifact hashing, evidence commit, and push. Explicitly authorize no retries and require a fresh run root.

- [ ] **Step 4: Commit the packet with a temporary base-commit marker**

```bash
git add M3_PAIRED_GPTQ_AWQ_TASK_ISOLATED_HANDOFF.md docs/superpowers/specs/2026-07-14-m3-task-isolated-paired-quality-rerun-design.md
git commit -m "docs(m3): authorize generated reasoning rerun"
```

- [ ] **Step 5: Resolve and amend the packet's exact base commit**

Use a content commit followed by a packet commit so the packet can name an immutable implementation base without self-reference:

```bash
IMPLEMENTATION_COMMIT=$(git rev-parse HEAD^)
```

Set `Base Git commit` to `$IMPLEMENTATION_COMMIT`, verify every command refers to the r4 matrix/config, then commit the packet update:

```bash
git add M3_PAIRED_GPTQ_AWQ_TASK_ISOLATED_HANDOFF.md
git commit -m "docs(m3): finalize r4 executor base revision"
```

- [ ] **Step 6: Final verification and push**

Run:

```bash
git diff --check HEAD~2..HEAD
git status --short --branch
git log -8 --oneline
git push origin duy-branch
```

Expected: clean tracked worktree, the implementation and packet commits on `duy-branch`, and the remote updated. Stop in `READY_FOR_EXECUTOR`; do not run cluster jobs locally.
