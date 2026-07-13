# MiniMax-M3 vLLM Quality Evaluation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a reproducible, paired, vLLM-first MiniMax-M3 quality matrix comparing BF16, in-house GPTQ, cyankiwi AWQ, and aquaman AutoRound in less than five hours, with downstream, distributional, generation-health, and checkpoint-fidelity metrics.

**Architecture:** Extend the existing `pipeline.evalsuite` boundaries: lm-eval remains the task executor, normalized sample rows become the stable comparison interface, and focused modules add sample selection, paired statistics, generation health, distributional probes, and checkpoint diagnostics. A MiniMax-M3 matrix controller validates provenance and merges shard outputs; concurrent `srun` arms execute on ten nodes (two 8xH100 nodes per BF16 arm and one per quantized arm) and return complete evidence through Git.

**Tech Stack:** Python 3.11+, PyYAML, EleutherAI lm-evaluation-harness, vLLM, Transformers tokenizer, safetensors metadata, pytest, Bash, Slurm `srun`, JSON/JSONL artifacts.

## Global Constraints

- Model quality/accuracy is the primary milestone; serving performance is deferred.
- vLLM is the accepted backend for this milestone; preserve the existing SGLang boundary.
- Compare exactly four default checkpoints: BF16, passing in-house GPTQ, cyankiwi AWQ INT4, and aquaman AutoRound 3.2-bit.
- Every checkpoint must use identical prompts, sample identities, chat-template behavior, thinking mode, and deterministic decoding.
- The production matrix must target less than five hours of GPU wall-clock time.
- Use concurrent background `srun --exclusive`; do not use `sbatch`.
- Missing tasks, samples, metrics, checkpoints, or provenance are failures and must never be silently skipped.
- Do not report bounded top-k proxies as full-vocabulary KL divergence.
- Results must be resumable, schema-versioned, and suitable for Git handoff from the capable cluster.
- Follow TDD for every implementation task and preserve unrelated worktree changes.

---

## File Structure

### Existing files to modify

- `pipeline/config.py`: exact-sample manifest and deterministic bootstrap configuration.
- `pipeline/lmeval_runner.py`: pass exact sample maps into lm-eval and reject `limit`/sample conflicts.
- `pipeline/evalsuite/static.py`: namespaced stable sample IDs and richer normalized sample rows.
- `pipeline/evalsuite/compare.py`: consume stable IDs and expanded paired statistics.
- `pipeline/evalsuite/report.py`: render quantization-fidelity and gate results.
- `pipeline/evalsuite/cli.py`: task-shard and sample-manifest CLI overrides.
- `pipeline/README.md`: MiniMax-M3 quality matrix usage and artifact contract.

### New focused modules

- `pipeline/evalsuite/sampling.py`: exact sample-manifest schema, validation, hashing, and stratified selection.
- `pipeline/evalsuite/stats.py`: exact McNemar and deterministic paired-bootstrap metrics.
- `pipeline/evalsuite/health.py`: generation degeneration and output-health diagnostics.
- `pipeline/evalsuite/probe_corpus.py`: immutable calibration-disjoint teacher-forced corpus construction.
- `pipeline/evalsuite/distributional.py`: pure calculations over teacher-forced top-k records.
- `pipeline/m3_distributional_probe.py`: GPU vLLM prompt-logprob probe runner.
- `pipeline/m3_checkpoint_diagnostics.py`: generic BF16/compressed/GPTQ/AutoRound metadata diagnostics.
- `pipeline/m3_quality_eval.py`: model/shard matrix manifest, preflight, merge, gates, and root report orchestration.
- `pipeline/configs/eval_minimax_m3_quality.yaml`: canonical deterministic quality task profile.
- `pipeline/configs/minimax_m3_quality_matrix.yaml`: four models, two shards, paths, and gate defaults.
- `pipeline/slurm/test_m3_quality_eval_arm.sh`: one resumable model/shard worker.
- `pipeline/slurm/run_m3_quality_eval_srun.sh`: smoke-gated eight-arm concurrent `srun` launcher.

### New tests

- `pipeline/tests/test_eval_sampling.py`
- `pipeline/tests/test_eval_stats.py`
- `pipeline/tests/test_eval_health.py`
- `pipeline/tests/test_eval_distributional.py`
- `pipeline/tests/test_m3_checkpoint_diagnostics.py`
- `pipeline/tests/test_m3_quality_eval.py`
- `pipeline/tests/test_m3_quality_eval_runner.py`

---

### Task 1: Exact Sample Manifests and Stable Grouped-Task Identity

**Files:**
- Create: `pipeline/evalsuite/sampling.py`
- Modify: `pipeline/config.py`
- Modify: `pipeline/lmeval_runner.py`
- Modify: `pipeline/evalsuite/static.py`
- Modify: `pipeline/evalsuite/cli.py`
- Test: `pipeline/tests/test_eval_sampling.py`
- Test: `pipeline/tests/test_lmeval_runner.py`
- Test: `pipeline/tests/test_static_checkpoint.py`

**Interfaces:**
- Produces: `load_sample_manifest(path: Path) -> SampleManifest`
- Produces: `sample_map_for_task(manifest: SampleManifest, task_name: str) -> dict[str, list[int]] | None`
- Produces: `stable_sample_uid(task: str, subtask: str, doc_id: object) -> str`
- Produces: `build_stratified_indices(sizes: dict[str, int], total: int, seed: int) -> dict[str, list[int]]`
- Changes: `EvalConfig.samples_manifest: str | None`, `EvalConfig.bootstrap_seed: int`, `EvalConfig.bootstrap_iters: int`
- Consumes: lm-eval `simple_evaluate(samples={leaf_task: [indices...]})`

- [ ] **Step 1: Write failing sample-manifest tests**

```python
def test_stratified_indices_are_deterministic_and_balanced():
    sizes = {"mmlu_pro_math": 100, "mmlu_pro_history": 50}
    a = build_stratified_indices(sizes, total=30, seed=42)
    b = build_stratified_indices(sizes, total=30, seed=42)
    assert a == b
    assert sum(map(len, a.values())) == 30
    assert len(a["mmlu_pro_math"]) == 20
    assert len(a["mmlu_pro_history"]) == 10
    assert a["mmlu_pro_math"] == sorted(set(a["mmlu_pro_math"]))

def test_manifest_hash_rejects_mutation(tmp_path):
    path = tmp_path / "samples.json"
    path.write_text(json.dumps({"schema_version": 1, "tasks": {"mmlu_pro": {"mmlu_pro_math": [1, 3]}}}))
    manifest = load_sample_manifest(path)
    assert sample_map_for_task(manifest, "mmlu_pro") == {"mmlu_pro_math": [1, 3]}
    assert len(manifest.sha256) == 64

def test_stable_sample_uid_namespaces_subtasks():
    assert stable_sample_uid("mmlu", "mmlu_math", 0) != stable_sample_uid("mmlu", "mmlu_history", 0)
```

- [ ] **Step 2: Run the focused tests and verify they fail**

Run: `python -m pytest pipeline/tests/test_eval_sampling.py -q`

Expected: FAIL because `pipeline.evalsuite.sampling` does not exist.

- [ ] **Step 3: Implement the immutable manifest model and stratified selector**

```python
@dataclass(frozen=True)
class SampleManifest:
    schema_version: int
    seed: int
    tasks: dict[str, dict[str, tuple[int, ...]]]
    sha256: str

def stable_sample_uid(task: str, subtask: str, doc_id: object) -> str:
    payload = json.dumps([task, subtask, doc_id], ensure_ascii=False, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()

def build_stratified_indices(sizes: dict[str, int], total: int, seed: int) -> dict[str, list[int]]:
    if total <= 0 or total > sum(sizes.values()):
        raise ValueError("total must be between 1 and available examples")
    # Allocate proportional floors, then largest remainders with lexical tie-breaks.
    # Sample without replacement from each leaf using random.Random(seed + sha256(leaf)).
```

The JSON writer must include `schema_version`, `seed`, `tasks`, source task revision metadata, and a canonical SHA-256 over the content excluding the `sha256` field. The loader recomputes and validates that hash when it is present.

- [ ] **Step 4: Add exact-sample configuration and runner wiring**

Add to `EvalConfig`:

```python
samples_manifest: str | None = None
bootstrap_seed: int = 42
bootstrap_iters: int = 10_000
```

In `evaluate_tasks`, load the manifest once, obtain the current task's leaf mapping, and enforce:

```python
sample_map = sample_map_for_task(manifest, task.name) if manifest else None
if sample_map is not None and task.limit is not None:
    raise ValueError(f"task {task.name}: exact samples and limit are mutually exclusive")
if sample_map is not None:
    kwargs["samples"] = sample_map
elif task.limit is not None:
    kwargs["limit"] = task.limit
```

Add `--tasks comma,separated` and `--samples-manifest PATH` to `evalsuite run`; filter configured tasks in requested order and fail on unknown names.

- [ ] **Step 5: Preserve subtask names in normalized samples**

Change `_collect_task_samples` to copy each raw row and add `_eval_subtask` equal to the actual leaf task key. Change `_extract_sample_row` to emit:

```python
{
    "sample_uid": stable_sample_uid(task.name, subtask, doc_id),
    "task": task.name,
    "subtask": subtask,
    "doc_id": doc_id,
    "target": sample.get("target"),
    "response": _first_response(sample),
    "metric": used_metric or base,
    "metric_value": metric_value,
    "correct": correct,
}
```

- [ ] **Step 6: Add runner and grouped-ID regressions**

```python
def test_evaluate_tasks_passes_exact_samples(monkeypatch, tmp_path):
    # Stub lm_eval.simple_evaluate and assert samples={"mmlu_pro_math": [0, 4]}.

def test_group_rows_with_same_doc_id_get_distinct_uids(tmp_path):
    # mmlu_math/doc_id=0 and mmlu_history/doc_id=0 must both survive JSONL output.
```

- [ ] **Step 7: Run Task 1 tests**

Run: `python -m pytest pipeline/tests/test_eval_sampling.py pipeline/tests/test_lmeval_runner.py pipeline/tests/test_static_checkpoint.py -q`

Expected: PASS.

- [ ] **Step 8: Commit Task 1**

```bash
git add pipeline/config.py pipeline/lmeval_runner.py pipeline/evalsuite/cli.py pipeline/evalsuite/static.py pipeline/evalsuite/sampling.py pipeline/tests/test_eval_sampling.py pipeline/tests/test_lmeval_runner.py pipeline/tests/test_static_checkpoint.py
git commit -m "feat(eval): add exact paired sample manifests"
```

---

### Task 2: Quantization-Specific Paired Statistics

**Files:**
- Create: `pipeline/evalsuite/stats.py`
- Modify: `pipeline/evalsuite/compare.py`
- Modify: `pipeline/config.py`
- Test: `pipeline/tests/test_eval_stats.py`
- Test: `pipeline/tests/test_compare.py`

**Interfaces:**
- Produces: `exact_mcnemar(regressions: int, recoveries: int) -> dict[str, float | int | str]`
- Produces: `paired_bootstrap(a: Sequence[float], b: Sequence[float], statistic: Callable, seed: int, iters: int) -> dict[str, float]`
- Produces: `pair_binary(rows_a, rows_b, *, seed, iters) -> dict[str, Any]`
- Changes: comparison row identity defaults to `sample_uid` with legacy fallback only for old artifacts.

- [ ] **Step 1: Write failing statistical tests**

```python
def test_exact_mcnemar_all_one_sided_flips():
    result = exact_mcnemar(6, 0)
    assert result == {"method": "exact_binomial", "discordant": 6, "p_value": 0.03125}

def test_pair_binary_reports_quantization_conditionals():
    a = rows({"a": 1, "b": 1, "c": 0, "d": 0})
    b = rows({"a": 1, "b": 0, "c": 1, "d": 0})
    out = pair_binary(a, b, seed=7, iters=200)
    assert out["conditional_regression_rate"] == 0.5
    assert out["conditional_recovery_rate"] == 0.5
    assert out["net_harmful_flips"] == 0
    assert out["score_recovery_ratio"] == 1.0

def test_bootstrap_is_reproducible():
    assert paired_bootstrap([1, 0], [0, 0], delta_mean, 42, 1000) == paired_bootstrap([1, 0], [0, 0], delta_mean, 42, 1000)
```

- [ ] **Step 2: Run and verify RED**

Run: `python -m pytest pipeline/tests/test_eval_stats.py pipeline/tests/test_compare.py -q`

Expected: FAIL on missing module and fields.

- [ ] **Step 3: Implement exact and asymptotic McNemar methods**

Use a numerically stable exact two-sided binomial tail for fewer than 25 discordant pairs; retain the continuity-corrected chi-square method for 25 or more. Return the selected `method` and never overwrite one method with the other.

```python
def exact_mcnemar(regressions: int, recoveries: int) -> dict:
    n = regressions + recoveries
    if n == 0:
        return {"method": "exact_binomial", "discordant": 0, "p_value": 1.0}
    k = min(regressions, recoveries)
    p = min(1.0, 2.0 * sum(math.comb(n, i) for i in range(k + 1)) / 2**n)
    return {"method": "exact_binomial", "discordant": n, "p_value": p}
```

- [ ] **Step 4: Implement deterministic paired bootstrap intervals**

Sample paired indices with `random.Random(seed)`, compute percentile 2.5/97.5 bounds, and return `estimate`, `ci95_low`, `ci95_high`, `seed`, and `iterations`. Reject unequal or empty input arrays explicitly.

- [ ] **Step 5: Expand binary comparison fields**

Add:

```python
conditional_regression_rate = regressions / (both_correct + regressions) if baseline_correct else None
conditional_recovery_rate = recoveries / (both_wrong + recoveries) if baseline_wrong else None
score_recovery_ratio = acc_b / acc_a if acc_a else None
net_harmful_flips = regressions - recoveries
```

Compute paired-bootstrap intervals for accuracy delta, flip indicator mean, and regression indicator mean. Include missing-in-A, missing-in-B, duplicate-ID, and paired-coverage counts. Duplicate stable IDs must raise rather than last-write-win.

- [ ] **Step 6: Add continuous paired metrics**

Replace the perplexity-specific mean-only helper with a generic paired continuous helper that reports paired mean/median delta, relative delta, bootstrap interval, and raw metric direction. Preserve the explicit NLL/perplexity presentation fields for the distributional module.

- [ ] **Step 7: Run Task 2 tests**

Run: `python -m pytest pipeline/tests/test_eval_stats.py pipeline/tests/test_compare.py -q`

Expected: PASS.

- [ ] **Step 8: Commit Task 2**

```bash
git add pipeline/config.py pipeline/evalsuite/stats.py pipeline/evalsuite/compare.py pipeline/tests/test_eval_stats.py pipeline/tests/test_compare.py
git commit -m "feat(eval): add paired quantization fidelity statistics"
```

---

### Task 3: Generation-Health Diagnostics

**Files:**
- Create: `pipeline/evalsuite/health.py`
- Modify: `pipeline/evalsuite/static.py`
- Test: `pipeline/tests/test_eval_health.py`
- Test: `pipeline/tests/test_static_checkpoint.py`

**Interfaces:**
- Produces: `analyze_generation(text: str | None, *, token_ids: Sequence[int] | None, max_gen_toks: int | None, extracted_answer: object) -> dict[str, Any]`
- Produces: `summarize_generation_health(rows: Sequence[dict]) -> dict[str, Any]`
- Extends normalized sample rows with `health` and optional `response_token_ids`.

- [ ] **Step 1: Write failing loop and failure-shape tests**

```python
@pytest.mark.parametrize("ids,period", [([1,2]*16, 2), ([4,5,6]*12, 3), ([9]*20, 1)])
def test_detects_periodic_token_loop(ids, period):
    out = analyze_generation("x", token_ids=ids, max_gen_toks=len(ids), extracted_answer=None)
    assert out["periodic_loop"] is True
    assert out["loop_period"] == period
    assert out["length_cap_hit"] is True

def test_healthy_short_answer():
    out = analyze_generation("Paris.", token_ids=[1,2], max_gen_toks=64, extracted_answer="Paris")
    assert out["empty"] is False
    assert out["periodic_loop"] is False
    assert out["answer_extraction_failed"] is False
```

- [ ] **Step 2: Run and verify RED**

Run: `python -m pytest pipeline/tests/test_eval_health.py -q`

Expected: FAIL because `health.py` is absent.

- [ ] **Step 3: Implement token-period and repeated n-gram detection**

Check periods 1 through 16 over the final window, require at least four repeats and at least 16 repeated tokens, and separately compute repeated 3-gram/4-gram fractions. This explicitly covers the known eight-token MiniMax loop without flagging ordinary repeated words in short answers.

- [ ] **Step 4: Normalize generation metadata**

Extract token IDs and finish/cap information from lm-eval responses when present. If token IDs are unavailable, tokenize the final response with the checkpoint tokenizer in the arm post-processing step. Store the evidence source (`backend`, `tokenizer_reencode`, or `unavailable`) so cap-hit certainty is not overstated.

- [ ] **Step 5: Write per-task health summaries atomically**

After each generative task checkpoint, write `generation_health/<task>.json` containing counts/rates for empty, missing, extraction failure, cap hit, periodic loop, repeated n-gram, non-finite metric, output-token quantiles, and reasoning-token quantiles when separable.

- [ ] **Step 6: Run Task 3 tests**

Run: `python -m pytest pipeline/tests/test_eval_health.py pipeline/tests/test_static_checkpoint.py -q`

Expected: PASS.

- [ ] **Step 7: Commit Task 3**

```bash
git add pipeline/evalsuite/health.py pipeline/evalsuite/static.py pipeline/tests/test_eval_health.py pipeline/tests/test_static_checkpoint.py
git commit -m "feat(eval): detect quantization generation degeneration"
```

---

### Task 4: Distributional Fidelity Probe

**Files:**
- Create: `pipeline/evalsuite/probe_corpus.py`
- Create: `pipeline/evalsuite/distributional.py`
- Create: `pipeline/m3_distributional_probe.py`
- Test: `pipeline/tests/test_eval_distributional.py`

**Interfaces:**
- Produces: `build_probe_corpus(texts: Iterable[str], tokenizer, *, seed: int) -> list[dict[str, Any]]`
- Produces: `normalize_prompt_logprobs(request_output, prompt_meta) -> Iterable[dict]`
- Produces: `compare_distributional_records(reference: Iterable[dict], candidate: Iterable[dict]) -> dict[str, Any]`
- CLI: `python -m pipeline.m3_distributional_probe run --model ... --corpus ... --out ...`
- CLI: `python -m pipeline.m3_distributional_probe compare --reference ... --candidate ... --out ...`

- [ ] **Step 1: Write failing pure-calculation tests**

```python
def test_distributional_metrics_use_observed_token_and_topk():
    ref = records(observed_lp=-1.0, top=[(10,-0.1),(11,-1.0)])
    quant = records(observed_lp=-1.5, top=[(10,-0.2),(12,-0.9)])
    out = compare_distributional_records(ref, quant)
    assert out["paired_tokens"] == 1
    assert out["mean_observed_logprob_drift"] == pytest.approx(-0.5)
    assert out["top1_agreement"] == 1.0
    assert out["topk_jaccard"] == pytest.approx(1 / 3)
    assert "kl_divergence" not in out

def test_distributional_pairing_rejects_token_mismatch():
    with pytest.raises(ValueError, match="token identity"):
        compare_distributional_records(ref_for_token(1), quant_for_token(2))

def test_probe_corpus_has_fixed_length_buckets():
    corpus = build_probe_corpus(["one two three " * 40000], fake_tokenizer, seed=42)
    assert Counter(row["length_bucket"] for row in corpus) == {"short": 8, "8k": 4, "32k": 2}
    assert all(row["prompt_token_ids"] for row in corpus)
```

- [ ] **Step 2: Run and verify RED**

Run: `python -m pytest pipeline/tests/test_eval_distributional.py -q`

Expected: FAIL on missing module.

- [ ] **Step 3: Implement the immutable calibration-disjoint corpus builder**

Use `Salesforce/wikitext`, configuration `wikitext-2-raw-v1`, test split, with
the resolved dataset revision recorded. Concatenate non-empty test documents
with the BF16 tokenizer and select non-overlapping deterministic windows: eight
short (2,048-token), four 8,192-token, and two 32,768-token prompts. Persist
`prompt_token_ids`, source document/span metadata, tokenizer hash, dataset
revision, and a canonical corpus SHA-256. Reject any quantized checkpoint whose
tokenizer hash differs from BF16. Corpus generation runs during preflight and
requires no GPU.

- [ ] **Step 4: Implement schema-versioned normalized records**

Each JSONL row contains corpus hash, prompt ID, length bucket, position, observed token ID, observed log-probability, and ordered top-k `(token_id, logprob, rank)` values. Reject duplicate `(prompt_id, position)` keys and mismatched observed tokens.

- [ ] **Step 5: Implement pure paired metrics**

Report NLL, perplexity, perplexity ratio, bits-per-token delta, observed-token drift quantiles, top-1 agreement, top-5/top-20 Jaccard overlap, BF16 top-token retention, and the same breakdown by length bucket and position quartile. Never emit a `kl_divergence` field.

- [ ] **Step 6: Implement the vLLM GPU probe CLI**

Load the same checkpoint with TP=8 and MiniMax serving compatibility hooks, tokenize the immutable corpus, and call vLLM with:

```python
SamplingParams(
    temperature=0.0,
    max_tokens=1,
    prompt_logprobs=20,
    logprobs=20,
    seed=42,
)
```

The single generated token is discarded; prompt log-probabilities are the artifact. Write records incrementally per prompt and an atomic summary containing versions, model/config/tokenizer hashes, corpus hash, counts, and elapsed seconds. Resume only when all provenance hashes match.

- [ ] **Step 7: Add CPU normalization stubs for vLLM output variants**

Tests must cover dictionary-style logprobs, current `Logprob` objects, missing first-token logprob, and fewer than requested top-k entries.

- [ ] **Step 8: Run Task 4 tests**

Run: `python -m pytest pipeline/tests/test_eval_distributional.py -q`

Expected: PASS without importing vLLM at module import time.

- [ ] **Step 9: Commit Task 4**

```bash
git add pipeline/evalsuite/probe_corpus.py pipeline/evalsuite/distributional.py pipeline/m3_distributional_probe.py pipeline/tests/test_eval_distributional.py
git commit -m "feat(eval): add MiniMax distributional fidelity probe"
```

---

### Task 5: Generic Checkpoint Quantization Diagnostics

**Files:**
- Create: `pipeline/m3_checkpoint_diagnostics.py`
- Modify: `pipeline/verify_quant_checkpoint.py`
- Test: `pipeline/tests/test_m3_checkpoint_diagnostics.py`

**Interfaces:**
- Produces: `diagnose_checkpoint(path: Path, *, baseline_bytes: int | None) -> dict[str, Any]`
- Produces: `classify_module(module_name: str) -> str`
- CLI: `python -m pipeline.m3_checkpoint_diagnostics --checkpoint LABEL=PATH ... --out DIR`

- [ ] **Step 1: Write failing synthetic-index tests**

```python
def test_compressed_checkpoint_reports_coverage_and_fallback(tmp_path):
    write_index(tmp_path, {
        "language_model.layers.3.mlp.experts.0.gate_proj.weight_packed": "a.safetensors",
        "language_model.layers.3.mlp.experts.0.gate_proj.weight_scale": "a.safetensors",
        "language_model.layers.3.mlp.gate.weight": "a.safetensors",
    })
    write_quant_config(tmp_path, bits=4, group_size=128)
    out = diagnose_checkpoint(tmp_path, baseline_bytes=1000)
    assert out["quantization"]["weight_bits"] == 4
    assert out["coverage_by_component"]["routed_experts"]["quantized_modules"] == 1
    assert out["coverage_by_component"]["routers"]["plain_modules"] == 1

def test_plain_bf16_checkpoint_is_not_reported_as_broken(tmp_path):
    # No quantization_config => method "none", not infrastructure failure.
```

- [ ] **Step 2: Run and verify RED**

Run: `python -m pytest pipeline/tests/test_m3_checkpoint_diagnostics.py -q`

Expected: FAIL on missing module.

- [ ] **Step 3: Refactor reusable metadata helpers from the strict verifier**

Expose `_load_weight_keys`, module-prefix detection, and MiniMax component classification without changing the strict verifier's CLI exit semantics. Classification must recognize `language_model.layers`, `model.layers`, `mlp`, `block_sparse_moe`, shared experts, indexer, routers, norms, vision, and LM head aliases.

- [ ] **Step 4: Implement format-tolerant diagnostics**

Support plain BF16, compressed-tensors pack-quantized, GPTQ-style, AWQ-style, and AutoRound metadata. For unknown fields, write `{"status": "unavailable", "reason": "..."}`. Compute checkpoint bytes from actual shard file sizes, config/index hashes, compression ratio, stored bits per original parameter when shapes are available, module coverage, scale counts, and quantization provenance.

- [ ] **Step 5: Add deterministic scale/saturation sampling**

Open only a bounded deterministic shard/tensor sample. Report finite/zero scale counts and scale quantiles. Decode packed values only for recognized formats; otherwise make saturation explicitly unavailable. Do not load the whole 428B checkpoint into RAM.

- [ ] **Step 6: Run Task 5 tests**

Run: `python -m pytest pipeline/tests/test_m3_checkpoint_diagnostics.py -q`

Expected: PASS.

- [ ] **Step 7: Commit Task 5**

```bash
git add pipeline/m3_checkpoint_diagnostics.py pipeline/verify_quant_checkpoint.py pipeline/tests/test_m3_checkpoint_diagnostics.py
git commit -m "feat(eval): add MiniMax checkpoint fidelity diagnostics"
```

---

### Task 6: MiniMax-M3 Matrix Controller, Gates, and Reports

**Files:**
- Create: `pipeline/m3_quality_eval.py`
- Create: `pipeline/configs/eval_minimax_m3_quality.yaml`
- Create: `pipeline/configs/minimax_m3_quality_matrix.yaml`
- Modify: `pipeline/evalsuite/report.py`
- Test: `pipeline/tests/test_m3_quality_eval.py`
- Test: `pipeline/tests/test_compare.py`

**Interfaces:**
- Produces: `load_matrix(path: Path) -> MatrixSpec`
- Produces: `write_run_manifest(spec: MatrixSpec, root: Path) -> dict[str, Any]`
- Produces: `validate_and_merge(root: Path) -> dict[str, Any]`
- Produces: `evaluate_gates(matrix: dict[str, Any], thresholds: GateThresholds) -> dict[str, Any]`
- CLI subcommands: `preflight`, `manifest-arm`, `aggregate`, `report`

- [ ] **Step 1: Write failing matrix validation tests**

```python
def test_default_matrix_has_four_models_and_eight_arms():
    spec = load_matrix(Path("pipeline/configs/minimax_m3_quality_matrix.yaml"))
    assert [m.label for m in spec.models] == ["bf16", "inhouse_gptq", "cyankiwi_awq", "aquaman_autoround"]
    assert len(spec.expected_arms) == 8

def test_merge_rejects_sample_manifest_mismatch(tmp_path):
    write_arm(tmp_path, "bf16_reasoning", sample_sha="a")
    write_arm(tmp_path, "gptq_reasoning", sample_sha="b")
    with pytest.raises(ValueError, match="sample manifest"):
        validate_and_merge(tmp_path)

def test_quality_failure_is_distinct_from_infrastructure_failure():
    gates = evaluate_gates(matrix_with_task_delta(-0.03), GateThresholds(max_task_drop=0.02))
    assert gates["infrastructure_ok"] is True
    assert gates["quality_ok"] is False
```

- [ ] **Step 2: Run and verify RED**

Run: `python -m pytest pipeline/tests/test_m3_quality_eval.py -q`

Expected: FAIL on missing controller/config.

- [ ] **Step 3: Add canonical quality config**

Define vLLM TP=8, MiniMax compatibility defaults, chat template enabled, deterministic generation, and these configured task aliases:

```yaml
tasks:
  - {name: gpqa_diamond, metric: "acc_norm,none", num_fewshot: 0, limit: null}
  - {name: ifeval, metric: "prompt_level_strict_acc,none", num_fewshot: 0, limit: null}
  - {name: aime_2025, metric: "exact_match,none", num_fewshot: 0, limit: null}
  - {name: mmlu_pro, metric: "acc,none", num_fewshot: 5, limit: null}
  - {name: gsm8k, metric: "exact_match,strict-match", num_fewshot: 5, limit: null}
```

Preflight must resolve the installed harness's canonical identifiers and metrics. If an alias/metric differs, it writes a resolved config under the run root; it never edits the canonical source config or silently substitutes a different benchmark.

- [ ] **Step 4: Add the four-model/two-shard matrix config**

Use exact NFS defaults from the design. Define `reasoning` and `broad` shards, with the distributional probe assigned to the less expensive shard after smoke-derived timing. Define threshold defaults: 2 percentage-point maximum task drop, 98% macro recovery, 5% conditional regression, 10% perplexity increase, and zero degeneration failures.

- [ ] **Step 5: Implement preflight and manifests**

Preflight checks checkpoint paths, index/config readability, tokenizer/config/chat-template hashes, installed versions, task resolution, sample-map coverage, GPU-free checkpoint diagnostics, repository commit/status, and output writability. `run_manifest.json` is written before GPU launch and includes all provenance required by the design.

- [ ] **Step 6: Implement strict shard merge and pairwise comparison**

Validate schema/provenance hashes, merge JSONL by stable sample UID, reject conflicts and incomplete coverage, then call the expanded comparison module for BF16 versus each quantized model. Merge generation-health and distributional artifacts and retain unavailable fields explicitly.

- [ ] **Step 7: Implement gates and root report**

Render a four-model benchmark table, BF16-relative flip/regression table, confidence intervals, subgroup worst cases, distributional fidelity, generation health, checkpoint diagnostics, gate decisions, runtime, and failures. Remove the old “serving performance deferred” boilerplate from pairwise reports only when the new root report already states the phase boundary.

- [ ] **Step 8: Add self-comparison acceptance tests**

BF16 versus a byte-identical copy must produce zero score delta, zero flips, zero distributional drift, no degeneration, kappa 1 where defined, identical sample hashes, and passing gates.

- [ ] **Step 9: Run Task 6 tests**

Run: `python -m pytest pipeline/tests/test_m3_quality_eval.py pipeline/tests/test_compare.py -q`

Expected: PASS.

- [ ] **Step 10: Commit Task 6**

```bash
git add pipeline/m3_quality_eval.py pipeline/configs/eval_minimax_m3_quality.yaml pipeline/configs/minimax_m3_quality_matrix.yaml pipeline/evalsuite/report.py pipeline/tests/test_m3_quality_eval.py pipeline/tests/test_compare.py
git commit -m "feat(eval): aggregate MiniMax quality matrix"
```

---

### Task 7: Resumable Eight-Node `srun` Orchestration

**Files:**
- Create: `pipeline/slurm/test_m3_quality_eval_arm.sh`
- Create: `pipeline/slurm/run_m3_quality_eval_srun.sh`
- Test: `pipeline/tests/test_m3_quality_eval_runner.py`

**Interfaces:**
- Environment: `RUN_ID`, `MODEL_LABEL`, `SHARD`, `MATRIX_CONFIG`, `RESULTS_ROOT`, `EVIDENCE_ROOT`, `TIME_LIMIT`, `SRUN_ARGS`, `DRY_RUN`
- Output: one arm directory matching the design artifact tree.

- [ ] **Step 1: Write failing dry-run topology tests**

```python
def test_launcher_dry_run_has_eight_exclusive_tp8_arms():
    out = run_launcher(DRY_RUN="1", RUN_ID="test-run")
    lines = [line for line in out.splitlines() if line.startswith("srun ")]
    assert len(lines) == 8
    assert all("--exclusive" in line and "--gres=gpu:8" in line for line in lines)
    assert "MODEL_LABEL=bf16" in out
    assert "MODEL_LABEL=inhouse_gptq" in out

def test_resume_skips_only_validated_complete_arms(tmp_path):
    # Completed arm with matching manifest is omitted; failed or mismatched arm is relaunched.
```

- [ ] **Step 2: Run and verify RED**

Run: `python -m pytest pipeline/tests/test_m3_quality_eval_runner.py -q`

Expected: FAIL because scripts are missing.

- [ ] **Step 3: Implement one-arm worker**

The worker activates `/mnt/nfs/hoangduy/venvs/quant`, sources cluster environment, writes an arm manifest, runs its selected lm-eval tasks with exact samples, runs the distributional probe when assigned, copies stdout/stderr/return code into the evidence tree, and validates required files before writing `arm_complete.json` atomically.

- [ ] **Step 4: Implement concurrent launcher**

Use:

```bash
srun --exclusive --nodes=1 --ntasks=1 --gres=gpu:8 \
  --time="$TIME_LIMIT" --kill-on-bad-exit=0 \
  env RUN_ID="$RUN_ID" MODEL_LABEL="$model" SHARD="$shard" \
  bash pipeline/slurm/test_m3_quality_eval_arm.sh
```

Launch all selected arms in the background, retain PID-to-arm mapping, wait without fail-fast, write every return code, aggregate after waits, and return nonzero for infrastructure or quality gate failure. `DRY_RUN=1` prints shell-escaped commands and performs no filesystem mutation except optional temporary test roots.

- [ ] **Step 5: Add resume and retry semantics**

The launcher calls controller validation before skipping an arm. A mere file/directory presence is insufficient. `ARMS=model_shard,...` permits targeted retries without changing the run manifest.

- [ ] **Step 6: Run shell and runner tests**

Run:

```bash
bash -n pipeline/slurm/test_m3_quality_eval_arm.sh
bash -n pipeline/slurm/run_m3_quality_eval_srun.sh
python -m pytest pipeline/tests/test_m3_quality_eval_runner.py -q
DRY_RUN=1 RUN_ID=plan-smoke bash pipeline/slurm/run_m3_quality_eval_srun.sh
```

Expected: syntax PASS, pytest PASS, and exactly eight escaped `srun` commands.

- [ ] **Step 7: Commit Task 7**

```bash
git add pipeline/slurm/test_m3_quality_eval_arm.sh pipeline/slurm/run_m3_quality_eval_srun.sh pipeline/tests/test_m3_quality_eval_runner.py
git commit -m "feat(eval): launch MiniMax quality matrix with srun"
```

---

### Task 8: Documentation, Full Regression, and Capable-Cluster Handoff

**Files:**
- Modify: `pipeline/README.md`
- Create: `MINIMAX_M3_EVAL_HANDOFF.md`
- Modify: `BUGS_AND_FIXES.md`
- Modify: `docs/superpowers/plans/2026-07-12-minimax-m3-vllm-quality-evaluation.md`

**Interfaces:**
- Handoff command: preflight, one-model smoke, four-model smoke, then production matrix.
- Return contract: results tree, logs, version/task resolution, sample manifest, and concise runtime/failure summary committed to Git.

- [ ] **Step 1: Document commands and artifact interpretation**

Add exact commands:

```bash
python -m pipeline.m3_quality_eval preflight \
  --matrix pipeline/configs/minimax_m3_quality_matrix.yaml \
  --run-id "$RUN_ID"

PROFILE=smoke MODELS=bf16 \
  RUN_ID="$RUN_ID" bash pipeline/slurm/run_m3_quality_eval_srun.sh

PROFILE=smoke RUN_ID="$RUN_ID" \
  bash pipeline/slurm/run_m3_quality_eval_srun.sh

PROFILE=production RUN_ID="$RUN_ID" \
  bash pipeline/slurm/run_m3_quality_eval_srun.sh
```

Explain model overrides, targeted arms, resume rules, gates, exact sample identities, and why the scores are quantization-fidelity comparisons rather than vendor-score reproductions.

- [ ] **Step 2: Write the external-agent handoff**

Require the capable agent to return:

- Preflight JSON and resolved task/config file.
- Every arm manifest, stdout/stderr, return code, and completion marker.
- Sample manifest and its hash.
- Per-model aggregates, samples, health, distributional records, and checkpoint diagnostics.
- Matrix/pairwise JSON and Markdown reports.
- `scontrol show job/step` or available `srun` environment evidence, host/GPU mapping, start/end times, and observed wall-clock.
- Exact deviations, retries, OOMs, task failures, and first/last relevant log excerpts.

The agent may dynamically rebalance shards after the one-model smoke only by editing a run-local resolved matrix and documenting the reason; it must not change samples, prompts, decoding, metrics, or gates.

- [ ] **Step 3: Run the full local regression suite**

Run:

```bash
python -m pytest pipeline/tests/test_eval_sampling.py \
  pipeline/tests/test_eval_stats.py \
  pipeline/tests/test_eval_health.py \
  pipeline/tests/test_eval_distributional.py \
  pipeline/tests/test_m3_checkpoint_diagnostics.py \
  pipeline/tests/test_m3_quality_eval.py \
  pipeline/tests/test_m3_quality_eval_runner.py \
  pipeline/tests/test_compare.py \
  pipeline/tests/test_lmeval_runner.py \
  pipeline/tests/test_static_checkpoint.py -q
python -m compileall -q pipeline
git diff --check
```

Expected: all selected tests PASS, compile PASS, diff check PASS.

- [ ] **Step 4: Run broader eval regressions**

Run: `python -m pytest pipeline/tests -q`

Expected: PASS, or document dependency-backed skips that already existed before this change. Any new failure blocks handoff.

- [ ] **Step 5: Run a CPU-only CLI fixture smoke**

Use synthetic checkpoint/task fixtures to execute preflight, arm validation, aggregation, and self-comparison without GPU. Expected: complete artifact tree and zero-delta self-report.

- [ ] **Step 6: Update plan status and commit docs**

```bash
git add pipeline/README.md MINIMAX_M3_EVAL_HANDOFF.md BUGS_AND_FIXES.md docs/superpowers/plans/2026-07-12-minimax-m3-vllm-quality-evaluation.md
git commit -m "docs: hand off MiniMax quality evaluation"
```

- [ ] **Step 7: Push the branch for the capable cluster**

Run: `git push origin duy-branch`

Expected: remote `duy-branch` advances to the verified handoff commit.

---

## Completion Evidence

Implementation is locally complete only when Tasks 1-8 pass their stated tests,
the dry-run emits exactly eight correct `srun` arms, the synthetic self-compare
has zero drift/flips, documentation names every artifact, and the branch is
pushed. The full user objective remains incomplete until the capable cluster
returns a validated four-checkpoint production-quality report within the target
wall-clock or provides evidence sufficient to tune the sharding/sample budget.

