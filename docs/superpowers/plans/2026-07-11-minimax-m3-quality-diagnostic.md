# MiniMax-M3 Paired Quality Diagnostic Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Produce a reproducible paired cyankiwi-versus-portable-AWQ quality experiment that identifies the first failing MiniMax-M3 serve boundary and returns a complete, compact evidence bundle through Git.

**Architecture:** Keep quality scoring and evidence classification in a pure-Python module, install dormant runtime diagnostics for every MiniMax-M3 checkpoint, and retain W4A8 execution patches behind the existing scheme guard. A shell runner launches the reference and candidate sequentially in eager mode, while a committed runbook defines the remote agent's preflight and return contract.

**Tech Stack:** Python 3.12, pytest, Bash, vLLM offline `LLM`, Torch, Safetensors, JSON/JSONL, Slurm-compatible cluster utilities.

## Global Constraints

- Quality work only; do not resume CUDA-graph RCA in this plan.
- Use eager execution for the paired comparison.
- Do not re-quantize, delete, or duplicate a checkpoint.
- Do not modify vLLM behavior to repair a suspected loader boundary before the paired evidence identifies it.
- Keep raw generations and provenance; a narrative summary is not sufficient.
- Never commit checkpoints, secrets, complete environment dumps, or unbounded logs.
- The reference and candidate must use the same commit, environment, node, topology, TP/EP settings, prompt suite, and sampling settings.
- Stop the live matrix when the cyankiwi reference is not a valid quality baseline.
- Runtime deviations are allowed when necessary, but every deviation must be recorded.

---

### Task 1: Pure output-quality assessment

**Files:**
- Create: `pipeline/m3_quality_evidence.py`
- Create: `pipeline/tests/test_m3_quality_evidence.py`

**Interfaces:**
- Produces: `M3_QUALITY_CASES: tuple[QualityCase, ...]`
- Produces: `assess_output(text: str, expected_any: tuple[str, ...]) -> dict[str, object]`
- Produces: `assess_quality_outputs(outputs: list[str]) -> dict[str, object]`
- Produces: `classify_pair(reference: dict, candidate: dict, evidence: dict) -> dict[str, object]`
- Consumes later: `pipeline.serve_verify.verify_serve` uses the prompt definitions and assessment.

- [ ] **Step 1: Write failing quality tests**

```python
def test_assess_output_accepts_factual_answer():
    result = assess_output(" Paris.", ("paris",))
    assert result["passed"] is True
    assert result["repetitive"] is False


def test_assess_output_rejects_concatenated_repetition():
    result = assess_output("arring" * 20, ("paris",))
    assert result["passed"] is False
    assert result["repetitive"] is True


def test_assess_output_rejects_token_repetition():
    result = assess_output("seringk " * 20, ("paris",))
    assert result["passed"] is False
    assert result["repetitive"] is True


def test_reference_failure_stops_pair():
    result = classify_pair(
        {"quality_ok": False},
        {},
        {},
    )
    assert result["verdict"] == "invalid_reference"
```

- [ ] **Step 2: Run tests and confirm the module is missing**

Run:

```bash
PYTHONPATH="$PWD" pytest -q pipeline/tests/test_m3_quality_evidence.py
```

Expected: collection fails because `pipeline.m3_quality_evidence` does not exist.

- [ ] **Step 3: Implement deterministic prompt and repetition assessment**

```python
@dataclass(frozen=True)
class QualityCase:
    case_id: str
    prompt: str
    expected_any: tuple[str, ...]


M3_QUALITY_CASES = (
    QualityCase("capital_france", "The capital of France is", ("paris",)),
    QualityCase(
        "arithmetic_2_plus_2",
        "What is 2 + 2? Answer with only the number.",
        ("4", "four"),
    ),
)
```

Detect whitespace-token domination and consecutive character chunks repeated
three or more times. Return raw text, normalized text, expected-answer match,
repetition reasons, and `passed`. Aggregate cases without discarding raw
outputs.

- [ ] **Step 4: Add decision-tree tests and implementation**

Cover these exact verdicts:

- `invalid_reference`
- `candidate_quality_pass`
- `lm_head_boundary`
- `shared_expert_boundary`
- `attention_indexer_boundary`
- `inconclusive_missing_evidence`

The classifier must prefer invalid reference, then candidate pass, then missing
required diagnostics, then `lm_head`, then shared expert, then
attention/indexer.

- [ ] **Step 5: Run focused tests**

Run:

```bash
PYTHONPATH="$PWD" pytest -q pipeline/tests/test_m3_quality_evidence.py
```

Expected: all tests pass.

- [ ] **Step 6: Commit**

```bash
git add pipeline/m3_quality_evidence.py pipeline/tests/test_m3_quality_evidence.py
git commit -m "test: classify MiniMax-M3 quality evidence"
```

---

### Task 2: MiniMax-M3-wide diagnostics and bounded parameter fingerprints

**Files:**
- Modify: `pipeline/slurm/patch_vllm_m3_serve.py`
- Modify: `pipeline/serve_verify.py`
- Modify: `pipeline/tests/test_patch_vllm_m3_serve.py`
- Modify: `pipeline/tests/test_serve_verify_m3_env.py`

**Interfaces:**
- Produces: `ensure_m3_quality_diagnostics(*, apply: bool = True) -> str`
- Produces log marker: `M3_PARAM_FINGERPRINT# {json}`
- Produces log marker: `M3_PARAM_FINGERPRINT_SUMMARY# {json}`
- Retains: `ensure_vllm_m3_patches()` only for W4A8.
- Consumes: environment flags `M3_LOAD_AUDIT`, `M3_MOE_PROBE`, and `M3_PARAM_FINGERPRINT`.

- [ ] **Step 1: Write failing installation-boundary tests**

Use monkeypatched diagnostic and W4A8 patch callables to prove:

```python
def test_m3_w4a16_installs_diagnostics_without_w4a8_patches(...):
    ...
    assert calls == ["quality_diagnostics"]


def test_m3_w4a8_installs_diagnostics_and_required_patches(...):
    ...
    assert calls == ["quality_diagnostics", "w4a8_patches"]
```

Extract the pre-vLLM setup into a CPU-testable helper rather than booting
`LLM` in these tests.

- [ ] **Step 2: Run the focused tests and verify they fail**

Run:

```bash
PYTHONPATH="$PWD" pytest -q pipeline/tests/test_patch_vllm_m3_serve.py pipeline/tests/test_serve_verify_m3_env.py
```

Expected: failures show that diagnostics are still inside the W4A8 branch and
the fingerprint interface is absent.

- [ ] **Step 3: Move dormant diagnostics outside the W4A8 guard**

For MiniMax-M3 checkpoints:

```python
diagnostic_status = ensure_m3_quality_diagnostics()
if _is_w4a8_moe_scheme(_read_quant_config(ckpt)):
    ensure_vllm_m3_patches()
```

The helper installs the load audit, MoE probe, and fingerprint block before
worker creation. Non-M3 checkpoints remain untouched. Explicitly enabled
diagnostics fail the serve preflight when their target model class cannot be
found; dormant diagnostics remain best effort.

- [ ] **Step 4: Add bounded parameter fingerprints to the injected block**

After successful model loading, inspect only selected named parameters:

- `lm_head`;
- shared and routed experts in layers 3 and 59;
- q/k/v and indexer parameters in layers 3 and 59.

For each selected local shard emit one JSON record containing case, rank, name,
category, layer, dtype, shape, numel, sampled finite fraction, sampled
`abs_max`, mean, standard deviation, and SHA-256 digest. Select at most 256
evenly spaced elements and never clone or gather a full parameter. Emit a
summary listing found and missing categories.

- [ ] **Step 5: Execute the injected diagnostic against fake model classes**

The regression test must `exec` the injected block with fake vLLM logger,
Torch, model classes, named parameters, and loader generators. Verify:

- source-to-target matches are recorded;
- fingerprint JSON is emitted;
- missing categories appear in the summary;
- original `weight_loader` attributes are restored after success and failure;
- the block remains idempotent.

- [ ] **Step 6: Run focused tests**

Run:

```bash
PYTHONPATH="$PWD" pytest -q pipeline/tests/test_patch_vllm_m3_serve.py pipeline/tests/test_serve_verify_m3_env.py
```

Expected: all tests pass.

- [ ] **Step 7: Commit**

```bash
git add pipeline/slurm/patch_vllm_m3_serve.py pipeline/serve_verify.py pipeline/tests/test_patch_vllm_m3_serve.py pipeline/tests/test_serve_verify_m3_env.py
git commit -m "feat: instrument MiniMax-M3 quality boundaries"
```

---

### Task 3: Integrate the quality prompt suite into offline serve verification

**Files:**
- Modify: `pipeline/serve_verify.py`
- Create: `pipeline/tests/test_serve_verify_quality.py`

**Interfaces:**
- Consumes: `M3_QUALITY_CASES` and `assess_quality_outputs`.
- Produces report fields: `generation_completed`, `quality_cases`, and `quality_ok`.
- Retains report fields: `sample_prompt`, `sample_output`, `sane_output`, and infrastructure `ok`.

- [ ] **Step 1: Write failing report tests with fake LLM output**

Test that MiniMax-M3 generates the fixed prompt suite in one greedy batch,
records every raw output, and rejects `arringarring...`. Test that a non-M3
checkpoint retains its configured single prompt.

- [ ] **Step 2: Run tests and verify the old nonempty gate fails them**

Run:

```bash
PYTHONPATH="$PWD" pytest -q pipeline/tests/test_serve_verify_quality.py
```

Expected: failure because `quality_cases` and `quality_ok` are absent.

- [ ] **Step 3: Implement report separation**

For MiniMax-M3, call:

```python
prompts = [case.prompt for case in M3_QUALITY_CASES]
outputs = llm.generate(prompts, SamplingParams(max_tokens=64, temperature=0.0))
quality = assess_quality_outputs([item.outputs[0].text for item in outputs])
```

Set `generation_completed` from nonempty responses. Set `ok` to successful
load plus completed generation so existing infrastructure automation does not
change meaning unexpectedly. Set `quality_ok` independently. Preserve the
first case in the legacy sample fields.

- [ ] **Step 4: Run tests**

Run:

```bash
PYTHONPATH="$PWD" pytest -q pipeline/tests/test_m3_quality_evidence.py pipeline/tests/test_serve_verify_quality.py
```

Expected: all tests pass.

- [ ] **Step 5: Commit**

```bash
git add pipeline/serve_verify.py pipeline/tests/test_serve_verify_quality.py
git commit -m "feat: separate MiniMax-M3 serve quality from readiness"
```

---

### Task 4: Paired runner and compact evidence bundle

**Files:**
- Create: `pipeline/slurm/test_m3_paired_quality.sh`
- Extend: `pipeline/m3_quality_evidence.py`
- Extend: `pipeline/tests/test_m3_quality_evidence.py`
- Create: `pipeline/tests/test_m3_paired_quality_runner.py`

**Interfaces:**
- Shell inputs: `REFERENCE_CKPT`, `CANDIDATE_CKPT`, `MODEL_ID`, `CONFIG`, `RESULTS_ROOT`, `RUN_ID`, and `DRY_RUN`.
- Python CLI: `python -m pipeline.m3_quality_evidence bundle --run-dir PATH`.
- Produces: `run_manifest.json`, per-case reports, extracted JSONL diagnostics, `comparison.json`, `artifact_index.json`, and bounded log excerpts.

- [ ] **Step 1: Write failing dry-run and classifier tests**

The dry-run test invokes the shell runner with temporary checkpoint stubs and
asserts:

- case order is reference then candidate;
- both commands use eager mode, TP=8, EP enabled, identical model length and GPU
  utilization;
- all three diagnostic flags are enabled;
- no GPU cleanup, vLLM launch, or NFS mutation occurs;
- the manifest contains the current commit and exact commands.

- [ ] **Step 2: Run tests and verify the runner is absent**

Run:

```bash
PYTHONPATH="$PWD" pytest -q pipeline/tests/test_m3_paired_quality_runner.py pipeline/tests/test_m3_quality_evidence.py
```

Expected: failure because the runner and bundle CLI do not exist.

- [ ] **Step 3: Implement the sequential runner**

Use a per-case symlink named `checkpoint` under the result directory so
`pipeline.run` writes metadata and `serve_report.json` into the case
directory without modifying the source checkpoint. Live cases run:

```bash
python -m pipeline.run --config "$CONFIG" --stage serve \
  --checkpoint "$CASE_DIR/checkpoint" \
  --set model.id="$MODEL_ID" \
  --set serve.tensor_parallel_size=8 \
  --set serve.enable_expert_parallel=true \
  --set serve.block_size=128 \
  --set serve.kv_cache_dtype=fp8 \
  --set serve.max_model_len="$MAX_MODEL_LEN" \
  --set serve.gpu_memory_utilization="$GPU_UTIL" \
  --set serve.enforce_eager=true \
  --set serve.disable_custom_all_reduce=true \
  --set eval.enabled=false
```

Run the existing owned-process GPU cleanup before each live case. Stop after an
invalid reference. Preserve partial evidence on every exit path.

- [ ] **Step 4: Implement provenance and extraction**

Collect allowlisted version commands, Git status/commit, scheduler identifiers,
`nvidia-smi` inventory/topology, config/index SHA-256, command arrays, timing,
return codes, retries, and deviations. Parse prefixed diagnostics into JSONL,
retain raw outputs in reports, create bounded warning/error excerpts, and index
large external logs with paths, sizes, and hashes.

Do not serialize tokens, credentials, arbitrary environment variables, or full
package/environment dumps.

- [ ] **Step 5: Implement bundle validation and comparison**

The bundle CLI rejects a complete status when required provenance or diagnostic
categories are missing. It writes `comparison.json` using `classify_pair`.
Missing required evidence yields `inconclusive_missing_evidence`, never a
guessed loader verdict.

- [ ] **Step 6: Run tests and dry-run manually**

Run:

```bash
PYTHONPATH="$PWD" pytest -q pipeline/tests/test_m3_quality_evidence.py pipeline/tests/test_m3_paired_quality_runner.py
DRY_RUN=1 RESULTS_ROOT=/tmp/m3-paired-quality-dryrun \
  bash pipeline/slurm/test_m3_paired_quality.sh
```

Expected: tests pass; dry-run creates a two-case manifest and no GPU processes.

- [ ] **Step 7: Commit**

```bash
git add pipeline/m3_quality_evidence.py pipeline/slurm/test_m3_paired_quality.sh pipeline/tests/test_m3_quality_evidence.py pipeline/tests/test_m3_paired_quality_runner.py
git commit -m "feat: add MiniMax-M3 paired quality evidence runner"
```

---

### Task 5: Remote-agent runbook and return checklist

**Files:**
- Create: `MINIMAX_M3_QUALITY_RUNBOOK.md`
- Modify: `MINIMAX_M3_HANDOFF.md`
- Modify: `BUGS_AND_FIXES.md`

**Interfaces:**
- Consumes: the exact paired runner command and bundle schema from Task 4.
- Produces: a copyable preflight/run/return handoff for the GPU-cluster agent.

- [ ] **Step 1: Write the runbook**

Include:

- objective and non-goals;
- prerequisite code commit and clean-worktree check;
- checkpoint and environment preflight;
- one exact live command;
- invariants versus runtime-adaptable details;
- stop conditions and retry/deviation rules;
- required result files;
- large-artifact retention and SHA-256 requirements;
- exact `git add` paths and result commit-message format;
- a final yes/no return checklist covering all six questions in the design's
  remote-agent handoff contract.

- [ ] **Step 2: Update current-status documentation**

Record that routed aliases were a real but insufficient repair, that the next
experiment is the paired eager comparison, and that CUDA-graph work remains
deferred until quality is resolved.

- [ ] **Step 3: Self-review the instructions**

Verify that an agent can run the experiment without reading conversation
history, while still being empowered to resolve scheduler/path/runtime-only
issues and record deviations.

- [ ] **Step 4: Commit**

```bash
git add MINIMAX_M3_QUALITY_RUNBOOK.md MINIMAX_M3_HANDOFF.md BUGS_AND_FIXES.md
git commit -m "docs: hand off MiniMax-M3 paired quality run"
```

---

### Task 6: Verification and handoff commit audit

**Files:**
- Verify all files changed in Tasks 1-5.

**Interfaces:**
- Produces: a verified code commit ready to push and run on the GPU cluster.

- [ ] **Step 1: Run focused tests**

```bash
PYTHONPATH="$PWD" pytest -q pipeline/tests/test_m3_quality_evidence.py pipeline/tests/test_m3_paired_quality_runner.py pipeline/tests/test_patch_vllm_m3_serve.py pipeline/tests/test_serve_verify_m3_env.py pipeline/tests/test_serve_verify_quality.py pipeline/tests/test_reexport_minimax_m3_vllm.py
```

Expected: all focused tests pass.

- [ ] **Step 2: Run syntax and shell checks**

```bash
python -m py_compile pipeline/m3_quality_evidence.py pipeline/serve_verify.py pipeline/slurm/patch_vllm_m3_serve.py
bash -n pipeline/slurm/test_m3_paired_quality.sh
git diff --check
```

Expected: every command exits 0.

- [ ] **Step 3: Run the final dry-run**

```bash
rm -rf /tmp/m3-paired-quality-final
DRY_RUN=1 RESULTS_ROOT=/tmp/m3-paired-quality-final \
  bash pipeline/slurm/test_m3_paired_quality.sh
python -m pipeline.m3_quality_evidence bundle \
  --run-dir "$(find /tmp/m3-paired-quality-final -mindepth 1 -maxdepth 1 -type d | head -1)"
```

Expected: the manifest has two identical-envelope cases, the bundle is marked
`dry_run`, and no live GPU action is reported.

- [ ] **Step 4: Audit repository state and handoff provenance**

```bash
git status --short
git log -6 --oneline
```

Expected: only intended changes are present, and the runbook names the final
diagnostic commit.

- [ ] **Step 5: Push only after local verification**

```bash
git push origin duy-branch
```

Expected: the remote branch contains the diagnostic implementation and the
GPU-cluster agent can begin with the committed runbook.
