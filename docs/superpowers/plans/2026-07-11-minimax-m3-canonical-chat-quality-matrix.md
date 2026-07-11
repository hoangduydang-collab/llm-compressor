# MiniMax-M3 Canonical Chat Quality Matrix Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build and hand off a four-node canonical-chat quality comparison of cyankiwi and the portable W4A8 MiniMax-M3 checkpoint through offline and HTTP serving.

**Architecture:** The existing offline verifier will render official MiniMax-M3 chat prompts before generation. A standard-library evidence module will normalize offline reports and OpenAI chat responses into one arm schema and aggregate four independent arms. Shell runners will execute one arm per node and a submission helper will launch or dry-run all four arms.

**Tech Stack:** Python 3.12 standard library, vLLM offline `LLM`, OpenAI-compatible vLLM HTTP server, Bash, Slurm, pytest-style CPU tests.

## Global Constraints

- Quality work remains ahead of CUDA-graph RCA.
- All four arms use eager TP8+EP, block size 128, FP8 KV cache, `max_model_len=2048`, GPU utilization 0.85, and disabled custom all-reduce.
- Both interfaces use the same two user messages, 64 output tokens, temperature 0, and `thinking_mode="disabled"`.
- `M3_LOAD_AUDIT=0`, `M3_MOE_PROBE=0`, and `M3_PARAM_FINGERPRINT=0` in every arm.
- No checkpoint edits, re-quantization, candidate repair, CUDA graphs, or silent retries.

---

### Task 1: Canonical offline chat prompts

**Files:**
- Modify: `pipeline/serve_verify.py`
- Modify: `pipeline/tests/test_serve_verify_quality.py`

**Interfaces:**
- Consumes: `llm.get_tokenizer().apply_chat_template(messages, tokenize=False, add_generation_prompt=True, thinking_mode="disabled")`
- Produces: MiniMax-M3 `quality_cases[*].rendered_prompt` and `prompt_mode="chat_template"`

- [ ] **Step 1: Write the failing test**

Extend `_FakeLLM` with a tokenizer that records template calls and returns distinct rendered prompts. Assert each MiniMax case is generated from one rendered prompt, the template receives one user message plus `thinking_mode="disabled"`, and each report case retains its rendered prompt. Keep the existing non-M3 raw-prompt assertion.

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest -q pipeline/tests/test_serve_verify_quality.py`

Expected: FAIL because `_run_generation_smoke` still sends raw prompts and does not record rendered prompts.

- [ ] **Step 3: Write minimal implementation**

Add a small local rendering helper in `_run_generation_smoke`. For MiniMax-M3 only, obtain the tokenizer, render each `QualityCase.prompt` as one user message, and generate sequentially from the rendered strings. Preserve the original user prompt as `prompt`, record `rendered_prompt`, and set `prompt_mode="chat_template"` in the result.

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest -q pipeline/tests/test_serve_verify_quality.py pipeline/tests/test_m3_quality_evidence.py`

Expected: all tests pass.

### Task 2: Unified arm evidence and matrix classification

**Files:**
- Create: `pipeline/m3_chat_quality.py`
- Create: `pipeline/tests/test_m3_chat_quality.py`

**Interfaces:**
- Consumes: offline `serve_report.json` or two saved OpenAI chat response JSON files
- Produces: `arm_report.json`, `arm_manifest.json`, and matrix `comparison.json`

- [ ] **Step 1: Write failing classifier tests**

Cover extraction of `choices[0].message.content`, `finish_reason`, response errors, semantic assessment through `assess_quality_outputs`, passing reference/candidate pairs for both interfaces, candidate failure with valid references, interface disagreement, and missing arms.

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest -q pipeline/tests/test_m3_chat_quality.py`

Expected: import failure because `pipeline.m3_chat_quality` does not exist.

- [ ] **Step 3: Implement the standard-library module**

Define the fixed arm names, `normalize_http_responses`, `normalize_offline_report`, `classify_matrix`, manifest creation, per-arm bundle creation, and CLI subcommands `manifest`, `bundle-arm`, and `aggregate`. Classification must distinguish infrastructure failure, invalid reference, candidate quality failure, interface disagreement, complete candidate pass, and missing evidence.

- [ ] **Step 4: Run focused tests**

Run: `python -m pytest -q pipeline/tests/test_m3_chat_quality.py pipeline/tests/test_m3_quality_evidence.py`

Expected: all tests pass.

### Task 3: One-arm runtime runner

**Files:**
- Create: `pipeline/slurm/test_m3_chat_quality_arm.sh`
- Create: `pipeline/tests/test_m3_chat_quality_runner.py`

**Interfaces:**
- Consumes: `MATRIX_ID`, one `ARM`, checkpoint paths, environment paths, and result roots
- Produces: one full-log arm directory and one compact evidence arm directory

- [ ] **Step 1: Write failing dry-run tests**

For all four arms, create temporary checkpoint metadata and run with `DRY_RUN=1`. Assert checkpoint/interface selection, eager envelope, diagnostics off, chat parameters, unique paths, and no GPU/server command execution. Assert unknown arms fail before creating evidence.

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest -q pipeline/tests/test_m3_chat_quality_runner.py`

Expected: FAIL because the runner is absent.

- [ ] **Step 3: Implement the runner**

Validate one arm and selected checkpoint, write the manifest, then in live mode record versions/GPU topology. Offline arms invoke `pipeline.run --stage serve`. HTTP arms start `run_vllm_http_serve_smoke.sh` with eager 2048/0.85 settings and unique log/PID paths, wait for `/health`, post both canonical chat requests with `thinking_mode=disabled`, save raw JSON, stop only their own server, and always call `bundle-arm` from a cleanup trap.

- [ ] **Step 4: Verify runner tests and shell syntax**

Run: `bash -n pipeline/slurm/test_m3_chat_quality_arm.sh && python -m pytest -q pipeline/tests/test_m3_chat_quality_runner.py`

Expected: all tests pass.

### Task 4: Parallel submission and handoff

**Files:**
- Create: `pipeline/slurm/submit_m3_chat_quality_matrix.sh`
- Create: `pipeline/tests/test_submit_m3_chat_quality_matrix.py`
- Modify: `MINIMAX_M3_QUALITY_RUNBOOK.md`
- Modify: `MINIMAX_M3_HANDOFF.md`
- Modify: `BUGS_AND_FIXES.md`

**Interfaces:**
- Consumes: shared matrix ID, optional Slurm options, and the per-arm runner
- Produces: four independent Slurm jobs or four dry-run commands, plus an executor return contract

- [ ] **Step 1: Write failing submission test**

Run the helper in dry-run mode and assert exactly four commands with the fixed arm names, one node/eight GPUs each, one shared matrix ID, and distinct output/error logs.

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest -q pipeline/tests/test_submit_m3_chat_quality_matrix.py`

Expected: FAIL because the submission helper is absent.

- [ ] **Step 3: Implement helper and documentation**

Implement `DRY_RUN=1` command emission and live `sbatch --parsable` submission. Update the runbook with preflight, submission, monitoring, aggregation, evidence checks, commit/push instructions, and dynamic scheduler allowances. Update the handoff and bug chronicle to mark raw completion invalid and name this matrix as the active quality boundary.

- [ ] **Step 4: Verify all focused tests**

Run the full focused MiniMax-M3 suite, `bash -n` on all new scripts, `python -m compileall`, `git diff --check`, and dry-run the matrix.

Expected: all tests and checks pass; dry-run emits four independent arms and no live commands.

- [ ] **Step 5: Commit and push**

Commit the implementation, tests, and handoff on `duy-branch`; push to `origin/duy-branch`; verify local and remote commit IDs match and the worktree is clean.
