# Mixed calibration implementation plan

> **For agentic workers:** Use subagent-driven-development to implement and review the source task; root handles integration documentation and final validation.

**Goal:** Offer reusable mixed generic/coding-agent calibration for both AWQ and GPTQ.
**Architecture:** CPU preparation produces a verified token bundle; shared calibration loader consumes it through an opt-in path.
**Tech Stack:** Python, Hugging Face datasets/transformers, existing YAML configuration.

## Global Constraints

- Existing single-source calibration behavior is unchanged when prepared_dataset is unset.
- Both AWQ and GPTQ reuse the same prepared bundle and existing rank partitioning.
- Quantization recipes, native W4AFP8 export and source-rtn MTP are unchanged.
- No model-scoring passes, no GPU allocations, no entropy experiment.
- Preparation is CPU-only; output uses actual fixed-length token windows.
- Preserve unrelated .gitignore, predictive_ptq/ and results/predictive-ptq/ changes.

### Task 1: Prepare and consume mixed calibration bundles

**Files:** pipeline/prepare_calibration.py (CLI/mixture), pipeline/calibration_bundle.py
(bundle verification and tokenizer identity; optional focused SWE adapter file),
pipeline/calibration.py, pipeline/config.py, pipeline/tests/test_mixed_calibration.py.
Root owns docs and example YAML configs, not source files.

**Interfaces:** calibration.prepared_dataset: str | None = None. Directory contains
data.jsonl (input_ids, attention_mask only) and manifest.json. Shared loader loads verified
tokens without retemplating/retruncating, preserving bundle order, then partitions normally.
Preparation CLI: python -m pipeline.prepare_calibration --config MIX.yaml --output DIR
--tokenizer ID_OR_PATH [--tokenizer-revision REV] [--trust-remote-code].
Mix YAML has num_samples, max_seq_length, seed, sources. Each source has name, weight,
dataset_id, dataset_split, optional dataset_config_name, dataset_revision, dataset_data_files,
format (messages, text, swe_chat), optional column (messages/text column) and id_column.
Defaults for source split=train, format=messages, column=messages; text defaults column=text.
No automatic production dataset download in tests. Use existing HF APIs and template support.

- [ ] Add meaningful failing tests for mixed fixed-token counts, determinism, corruption,
  tokenizer mismatch, config parsing, SWE session/tool reconstruction and rank coverage.
- [ ] Implement strict mix validation and deterministic largest-remainder integer quotas;
  weights normalize to requested windows; reject positive sources rounded to zero.
- [ ] Load HF datasets using ID/config/revision/data_files. For SWE-chat, use conversations
  config (explicit in YAML), gather lightweight session_id/turn_number index, shuffle session
  candidates deterministically, retrieve each selected session in turn order. Do not shuffle
  individual turns. Avoid loading all content into Python memory. At most one full window
  per eligible document/session; skip short/empty docs, fail clearly if quota unmet.
- [ ] Normalize SWE user/assistant/tool_use/tool_result rows to tokenizer messages. Keep
  reasoning text. Drop metadata. Use function tool calls with parsed argument dictionaries,
  matching IDs and tool results. Fail on malformed required tool fields/unknown role.
  Adapt only the published normalized table, not multiple raw agent transcript formats.
- [ ] Tokenize entire eligible sessions on CPU, select seeded offset, slice exact length.
  SWE-chat selection MUST span session positions, not default to prefixes: identify
  substantive assistant text/code/tool-call turns, choose among eligible anchors across
  the session, retain preceding context and meaningful assistant content in each window.
  Exclude metadata-only/tool-output-only stretches. Structural selection only, no model
  scoring. Record anchor turn and offset. Test a long synthetic trace where useful later
  work is selected, including a prefix made of boilerplate/large tool output.
  Shuffle combined windows deterministically. Record source ID, document/session ID, offset,
  source content identity, realized counts and source specifications. Pin/record resolved HF
  revisions when practical; always record selected content hashes. Use tokenizer identity
  hashing vocabulary, special token mapping and chat template. Verify byte hashes before read.
- [ ] Write bundle directory without overwriting an existing output; avoid a partially valid
  bundle on failure (temporary directory then publish or manifest written last).
- [ ] Add opt-in prepared_dataset config field and shared loader path, verify tokenizer,
  num_samples, max_seq_length, nonnegative integer tokens, full all-ones masks and row count;
  reject malformed/corrupt bundles before oneshot. Existing legacy branch remains unchanged.
- [ ] Run focused CPU tests and relevant config/distributed contract regressions with
  OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 PYTHONPATH=src:. .venv-prequant/bin/python -m pytest.
  Do not run tests/llmcompressor while root writes files (its file-count guard races).
- [ ] Commit only owned files, self-review, write implementation/test evidence report.

### Task 2: Examples, usage and completion (root)

- [ ] Add pipeline/configs/calibration/glm53_generic_agentic_mix.yaml with 50/50 weights,
  256x2048 windows and seed42. Sources UltraChat train_sft/messages and SWE-chat
  conversations/train/swe_chat. Document gated access and local parquet alternative.
- [ ] Add native AWQ/GPTQ full mixed example configs by retaining base recipes and using
  the same prepared_dataset directory, distinct run/output/offload paths. Test config parity.
- [ ] Document prepare/consume commands, weighted quotas, short-session policy, provenance,
  resource limits, tokenizer verification and source access. No full-run launch authorization.
- [ ] Inspect task diff and independent review; fix important findings, test changed behavior.
- [ ] Commit docs/evidence, review final feature diff, push duy-branch using dedicated
  hoangduydang-collab credential helper and verify remote SHA. Preserve unrelated files.
