> Historical initial implementation report for c6d5e163. Independent review
> required canonical assistant spans and constituent-turn provenance; the final
> implementation and regenerated integration-result.json include those fixes.
> See fix-report.md for the final source validation.

# Task 1 implementation report: mixed calibration bundles

Status: DONE

## Implemented interfaces

- Added `CalibrationConfig.prepared_dataset: str | None = None`.
- `build_calibration_dataset_with_partition` now has an opt-in prepared-bundle
  branch. It validates and loads fixed token rows, preserves bundle order, and
  then uses the existing rank partition calculation. The legacy dataset,
  shuffle, chat-template, and truncation path is unchanged when the field is
  unset.
- Added `pipeline.calibration_bundle`:
  - `tokenizer_identity(tokenizer, revision=None)` fingerprints vocabulary,
    special-token mapping, chat template, and canonical backend tokenizer
    configuration. Hub/local path and revision names are provenance only and do
    not affect the behavior hash. Mutable runtime truncation/padding settings are
    excluded.
  - `load_calibration_bundle(directory, tokenizer, num_samples,
    max_seq_length)` verifies `data.jsonl` SHA-256 before parsing, tokenizer
    identity, configured dimensions, row count, exact row fields, nonnegative
    integer token IDs, and full all-ones masks, then returns a Hugging Face
    `Dataset`.
- Added `pipeline.prepare_calibration` and its module CLI:
  `python -m pipeline.prepare_calibration --config MIX.yaml --output DIR
  --tokenizer ID_OR_PATH [--tokenizer-revision REV] [--trust-remote-code]`.
  It provides strict mix/source validation, deterministic largest-remainder
  quotas, deterministic source and combined shuffles, at most one window per
  document/session, selected-content hashes and source provenance, exact-length
  all-ones-mask rows, early overwrite refusal, and atomic temporary-directory
  publication.
- Added `pipeline.swe_chat`, a strict adapter for the published normalized
  `SALT-NLP/SWE-chat` conversations table. It groups session turns in
  `turn_number` order, drops metadata, merges contiguous assistant thinking,
  response, and tool-use records, preserves `reasoning_content`, parses tool
  arguments as dictionaries, and verifies tool call/result IDs.
- SWE windows structurally choose among substantive assistant text/code/tool-call
  messages throughout a session. One selected session needs at most three full
  tokenizations (session, prefix before anchor, prefix through anchor). The
  resulting exact-length window includes the anchor where possible and retains
  preceding context. It performs no model scoring and makes no GPU allocation.
- Hub `SALT-NLP/SWE-chat` requires explicit `conversations`; normalized local
  JSON/Parquet sources may omit a named dataset config.

## Red/green evidence

Initial failing test command:

```text
OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 PYTHONPATH=src:. \
  .venv-prequant/bin/python -m pytest -q \
  pipeline/tests/test_mixed_calibration.py
```

Initial result before source implementation:

```text
ModuleNotFoundError: No module named 'pipeline.calibration_bundle'
1 error during collection
```

Focused final command used the same environment and test path. Result:

```text
.............                                                            [100%]
13 passed in 2.79s
```

Coverage includes exact mixed counts and lengths, determinism, strict config and
quota behavior, byte corruption, tokenizer mismatch, dimensions, malformed
tokens and masks after recomputing the data hash, rank coverage/order, SWE
thinking/tool reconstruction, a later meaningful anchor after a large tool
output, quota shortage without partial publication, real Hugging Face local JSON
loading, and overwrite refusal before source loading.

Final calibration/config/distributed regression command:

```text
OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 PYTHONPATH=src:. \
  .venv-prequant/bin/python -m pytest -q \
  pipeline/tests/test_mixed_calibration.py \
  pipeline/tests/test_calibration_dataset.py \
  pipeline/tests/test_calibration_partition.py \
  pipeline/tests/test_distributed_quantize_contract.py \
  pipeline/tests/test_mixed_calibration_configs.py
```

Result:

```text
35 passed, 14 warnings in 16.11s
```

The final run also covers the explicit one-window-per-document-ID contract. The
warnings are existing `torch.jit.script_method` deprecation warnings. An
earlier regression attempt found two stale distributed test fixtures missing the
already-existing `mtp_policy` field; root updated those fixtures, and all eight
distributed contract tests pass in the final run.

Lint and compilation:

```text
.venv-prequant/bin/python -m ruff check <owned source/test files>
All checks passed!

.venv-prequant/bin/python -m py_compile <owned Python source files>
exit 0
```

## Real tokenizer integration evidence

The retained CPU integration artifact is
`results/mixed-calibration/20260912-cpu/integration-result.json`. It uses the
pinned `zai-org/GLM-5.3-BF16` tokenizer revision
`304b8051cfb2b260b61ce0cbe330e02a98e73639`, synthetic local generic/SWE inputs,
and no model forward pass or GPU.

Observed result: PASS; 8 windows of 128 tokens, split 4 generic/4 coding-agent.
The AWQ and GPTQ consumer paths produced the identical partition token hash
`6677aa560b0053bf8d7c51911395302ff290c97b641e010c971205b6b8036c24`.
Coding-agent offsets were 4175, 5760, 6497, and 6497 with anchor turns 4, 7, 9,
and 9. All contained assistant markup. Tokenizer identity remained stable after
runtime padding/truncation settings changed.

## Review notes and limits

- No quantization recipes, quantization math, native W4AFP8 export, or source-RTN
  MTP code changed.
- No production dataset download is performed by tests. The gated remote
  `SALT-NLP/SWE-chat` dataset was not downloaded in validation; its published
  normalized schema is covered through local JSON and synthetic Arrow rows.
- The shared worktree also contains root-owned documentation, YAML, config tests,
  integration results, and unrelated `.gitignore`/predictive-PTQ changes. They
  are intentionally excluded from this task's commit.
