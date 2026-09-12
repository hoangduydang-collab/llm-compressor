# Mixed calibration correctness fix report

Status: DONE

## Changes

- Normalized SWE messages now retain every original turn number merged into each
  message. Manifest samples record `anchor_turns`, `anchor_message_index`, the
  selected content/reasoning/tool-name field, optional tool-call index, and
  canonical half-open character/token spans.
- SWE windows render one canonical full session and one sentinel-marked full
  session. Removing preserved alphanumeric sentinels must reproduce the canonical
  rendering exactly. The canonical text is tokenized once with required character
  offsets, and every selected window is checked to overlap the substantive
  assistant span.
- The window begins about one-third of a window before the selected anchor when
  available. Generic source behavior is unchanged.

## Focused test

Command:

```text
OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 PYTHONPATH=src:. \
  .venv-prequant/bin/python -m pytest -q \
  pipeline/tests/test_mixed_calibration.py
```

Raw output after the final fixture strengthening:

```text
.................                                                        [100%]
17 passed in 2.63s
```

## Calibration/config/distributed regression set

Command:

```text
OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 PYTHONPATH=src:. \
  .venv-prequant/bin/python -m pytest -q \
  pipeline/tests/test_mixed_calibration.py \
  pipeline/tests/test_calibration_dataset.py \
  pipeline/tests/test_calibration_partition.py \
  pipeline/tests/test_distributed_quantize_contract.py \
  pipeline/tests/test_mixed_calibration_configs.py
```

Raw output:

```text
......................................                                   [100%]
=============================== warnings summary ===============================
pipeline/tests/test_distributed_quantize_contract.py: 14 warnings
  /home/e/e1129930/llm-compressor/.venv-prequant/lib/python3.12/site-packages/torch/jit/_script.py:365: DeprecationWarning: `torch.jit.script_method` is deprecated. Please switch to `torch.compile` or `torch.export`.
    warnings.warn(

-- Docs: https://docs.pytest.org/en/stable/how-to/capture-warnings.html
38 passed, 14 warnings in 18.51s
```

## Static checks

Commands:

```text
.venv-prequant/bin/python -m ruff check pipeline/swe_chat.py \
  pipeline/prepare_calibration.py pipeline/tests/test_mixed_calibration.py
.venv-prequant/bin/python -m py_compile pipeline/swe_chat.py \
  pipeline/prepare_calibration.py
git diff --check -- pipeline/swe_chat.py pipeline/prepare_calibration.py \
  pipeline/tests/test_mixed_calibration.py
```

Raw outputs:

```text
All checks passed!
```

`py_compile` and `git diff --check` exited 0 with no output.

## Integration handoff

Root independently reported that the corrected real GLM tokenizer CLI smoke and
the stricter decoded substantive-payload verifier passed. Root owns the retained
integration artifacts and documentation updates.
