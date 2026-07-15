# Final BF16 health fix report

Status: DONE

## Scope

- Implemented final whole-branch review finding 4 only.
- Did not touch replay code or documentation.

## Root cause and fix

`validate_and_merge()` wrote per-model generation health under each merged model
directory, but returned only model labels and pairwise comparisons in
`matrix.json`. A one-model BF16 run has no pairwise comparison, so
`evaluate_gates()` had no health evidence from which to build its advisory.

The merged matrix now includes a top-level `generation_health` mapping keyed by
model label. Paired comparisons reuse the same health records and retain their
existing baseline/candidate advisory representation. When there are no paired
comparisons, the advisory falls back to the merged per-model health, preserving
the sole baseline model's task evidence and finding count.

## TDD evidence

- RED: the BF16-only regression failed with `KeyError: 'generation_health'`.
- GREEN: the regression and the existing paired-advisory regression passed
  together (`3 passed`).
- The regression loads the committed
  `pipeline/configs/minimax_m3_bf16_reasoning_r4.yaml`, constructs its sole-model
  arm layout, injects a GPQA health finding, and verifies matrix plus advisory
  model/task evidence.

## Verification

- `python -m pytest -q pipeline/tests/test_m3_quality_eval.py pipeline/tests/test_m3_quality_evidence.py pipeline/tests/test_static_checkpoint.py`
  - 84 passed.
- `python -m pytest -q pipeline/tests/test_m3_quality_eval.py pipeline/tests/test_m3_quality_eval_runner.py pipeline/tests/test_m3_quality_evidence.py pipeline/tests/test_static_checkpoint.py`
  - 92 passed, 7 failed solely because `bash` is unavailable on this Windows
    host, matching the limitation recorded in the final branch review.
- `python -m ruff check pipeline/m3_quality_eval.py pipeline/tests/test_m3_quality_eval.py`
  - All checks passed.
- `git diff --check`
  - Passed.
