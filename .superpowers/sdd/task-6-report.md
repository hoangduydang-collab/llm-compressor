# Task 6 — Validate and publish complete AA-LCR results

## Implementation

- Added strict `build_summary(checkpoint)` validation. A headline is emitted only
  for the exact 300-unit contract population with successful candidate records,
  300 valid judgments under the pinned judge-contract hash, and no terminal judge
  failures.
- Recompute headline pass@1, question macro accuracy, per-repeat accuracy,
  checkpointed category breakdowns, token distributions, finish reasons, and
  candidate diagnostics from SQLite primitives.
- Persist document categories immutably during `generate_missing`; legacy
  checkpoints publish an explicit unattributed category and limitation instead of
  fabricating category values.
- Added `publish_results(checkpoint, out_dir)`, which refuses existing
  destinations and writes/fsyncs all artifacts in a temporary sibling directory
  before a single rename: `run-manifest.json`, `candidates.jsonl`,
  `judgments.jsonl`, `summary.json`, `report.md`, and `files.sha256`.

## TDD evidence

RED:

```text
py -3.12 -m pytest -q pipeline/tests/test_aa_lcr_v11_summary.py
4 failed
AttributeError: IncompleteRunError/build_summary/publish_results missing
```

An additional RED test for an absent final candidate content failed as expected:

```text
1 failed, 4 deselected
Failed: DID NOT RAISE IncompleteRunError
```

GREEN:

```text
py -3.12 -m pytest -q pipeline/tests/test_aa_lcr_v11_summary.py
6 passed
```

## Verification and review

```text
py -3.12 -m pytest -q pipeline/tests/test_aa_lcr_v11_candidate.py pipeline/tests/test_aa_lcr_v11_checkpoint.py pipeline/tests/test_aa_lcr_v11_dataset.py pipeline/tests/test_aa_lcr_v11_identity.py pipeline/tests/test_aa_lcr_v11_judge.py pipeline/tests/test_aa_lcr_v11_summary.py
87 passed
```

- `git diff --check`: clean.
- Edited-file diagnostics: clean.
- Self-review caught a no-live-calls violation: token counting initially loaded
  `tiktoken`'s remote BPE cache. Publication now uses a documented local
  whitespace-token proxy for answer/reasoning text; API usage fields remain the
  source for prompt and completion distributions.

## Concern

The existing v1.1 checkpoint schema had no category field. New generation records
the categories immutably before candidate work. Historical complete checkpoints
cannot reconstruct category labels from their stored primitives, so their
published category breakdown is explicitly `unattributed` with a limitation
banner; their headline is unaffected.
