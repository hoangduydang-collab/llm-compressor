# AA-LCR Runtime Canary Fix Report

## Scope

This change repairs only the pinned AA-LCR archive-member contract and the
offline `Start-AaLcrJob` runbook race. It does not alter the dataset revision,
archive digest, judge contract, dependency lock, or generic unexpected-member
policy. No live API, Kubernetes, or model calls were made.

## RED evidence

- `test_safe_extract_recovers_unflagged_utf8_name_and_normalizes_nfc` failed
  because the unflagged filename was decoded as CP437 mojibake and rejected as
  unexpected.
- `test_safe_extract_rejects_canonical_name_collision` failed because
  decomposed and NFC-equivalent unflagged names were treated as distinct.
- `test_prepare_dataset_allows_only_pinned_official_unreferenced_member` failed
  because the explicit archive allowlist constant did not exist.
- `test_runbook_rejects_nonterminal_same_run_job_before_creation` failed
  because the runbook made no same-run Job query.
- `test_runbook_waits_for_created_pod_before_following_job_logs` failed because
  the runbook followed logs immediately after Job creation.

## GREEN evidence

- The three new dataset tests passed: `3 passed`.
- The two new runbook semantic tests passed: `2 passed`.
- Focused dataset and CLI tests passed: `37 passed`.
- All AA-LCR tests passed in a fresh Python 3.12 venv:
  `138 passed in 38.06s`.
- Ruff passed for `pipeline/aa_lcr_v11.py` and all AA-LCR test modules.
- The credential scan found no credential literals. The scan excludes the two
  pre-existing `sk-do-not-record` / `sk-test-do-not-serialize` negative-test
  sentinels, which are asserted not to enter artifacts.
- `git diff --check` passed.

## Medium review follow-up

**Root cause:** The first decoder applied CP437-byte-to-UTF-8 recovery to every
unflagged ZIP name. A valid CP437 name whose bytes are also valid UTF-8 could
therefore be silently renamed outside the pinned archive contract.

**RED:** Two new tests failed: an ambiguous valid unflagged CP437 member was
renamed without `expected_members`, and rejected when its normal CP437 spelling
was explicitly expected. The pre-existing pinned-style recovery test continued
to specify the recovered NFC name as expected.

**GREEN:** The extractor now computes the NFC-safe normal `zipfile` candidate
first. It preserves that candidate with no expected-members contract, or when
the normal candidate is expected. Only after a normal mismatch does it try the
unflagged CP437-byte-to-UTF-8 candidate, selecting it only if that recovered
NFC-safe path is explicitly expected. Raw-header NUL validation and selected
canonical collision checks are unchanged.

**Final verification:** Dataset/CLI tests: `39 passed`. Full clean-venv AA-LCR
suite: `140 passed in 37.80s`; Ruff passed. `git diff --check` passed before
the separate follow-up commit.

## Tiktoken runtime reproducibility follow-up

**Root cause:** `tiktoken==0.14.0` fetches the `cl100k_base` vocabulary lazily.
The local real-prepare CA failure proved that the lock installation alone does
not stage it, while canary and full Jobs each have a fresh temporary cache.

**RED/GREEN:** Static manifest and runbook tests initially failed because no
shared `TIKTOKEN_CACHE_DIR`, prefetch, SHA-256 inventory, or fail-closed
consumer check existed. They pass after stage now materializes the vocabulary
on every run and consumers require a nonempty shared artifact before the
runner. Dataset/manifest tests: `27 passed in 1.19s`. Full clean-venv AA-LCR
suite: `141 passed in 39.34s`; Ruff passed. The credential scan and
`git diff --check` passed.
