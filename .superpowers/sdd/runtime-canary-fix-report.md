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

## Pinned tokenizer readiness follow-up

**Scope:** Readiness is now the exact `tiktoken==0.14.0` `cl100k_base` cache
file `9b5ad71b2ce5302211f9c61530b329a4922fc6a4`, with BPE SHA-256
`223921b76ee99bde995b7ff738513eef100fb51d18c93597a113bcffe865b2a7`.
Stage verifies that file after `get_encoding`, atomically writes the exact
filename-and-hash marker, and retains the broader inventory only as audit
metadata. Canary and full verify both marker contents and recomputed file
digest; unrelated or stale nonempty files cannot satisfy readiness.

**RED/GREEN:** The exact-cache test initially failed because
`CL100K_CACHE_FILE` was absent. It passed after all three manifests received
identical path, filename, digest, and marker environment values. Final
dataset/manifest tests: `27 passed in 1.27s`; full clean-venv AA-LCR suite:
`141 passed in 34.42s`; Ruff and `git diff --check` passed.

## Official CSV schema follow-up

**Root cause:** The pinned CSV's exact header is
`"", document_category, document_set_id, question_id, question, answer,
data_source_filenames, data_source_urls, input_tokens`, but `_load_questions`
read synthetic-only `category` after successfully using `document_category`
for archive paths.

**RED/GREEN:** The primary fixture now uses the exact official headers and a
new full-loading test failed at the missing `category` lookup. The loader now
uses official `document_category` for `Question.category` and official
`answer` for `Question.official_answer`, retaining the old aliases only for
fixture compatibility. Dataset tests: `18 passed in 1.83s`; full clean-venv
AA-LCR suite: `142 passed in 38.46s`; Ruff and `git diff --check` passed.

## Pinned input-token metadata follow-up

**Root cause:** Five upstream v1.1 `input_tokens` values are stale metadata,
not prompt or tiktoken-version drift. The official loader preserves raw
`row["question"]`, whereas the former local helper stripped trailing text.

**Contract:** Candidate prompts and judge answers now preserve raw CSV text.
Only the five pinned published/actual tuples are accepted; new, missing, or
changed discrepancies fail closed. `Question`, the immutable checkpoint
contract, and publication summary retain published versus actual prompt-token
provenance. Targeted dataset/summary tests: `41 passed in 26.23s`; full
clean-venv AA-LCR suite: `147 passed in 43.22s`; Ruff, credential, and
`git diff --check` scans passed.

## Raw official answer follow-up

**RED/GREEN:** An official-schema fixture with trailing answer spaces failed
while the loader used the stripping metadata helper. The production path now
loads official `answer` with `_raw_row_value`, preserving judge input exactly.
Dataset tests: `23 passed in 2.21s`; full clean-venv AA-LCR suite:
`148 passed in 36.43s`; Ruff and `git diff --check` passed.
