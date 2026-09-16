# AA-LCR v1.1 Final Review Fix Report

Date: 2026-09-13

## Outcome

- Added an explicit `run --canary --limit 1 --repeats 1` mode.
  It always validates/reuses the immutable judge preflight before candidate
  generation, persists one candidate and one judgment, prints durable
  checkpoint status, and never summarizes or publishes.
- Rejected every other partial combined `run` population before dataset,
  candidate, or judge network work. Canary is rejected outside `phase=run`
  and with `--plan-only`.
- Pinned the OpenAI SDK client to
  `https://api.openai.com/v1`, independent of `OPENAI_BASE_URL`, and recorded
  the endpoint in judgment, preflight, summary, and publication identity.
- Required returned judge model identity to be exactly `gpt-5.6-luna` before
  verdict parsing. Mismatches use bounded `JudgeProtocolError` retries and
  terminal incomplete semantics.
- Pinned candidate request, contract, `/v1/models`, before/after snapshots,
  offline identity files, and publication identity to exactly
  `glm-5.3-w4afp8`, with expected and observed model fields.
- Added an immutable SQLite `judge_preflight` singleton with idempotent query
  and conflict rejection. Strict summary/publication requires a complete,
  valid audit and exports it.
- Added `judge_retry_count`, including successful judgment and preflight
  retries, plus explicit `judge_terminal_failure_count == 0` for publishable
  runs.
- Updated the canary manifest, design, implementation plan, and runbook.

## Strict TDD Evidence

Every production behavior change followed test-first RED, minimal
implementation, then GREEN:

1. Canary/population gate:
   RED `4 failed, 3 passed`; GREEN `7 passed`.
2. Explicit OpenAI endpoint and returned judge identity:
   RED `3 failed`; GREEN `3 passed`.
3. Candidate expected/observed identity and drift:
   RED `3 failed`; GREEN `3 passed`.
4. Immutable preflight persistence/publication:
   RED `6 failed`; GREEN `6 passed`.
5. Judge summary diagnostics:
   RED `2 failed`; GREEN `2 passed`.
6. Canary manifest and durable-checkpoint runbook:
   RED `2 failed`; GREEN `2 passed`.
7. Preflight failure ordering regression found by the scoped suite:
   RED `2 failed`; GREEN `2 passed`.
8. Preflight resume and automatic full-run preflight:
   RED `2 failed`; GREEN `2 passed`.
9. Durable canary completion status:
   RED `1 failed`; GREEN `1 passed`.
10. Canary plus `--plan-only` rejection:
    RED `1 failed, 6 passed`; GREEN `7 passed`.
11. Offline wrong candidate identity:
    RED `1 failed`; GREEN `1 passed`.
12. Strict publication candidate identity:
    RED `1 failed`; GREEN `1 passed`.
13. Complete preflight metadata contract:
    RED `1 failed`; GREEN `2 passed` including resume coverage.

Commands used absolute PowerShell-expanded test paths and:

```powershell
--confcutdir "$root\pipeline\tests"
```

No live OpenAI, candidate-model, GPU, or Kubernetes request was made.

## Final Verification

- Clean isolated environment: Python 3.12 `.venv-review`.
- Scoped tests: `133 passed in 37.05s`.
- Ruff: `All checks passed!`
- Credential scan: `0 suspicious literals`.
- `git diff --check`: clean.
- IDE diagnostics: no errors.

## Self-review

- Canary returns before `_ensure_published_result` and `publish_results`.
- Population validation runs before `_prepare_run`, so rejected partial runs
  cannot download the dataset or contact either endpoint.
- Preflight resume reads and validates the immutable audit instead of issuing
  a second API request that would conflict on timestamps/request IDs.
- Strict publication validates candidate contract plus both server snapshots,
  preflight identity, 300 candidate units, 300 successful judgment units, and
  zero terminal judge failures.
- Judgment model identity is checked before output/verdict acceptance.
- Candidate request model and observed `/v1/models` identity share one pinned
  constant.

## Remaining concern

Low severity: `summarize` still re-prepares the pinned official dataset before
opening the existing SQLite checkpoint, so it requires dataset-network
availability. This limitation is documented in the design, plan, and runbook;
changing checkpoint loading was deliberately deferred to avoid expanding the
final-review fix scope.
