# MiniMax-M3 Guarded Full Quantization Matrix Implementation Plan

**Goal:** Run three full MiniMax quantization hypotheses concurrently with durable,
low-overhead, per-layer diagnostics and fail-fast behavior.

**Architecture:** Add an opt-in post-propagation callback to the sequential pipeline,
implement bounded sketch/metric and abort primitives, wrap AWQ/quantization modifiers
for full-model evidence, and launch each arm through detached tmux plus exclusive srun.

## Constraints

- Production behavior is unchanged when no callback is installed.
- Diagnostics reuse calibration/propagation and retain bounded sketches only.
- Persist evidence before aborting; one arm never cancels another.
- Do not start serving/CUDA-graph diagnosis in this run.

### Task 1: Finish structural trace preflight

- Complete the trace-only CLI and focused tests.
- Add detached exclusive-node launcher and required evidence handoff.

### Task 2: Post-propagation callback

- Add a failing sequential-pipeline test proving the callback runs after propagation
  and before the next subgraph.
- Implement an opt-in session-state callback with subgraph index/count/modules.
- Verify no-callback behavior remains unchanged.

### Task 3: Bounded diagnostics core

- Add failing CPU tests for deterministic sketches, continuous metrics, qparam/scale
  summaries, weight reconstruction metrics, variant-aware checks, atomic layer writes,
  heartbeat updates, and informative abort reports.
- Implement the pure CPU-testable functions.

### Task 4: Full guarded recipe and runner

- Add failing tests for `offsetfix`, `nosmooth`, and `quant_only` recipe isolation.
- Implement full-model audited AWQ and quantization modifiers using the production
  loader, patch, mappings, dataset, recipe contract, and checkpoint save path.
- Install boundary hooks for all decoder layers, persist after propagation, enforce
  variant-aware mapping/quality checks, and preserve partial evidence on exceptions.

### Task 5: Three-arm tmux/srun launcher and handoff

- Launch all arms concurrently on separate top-level exclusive nodes.
- Add smoke/dry-run validation, independent return-code files, aggregation, and durable
  tmux controller behavior.
- Update `MINIMAX_M3_HANDOFF.md` and the active rerun report with exact commands,
  interpretation, abort contract, and returned-artifact requirements.

### Task 6: Verification and delivery

- Run focused sequential, trace, diagnostic, recipe, launcher, config, and offset-norm
  tests; shell syntax checks; compile checks; and `git diff --check`.
- Commit and push for executor smoke, then stop for returned evidence analysis.
