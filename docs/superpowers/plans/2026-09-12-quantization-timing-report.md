# Quantization Timing Report Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development or superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Make existing AWQ/GPTQ phase evidence usable without visibly slowing quantization.

**Architecture:** Reuse existing JSONL and phase summarization; put reporting in a CPU-only module. Add only lightweight coarse spans to production, and generate reports outside the quantize_run span without waiting for peers.

**Tech Stack:** Python standard library, existing Loguru instrumentation, pytest.

## Global Constraints

- No CUDA synchronization/events, tensor inspection, new distributed collectives/barriers, profiler dependency, per-token or per-grid-point instrumentation.
- New spans skip resource snapshots; existing default snapshots remain unchanged.
- Report generation runs outside quantize_run and never masks work exceptions or fails successful quantization. Automatic aggregation reads at most 64 MiB of total log inputs; above that it writes a deferred notice and offline CLI command without parsing payloads. CLI aggregation is uncapped.
- Rank timing includes waits; nested phases must not be added as elapsed; clocks from different nodes must not be compared.
- Missing/reset counters remain unavailable; process I/O is not unique shared-storage traffic.
- Work on duy-branch; preserve unrelated .gitignore, predictive_ptq/, results/predictive-ptq/. No GPU allocation.

### Task 1: Offline reporting and lightweight integration

**Files:** Create `pipeline/timing_report.py`, `pipeline/tests/test_timing_report.py`; modify `pipeline/metrics.py` only for shared parser/summary reuse if necessary, `pipeline/quantize.py`, `src/llmcompressor/utils/metric_logging.py`, `src/llmcompressor/pipelines/sequential/pipeline.py`, `src/llmcompressor/modifiers/transform/awq/base.py`, `tests/llmcompressor/utils/test_phase_logging.py`.

**Interfaces:**
- `build_timing_report(paths) -> dict`: explicit rank JSONL paths, no torch import in reporting module; reuse summarize_phases for canonical completeness and elapsed. Additional grouping must preserve subgraph/module identity, one group per phase plus identity and per-rank durations; list slow groups. Summarize selected process I/O deltas, RSS and GPU allocator observations compactly; retain raw source paths.
- `write_timing_report(paths, output_prefix) -> dict`: write `<prefix>.json` and `<prefix>.md`, return report. Tables have phase, observed-rank count, slowest rank, min/median/max duration and spread. Per-layer/subgraph details and incomplete/failed status appear in both artifacts.
- CLI `python -m pipeline.timing_report LOG [LOG ...] --output-prefix PATH`: offline after launcher exits; standard library only, tolerate partial JSON tail and missing files, make absent evidence explicit.
- `compression_phase(name, *, collect_snapshot=True, **identity)`: False avoids all snapshot reads, preserving paired timing/status and nesting.

- [x] Write focused synthetic tests before implementation: nested/overlap totals, unequal-rank durations, multi-node clock offsets, module decoder-layer vs subgraph grouping, failed/open spans, truncated tail/missing logs, missing/reset I/O and gauge semantics. Missing logs must not imply zero work; per-rank phase imbalance must not claim collective latency.
- [x] Implement report using existing `_iter_records`, `_union_duration`, `summarize_phases`, `_decoder_layer` helpers, avoiding a second inconsistent completeness algorithm. Avoid retaining full raw snapshots; retain compact needed fields only. Detect duplicate span boundaries/multiple runs if encountered and flag ambiguity instead of silently merging.
- [x] Add `collect_snapshot=False` API and no-read/no-sync test. Wrap the existing sequential_epoch_end callback in `sequential_compression` with subgraph identity, and `_compute_best_scale` invocation in `awq_search` with smooth module identity. Add only coarse MTP preflight/side-artifact/existing barrier spans. Do not duplicate existing GPTQ per-module timers.
- [x] Add `write_automatic_timing_report(paths, output_prefix, max_input_bytes=64*1024*1024)` to stat input sizes and defer aggregation above the automatic byte budget. Write JSON/Markdown notices with sizes, deferred status and offline command; test that deferred reporting never calls the parser. This limit follows local measured parse cost and the owner no-visible-overhead constraint.
- [x] In `run_quantize`, retain existing capture and summary. In a guarded finally after capture closes, source rank generates a one-shot timing report from paths constructed using declared world size and DistributedContext.rank_path convention. Peers do not aggregate. Do not wait or poll. Warn on report failure while preserving original exception; successful return unchanged. Read each rank log only as needed. Automatic report may be incomplete until manual CLI regeneration after launcher exits.
- [x] Test successful, failing and report-failing run integration, automatic all-rank path inclusion, and no new barrier calls. Test AWQ search span around real mapping search where existing tests permit. Run `OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 PYTHONPATH=src:. .venv-prequant/bin/python -m pytest -q pipeline/tests/test_metrics.py pipeline/tests/test_timing_report.py tests/llmcompressor/utils/test_phase_logging.py` plus the existing native AWQ roundtrip test and relevant sequential test coverage for changed paths.
- [x] Run focused ruff and `git diff --check`. Commit only owned source/test files. Write report with exact test commands/results and any limitations.

### Task 2: Overhead evidence and operator documentation (root)

**Files:** Create `docs/glm53-quantization-timing.md` and small raw evidence under `results/glm53-timing/20260912-cpu/`; update `GLM53_QUANTIZATION_OBJECTIVES.md` with report status and usage link.

- [x] Benchmark empty lightweight spans with the real JSONL sink after warmup over repeated trials; include baseline context-manager loop and existing snapshot spans. Save exact command/environment, raw samples and medians. No claimed GPU or CephFS measurement.
- [x] Generate and inspect an example report, measure offline aggregation separately, document no added synchronization and partial automatic report semantics. Show a copy-ready offline command using existing per-rank paths.
- [x] Add executor follow-up guidance for existing authorized runs: preserve all raw rank logs and regenerate after launcher termination. No new cluster experiment or allocation.
- [x] Review complete diff for timing semantics, algorithm invariance and overhead. Fix material findings, run relevant covering tests, commit docs/evidence and push duy-branch with the dedicated hoangduydang-collab credential helper.

Final qualification: source commit `7f09ab1d`; 56 distinct passing CPU tests,
operation-order comparison clean, final review PASS/APPROVE with no blocking
findings. Evidence: `results/glm53-timing/20260912-cpu/`. The optional minor
resource-sample ordering observation does not affect the stored timing values.
