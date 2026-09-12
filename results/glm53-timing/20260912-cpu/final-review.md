# Independent final review: objective 3 quantization timing

Review base: `a150f6902d7938f8b2b9d4e57be95e062a1be0c3`
Review head: `710028e0`

Spec gate: **PASS**
Quality gate: **APPROVE**

No blocking correctness, integration, algorithm-order, or scope finding was identified. One low-severity reproducibility improvement is recorded below; it does not block this change.

## Review scope and method

Reviewed the supplied commit list/stat/full-context diff, approved design, implementation report, production and test changes, operator documentation, benchmark script and preserved evidence. Inspected the existing `summarize_phases` implementation specifically for canonical completeness, elapsed-clock semantics, and resource-retention compatibility. No source, index, or HEAD mutation was performed. The only review artifact written is this file. No GPU allocation, cluster action, push, or suite rerun was performed.

## Findings

- **Low / optional — deterministic resource-observation ordering:** `pipeline/timing_report.py:177` traverses the intersection of boundary-key sets, and resource observations retain that iteration order. `pipeline/timing_report.py:375` then displays the first 20 observations. Consequently separate Python processes can produce different JSON observation ordering and different Markdown resource examples for the same input. Timing calculations and the full JSON observation set are unaffected. Consider sorting paired keys or sorting observations by a stable phase/rank/identity key before rendering; this would make regenerated artifacts easier to compare. This is not an approved-spec violation.

## Spec and correctness assessment

- Reporting remains standard-library-only, reuses the canonical parser and phase summarizer, retains explicit source paths and preserves the existing metadata summary. The parser extension is backward compatible with existing callers.
- Phase/identity groups and decoder-layer rollups retain module/subgraph distinctions. Rank duration intervals are unioned rather than summed when they overlap; rank spread is explicitly qualified as work imbalance rather than collective latency. Canonical elapsed estimates compare monotonic timestamps only within each recorded node.
- Paired failed spans remain usable evidence; open, orphaned, invalid, duplicate, multi-run and malformed/truncated evidence prevents or qualifies completeness. Missing ranks are not represented as zero-duration work. JSON retains canonical diagnostic details; Markdown exposes the main missing/failed/open/ambiguous evidence.
- Resource reporting uses logged boundary snapshots only, keeps counter resets and missing counters unavailable, and presents RSS and allocator observations as gauges. It does not equate process I/O with unique shared-storage traffic. The new lightweight snapshot shape is `{}`, preserving consumers expecting a mapping.
- Automatic reporting executes after capture closure in the outer finally, on source rank only. Expected rank paths are built with the existing distributed convention. It neither polls nor waits for peers. Summary/report exceptions are guarded and successful work returns its original checkpoint path; failed work retains its original exception.
- The automatic reporter stats observed sizes and defers above 64 MiB before calling the parser. The deferred artifacts include a shell-quoted offline command. The offline CLI is uncapped. This is an observed-size guard, not a hard bound on a concurrently growing file or elapsed I/O time; the documentation appropriately describes the guarantee as an observed-input budget and requires final regeneration after launcher exit.
- Added spans surround the sequential callback, AWQ mapping search, native MTP preflight, side-artifact work and existing barriers. They use `collect_snapshot=False`; no new CUDA synchronization/events, resource reads, tensor inspection, collectives, per-token hooks, grid-point timers, or profiler dependency appears in the change.
- The operation-order evidence reports identical work ASTs for all three affected work functions after removing timer wrappers. Inspection of the diff agrees with that result: the original quantization calls remain in order.

## Validation and evidence assessment

Independently checked evidence integrity without rerunning the reported suites:

- All six production-source SHA-256 values in `validation.json` match the reviewed files.
- Preserved final JUnit files contain 39 focused passes and 17 integration passes, with zero failures/errors; additional isolated native AWQ and sequential runs contain 1 and 12 passes. Deduplicating test identities gives **56 distinct passing tests**.
- The decompressed raw benchmark sample matches the manifest SHA-256 and byte count.
- The operation-order report binds its three comparisons to matching reviewed source hashes.
- Focused tests cover overlap/rank/multi-node clocks, identity grouping, interrupted/failed/duplicate evidence, reset counters, CLI artifacts, over-budget parser avoidance, lightweight snapshot compatibility, and report failure isolation. Native AWQ and MTP integration evidence exercises actual changed paths; sequential helper evidence is appropriately qualified rather than presented as a full distributed run.
- The benchmark includes warmup, repeated trials, a context-manager baseline, real buffered Loguru JSONL writes and a snapshot comparison. Its documented 216.0 microseconds per lightweight span, 1,816.6 microseconds per snapshot span, and 0.23-second aggregation measurement are scoped to CPU observer work. The 100,000-span/10-hour comparison is expressly an extrapolation.
- The initial integration teardown guard failure is preserved and explained, and the identical final suite is clean. Existing Torch deprecation warnings are disclosed. No evidence was silently relabeled as GPU or CephFS performance.

## Limits

This review and the preserved tests do not establish full GLM GPU runtime slowdown or shared-storage throughput. The automatic report can be partial while peers finish; final all-rank diagnosis requires explicit offline regeneration and complete executor traces. Large uncapped offline reports necessarily consume memory proportional to retained boundary/group evidence. The byte budget does not prove a hard wall-time limit on arbitrary storage. These limits are documented and do not require a new allocation to accept this instrumentation/reporting change.
