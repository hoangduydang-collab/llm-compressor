# Objective 3 timing report design

Approved in conversation on 2026-09-12, with the additional requirement that instrumentation must not visibly increase quantization time.

Reuse `compression_phase`, existing JSONL capture, and `summarize_phases`. Existing code already records most phases and process/cgroup memory and I/O snapshots. PyTorch Profiler supports finer CPU/GPU investigation but adds overhead (https://docs.pytorch.org/tutorials/recipes/recipes/profiler_recipe.html); reserve it for later targeted diagnosis.

Produce offline JSON and Markdown timing reports from explicit rank log paths. The automatic source-rank report runs after its capture closes, scans expected rank paths once, never waits for peers, and labels incomplete evidence. A CLI regenerates the final all-rank report after the launcher exits, or reads interrupted logs. Preserve existing quantization metrics metadata.

Show phase totals per rank, slowest rank and duration spread; group phases by recorded subgraph identity and module decoder layer where available. Do not equate subgraph number with decoder layer. Include top slow groups, failed/open spans and missing rank evidence. Summed rank work and nested phases are not end-to-end elapsed. Compare monotonic clocks only within a node. Summarize available process I/O deltas and boundary RSS/GPU allocator values without reading checkpoint tensors. Missing or reset counters are unavailable, and process I/O does not identify unique Ceph traffic. Keep resource scope explicit and preserve raw logs.

Add coarse lightweight spans for sequential compression callbacks, AWQ mapping search, native MTP preflight, checkpoint side artifacts and existing barrier calls. New spans must skip resource snapshots; no CUDA events/synchronization, tensor inspection, extra collectives, per-token/per-grid-point instrumentation or profiler dependency. Preserve all algorithm work/order. Existing default snapshot collection is unchanged.

Reporting must never mask an original quantization exception or make successful quantization fail. Test synthetic overlap/skew, multi-node clocks, incomplete/malformed tails, per-layer identity, unavailable resources, rank-log discovery and failure handling. Measure added lightweight span overhead with a real local JSONL sink, and report its scope honestly; CPU measurement does not prove full shared-storage runtime overhead. No new cluster allocation is authorized.
