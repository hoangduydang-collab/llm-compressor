# Quantization timing report

Evidence status: **complete**

A source-rank report may be partial until regenerated after the launcher exits.

Rank durations include waits. Nested phases and rank work must not be added as elapsed. Clock values are compared only within a node. Rank imbalance does not identify collective latency.

Process I/O is not unique shared-storage traffic. Missing or reset counters are unavailable; RSS and GPU allocator values are boundary gauges.

## Source evidence

- `/tmp/glm53-timing-overhead-final-20260912/lightweight-1.jsonl`

## Phase totals

| Phase | Ranks | Slowest rank | Min (s) | Median (s) | Max (s) | Spread (s) |
|---|---:|---|---:|---:|---:|---:|
| awq_search | 1 | xcnc14:0 | 0.189729 | 0.189729 | 0.189729 | 0.000000 |
| quantize_run | 1 | xcnc14:0 | 0.445466 | 0.445466 | 0.445466 | 0.000000 |

## Per-layer and subgraph groups

| Phase | Identity | Decoder layer | Ranks | Max (s) | Spread (s) | Status |
|---|---|---:|---:|---:|---:|---|
| awq_search | `decoder-layer rollup` | 3 | 1 | 0.189729 | 0.000000 | complete |

## Slowest layer and subgraph groups

- `awq_search` `{"module": "model.layers.3.mlp.experts.0.up_proj"}`: 0.189729 s max, 0.000000 s spread
- `awq_search` `{"decoder_layer": 3}`: 0.189729 s max, 0.000000 s spread

## Incomplete, failed, or ambiguous evidence

- missing paths: `[]`
- missing ranks: `[]`
- open spans: `[]`
- orphan ends: `[]`
- invalid spans: `[]`
- duplicate boundaries: `[]`
- multiple/missing roots: `{}`
- malformed/truncated lines: `0`
- failed spans: `0`

## Resource observations

- No usable paired boundary resource observations were present.
