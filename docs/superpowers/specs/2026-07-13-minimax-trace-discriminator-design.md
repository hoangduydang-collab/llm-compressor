# MiniMax-M3 Sequential Trace Discriminator Design

## Goal

Determine why 60 correctly matched MiniMax decoder targets produce one sequential
subgraph, without running oneshot, calibration, AWQ, quantization, or quality evaluation.

## Design

Add an opt-in diagnostics sink to the existing `trace_subgraphs` implementation so the
experiment observes the exact production tracer rather than a copied approximation.
The sink records matched target paths, ancestor count, raw FX nodes by operation,
`call_module` targets, sequential target nodes, partition sizes, subgraph count, and
generated graph code. Existing callers that omit the sink behave identically.

A MiniMax-specific command loads the model through the production loader, applies the
existing text-calibration patch, builds and collates the same first calibration batch as
oneshot, and traces two roots against that batch:

1. the full `MiniMaxM3SparseForConditionalGeneration` wrapper; and
2. its live language-model subtree.

Each root gets an independent artifact directory containing `report.json`, `graph.py`,
and `nodes.json`. A root-level summary classifies only structural observations; it does
not claim an AWQ root cause. Import paths and package versions are recorded to prove
which tracer implementation executed.

## Failure handling

Trace exceptions are captured per root with type, message, and traceback, then the other
root is still attempted. The command returns zero only when both roots trace and writes
an aggregate report in all cases. It performs no model forwards after tracing and never
constructs recipe modifiers.

## Interpretation

- zero target nodes in the full graph but nonzero in the subtree localizes the fault to
  the multimodal wrapper/autowrap boundary;
- zero target nodes in both graphs localizes it to full-size/config/input/environment
  tracing behavior;
- target nodes present with one partition localizes it to graph partitioning; and
- expected target nodes and partitions falsify the reported collapse under this exact
  trace-only setup and require comparison with oneshot lifecycle state.

## Testing

CPU tests use tiny modules and synthetic FX graphs to verify diagnostic inventories,
unchanged no-sink behavior, subtree sample filtering, atomic artifacts, exception
persistence, and aggregate exit status. The executor first runs a short smoke using the
real model and returns all artifacts before any further AWQ work.
