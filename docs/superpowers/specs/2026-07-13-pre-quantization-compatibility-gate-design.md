# Pre-Quantization Compatibility Gate Design

## Goal

Fail before calibration when an original model and an llm-compressor AWQ or GPTQ
recipe are structurally incompatible. The gate complements the post-quantization
serving ABI checker; it does not predict calibration quality or runtime accuracy.

## Architecture

The library entry point accepts an instantiated model, preferably built with meta
tensors, and the exact modifier list that quantization will use. It invokes the same
quantization configuration, target matching, group-divisibility, dynamic AWQ mapping,
and AWQ mapping-resolution code used by llm-compressor. It stops before observers,
forward hooks, activation caching, Hessians, grid search, or weight mutation.

A pipeline CLI builds the configured model under `accelerate.init_empty_weights`,
constructs the real recipe, runs the analyzer, writes a versioned JSON report, and
returns nonzero for hard incompatibilities. Reports retain method, module-class and
target inventories, resolved AWQ mappings, norm-calibration adapter coverage,
warnings, failures, and checks that remain unverified statically.

## Initial checks

- Recipe configuration resolves without conflict.
- Quantization targets and ignores match at least one eligible module.
- Group and tensor-group weight schemes satisfy group-size divisibility.
- GPTQ resolves at least one non-Embedding weighted target.
- AWQ infers or accepts mappings, resolves at least one mapping, and records mappings
  skipped by missing targets or incompatible shapes.
- Every norm used as an AWQ smooth layer records whether llm-compressor will replace
  it through `NormCalibrationModule`. Known offset-norm classes must have an adapter;
  custom unclassified norms are warnings, not silently classified as ordinary norms.
- Resolved AWQ hook targets exist and balance modules have weights and compatible
  feature dimensions through the real resolver.

MiniMax-M3 is the first regression profile: its `MiniMaxM3VLRMSNorm` smooth layers
must resolve through `CalibrationOffsetNorm`, and its configured MoE/attention AWQ
mappings must resolve on the model representation used by quantization.

## Scope and safety

Version one supports AWQ plus its quantization modifier and GPTQ. Other modifiers are
reported as unsupported instead of guessed. The analyzer may attach temporary
quantization-scheme metadata to the supplied model, so callers should use a disposable
meta model. It never starts calibration or installs hooks.

Static success proves planner compatibility only. Calibration dataset suitability,
activation statistics, quantization error, representative-layer quality, checkpoint
ABI, vLLM loading, and downstream quality remain later gates.

## Testing

Small synthetic meta models cover passing GPTQ/AWQ, empty targets, indivisible groups,
unresolved AWQ mappings, missing offset-norm adapter detection, and JSON report
stability. A MiniMax-named synthetic regression avoids downloading the full model in
unit tests while exercising the same norm registry and real AWQ resolver.
