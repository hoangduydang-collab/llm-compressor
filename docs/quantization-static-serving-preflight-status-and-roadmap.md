# Quantization Static Serving Preflight: Status and Roadmap

## Status

The CPU-only static serving ABI checker is proven for the MiniMax-M3
compressed-tensors checkpoints in this repository. It is mandatory for the
current M3 quality workflow, but it is not yet a model-agnostic checker.

Current entry point:

```bash
python -m pipeline.m3_serve_abi --checkpoint /path/to/checkpoint \
  --out /path/to/serving_abi.json
```

The command reads only `config.json` and `model.safetensors.index.json`. It
does not allocate a GPU or open tensor payloads.

## Proven MiniMax-M3 result

The original in-house GPTQ checkpoint contained 21,888 packed routed-expert
modules while its 57 routers and 171 shared-expert projections remained plain.
Its persisted ignore rules matched Transformers names (`mlp.gate` and
`mlp.shared_experts`) but not the vLLM runtime names
(`block_sparse_moe.gate` and `block_sparse_moe.shared_experts`).

The checker rejected the checkpoint with exactly 228
`plain_runtime_module_not_ignored` errors before task preparation or GPU work.
This predicted the previously observed garbage-serving failure.

A metadata-only overlay added the two vLLM aliases. The overlay changed only
`config.json`; source and overlay Safetensors-index hashes remained identical
and `tensor_payload_unchanged` was recorded as true. The checker then accepted
BF16, repaired GPTQ, and cyankiwi AWQ in one aggregate preflight.

Subsequent runtime evidence validated the static diagnosis. Repaired GPTQ
produced coherent generations in two independent smoke runs, with no empty
outputs or periodic loops. Against cyankiwi AWQ on the paired 2,047-token probe,
top-1 agreement was about 86.4% and the perplexity ratio was 1.014–1.022. The
checker therefore caught a real serving ABI defect rather than a harmless
metadata difference.

Primary evidence:

- `M3_STATIC_ABI_GATE_REPORT.md`
- `M3_GPTQ_REPAIRED_ABI_PREFLIGHT_REPORT.md`
- `M3_TMUX_SMOKE_FINAL_REPORT.md`
- `M3_3MODEL_GPTQ_AWQ_FINAL_REPORT.md`

## Current checks

For the supported M3 compressed-tensors layout, the checker:

- inventories packed weights, scales, and plain weights from index keys;
- translates known MiniMax Transformers/vLLM module aliases;
- evaluates every ignore rule in source and runtime namespaces;
- rejects malformed ignore regular expressions;
- rejects quantization groups that do not target `Linear`;
- rejects modules containing both packed and plain weights;
- rejects plain quantizable runtime modules not covered by an ignore rule;
- rejects packed modules covered by an ignore rule;
- rejects packed modules missing a scale; and
- reports counts by MiniMax component and per-pattern source/runtime matches.

The quality preflight writes a report for every active model and only then
raises an aggregate failure. Static failure occurs before dataset imports,
task preparation, model loading, or GPU allocation.

## What it does not prove

A valid static report is necessary, not sufficient. It does not prove:

- tensor values are finite or numerically accurate;
- scale/zero-point shapes, dtypes, group sizes, or dequantization semantics are
  correct;
- activation quantization semantics match the runtime kernel;
- calibration data or AWQ/GPTQ optimization produced acceptable accuracy;
- the installed vLLM version and hardware support the selected kernel;
- generation, KV cache, distributed execution, or CUDA graphs work; or
- task quality meets acceptance thresholds.

Representative-layer checks, independent dequantization, teacher-forced probes,
and end-to-end evaluation remain required after static validation.

## Why the implementation is not general yet

`pipeline/m3_serve_abi.py` intentionally contains MiniMax-specific assumptions:

- module classification comes from the M3 checkpoint diagnostics;
- namespace translation hard-codes M3 MoE, shared-expert, router, and MSA
  indexer aliases;
- quantizable component policy is a fixed M3 category set;
- packed tensors are recognized through compressed-tensors suffixes such as
  `.weight_packed` and `.weight_scale`;
- metadata is expected under `quantization_config.ignore` and
  `config_groups[*].targets`;
- every checkpoint is expected to have a Safetensors index; and
- only `Linear` target groups are accepted.

These assumptions do not safely cover ordinary AWQ/GPTQ `qweight` layouts,
FP8/block-FP8, MXFP8, NVFP4, AutoRound mixed-bit checkpoints, unsharded files,
architecture-specific fusions, tied weights, embeddings, convolutions, or
other serving frameworks. Applying the current M3 policy blindly to another
model could create false failures or, worse, false passes.

## Deferred model-agnostic design

Future work should preserve the static-first principle while separating four
independent concerns:

1. **Checkpoint inventory adapters** identify logical modules and required
   companions for each storage format: compressed-tensors pack-quantized,
   AWQ, GPTQ, FP8/block-FP8, MXFP8/NVFP4, and mixed-bit formats.
2. **Architecture/runtime namespace adapters** map source module names to the
   names constructed by a specific serving backend and version. Profiles must
   be explicit and versioned; unknown architectures fail as unsupported rather
   than guessing.
3. **Precision-policy evaluation** decides which modules should be quantized,
   ignored, or exempt and compares that decision with the actual inventory.
4. **Backend capability checks** validate format, activation scheme, hardware
   capability, and required kernels against the intended runtime environment.

A generic report should record the selected inventory adapter, architecture
profile, runtime/backend version, policy source, unsupported assumptions, and
stable error codes. The M3 implementation should become one registered profile,
with `pipeline.m3_serve_abi` retained as a compatibility wrapper.

## Proposed implementation stages

1. Extract a format-neutral inventory and report schema without changing M3
   behavior; run the existing M3 fixtures as compatibility tests.
2. Add explicit format adapters with synthetic positive and negative fixtures.
3. Add versioned model/runtime namespace profiles, starting with a second MoE
   model and a dense model to prevent M3-shaped abstractions.
4. Validate companion tensor names, shapes, dtypes, group size, zero-point, and
   activation metadata without loading full tensors.
5. Add optional sampled tensor checks and independent dequantization as a
   separate, slower tier.
6. Integrate the generic gate before and after every new quantization/export
   recipe and make unsupported profiles a hard preflight failure.

## Acceptance criteria for general use

The checker is ready to apply automatically to every new model only when:

- an unknown model, format, or backend cannot silently pass;
- each supported format has corrupted and valid fixture coverage;
- architecture aliases are versioned and traceable to runtime loader behavior;
- direct and exported checkpoints can be compared without tensor mutation;
- reports distinguish metadata, tensor-contract, capability, and unsupported
  failures;
- the current MiniMax fail/repair evidence remains reproducible; and
- static pass is always followed by a short runtime quality smoke.

Until those criteria are met, use the current checker only for the documented
MiniMax-M3 compressed-tensors profiles.


## Pre-quantization companion gate

The original-model companion is implemented at
`llmcompressor.preflight.quantization` with the pipeline entry point:

```bash
python -m pipeline.prequant_compatibility --config <pipeline.yaml> --output <report.json>
```

It operates on a disposable meta model and the exact AWQ/GPTQ modifier list. Unlike
the serving ABI checker, it runs before weights or calibration data are loaded and
asks whether llm-compressor can structurally apply the requested method: recipe
resolution, target/ignore coverage, group-size divisibility, real AWQ mapping
resolution, hook-target existence, balance-layer shape compatibility, and offset-norm
adapter coverage. MiniMax-M3 is the first regression profile.

The two gates cover different boundaries. A pre-quantization pass does not inspect the
produced checkpoint or vLLM, while a post-quantization ABI pass does not prove that the
calibration planner interpreted the original architecture correctly. Both reports
explicitly retain unverified numerical/runtime properties. The intended lifecycle is:

1. pre-quantization planner compatibility;
2. representative-layer canary when the recipe or architecture is new;
3. full quantization;
4. post-quantization serving ABI compatibility;
5. bounded runtime smoke; and
6. paired model-quality evaluation.

Version one intentionally supports AWQ and GPTQ only. Generalization should add method
adapters around real modifier planner APIs, never parallel reimplementations of their
target or mapping rules.
