# GLM-5.3: early SGLang W4AFP8 alignment

Date: 2026-09-09. Scope: the first bounded implementation for
[objective 2](../GLM53_QUANTIZATION_OBJECTIVES.md#2-produce-the-serving-required-w4afp8-formats-during-quantization).
Objective 2 remains open. EP GPTQ remains the priority; this change does not
alter its implementation or the pinned H100 test.

## Findings and reuse

The repository already has the required FP8 quantizer, integer repacker, indexer
repair, MTP graft and converted-checkpoint verifier. The missing connection was
between the full AWQ recipe and those export utilities. In particular:

- `glm53_distributed_w4afp8_awq_full.yaml` already used `FP8_BLOCK` for attention,
  shared experts and dense MLPs, but omitted indexer `wk` and `wq_b`.
- `glm52_awq_1layer_indexer_fp8.yaml` already demonstrated the intended indexer
  target patterns. Reuse those patterns with the full recipe's block preset.
- `to_sglang_w4afp8.py` unconditionally rebuilt FP8-rest weights from the base,
  reconstructing AWQ folds. Its source-relative check also assumed per-channel
  scales. That was appropriate for historical per-channel output, but wrong for
  native block output: it changes already-quantized numbers and can encounter
  incompatible scale shapes during the old check.
- Installed compressed-tensors `FloatQuantizationCompressor` already emits
  E4M3 weights plus the 128x128 `weight_scale` grid. Its `FP8_BLOCK` preset and
  llm-compressor's observers supply the numerical implementation. No new FP8
  quantizer or integer packing convention is needed.

Prior-work sources inspected:

- [SGLang v0.5.17 W4AFp8Config and MoE method](https://github.com/sgl-project/sglang/blob/v0.5.17/python/sglang/srt/layers/quantization/w4afp8.py).
- [SGLang v0.5.17 FP8 linear loader](https://github.com/sgl-project/sglang/blob/v0.5.17/python/sglang/srt/layers/quantization/fp8.py).
- [SGLang v0.5.17 DSA indexer](https://github.com/sgl-project/sglang/blob/v0.5.17/python/sglang/srt/layers/attention/dsa/dsa_indexer.py).
- [Upstream llm-compressor schemes](https://github.com/vllm-project/llm-compressor/blob/main/docs/guides/compression_schemes.md)
  and [model-free PTQ](https://github.com/vllm-project/llm-compressor/tree/main/examples/model_free_ptq),
  plus this fork's `pipeline/recipe.py`, converter, kernels, patch and graft.

The [existing Rancher guide](glm53-w4afp8-rancher-evaluation.md) names
`lmsysorg/sglang:v0.5.17`. This is the source reference for this work; an actual
execution packet must still record the deployed image digest/revision and flags.
The source inspection and CPU conformance tests are not a serving qualification.

## Contract and implementation boundary

| Module/component | Required representation on the referenced path | Status after this change |
| --- | --- | --- |
| Routed experts | INT4 group 128; signed nibble pairs in INT8 storage; renamed weight scales | Existing lossless CT-unpack/SGLang-repack retained |
| Attention, shared experts, dense MLPs | E4M3 weights, 128x128 block multipliers under `weight_scale_inv` | Existing FP8_BLOCK recipe; converter now preserves native bytes and values |
| Indexer `wk`, `wq_b` | Block FP8 on the unfused GLM indexer path | Added to the full AWQ recipe's FP8 pass |
| Indexer `weights_proj`, norms, router, embeddings/head | Remain unquantized where the model builds them without quantization | Existing scope retained; indexer exclusions tested |
| Expert activations | Standard W4AFP8 MoE path uses static scales, reduced to one maximum per gate/up or down group | Still differs from the dynamic per-token CT preset |
| MTP layer 78 | Actual draft-head tensors plus honest config metadata | Still supplied by the existing graft; absent MTP remains marked absent |

`W4AFp8Config.from_config` does not propagate `ignored_layers` in the inspected
version. Therefore an ignore-list declaration alone cannot make indexer `wk` or
`wq_b` BF16. The model's actual construction matters. DeepEP low-latency and fused
indexer branches have additional behavior; this note does not qualify them.

The selected quick change repairs the recipe and adapts the existing converter.
Merely chaining all the old post-processing commands would retain the unnecessary
rounding and source reads. A direct native exporter could also remove the second
whole-checkpoint write, but requires a larger integration with distributed save,
activation semantics and MTP handling.

## New behavior

For a source module explicitly declared symmetric static 128x128 block FP8,
conversion checks E4M3 dtype, weight/scale geometry and finite positive scales.
It copies weight bytes unchanged and renames `weight_scale` to
`weight_scale_inv`. BF16/FP16 scale values widen exactly to FP32. It does not
recover a fold or read BF16 weights on this path. Shape alone cannot select the
fast path: a small matrix may have identical channel and block scale shapes.

Legacy per-channel FP8 and BF16 indexers still use the existing conversion
kernels and fold checks. They require the original BF16 source; a missing base
weight now fails instead of copying an unloadable tensor and reporting success.
The verifier checks sampled native-block weights and scales exactly, retaining
the historical tolerance only for legacy conversions.

For a new aligned checkpoint, the same converter can run without `--base`:

```bash
python -m pipeline.to_sglang_w4afp8 --ckpt /path/to/checkpoint --out /path/to/new-output
python -m pipeline.verify_sglang_w4afp8 --src /path/to/checkpoint --dst /path/to/new-output
```

These are interface examples, not authorization for a full conversion or a GPU
run. `conversion_manifest.json.fp8_paths` and the log distinguish
`preserved_block` from `rebuilt_from_base`. An entirely aligned main-model export
should report zero rebuilds. A mixed historical artifact may still need `--base`.
The converter still rewrites the output checkpoint and retains expert packing.
No time saving on shared storage has been measured for this change.

## Validation

**116 tests passed in 32.94 seconds** (CPU). Evidence: [pytest log](../results/glm53-sglang-early-alignment/20260909-cpu/pytest.log)
and [reproduction details](../results/glm53-sglang-early-alignment/20260909-cpu/README.md).
Tests use the installed native CT FP8 and INT4 compressors, including a folded
weight and a partially filled block. They prove conversion works without a BF16
source or an FP8 requantizer, check exact output, and require failure after a
one-bit weight change or one-ULP scale change. Invalid block metadata, dtype,
geometry and scales fail. Existing legacy conversion, indexer repair, MTP graft
and kernel tests remain part of the focused suite.

## Remaining work for objective 2

1. Build a runtime-specific pre-calibration contract from the actual model/module
   inventory and source headers. Validate all custom recipes, not only the
   default GLM-5.3 recipe covered here. Reuse the scope comparison and preflight
   infrastructure already in this fork.
2. Align expert activation QDQ with the intended SGLang execution path. Existing
   static observers are building blocks, but simply setting `dynamic=False`
   leaves per-expert versus layer-wide scale reduction unresolved. Changing
   scales after calibration would recreate the mismatch. Qualify this separately
   for AWQ and EP GPTQ; EP's current supported recipe remains unchanged.
3. Reuse the repacking/rename code in a bounded native save adapter, so writing
   compressed-tensors output followed by another full write is unnecessary.
   Retain independent integer conformance checks and existing distributed save
   coordination. Do not invent a replacement safetensors writer.
4. Incorporate the existing MTP graft into final assembly from the same source
   revision, with explicit MTP policy and complete tensor coverage. The main
   Transformers model does not instantiate that layer, so recipe targets alone
   cannot preserve it.
5. Prepare a bounded execution packet against the pinned runtime: actual recipe
   initialization, quantize/save/reload, indexer and folded-weight checks, then
   SGLang load and paired output checks. Record phase times and bytes. Only then
   consider a full run. CPU results here establish serialization conformance,
   not full-model accuracy or the absence of GPU indexer/offload problems.

The executor should read this note through the shared objectives brief. No new
full AWQ rerun, conversion of the 394.6 GB checkpoint, or SGLang GPU evaluation
was launched for this change.
