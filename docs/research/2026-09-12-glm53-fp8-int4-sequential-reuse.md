# Reusing sequential FP8-weight / INT4-weight calibration

Date: 2026-09-12. Research snapshot before implementation.
For current implementation and qualification status, see
[FP8-before-GPTQ implementation](../glm53-fp8-before-gptq-implementation.md).
Local baseline: `383a4411`. [Source revisions and file hashes](../evidence/2026-09-12-fp8-int4-reuse-sources.json).

## Owner clarification

Group A is attention, shared experts, dense-prefix MLPs and the explicitly targeted
indexer weights, destined for block FP8. Group B is routed-expert INT4 weights.
B must be calibrated using inputs produced with upstream A already FP8-rounded.
Both groups' weight errors must reach later decoder layers. Activation quantization
optimization is a separate question. Direct SGLang export remains the final target.

## Findings from primary implementations

### Closest precedent: IST-DASLab MoE-Quant

Inspected revision `5a3b298cfb5c475a9b6584d48b43fcebc4ddfb2f`.
Its [quant.py](https://github.com/IST-DASLab/MoE-Quant/blob/5a3b298cfb5c475a9b6584d48b43fcebc4ddfb2f/quant.py#L198-L279)
loads a native FP8 decoder block and dequantizes it before collecting Hessians.
With `--quantize_only_experts`, only routed experts acquire GPTQ handles. Non-expert
weights retain the values reconstructed from FP8, even though their working dtype
is FP16/BF16. Consequently, attention's FP8 rounding is already present when expert
inputs are observed. After solving experts, the block is replayed to produce the
next block's inputs (`quant.py:413`).
Its [FP8 dequantizer](https://github.com/IST-DASLab/MoE-Quant/blob/5a3b298cfb5c475a9b6584d48b43fcebc4ddfb2f/src/quant_utils.py#L267-L297)
expands 128x128 block multipliers and crops padding.

This is a close numerical precedent, not the exact BF16-to-new-FP8 preparation
step: A starts quantized in the source checkpoint. The runner also assumes
DeepSeek and a 163-shard source layout. Reuse its ordering and weight-materialization
pattern; keep this fork's existing GLM EP adapter, calibration policy and solver.
This repository is identified by its URL; do not conflate it with similarly named
MoEQuant papers.

### Established within-block ordering: original GPTQ

Inspected revision `2d65066eeb06a5c9ff5184d8cebdf33662c67faf`.
[llama.py:75-125](https://github.com/IST-DASLab/gptq/blob/2d65066eeb06a5c9ff5184d8cebdf33662c67faf/llama.py#L75-L125)
implements `true_sequential`: Q/K/V, attention output, MLP gate/up, then MLP down.
Each group receives a new calibration forward after earlier groups have been
quantized. A final replay supplies the next decoder block. This establishes the
ordering mechanism, although this runner does not implement mixed block-FP8/INT4.
[Hugging Face's GPTQConfig documentation](https://huggingface.co/docs/transformers/main/en/main_classes/quantization#transformers.GPTQConfig)
also documents this within-block meaning of `true_sequential`.

### Reuse directly: llm-compressor and compressed-tensors

Inspected upstream llm-compressor `b52e76d66a6f47275c33dd59342a90ae7c50d34a`
and compressed-tensors `ddd5a568f435eb71bde3d71846ee55cab7be0d4b`.
The [mixed-modifier example](https://github.com/vllm-project/llm-compressor/blob/b52e76d66a6f47275c33dd59342a90ae7c50d34a/examples/quantization_non_uniform/quantization_multiple_modifiers.py)
provides composition infrastructure, but composition alone does not prove A-before-B.
The inspected upstream sequential loop, like this fork, wraps calibration and
propagation in `DisableQuantization(model)`.

[compressed-tensors forward.py](https://github.com/vllm-project/compressed-tensors/blob/ddd5a568f435eb71bde3d71846ee55cab7be0d4b/src/compressed_tensors/quantization/lifecycle/forward.py)
already provides `quantize`, `dequantize` and `fake_quantize`, including BLOCK
dispatch. This fork already uses its fake-quant primitive inside GPTQ. Existing
`observe`/`update_qparams` provide weight scales; existing offload utilities publish
updated tensors. No new FP8 numerical kernel or GPTQ solver is needed.

### NVIDIA Model Optimizer: useful reference, not a drop-in fix

Inspected revision `51de53e48ccae8804f8fe1198b7cf89475c5c4f4`.
Its [layerwise configuration](https://github.com/NVIDIA/Model-Optimizer/blob/51de53e48ccae8804f8fe1198b7cf89475c5c4f4/modelopt/torch/quantization/config.py)
exposes `get_qdq_activations_from_prev_layer`, defaulting true for GPTQ, to carry
quantized outputs into later layers. However, its
[GPTQ calibration function](https://github.com/NVIDIA/Model-Optimizer/blob/51de53e48ccae8804f8fe1198b7cf89475c5c4f4/modelopt/torch/quantization/model_calib.py#L2234-L2315)
disables weight quantizers during Hessian collection. That alone does not provide
FP8 fake-quantized A inside the currently calibrated block. Replacing our framework
would therefore not establish the requested behavior without further integration.

## Recommended integration to evaluate

Apply the MoE-Quant pattern using our existing dependencies. On entering each
resident decoder block, observe and block-FP8-quantize A using existing primitives;
materialize its dequantized working values before GPTQ's first calibration forward.
Then collect and solve B through the current EP path and replay the combined block
using existing `propagate_error=True` machinery. This also handles same-block
attention before experts without changing EP to Linear-level sequential targets,
which the current EP preflight rejects.

Keep the FP8 payload and scales consistent with the working values and final
export. Do not recompute scales from already-rounded A at epoch end or export:
that could change the effective quantizer. Integrate with existing collective
offload/publication ordering and process one resident block at a time. Shared
experts affect the summed block output, not the routed experts' input in that same
block. Keep activation QDQ disabled for this weight-ordering comparison.

Before full execution, verify on a tiny fixture that B's captured inputs/Hessian
match an explicit A-FP8 reference, both groups affect next-layer cached inputs,
EP and DDP agree, disk save/reload preserves values, and final exported FP8
weights/scales reconstruct the calibration values. Then compare representative
quality and phase timing against the current A-BF16/B-INT4 baseline.

The recipe documents GLM-5.3 BF16 as an upcast of FP8 weights. Measure the actual
A round-trip difference: if the chosen block quantizer reproduces those values,
this change may have little numerical effect for this source. That does not make
BF16 storage evidence of an unrounded original, nor establish an accuracy gain.

No turnkey implementation was established for BF16-source GLM-5.3, block-FP8 A,
EP GPTQ INT4 B, and direct SGLang export together. This is a bounded search finding.
The ordering has prior implementations; the remaining work is integrating existing
numerics into this fork and validating them. No production code or GPU runs were
changed by this research.
