# Hopper Packed-NVFP4 W4A8 Fallback Design

**Date:** 2026-07-19
**Scope:** vLLM serving compatibility and kernel research for H100/H200
**Status:** Approved design; implementation not started

## Problem

Official model releases may provide only NVFP4 checkpoints intended for native
Blackwell FP4 execution. Hopper supports FP8 Tensor Cores but not native FP4
Tensor Core multiplication. vLLM can already serve NVFP4 on Hopper through
Marlin W4A16 or software dequantization, but it has no packed-weight NVFP4 W4A8
path.

Loading an NVFP4 checkpoint and permanently expanding every weight to FP8 would
reuse existing Hopper W8A8 kernels, but it nearly doubles persistent weight
memory and loses the principal bandwidth advantage of the downloaded format.
The desired fallback instead keeps weights packed at four bits in GPU memory,
converts the needed weight fragments to FP8 inside the GEMM, and performs FP8
Tensor Core multiplication.

This project is both a release-day compatibility/performance investigation and
a baseline for comparing newly developed quantized models. It must not weaken
the reliable upstream W4A16 compatibility path.

## Existing resources and search conclusion

The design deliberately builds on existing work:

- vLLM already loads NVFP4 packed E2M1 weights, per-group E4M3 scales, and a
  global scale, then repacks them for Marlin W4A16.
- vLLM Marlin already has vectorized E2M1-to-FP16/BF16 conversion and an
  E2M1-to-E4M3 register conversion primitive.
- vLLM already provides dynamic per-token FP8 activation quantization.
- vLLM's existing Hopper CUTLASS W4A8 path provides serving integration,
  activation-scale, epilogue, and dispatch patterns, but currently accepts
  uniform signed INT4 weights with group size 128 rather than NVFP4 E2M1 with
  group size 16.
- NVIDIA CUTLASS provides SM90 mixed-input GEMM machinery with fast numerical
  converters, group scaling, TMA, and WGMMA.

The current Marlin FP8 `mma.sync` path is not the production foundation for this
goal. vLLM builds it only for SM89 and SM12x; its generator documents that the
same instruction route is simulated through FP16 MMA on SM90/SM100 and does not
accelerate Hopper. A useful Hopper kernel therefore needs an SM90 WGMMA
mainloop. Marlin's checkpoint semantics and conversion logic remain reusable,
but its final packed layout must not be assumed compatible with WGMMA without a
layout proof.

Primary references:

- <https://github.com/vllm-project/vllm/blob/main/vllm/model_executor/layers/quantization/utils/marlin_utils_fp4.py>
- <https://github.com/vllm-project/vllm/blob/main/csrc/quantization/marlin/dequant.h>
- <https://github.com/vllm-project/vllm/blob/main/csrc/quantization/marlin/generate_kernels.py>
- <https://github.com/vllm-project/vllm/blob/main/vllm/model_executor/kernels/linear/mixed_precision/cutlass.py>
- <https://github.com/NVIDIA/cutlass/blob/main/CHANGELOG.md>
- <https://docs.nvidia.com/cutlass/4.5.2/media/docs/pythonDSL/mma_docs/wgmma_programming.html>

## Goals

1. Load an official NVFP4 checkpoint without offline conversion.
2. Keep persistent linear weights packed as E2M1 at approximately NVFP4/Marlin
   memory cost throughout serving.
3. Dynamically quantize activations per token to FP8 E4M3.
4. Convert and scale packed NVFP4 fragments to FP8 within an SM90 GEMM mainloop.
5. Execute FP8 WGMMA with FP32 accumulation and BF16 output.
6. Preserve the upstream W4A16 Marlin fallback for unsupported shapes, failed
   preconditions, and release-day compatibility.
7. Make the dense proof of concept earn MoE production work through explicit
   correctness, memory, and performance gates.

## Non-goals

- Native FP4 execution on Hopper.
- Replacing W4A16 Marlin before W4A8 qualification.
- Supporting every FP4 checkpoint dialect in the first proof of concept.
- INT8 activation support in the first implementation.
- Fusing activation quantization into RMSNorm or another preceding operator in
  the first implementation.
- Writing a clean-sheet GEMM when CUTLASS mixed-input components can be adapted.
- Implementing fused MoE before the dense continuation gate passes.
- Claiming direct public-benchmark comparability from a paired subset or kernel
  microbenchmark.

## Chosen approach

Implement an opt-in dense SM90 W4A8 path by adapting CUTLASS mixed-input/TMA/
WGMMA machinery while reusing vLLM's NVFP4 checkpoint interpretation and
serving integration patterns.

The initial path will use vLLM's existing standalone dynamic per-token FP8
activation quantizer. It will not attempt to compute a per-token maximum inside
the GEMM because that reduction must finish before the corresponding matrix
multiplication begins and would complicate the mainloop.

The kernel will consume packed E2M1 weights and their group-of-16 E4M3 scales.
For each mainloop fragment it will:

1. load packed weights and the corresponding group scales;
2. decode E2M1 values into an FP8-capable register representation;
3. apply the correct group normalization before terms from differently scaled
   groups are combined;
4. round and clamp the scaled fragment to E4M3;
5. issue FP8 WGMMA with FP32 accumulation;
6. apply activation, channel/global weight, and output scaling in the epilogue;
7. write BF16 output.

The implementation may introduce an SM90-specific load-time repacker if the
Marlin-repacked tensor does not satisfy WGMMA's shared-memory descriptor and
operand-layout requirements. Such a repacker changes layout only; it must not
dequantize or persistently expand the weights.

## Numerical contract

For activation row `m`, output channel `n`, and weight group `g`:

```text
A8[m, k] = Q_E4M3(A[m, k] / A_scale[m])
B8[n, k] = Q_E4M3(E2M1(B4[n, k]) * B_group_scale[n, g]
                  / B_channel_scale[n])
C[m, n]  = BF16(A_scale[m] * B_channel_scale[n] * B_global_scale
                 * AccFP32_k(A8[m, k] * B8[n, k]))
```

`B_channel_scale[n]` is derived once at load time, without expanding the
weights: it is the maximum absolute value of
`E2M1(B4[n, k]) * B_group_scale[n, g]` over `k`, divided by the largest finite
E4M3 magnitude. An all-zero channel uses scale `1`. The deterministic reference
and kernel must use the same rounding, saturation, and zero-channel rule.

Equivalent scale factorizations are allowed only when a deterministic reference
proves they produce the declared W4A8 semantics. Raw E2M1 values are exactly
representable in E4M3, but multiplying by arbitrary NVFP4 E4M3 group scales can
require rounding. The W4A8 path is therefore not expected to be bit-identical to
W4A16 Marlin; it must be compared against a reference that performs the same
FP8 rounding and scale factoring.

## Components and boundaries

### NVFP4 loader adapter

Owns checkpoint recognition, tensor names, TP slicing, group/global scale
interpretation, shape metadata, and selection of an SM90 layout transformation.
It depends on existing vLLM NVFP4/compressed-tensors loaders. It does not own
activation quantization or GEMM policy.

### Activation quantizer

Uses the existing vLLM dynamic per-token E4M3 quantizer and returns FP8 values
plus FP32 per-token scales. A later project may fuse this work with a preceding
operator after the GEMM path is proven.

### Dense SM90 W4A8 kernel

Owns packed-weight loading, group-scale loading, in-mainloop conversion,
WGMMA, FP32 accumulation, and the scale-aware BF16 epilogue. It depends on
CUTLASS SM90 mixed-input components rather than Marlin's warp-level FP8 MMA.

### Kernel selector and fallback

Enables W4A8 only through an explicit experimental option and only on verified
SM90 devices, checkpoint schemes, layouts, dtypes, and aligned shapes. Every
unsupported case routes to the existing W4A16 Marlin path with a structured
reason. A conversion or correctness error fails closed; it must not silently
serve numerically suspect output.

### Reference and benchmark harness

Owns deterministic operand generation, reference W4A8 semantics, correctness
metrics, persistent-memory accounting, kernel latency, and model-serving
measurements. Raw measurements remain separate from the decision report.

## Staged execution and continuation gates

### Stage 0: upstream and version freeze

Before implementation, re-search vLLM, CUTLASS, FlashInfer, and related active
pull requests for an equivalent NVFP4-on-SM90 W4A8 path. Record exact vLLM,
CUTLASS, CUDA, driver, and H100/H200 versions. Adopt or rebase onto reputable
work if it exists. Bespoke work proceeds only if the exact combination remains
absent.

### Stage 1: deterministic reference and layout proof

Implement a framework-level reference for the numerical contract and prove
whether Marlin-repacked weights can be consumed by the selected CUTLASS/WGMMA
layout. If not, specify and test one SM90 repack transformation. This stage uses
small deterministic tensors and does not claim performance.

### Stage 2: dense kernel proof of concept

Support dense BF16-output linears on SM90 for all aligned rank-local projection
shapes found in the target model configuration. Benchmark every unique shape at
`M = 1, 8, 32, 128, 512, 2048`, covering decode through prefill behavior.

Correctness gates for every supported shape and `M`:

- no NaN or infinity in output or scales;
- cosine similarity to the deterministic W4A8 reference at least `0.999`;
- relative L2 error at most `0.02`;
- deterministic repeat results under the same inputs and configuration;
- unsupported or misaligned shapes select W4A16 before the first request.

Memory gate:

- persistent weight plus weight-scale bytes no more than `1.10x` the W4A16
  Marlin representation for the same unpadded logical tensor.

Performance continuation gate:

- after warmup, median kernel latency improves by at least `15%` over W4A16
  Marlin in at least one decision-relevant `M` bucket and is no more than `10%`
  slower in the primary serving bucket selected by the measured target workload;
- load-expanded W8A8 is measured as a throughput ceiling/control, not as a
  required result to beat;
- activation quantization time is included in W4A8 latency.

If correctness or memory fails, stop. If correctness and memory pass but the
performance continuation gate fails, retain the prototype and evidence but do
not begin fused-MoE production work.

### Stage 3: dense serving qualification

Integrate the qualified kernel behind an opt-in vLLM setting and compare W4A8,
W4A16 Marlin, and load-expanded W8A8 under the same model, prompts, topology,
scheduler settings, and request mix. Measure throughput, inter-token latency,
time to first token, peak HBM, and startup time.

The execution packet must satisfy the repository evaluation-harness contract
before GPU launch: record and verify tokenizer/chat-template hashes, reasoning
mode, task aliases and harness version, few-shot counts, metrics, generation and
sampling parameters, serving backend/topology, and sample-manifest hash. It must
state separately whether any quality run is directly comparable to a named
public recipe.

The model-level quality gate uses paired outputs under the existing evaluation
pipeline. It must show no statistically supported regression relative to the
predeclared paired tolerance; the packet must name that tolerance before launch
rather than selecting it after seeing results.

### Stage 4: MoE decision and productionization

Only after Stages 2 and 3 pass may a separate design cover fused MoE, expert
parallelism, routing, gate/up fusion, mixed-precision layers, and broader model
families. That work receives its own spec and implementation plan; this design
does not pre-authorize it.

## Alternatives considered

### Enable NVFP4 plus FP8 in current Marlin

This reuses more code and may be useful as a numerical experiment, but Marlin's
FP8 `mma.sync` path does not accelerate SM90. It is rejected as the production
architecture and may be used only as a bounded diagnostic if it reduces risk.

### Expand NVFP4 to FP8 at load time

This reuses mature Hopper W8A8 kernels and remains a useful performance ceiling.
It is rejected as the primary fallback because it nearly doubles persistent
weight memory and discards packed-weight bandwidth savings.

### Use only W4A16 Marlin

This is the reliable release-day compatibility baseline and remains the default
fallback. It does not test whether Hopper FP8 throughput can improve packed
NVFP4 serving, so it cannot satisfy the research goal by itself.

### Start with fused MoE

This would combine numerical, layout, routing, scheduling, and kernel questions
before the core GEMM is proven. It is rejected by the benchmark-gated scope.

## Error handling and rollout

- W4A8 is opt-in until dense serving qualification passes.
- Capability, scheme, shape, dtype, and layout checks run during model loading.
- Known unsupported cases route explicitly to W4A16 with a recorded reason.
- A failed scale/layout invariant stops loading rather than falling through after
  partial transformation.
- Runtime non-finite detection belongs in tests and diagnostic builds; production
  selection relies on preflight qualification rather than per-token scans.
- The original checkpoint is never modified.
- Evidence runs use fresh result roots and follow `PLANNER_EXECUTOR_PROTOCOL.md`.

## Estimated effort

- deterministic reference, layout proof, and bounded diagnostic: `1–2 weeks`;
- dense SM90 CUTLASS/WGMMA proof of concept: `4–8 weeks` for an experienced
  CUDA/CUTLASS engineer;
- dense vLLM integration and serving qualification: approximately 2–3
  person-months total including the proof of concept;
- fused MoE, TP/EP hardening, and broad-model production coverage: a separate
  `3–5 person-month` project if the continuation gate passes.

These are planning ranges, not schedule commitments. Stage 0 may materially
reduce them if equivalent upstream work appears.

## Acceptance criteria

- The repository records packed NVFP4 W4A8 as a benchmark-gated long-term goal.
- The implementation plan starts with an upstream search and dense reference,
  not bespoke fused MoE work.
- The dense kernel retains packed four-bit persistent weights and uses SM90 FP8
  WGMMA with FP32 accumulation and BF16 output.
- W4A16 Marlin remains the default fallback until qualification completes.
- Correctness, memory, and performance gates are machine-readable and evaluated
  before MoE work is authorized.
- Cluster quality evaluation is fail-closed under the repository harness
  contract and never conflates paired internal evidence with a public score.
- Failure to pass the continuation gate stops the project without displacing the
  working W4A16 release-day path.
