# MiniMax-M3 Hopper W4A8 Kernel Investigation

**Status:** planner investigation; no additional cluster work authorized

**Date:** 2026-07-25

**Target:** MiniMax-M3 routed-expert MoE on H100/SM90, TP8 + EP

**Checkpoint contract:** symmetric GPTQ INT4 weights, group size 128,
dynamic per-token activation quantization, BF16 output

**Implementation baselines audited:**

- vLLM v0.24.0 at
  [`ee0da84`](https://github.com/vllm-project/vllm/commit/ee0da84ab9e04ac7610e28580af62c365e898389)
- vLLM `main` at
  [`0ba2aa3`](https://github.com/vllm-project/vllm/commit/0ba2aa35a81dcc3246b26291368b53fa2389c7d7)
- Humming v0.1.10 at
  [`4351af3`](https://github.com/inclusionAI/humming/commit/4351af3a8fcdce1a8dee50104ba49566af2427fb)

This document answers two questions:

1. Which existing vLLM-accessible kernels can plausibly serve the checkpoint's
   W4A8 path efficiently on Hopper?
2. What computation do Marlin W4A8-INT8 and W4A8-FP8 actually perform, and why
   does vLLM reject Marlin W4A8-FP8 on SM90?

The active qualification procedure remains
[`M3_HUMMING_W4A8_HANDOFF.md`](M3_HUMMING_W4A8_HANDOFF.md). This investigation
does not replace it or authorize extra GPU allocations.

## Executive conclusion

For the exact **packed INT4 weight + per-token FP8 activation** contract on
H100, the production candidates already present in the audited software are:

1. **CUTLASS W4A8-FP8** — the current stable baseline and the only backend in
   vLLM's generic W4A8 MoE oracle.
2. **Humming W4A8-FP8** — a Hopper-native WGMMA/TMA implementation with direct
   GPTQ loading and vLLM integration. Its indexed mode is the active
   qualification candidate; grouped/automatic scheduling is a later optimization
   only after indexed correctness is established.

The useful adjacent paths are:

- **Marlin W4A8-INT8** — a genuine W4A8 integer Tensor Core path, not W4A16.
  It retains packed 4-bit checkpoint weights, dynamically quantizes activations
  to INT8, expands weight nibbles to signed INT8 only in registers, executes
  `s8 × s8 -> s32` MMA, and applies activation and weight scales around the
  accumulated result. It is a valid third benchmark arm, but it changes the
  activation format from FP8 to INT8 and is therefore not an exact
  W4A8-FP8 comparison.
- **Marlin W4A16** — the existing weight-only fallback. It is not W4A8.
- **Load-expanded W8A8-FP8** through an existing FP8 MoE backend — a useful
  throughput ceiling or diagnostic baseline, but it doubles persistent weight
  storage relative to packed W4 and is not the requested W4A8 serving path.

There is no justification yet for writing a new CUDA kernel. CUTLASS, Humming,
and Marlin INT8 cover the meaningful existing implementation space. A bespoke
kernel should be considered only if those measured paths fail the required
latency/throughput target and profiling identifies a kernel-local gap.

## Terminology: what “W4A8” means here

The labels describe the quantized operands presented to the matrix-multiply
pipeline, not necessarily the bit width of every temporary register:

- **W4**: weights are stored persistently as packed 4-bit values, with their
  quantization metadata.
- **A8**: the activation operand is quantized to an 8-bit format at runtime.
- **W4A16**: packed W4 weights are reconstructed into FP16/BF16 fragments and
  multiplied by FP16/BF16 activations.
- **W4A8-INT8**: A is INT8; the 4-bit integer weight code is unpacked into an
  8-bit integer register representation required by the integer Tensor Core
  instruction.
- **W4A8-FP8**: A is FP8; the 4-bit weight is converted into an FP8 fragment
  required by the FP8 Tensor Core instruction.

Register expansion does **not** turn a packed W4 checkpoint into W8 persistent
storage. It does affect the instruction and register-level implementation, so
the exact MMA instruction must be inspected rather than inferred from the name.

## Candidate matrix

| Backend/path | Persistent weight | Runtime activation | Tensor Core math | SM90 status | Exact target? | Assessment |
|---|---|---|---|---|---|---|
| CUTLASS W4A8 | packed signed INT4 after load-time reorder | dynamic per-token E4M3 | Hopper mixed-input W4/FP8 implementation | supported | yes | Stable baseline; current generic vLLM W4A8 MoE oracle |
| Humming indexed W4A8 | packed GPTQ INT4 after Humming transform | dynamic per-token E4M3 | SM90 WGMMA/TMA with register-side W4 conversion/scaling | supported | yes | Highest-priority challenger; active executor test |
| Humming grouped/auto W4A8 | same as indexed | same | same primitive with alternative scheduling | supported by Humming; integration must be qualified | yes | Test only after indexed correctness |
| Marlin W4A8-INT8 | packed GPTQ INT4 | dynamic per-token INT8 | `s8 × s8 -> s32`, then scale/reduce | supported | no: A8 format differs | Real W4A8; useful third arm |
| Marlin W4A8-FP8 | packed GPTQ INT4 | dynamic per-token E4M3 | warp-level `mma.sync` E4M3 | intentionally not built on SM90 | conceptually yes | Not a viable H100 candidate in current Marlin |
| Marlin W4A16 | packed GPTQ INT4 | BF16/FP16 | W4 reconstructed to BF16/FP16 MMA operand | supported | no | Existing fallback; strong low-concurrency reference |
| Triton MoE W8A8 | FP8/INT8 W8 | dynamic FP8/INT8 | W8A8 Triton dot | supported where scheme allows | no | Requires load expansion/requantization to W8 |
| FlashInfer CUTLASS MoE W8A8 | FP8 W8 | FP8 | FP8 CUTLASS | supported on SM90 | no | W8A8 alternative, not GPTQ W4A8 |
| DeepGEMM MoE | FP8 W8 | FP8 | FP8 GEMM | supported configurations only | no | W8A8 alternative, not W4A8 |
| Machete | packed W4 | BF16/FP16 | Hopper WNA16 | SM90 dense linear only | no | W4A16 and not a fused-MoE replacement |
| FlashInfer MXFP4/CuTeDSL | MXFP4 | A16 or backend-specific | FP4-family kernels | relevant fast paths target SM100 | no | Different format/architecture contract |
| TRTLLM MXINT4 MoE | MXINT4, group 32 | A16 | MXINT4 MoE | vLLM restricts to SM100 | no | Wrong architecture, group size, and activation type |
| INT4 emulation | expanded BF16 weights | BF16 | Triton higher-precision GEMM | available on newer main | no | Debug/correctness path, not a performance candidate |
| ExLlama / AllSpark / Conch | backend-specific | primarily A16 | dense mixed-precision paths | partial | no | Not a direct fused-MoE W4A8 path |
| AITER | backend-specific | backend-specific | ROCm | not NVIDIA | no | Excluded |

## 1. CUTLASS W4A8-FP8

### What can be reused

The audited vLLM release contains both dense and fused-MoE W4A8 support:

- The
  [W4A8 MoE oracle](https://github.com/vllm-project/vllm/blob/v0.24.0/vllm/model_executor/layers/fused_moe/oracle/w4a8.py)
  recognizes static INT4 weights plus dynamic per-token FP8 activations.
- The
  [CUTLASS MoE experts wrapper](https://github.com/vllm-project/vllm/blob/v0.24.0/vllm/model_executor/layers/fused_moe/experts/cutlass_moe.py)
  owns the fused expert execution.
- The
  [compressed-tensors W4A8 method](https://github.com/vllm-project/vllm/blob/v0.24.0/vllm/model_executor/layers/quantization/compressed_tensors/compressed_tensors_moe/compressed_tensors_moe_w4a8_fp8.py)
  connects checkpoint metadata to that oracle.
- The
  [dense linear wrapper](https://github.com/vllm-project/vllm/blob/v0.24.0/vllm/model_executor/kernels/linear/mixed_precision/cutlass.py)
  provides the corresponding non-MoE path.

At load/prepare time the oracle:

1. converts the checkpoint's biased unsigned packed representation to signed
   INT4;
2. performs the CUTLASS encode/reorder;
3. converts and packs weight scales into the expected FP8/scaled layout; and
4. constructs a quantization configuration with per-token activation and
   per-output-channel scaling.

The CUDA implementation is split across:

- [grouped W4A8 entry point](https://github.com/vllm-project/vllm/blob/v0.24.0/csrc/libtorch_stable/quantization/cutlass_w4a8/w4a8_grouped_mm_entry.cu)
- [dense W4A8 entry point](https://github.com/vllm-project/vllm/blob/v0.24.0/csrc/libtorch_stable/quantization/cutlass_w4a8/w4a8_mm_entry.cu)
- [weight/scale encode and reorder utilities](https://github.com/vllm-project/vllm/blob/v0.24.0/csrc/libtorch_stable/quantization/cutlass_w4a8/w4a8_utils.cu)

### Limitation

The generic vLLM W4A8 MoE oracle contains only one enum value:
`CUTLASS`. This is still true in the audited
[`main` oracle](https://github.com/vllm-project/vllm/blob/0ba2aa35a81dcc3246b26291368b53fa2389c7d7/vllm/model_executor/layers/fused_moe/oracle/w4a8.py).
Humming is not an alternative selected by that oracle; it enters through the
separate Humming quantization method.

CUTLASS is therefore the correct stable control, but not evidence that no other
installed kernel can execute the same numerical contract.

## 2. Humming W4A8-FP8

### Why it is the strongest direct challenger

Humming is designed around Hopper's warpgroup machinery rather than Marlin's
warp-level MMA schedule. Its implementation includes:

- direct GPTQ schema/loading support;
- dynamic input quantization;
- indexed and grouped MoE wrappers;
- SM90 tuning logic;
- TMA-oriented data movement and WGMMA execution; and
- W4-to-compute-fragment conversion within the kernel rather than permanent
  expansion of the checkpoint.

The vLLM-side integration is already in the audited release:

- [Humming quantization method](https://github.com/vllm-project/vllm/blob/v0.24.0/vllm/model_executor/layers/quantization/humming.py)
- [fused Humming MoE wrapper](https://github.com/vllm-project/vllm/blob/v0.24.0/vllm/model_executor/layers/fused_moe/experts/fused_humming_moe.py)
- [dense Humming wrapper](https://github.com/vllm-project/vllm/blob/v0.24.0/vllm/model_executor/kernels/linear/mixed_precision/humming.py)

The Humming implementation audited at v0.1.10 is:

- [GPTQ checkpoint adapter](https://github.com/inclusionAI/humming/blob/4351af3a8fcdce1a8dee50104ba49566af2427fb/humming/schema/gptq.py)
- [activation quantization](https://github.com/inclusionAI/humming/blob/4351af3a8fcdce1a8dee50104ba49566af2427fb/humming/ops/input.py)
- [MoE Python wrapper](https://github.com/inclusionAI/humming/blob/4351af3a8fcdce1a8dee50104ba49566af2427fb/humming/ops/moe.py)
- [JIT/configuration layer](https://github.com/inclusionAI/humming/blob/4351af3a8fcdce1a8dee50104ba49566af2427fb/humming/kernel/humming.py)
- [main CUDA kernel](https://github.com/inclusionAI/humming/blob/4351af3a8fcdce1a8dee50104ba49566af2427fb/humming/include/humming/kernel/humming.cuh)
- [WGMMA primitives](https://github.com/inclusionAI/humming/blob/4351af3a8fcdce1a8dee50104ba49566af2427fb/humming/include/humming/mma/wgmma.cuh)
- [generic dequantization](https://github.com/inclusionAI/humming/blob/4351af3a8fcdce1a8dee50104ba49566af2427fb/humming/include/humming/datatype/dequant.cuh)
- [fused MXFP4 conversion](https://github.com/inclusionAI/humming/blob/4351af3a8fcdce1a8dee50104ba49566af2427fb/humming/include/humming/datatype/dequant_fused.cuh)
- [SM90 tuning rules](https://github.com/inclusionAI/humming/blob/4351af3a8fcdce1a8dee50104ba49566af2427fb/humming/tune/sm90.py)
- [upstream benchmark-result tree](https://github.com/inclusionAI/humming/tree/v0.1.10/benchmarks/results)

### What remains to qualify

The existence of the implementation does not prove it is faster for MiniMax-M3.
Performance depends on the model's local expert shapes, routed-token histogram,
TP8/EP topology, kernel scheduling mode, launch overhead, and the amount of
non-GEMM MoE work.

The active executor test should therefore answer, in order:

1. Does the installed Humming build load this exact GPTQ group-128 checkpoint?
2. Does indexed Humming match the reference output within the agreed tolerance?
3. Is the requested backend actually selected for both expert GEMMs?
4. Does it improve the existing serving benchmark at the target concurrency
   points?
5. Only after 1–4 pass: does grouped/automatic scheduling improve the relevant
   routed-token distributions?

## 3. Marlin W4A8-INT8 is genuine W4A8

### Short answer

**Yes. Marlin W4A8-INT8 is an actual W4A8 integer compute path. It does not
fall back to W4A16.**

It is important to distinguish three representations:

1. The checkpoint and persistent device weight tensor remain packed INT4.
2. Each packed 4-bit weight code is expanded transiently into the signed INT8
   register layout required by the Tensor Core instruction.
3. The activation is dynamically quantized to INT8 per token.

This is normal W4A8 implementation behavior. An `s8 × s8` Tensor Core
instruction cannot directly consume a four-bit C++ register operand, so the
kernel reconstructs the W4 codes into the instruction's eight-bit operand
encoding at the last responsible moment.

### End-to-end source trace

#### A. Selection and activation quantization

[`get_marlin_input_dtype`](https://github.com/vllm-project/vllm/blob/v0.24.0/vllm/model_executor/layers/quantization/utils/marlin_utils.py)
reads `VLLM_MARLIN_INPUT_DTYPE`:

- unset: returns no eight-bit input type, retaining the W4A16 path;
- `int8`: returns `torch.int8`;
- `fp8`: requests `torch.float8_e4m3fn`, subject to the architecture guard
  discussed below.

The same file's `marlin_quant_input` calls `per_token_quant_int8` for the INT8
case. The
[Marlin MoE wrapper](https://github.com/vllm-project/vllm/blob/v0.24.0/vllm/model_executor/layers/fused_moe/experts/marlin_moe.py)
uses it before both expert GEMMs:

- hidden state to gate/up projection; and
- post-activation intermediate to down projection.

The resulting per-token activation scales are passed into
`ops.moe_wna16_marlin_gemm`. Thus, selecting INT8 changes the operand supplied
to the compiled kernel; it is not a label applied to an unchanged BF16 input.

#### B. Compiled kernel variant

The
[Marlin MoE kernel generator](https://github.com/vllm-project/vllm/blob/v0.24.0/csrc/libtorch_stable/moe/marlin_moe_wna16/generate_kernels.py)
contains a GPTQ-INT4 variant with:

- `a_type = kS8`;
- `b_type = kU4B8`; and
- BF16/FP16 output variants.

That is distinct from the ordinary BF16/FP16-activation Marlin
instantiations. The dense generated variants are defined by the corresponding
[dense Marlin generator](https://github.com/vllm-project/vllm/blob/v0.24.0/csrc/libtorch_stable/quantization/marlin/generate_kernels.py).

#### C. Weight unpacking is register-local

The
[`dequant<int32_t, kU4B8, true>` specialization](https://github.com/vllm-project/vllm/blob/v0.24.0/csrc/libtorch_stable/quantization/marlin/dequant.h)
extracts packed GPTQ nibbles and constructs signed INT8 lanes in registers. The
packed source tensor remains W4. There is no load-time conversion of the entire
weight matrix to W8 for this path.

#### D. The MMA is integer, not BF16/FP16

The
[Marlin MMA implementation](https://github.com/vllm-project/vllm/blob/v0.24.0/csrc/libtorch_stable/quantization/marlin/marlin_mma.h)
selects:

```text
mma.sync.aligned.m16n8k32.row.col.s32.s8.s8.s32.satfinite
```

for the relevant eight-bit integer path. Both Tensor Core operands are signed
INT8 and the immediate MMA accumulator is signed INT32. This is direct evidence
that W4A8-INT8 does not execute the W4A16 floating-point MMA.

#### E. Scaling and output

The
[Marlin MoE kernel template](https://github.com/vllm-project/vllm/blob/v0.24.0/csrc/libtorch_stable/moe/marlin_moe_wna16/marlin_template.h)
does the following:

1. accumulates the `s8 × s8` MMA result in INT32;
2. converts the integer partial result to FP32;
3. incorporates weight group scales while combining group partials;
4. multiplies by the per-token activation scale;
5. performs the required cross-block reduction; and
6. converts the final result to the configured BF16/FP16 output.

For groupwise weights, “FP32 accumulation” therefore needs a precise
qualification: each Tensor Core dot-product fragment accumulates in INT32, and
scaled group contributions are then combined in FP32. It is not a single
unscaled INT32 accumulator spanning arbitrarily different weight-scale groups.

### Operational caveats

- This path requires an explicit Marlin-selected model/quantization arm plus
  `VLLM_MARLIN_INPUT_DTYPE=int8`. The generic compressed-tensors W4A8-FP8
  oracle still selects CUTLASS; setting the environment variable alone does not
  redirect that oracle to Marlin.
- If the variable is missing or ignored, the same Marlin arm remains W4A16.
  Benchmark evidence must therefore record the environment, activation dtype,
  selected kernel, and preferably a profiler/kernel-name attestation.
- INT8 and E4M3 have different dynamic range and quantization error. The Marlin
  INT8 result is a valid systems comparison, but quality parity with the
  official W4A8-FP8 recipe must be measured rather than assumed.

## 4. Why Marlin W4A8-FP8 is unavailable on SM90

### Short answer

H100 supports native FP8 Tensor Core computation, but **the current Marlin FP8
path uses the wrong instruction family to exploit Hopper efficiently**.
Marlin emits warp-level `mma.sync` FP8 instructions. vLLM's generator states
that the relevant instruction is fully accelerated on SM89 and SM12x, while on
SM90/SM100 it can be accepted but is simulated using FP16 MMA and provides no
acceleration. vLLM therefore deliberately omits the FP8 Marlin instantiations
on SM90 and rejects the request in Python.

This is an implementation/scheduling limitation of Marlin, not a statement that
H100 lacks FP8 hardware.

### Source trace

#### A. Python rejects SM90

The
[`get_marlin_input_dtype` guard](https://github.com/vllm-project/vllm/blob/v0.24.0/vllm/model_executor/layers/quantization/utils/marlin_utils.py)
accepts Marlin FP8 activations only for:

- exact SM89; or
- the SM12x family.

For other architectures it raises an error saying W4A8-FP8 is slower than
W4A16 and suggests `VLLM_MARLIN_INPUT_DTYPE=int8`.

#### B. The build generator omits SM90 variants

The
[generated-kernel rules](https://github.com/vllm-project/vllm/blob/v0.24.0/csrc/libtorch_stable/moe/marlin_moe_wna16/generate_kernels.py)
set `SUPPORT_FP8` only for SM89 or SM12x. Their accompanying source comment is
the decisive implementation rationale:

- SM89 and SM12x fully support the selected
  `mma.sync.aligned.m16n8k32...e4m3...` instruction;
- SM90 and SM100 may accept it, but it is simulated through FP16 MMA and cannot
  deliver acceleration.

When `SUPPORT_FP8` is false, the generator skips FP8-activation template
instantiations and emits a runtime selector failure saying the FP8-activation
Marlin kernel was not built.

#### C. Marlin uses warp-level MMA

The
[PTX in `marlin_mma.h`](https://github.com/vllm-project/vllm/blob/v0.24.0/csrc/libtorch_stable/quantization/marlin/marlin_mma.h)
uses forms such as:

```text
mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32
```

This is the warp-level `mma.sync` family. Hopper's high-throughput FP8 route is
the warpgroup asynchronous `wgmma.mma_async` family, normally paired with
Hopper-oriented scheduling and data movement. Humming and Hopper-targeted
CUTLASS kernels are structured around that machinery; current Marlin is not.

The architecture distinction is documented in NVIDIA's:

- [PTX ISA documentation](https://docs.nvidia.com/cuda/parallel-thread-execution/)
- [Hopper tuning guide](https://docs.nvidia.com/cuda/hopper-tuning-guide/index.html)

#### D. Removing the guard is not a performance fix

Deleting the Python check or forcing the FP8 template to compile would at best
expose the same non-accelerated instruction mapping that the generator warns
about. It would not convert Marlin's warp-level pipeline into an SM90
WGMMA/TMA kernel. Doing that properly would be a substantial new kernel design,
not a one-line enablement.

For H100, the practical choices are therefore:

- CUTLASS or Humming for W4A8-FP8;
- Marlin for W4A8-INT8; or
- Marlin W4A16 as the existing weight-only fallback.

## 5. Other vLLM-accessible paths

### Triton MoE

The
[Triton experts implementation](https://github.com/vllm-project/vllm/blob/v0.24.0/vllm/model_executor/layers/fused_moe/experts/triton_moe.py)
supports native W8A8 schemes:

- static channel INT8 weights + dynamic per-token INT8 activations; and
- FP8 weights + dynamic FP8 activations, including supported blockwise forms.

Its packed INT4 branch is explicitly `use_int4_w4a16`. The source does not
offer a combined GPTQ INT4 + FP8 activation W4A8 path. Triton is therefore
relevant only after expanding/requantizing the weights to W8, or as a W4A16
fallback—not as a drop-in replacement for the target kernel.

### FlashInfer CUTLASS MoE

The current-main
[FlashInfer CUTLASS MoE wrapper](https://github.com/vllm-project/vllm/blob/0ba2aa35a81dcc3246b26291368b53fa2389c7d7/vllm/model_executor/layers/fused_moe/experts/flashinfer_cutlass_moe.py)
offers useful FP8 W8A8 and other supported quantized MoE paths. It does not
provide the target GPTQ INT4 group-128 + dynamic FP8 combination on SM90.

FlashInfer's CuTeDSL/NVFP4 and TRTLLM-style FP4 integrations should not be
conflated with this GPTQ INT4 checkpoint. Their weight encoding, scale
granularity, supported architectures, and activation contracts differ.

### DeepGEMM

The
[DeepGEMM MoE wrapper](https://github.com/vllm-project/vllm/blob/v0.24.0/vllm/model_executor/layers/fused_moe/experts/deep_gemm_moe.py)
is an FP8 W8A8 backend, not an INT4-weight backend. It can be part of a
load-expanded W8A8 ceiling experiment, but cannot directly consume this packed
GPTQ W4 checkpoint as W4A8.

### Machete

The
[Machete dense wrapper](https://github.com/vllm-project/vllm/blob/v0.24.0/vllm/model_executor/kernels/linear/mixed_precision/machete.py)
and
[Machete utilities](https://github.com/vllm-project/vllm/blob/v0.24.0/vllm/model_executor/layers/quantization/utils/machete_utils.py)
target Hopper mixed-precision dense linear operations. The relevant packed
low-bit configurations use FP16/BF16 activations: W4A16. There is no fused-MoE
W4A8 replacement to select for this target.

### TRTLLM MXINT4 MoE

The
[TRTLLM MXINT4 MoE wrapper](https://github.com/vllm-project/vllm/blob/v0.24.0/vllm/model_executor/layers/fused_moe/experts/trtllm_mxint4_moe.py)
is restricted by vLLM to SM100-family use and carries a different MXINT4,
group-32, A16 contract. It is not an H100 GPTQ group-128 W4A8 option.

### INT4 emulation

Current vLLM main includes an
[INT4 emulation MoE path](https://github.com/vllm-project/vllm/blob/0ba2aa35a81dcc3246b26291368b53fa2389c7d7/vllm/model_executor/layers/fused_moe/experts/int4_emulation_moe.py).
It expands weights to BF16 and runs higher-precision Triton computation. This is
useful for diagnosis or correctness isolation, but it discards the memory and
compute objective of W4A8.

### Dense-only and non-NVIDIA backends

- [AllSpark](https://github.com/vllm-project/vllm/blob/v0.24.0/vllm/model_executor/kernels/linear/mixed_precision/allspark.py),
  [Conch](https://github.com/vllm-project/vllm/blob/v0.24.0/vllm/model_executor/kernels/linear/mixed_precision/conch.py),
  and
  [ExLlama](https://github.com/vllm-project/vllm/blob/v0.24.0/vllm/model_executor/kernels/linear/mixed_precision/exllama.py)
  do not supply the required fused-MoE GPTQ W4A8-FP8 path.
- AITER is a ROCm backend and is outside the H100 target.
- BitBLAS was not present as a selectable implementation in the audited vLLM
  v0.24.0 source tree.

## 6. Recommended experiment order

No new benchmark harness is needed; use the existing serving benchmark and
preserve the same model, topology, request set, concurrency points, and
measurement protocol.

| Priority | Arm | Purpose | Gate |
|---|---|---|---|
| 1 | Existing CUTLASS W4A8-FP8 | Stable control | Already established |
| 2 | Humming indexed W4A8-FP8 | Exact-contract Hopper challenger | Loader, correctness, backend attestation, then performance |
| 3 | Humming grouped/auto | Scheduling optimization | Only if indexed passes |
| 4 | Marlin W4A8-INT8 | Existing-kernel alternative with true A8 integer math | Explicit env/dtype/kernel attestation and quality check |
| 5 | Load-expanded W8A8-FP8 | Estimate whether W4 unpack/dequant is the limiting cost | Treat as a different memory contract |
| 6 | New kernel work | Last resort | Only after profiling proves an uncovered kernel-local bottleneck |

### Required interpretation discipline

- Compare Humming and CUTLASS directly as W4A8-FP8 implementations.
- Compare Marlin INT8 as **W4A8-INT8**, not as proof about FP8.
- Compare Marlin's ordinary mode as **W4A16**, even if it wins latency.
- Report persistent weight memory separately for any W8A8 expansion arm.
- Do not attribute end-to-end serving differences solely to GEMM speed without
  checking routing, sorting, activation quantization, collective communication,
  and CUDA-graph behavior.

## 7. Decision

Continue the current Humming indexed qualification. If it passes correctness but
does not beat CUTLASS:

1. profile the two paths at representative routed-token counts;
2. test Humming's grouped/automatic scheduling if its integration is supported;
3. add one bounded Marlin W4A8-INT8 arm using the already-existing Marlin
   implementation;
4. optionally test load-expanded W8A8-FP8 as a diagnostic ceiling; and
5. consider bespoke CUDA/CuTeDSL work only if the evidence identifies a specific
   gap not covered by these implementations.

This ordering follows the repository's prime directive: use and measure
reputable existing implementations before designing a new kernel.
