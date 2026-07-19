# Hopper Packed-NVFP4 W4A8 Fallback Design

**Date:** 2026-07-19
**Scope:** vLLM serving compatibility and kernel research for H100/H200
**Status:** Revised implementation-grade reuse map; awaiting review before the
implementation plan and kernel work

## Problem

Official model releases may provide only NVFP4 checkpoints intended for native
Blackwell FP4 execution. Hopper supports FP8 Tensor Cores but not native FP4
Tensor Core multiplication. vLLM can already serve NVFP4 on Hopper through
Marlin W4A16 or software dequantization. The initial two-reference audit appeared
to show no packed-weight NVFP4 W4A8 path, but the broader upstream search found
one in the separately packaged Humming backend. Humming supplies the closest
execution machinery, but its current FP8 path cannot apply two NVFP4 group-16
scales inside one K=32 WGMMA step. The remaining work is therefore a bounded
Humming specialization plus qualification and target-version integration, not
clean-sheet kernel design.

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
- Humming `0.1.10`, pinned by current upstream vLLM, already supports E4M3
  activations with arbitrary signed FP weights including E2M1 on SM89+, packed
  weight repacking, compact group scales, global scales, dynamic input
  quantization, and WGMMA. Its compressed-tensors input adapter explicitly maps
  NVFP4 A4 input to per-token E4M3 on SM90.
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
- <https://github.com/vllm-project/vllm/blob/main/csrc/libtorch_stable/quantization/marlin/dequant.h>
- <https://github.com/vllm-project/vllm/blob/main/vllm/model_executor/kernels/linear/mixed_precision/cutlass.py>
- <https://github.com/vllm-project/vllm/blob/main/csrc/libtorch_stable/quantization/cutlass_w4a8/w4a8_mm_entry.cu>
- <https://github.com/vllm-project/vllm/blob/main/vllm/model_executor/layers/quantization/humming.py>
- <https://github.com/inclusionAI/humming>
- <https://github.com/NVIDIA/cutlass/blob/main/CHANGELOG.md>
- <https://docs.nvidia.com/cutlass/4.5.2/media/docs/pythonDSL/mma_docs/wgmma_programming.html>

### Reference snapshot

The serving environment evidenced in this repository is vLLM `0.24.0` from
`toncao/vllm@minimax-m3-compressed-tensors`. The latest-upstream audit for this
reuse map inspected `vllm-project/vllm` commit
`b6ff8a2f509cc7ac9c58176f5115a836aa1e08bd` on 2026-07-19. Its native CUTLASS
NVFP4 path is hardware-gated and its Hopper-compatible Marlin path is W4A16,
but its Humming backend is a direct candidate for this goal. Upstream vLLM pins
`humming-kernels==0.1.10`; the corresponding Humming tag is commit
`4351af3a8fcdce1a8dee50104ba49566af2427fb`. It contains an H200 benchmark for
E4M3 activations times packed E2M1 weights and generic tests for E2M1, group-16
scales, E4M3 scales, global scales, and WGMMA. The exact NVFP4 cross-product
(E2M1 + group-16 E4M3 + global scale + dynamic A8) is not represented by one
single inspected test or benchmark, so it must be qualified rather than assumed.

The paths in this document follow that current-upstream snapshot. The vLLM
`0.24.0` target may use the pre-`libtorch_stable` C++ path names. Stage 0 must
pin the exact target vLLM fork commit and map these symbols before any patch is
written; version strings alone are not sufficient.

## Goals

1. Load an official NVFP4 checkpoint without offline conversion.
2. Keep persistent linear weights packed as E2M1 at approximately NVFP4/Marlin
   memory cost throughout serving.
3. Dynamically quantize activations per token to FP8 E4M3.
4. Use an existing optimized kernel to convert packed NVFP4 fragments to FP8
   registers and apply compact group/global scales around SM90 FP8 WGMMA.
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
- Writing a clean-sheet GEMM while an existing reputable Humming path remains
  untested.
- Implementing fused MoE before the dense continuation gate passes.
- Claiming direct public-benchmark comparability from a paired subset or kernel
  microbenchmark.

## Chosen approach

Extend and qualify the existing **Humming W4A8 path first**. On the inspected
upstream, launching an NVFP4 compressed-tensors checkpoint with
`--quantization humming` causes `HummingConfig` to parse both the NVFP4 weight
and input schemas. Humming preserves packed E2M1 weights, recognizes group-16
E4M3 plus global weight scales, and maps the unsupported A4 request to dynamic
per-token E4M3 on SM90 through `get_fallback_input_dtype`.

Unmodified Humming is close but not sufficient. For non-MXMMA FP8 execution,
`HummingKernel.check_scale` requires every weight-scale group to be at least the
MMA K quantum, `256 / activation_bits = 32`. NVFP4 group size is 16. Its generic
scale tests deliberately skip this combination, and its published H200
E2M1/E4M3 benchmark uses channelwise (`g0`) weight scaling. The exact missing
mechanism is therefore a group-16 specialization: each K=32 WGMMA weight
fragment must load two E4M3 scales, apply each scale to the corresponding
16-wide half after E2M1-to-E4M3 register conversion, and then issue FP8 WGMMA.

Do not confuse this with `--linear-backend humming`: the inspected generic
`HummingNvFp4LinearKernel` adapter passes only the weight schema and therefore
defaults to an unquantized input schema. The full `--quantization humming`
override is the correct integration route **after** the group-16 specialization
because it carries the checkpoint input schema through Humming's A4-to-A8
platform fallback.

The first execution packet will verify whether the target
`toncao/vllm@minimax-m3-compressed-tensors` checkout contains the same Humming
integration and dependency. If it does, implementation is limited to launch
configuration, one bounded Humming kernel specialization, exact-combination
regression tests, evidence capture, and possibly selector telemetry. If it does
not, backport the reputable upstream Humming integration first. A bespoke
CUTLASS datatype variant is authorized only if the Humming specialization is
absent or fails a predeclared correctness, memory, or performance gate for
reasons that cannot be fixed locally.

## Direct implementation reuse matrix

| Component | CUTLASS W4A8 reference | NVFP4/Marlin reference | Humming direct implementation | Work remaining in this project |
|---|---|---|---|---|
| Checkpoint recognition | Mixed-precision selector pattern only | NVFP4 tensor names and format detection | `HummingConfig` reads compressed-tensors weight and input schemes directly | Verify the exact target fork; backport upstream integration only if absent |
| Packed weight allocation/TP loading | Packed-parameter metadata conventions | Authoritative `uint8 [N,K/2]` E2M1 checkpoint layout | `CompressedTensorsWeightSchema` consumes `nvfp4-pack-quantized` and views/reorders it without persistent expansion | Exact fused/unfused TP round-trip tests |
| NVFP4 group/global scale semantics | The `/8` group-scale headroom and compensating epilogue factor are reusable | Group-16 E4M3 and inverse global-scale meaning | Maps NVFP4 to `GROUP_TENSOR`, retains E4M3 scales, and inverts/folds checkpoint global scale | Apply `/8` while constructing scaled B registers and compensate with `global_scale * 8`; initially require one shared scalar global scale per fused physical layer and route mismatches to Marlin; exact fused-width and rounding tests |
| A4-to-A8 fallback | Dynamic `QuantFP8` shows desired per-token policy | Stored NVFP4 input-global scale documents original A4 recipe | `CompressedTensorsInputSchema.convert_humming` calls `get_fallback_input_dtype`; E2M1 activation becomes E4M3 with group size `0` on SM90 | Verify runtime metadata and prove stored A4 scale is not accidentally used |
| Dynamic activation quantization | Existing vLLM per-token E4M3 behavior | None | `HummingMethod` quantizes input dynamically when the prepared input schema is E4M3 | Accuracy/timing measurement including quantization; optional future fusion only after qualification |
| Weight reorder | CUTLASS reordered-layout concept | Marlin nibble semantics/layout oracle | `prepare_humming_weight`/`repack_weight` already handle packed E2M1 for FP8 execution | Layout/codepoint regression against logical NVFP4; no new repacker unless a bug is found |
| Compact group-scale layout | Existing W4A8 eight-way pack is unsuitable at g16 | Marlin proves compact scale storage is viable | `prepare_humming_weight_scale` permutes in place with unchanged tensor shape rather than eight-way replication | Reuse the compact tensor and loader infrastructure, but extend the WGMMA register mapping to provide two scales per K=32 fragment; enforce the `1.10x` memory gate |
| E2M1-to-E4M3 conversion | CUTLASS indicates the SM90 mixed-input/WGMMA placement | Marlin supplies an independent E2M1 conversion oracle | Humming WGMMA already loads packed E2M1 and dequantizes it to the E4M3 operand representation in registers | Reuse unchanged, then add scale-and-round on each 16-wide register half; exact all-16-code/edge-scale tests and SASS confirmation |
| SM90 WGMMA/TMA pipeline | Mature schedules and performance comparison | Not reused | Humming provides WGMMA, TMA, warp specialization, scale loads, and tuning/JIT infrastructure, but its ordinary FP8 path requires scale groups at least K=32 | Add one guarded E2M1/A8/g16 policy and tune it; do not build a new GEMM framework |
| FP32 accumulation/BF16 output | Existing CUTLASS contract | W4A16 baseline | Humming supports BF16 output and non-F16 accumulation by default | Assert `VLLM_HUMMING_USE_F16_ACCUM` is disabled and compare to deterministic reference |
| Bias/reshape/CUDA graph | Existing vLLM patterns | Existing NVFP4 interface | Humming vLLM method owns flatten/restore and kernel forwarding | Bias, capture, repeatability, and serving smoke tests |
| Backend selection | INT4 W4A8 selector is not reused | Normal NVFP4 selection reaches Marlin on Hopper | `--quantization humming` is the current explicit opt-in | Add fail-closed preflight and structured metadata; optional dedicated selector only after success |
| Compatibility fallback | Existing failure-reason pattern | W4A16 Marlin remains the reliable path | Separate launch arm, not an in-kernel retry | Preflight selects the whole arm before model mutation; retain normal NVFP4/Marlin command |
| Existing evidence | INT4 W4A8 tests | Marlin NVFP4 tests | H200 W4A8 E2M1/E4M3 g0 benchmark plus orthogonal datatype/scale/global/WGMMA tests; `check_scale` rejects A8/g16 | Exact NVFP4 cross-product kernel test and target-shape benchmark are genuinely new |
| Fused MoE | Grouped W4A8 is a later reference | NVFP4 MoE loaders exist | Humming supports MoE, but this design does not qualify it | Out of scope until dense gates pass |

## Genuinely new work on the chosen path

| Work package | Size/risk | Definition of done |
|---|---|---|
| H1. Target-version capability audit | Small | Exact vLLM fork commit, Humming integration files, `humming-kernels` version, CUDA/PyTorch/NVCC ABI, and H100/H200 support are recorded; absence triggers a backport decision |
| H2. Humming A8/g16 WGMMA specialization | Large/high | For E2M1 weights only, load two compact E4M3 scales per K=32 fragment, apply each to its 16-wide converted half with declared rounding/headroom, avoid double-applying group scale on accumulators, and keep FP32/global-scale semantics; initially reject differing fused global scales |
| H3. Launch/preflight integration | Small | An opt-in Humming arm and a normal Marlin arm are selected before loading, emit structured backend/dtype/group/accumulator metadata, and fail closed |
| H4. Exact regression and dense qualification | Medium-large | All-codepoint/edge-scale reference plus every rank-local dense shape passes correctness and `1.10x` memory gates; latency includes activation quantization and is compared to Marlin and expanded W8A8 |
| H5. Optional upstream adapter/selector patch | Medium | Needed only if `--quantization humming` cannot coexist with the target fork's model-specific loader; patch is based on current upstream rather than recreating Humming |

No new GEMM framework is justified, but H2 is genuine CUDA kernel work inside
Humming. Compared with the CUTLASS contingency, it reuses the packed E2M1
repacker, compact scale layout/loaders, E2M1-to-E4M3 converter, activation
quantizer, WGMMA/TMA pipeline, tuning framework, global-scale epilogue, and vLLM
integration. The new mechanism is narrowly the sub-MMA group-scale application
and its rounding/headroom policy.

### Chosen-path implementation touchpoints and order

| Order | Path or symbol | Action |
|---|---|---|
| 1 | Target vLLM checkout plus `requirements/cuda.txt` | Pin the fork commit and confirm `humming-kernels[cu12/cu13]==0.1.10` or record the exact compatible replacement |
| 2 | `vllm/model_executor/layers/quantization/humming.py::HummingConfig` | Verify `--quantization humming` reads both `weights` and `input_activations` from the checkpoint; no edit if current behavior matches upstream |
| 3 | Humming `schema/compressed_tensors.py::CompressedTensorsWeightSchema` | Assert NVFP4 maps to packed `float4e2m1`, E4M3 group size 16, and `GROUP_TENSOR` global scaling |
| 4 | Humming `schema/base.py::get_fallback_input_dtype` and `CompressedTensorsInputSchema.convert_humming` | Assert NVFP4 A4 maps to dynamic per-token `float8e4m3` with input group size 0 on SM90 |
| 5 | Humming `kernel/humming.py::check_scale` | Add a tightly guarded allowance only for A8 + E2M1 + group-16 once the specialized arithmetic is compiled; retain all other minimum-group assertions |
| 6 | Humming `memory/*/loader_bs.cuh`, `mma/wgmma.cuh::transform_b`, and `arith/mainloop_arith.cuh` | Load/map two scales per K=32 fragment, scale the corresponding converted E2M1 halves before WGMMA, and suppress the ordinary post-accumulator weight-group scaling for this policy |
| 7 | New exact-combination regression in Humming plus a vLLM integration test | Test all-codepoint E2M1, g16 E4M3, shared global scale, rejection of differing fused global scales, dynamic A8, BF16, FP32 accumulation, compact post-transform bytes, and rejection of unintended dtype/group combinations |
| 8 | `pipeline` preflight/evidence layer in this repository | Add two explicit arms: Humming W4A8 (`--quantization humming`) and normal NVFP4/Marlin W4A16; record resolved layer metadata and fail closed |
| 9 | One-layer GPU packet | Compile/JIT one representative target shape, capture generated config/SASS, compare reference, and measure transformed tensor bytes before model loading |
| 10 | Dense shape matrix and serving packet | Run the declared M buckets and only then model-serving/quality qualification |
| 11 | `vllm/model_executor/kernels/linear/nvfp4/humming.py` or selector registry | Modify only if the full Humming quantization override cannot be used; do not mistake the current weight-only generic adapter for the A8 path |

The implementation plan must stop after order 1 if the target source differs
materially and remap against that exact checkout. It must stop after order 9 if
correctness or compact memory fails. Only then may it activate the custom
CUTLASS contingency below.

## Contingency: custom CUTLASS datatype variant

The following design remains an implementation map, not the first work item. It
is entered only after H1-H4 produce a concrete blocker that cannot be resolved
by extending Humming at the bounded group-16 specialization point.

The contingency would implement a separate NVFP4 datatype variant of
`CutlassW4A8LinearKernel`. It must not modify the existing INT4 W4A8 operation
in place: its signed-INT4 encoder and group-128 scale-packing contract remain
stable. Shared code is factored only after both paths have independent tests.

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

The first arithmetic proof may temporarily use the existing W4A8 eight-way
scale packing to isolate converter correctness. That representation is not a
production candidate for group size 16: it stores eight E4M3 copies per logical
scale, making scale storage `8 / 16 = 0.5` byte per weight. Together with the
packed four-bit weight, this is about `1.0` byte per weight versus about
`0.5625` byte for compact NVFP4 weight plus one E4M3 scale per group of 16,
before small global/channel metadata. It would therefore be roughly `1.78x`
the compact representation and fail the `1.10x` memory gate. A compact group-16
scale feed is mandatory before performance qualification.

### Contingency approach comparison

Three implementation shapes were considered:

| Approach | Reuse | Main drawback | Decision |
|---|---|---|---|
| Fork the existing SM90 CUTLASS W4A8 path and replace its weight datatype, reorder, and group-16 scale feed | Reuses activation quantization, TMA/WGMMA skeleton, epilogue, schedules, dispatch, and serving ABI patterns | Requires a new E2M1 converter and compact group-16 scale integration | **Contingency choice** if Humming cannot satisfy the gates |
| Enable the Marlin FP8 `mma.sync` branch for NVFP4 | Reuses NVFP4 layout and E2M1 conversion most directly | The existing FP8 instruction route does not accelerate Hopper and is not the required SM90 WGMMA architecture | Diagnostic only |
| Expand NVFP4 weights to FP8 at load time | Reuses mature W8A8 GEMMs | Nearly doubles persistent weight bytes and loses the packed-weight bandwidth goal | Control/throughput ceiling only |

The contingency boundary is intentionally asymmetric: CUTLASS W4A8 owns execution;
Marlin/NVFP4 owns checkpoint meaning and donates conversion logic. Marlin's
final repacked layout is not an input contract for the new kernel unless a
round-trip layout test proves equivalence.

### Contingency component reuse matrix

Legend: **unchanged** means call or instantiate the existing component;
**parameterize** means retain its structure while adding an NVFP4-specific
parameter or specialization; **algorithm** means reuse the proven semantics but
implement it in the new component; **new** means no inspected reference supplies
the required component.

### Loader, selection, and persistent representation

| Component | From `CutlassW4A8LinearKernel` | From NVFP4/Marlin | Decision for the new path | Genuinely new work or proof required |
|---|---|---|---|---|
| Checkpoint recognition | Mixed-precision kernel selection and `can_implement` pattern | `CompressedTensorsW4A4Fp4`, ModelOpt/Quark adapters, packed tensor names, and NVFP4 format detection | Reuse the NVFP4 scheme; select a new Hopper kernel inside the NVFP4 registry rather than pretending the checkpoint is INT4 W4A8 | Permit dynamic FP8 execution without changing the on-disk NVFP4 declaration; define behavior for both NVFP4 and NVFP4A16 checkpoint variants |
| Weight allocation and TP slicing | Parameter-layout metadata conventions | `ModelWeightParameter` packed `uint8 [N,K/2]`, `GroupQuantScaleParameter` E4M3 group-16 scales, `PerTensorScaleParameter` global scale, and existing loaders | **Unchanged from NVFP4** | Shape/TP round-trip tests for fused and unfused dense linears; no new storage schema |
| Global-scale interpretation | Per-channel FP32 epilogue scale input | Compressed-tensors inverse-scale handling, fused-projection consistency checks, and Marlin global-scale semantics | Reuse NVFP4 inversion; fold the scalar weight-global scale into the new per-channel epilogue scale | New adapter and tests for multiple logical widths; fail loading on a disallowed global-scale mismatch rather than silently changing math |
| Activation checkpoint scale | Dynamic per-token W4A8 does not need a checkpoint scale | NVFP4 W4A4 checkpoints may contain an A4 input-global scale | Ignore it for arithmetic in this experimental dynamic-A8 path but preserve loader compatibility; record this semantic change | Explicit selector/telemetry and a reference test proving that A8 uses runtime token scales, not the stored A4 scale |
| Persistent weight dtype | Four-bit packed B operand | E2M1 nibble codes remain packed two per byte | Keep raw E2M1 codes; expose a distinct FP4 scalar type/trait to C++ | New type specialization or wrapper; never call `convert_packed_uint4b8_to_signed_int4_inplace` |
| Weight layout transform | `compute_memory_reordering_atom<MmaType>()`, `tile_to_shape`, and `reorder_tensor` | NVFP4 nibble order and Marlin repack tests provide the semantic oracle | Reuse the CUTLASS reordered layout **algorithm**, omitting the INT4 unified encoding LUT | New `reorder_fp4_e2m1` preprocessing op plus forward/inverse layout test for all 16 code points and rank-local shapes |
| Group-scale normalization | `convert_bf16_scales_to_fp8` factors group scales into E4M3 mainloop scales plus FP32 channel scales and reserves headroom by dividing/multiplying by 8 | E4M3 group-16 scale and FP32 global-scale meaning | Reuse the factorization algorithm after a one-time cast from checkpoint E4M3; multiply the resulting channel scale by the NVFP4 global scale | Small NVFP4 adapter; deterministic rounding, zero-channel, overflow, and fused-width tests |
| Persistent group-scale layout | `cutlass_pack_scale_fp8` and `ScalePackSize=8` | NVFP4/Marlin stores one compact scale per group of 16 | Existing eight-way pack is allowed only for a correctness spike; production must retain compact scale density | **New high-risk work:** compact group-16 scale layout, addressing, TMA/shared-memory staging, and memory accounting |
| Capability and shape checks | Exact SM90, E4M3 activation, BF16 output, no zero points/`g_idx`, K/N alignment checks | Marlin support check and existing fallback path | Parameterize for E2M1/group-16; preflight before destructive transforms | Prove actual K/N alignment constraints for every target rank-local dense shape; return structured fallback reasons |
| Fallback policy | Kernel-selection failure reasons | `MarlinNvFp4LinearKernel` is the reliable W4A16 path | New kernel is opt-in and ordered before Marlin only when every precondition passes | New experimental setting/registry entry and fail-closed selection tests; no runtime retry after partial transformation |

### Runtime GEMM

| Component | From CUTLASS W4A8 | From Marlin | Decision for the new path | Genuinely new work or proof required |
|---|---|---|---|---|
| Activation quantization | `QuantFP8(static=False, GroupShape.PER_TOKEN)` and FP32 token scales | None | **Reuse unchanged** | Cross-check output/scale layout and CUDA-graph behavior with the target vLLM version |
| Python call path | Flatten/restore shape, optional bias, operator call, BF16 output expectation | NVFP4 kernel base/registry interface | Reuse structure in a new `HopperNvFp4W4A8LinearKernel` | New class and op name so INT4 W4A8 remains ABI-stable |
| Operator ABI and registration | `cutlass_w4a8_mm` schema, fake/meta registration pattern, stable-Torch binding, CMake organization | NVFP4 kernel registry | Reuse pattern, not the existing op | New `cutlass_nvfp4_w4a8_mm` plus preprocessing op, fake/meta functions, build registration, and compile guards |
| Kernel template and schedules | `W4A8GemmKernel`, SM90 cooperative TMA schedule, TMA epilogue, ten tile/cluster instantiations, M/N dispatch heuristic | None | Fork initially; tune only after correctness | Compilation/resource proof with group-16 scale traffic; later benchmark may change schedules |
| A operand movement | Row-major E4M3 A layout, alignment, TMA staging | None | **Reuse unchanged** | Alignment and odd-M tests only |
| B global-memory reorder | `LayoutAtomQuant` and `LayoutB_Reordered` align sub-byte values with converter consumers | Marlin proves E2M1 nibble identity, not CUTLASS layout | **Parameterize** for E2M1 while preserving the CUTLASS consumer layout | Reorder-only preprocessing and descriptor proof; Marlin-repacked B is not reused by assumption |
| B element type | Packed `cutlass::int4b_t` | `scalar_types.float4_e2m1f` and E2M1 type identity | Replace INT4 with an E2M1 sub-byte type in the mixed-input collective | New CUTLASS trait/specialization if the pinned CUTLASS release cannot instantiate E2M1 directly |
| Nibble-to-FP8 conversion | CUTLASS mixed-input converter placement and vectorization | `dequant<__nv_fp8x4_e4m3, kFE2M1f, true>` supplies tested bit math and all-codepoint semantics | Reuse the Marlin **algorithm** inside the CUTLASS converter boundary; do not include Marlin kernel headers directly | **New core work:** vectorized E2M1-to-E4M3 converter integrated with the SM90 collective and validated for nibble order/sign/zero |
| Group-scale application | Tupled B/scale collective and runtime `group_size`/`StrideScale`; the C++ op already computes `ceil_div(K, group_size)` | Exact NVFP4 group size 16 and scale meaning | Retain the collective concept and runtime group size | **New core work:** make group-16 compact scales reach each converted fragment without eight-way persistent replication; prove K-tile boundary behavior |
| FP8 rounding and saturation | E4M3 MMA operand type and CUTLASS numerical conversion behavior | E2M1 values and E4M3 group scales | Declare one exact conversion mode in the reference and kernel | Exhaustive codepoint/edge-scale test, including signed zero, maximum, subnormal, saturation, and non-finite rejection |
| Tensor Core operation | SM90 FP8 WGMMA selected through `MmaType=float_e4m3_t` | None | **Reuse unchanged** after B is materialized as E4M3 fragments | SASS/profiler evidence that the selected target uses FP8 WGMMA, not an FP16/TF32 fallback |
| Accumulation | FP32 accumulator | Marlin W4A16 is comparison only | **Reuse unchanged** | Numerical reference comparison |
| Epilogue scales | Existing `ScaledEpilogue` combines FP32 per-token and per-channel scales | Weight-global scale semantics | Reuse unchanged after folding global scale into channel scale | Verify scale shapes for fused QKV/gate-up widths and TP partitions |
| Output and bias | BF16 D tensor and Python in-place bias add | Existing NVFP4 kernel interface | **Reuse unchanged** | Bias/no-bias and reshape tests |
| Dispatch heuristic | Existing M/N schedule table | None | Reuse as the first baseline | Benchmark every target shape; retune only with evidence because group-16 scale bandwidth changes the optimum |

### Reference, tests, and qualification

| Component | CUTLASS W4A8 donation | Marlin/NVFP4 donation | Decision | New work |
|---|---|---|---|---|
| Deterministic arithmetic reference | Activation quantization and scale-factorization utilities | E2M1 decode, group/global scale semantics, and existing emulation utilities | Compose a framework reference for the exact W4A8 contract | New reference function; it must model FP8 rounding at the same point as the kernel rather than compare only to W4A16 |
| Layout oracle | INT4 W4A8 encode/reorder tests | Marlin pack/repack/dequant tests | Add E2M1 reorder round-trip and kernel-vs-logical layout tests | New exhaustive nibble/layout suite |
| Kernel unit tests | `test_cutlass_w4a8.py` shape, dispatch, fake-op, and CUDA-graph patterns | `test_marlin_gemm.py` NVFP4 parameterization | Clone and specialize the test matrix | New group-16, global-scale, all-codepoint, fallback, memory, and deterministic-repeat cases |
| Correctness control | Existing INT4 W4A8 kernel guards against shared-code regressions | W4A16 Marlin and software emulation | Run all three: new W4A8 reference, W4A16 compatibility, and emulation | New comparison report with semantics labeled; W4A16 is not expected to be bit-identical |
| Microbenchmark | Existing W4A8 dispatch shapes and timing style | W4A16 baseline | Include activation quantization and persistent scale bytes | New benchmark harness/results for each target `(M,N,K)` and load-expanded W8A8 control |
| Serving integration | Kernel selection/logging conventions | NVFP4 dense registry and Marlin fallback | Opt-in dense qualification first | New structured selection telemetry and model-level evidence packet |
| Fused MoE | Grouped W4A8 is a later execution reference | NVFP4 MoE loader/layouts exist | **Out of scope** until dense gates pass | Separate design; do not generalize the dense op prematurely |

### Contingency implementation touchpoints

The implementation plan should resolve the vLLM `0.24.0` equivalents of these
current-upstream paths and symbols:

| Path or symbol | Planned action | Source role |
|---|---|---|
| `vllm/model_executor/kernels/linear/nvfp4/base.py` | Reuse the NVFP4 layer contract; extend configuration only if the backend needs explicit A8/output metadata | NVFP4 |
| `vllm/model_executor/kernels/linear/nvfp4/hopper_w4a8.py` | **Add** `HopperNvFp4W4A8LinearKernel`: dynamic per-token A8, load transforms, op call, bias/reshape | New integration glue |
| `vllm/model_executor/kernels/linear/__init__.py` | Register the new backend ahead of Marlin only for opt-in SM90 qualification; retain Marlin fallback | NVFP4 selector |
| `vllm/model_executor/layers/quantization/compressed_tensors/schemes/compressed_tensors_w4a4_nvfp4.py` | Reuse weight/scale allocation and inverse global-scale handling; avoid changing checkpoint tensor names | NVFP4 loader |
| `vllm/model_executor/kernels/linear/mixed_precision/cutlass.py` | Reference or factor activation/call-path helpers; do not widen INT4 `can_implement` or reuse its signed conversion | CUTLASS W4A8 |
| `vllm/model_executor/layers/quantization/utils/quant_utils.py::convert_bf16_scales_to_fp8` | Reuse scale factorization after a documented one-time dtype cast; add an NVFP4 wrapper if changing the generic name would be misleading | CUTLASS W4A8 plus new adapter |
| `csrc/libtorch_stable/quantization/cutlass_w4a8/w4a8_mm_entry.cu` | Fork the dense kernel/template/dispatch into a separately registered NVFP4 op; keep the INT4 op untouched | CUTLASS W4A8 execution foundation |
| `csrc/libtorch_stable/quantization/cutlass_w4a8/w4a8_utils.{cu,cuh}` | Reuse the reorder structure; add a raw-E2M1 reorder that omits `unified_encode_int4b` | CUTLASS layout plus new preprocessing |
| `csrc/libtorch_stable/quantization/marlin/dequant.h::dequant<__nv_fp8x4_e4m3,kFE2M1f,true>` | Port the bit conversion into a small NVFP4/CUTLASS converter header with an attribution comment and independent tests | Marlin conversion algorithm |
| `vllm/_custom_ops.py`, stable-Torch schemas/bindings, and CMake source lists | Add the GEMM and reorder ops, fake/meta implementations, capability guards, and build wiring | Existing registration patterns plus new ops |
| `tests/kernels/quantization/test_cutlass_w4a8.py` | Preserve as regression coverage and copy relevant patterns | CUTLASS W4A8 tests |
| New `tests/kernels/quantization/test_cutlass_nvfp4_w4a8.py` | Add reference, codepoint/layout, shape, scale, fallback, memory, CUDA-graph, and dispatch tests | New qualification |

### Contingency new work packages

| Work package | Size/risk | Definition of done |
|---|---|---|
| N1. Raw-E2M1 CUTLASS reorder | Medium | No numerical recoding; exhaustive nibble and random tensor round trips match logical NVFP4 order for every target alignment |
| N2. E2M1-to-E4M3 mixed-input converter | Large/high | All 16 codes and edge scales match the declared reference; compiled SM90 kernel reaches FP8 WGMMA with no FP16/TF32 conversion detour |
| N3. Compact group-16 scale feed | Large/highest | Correct scale selected across K tiles and groups, persistent bytes pass the `1.10x` Marlin gate, and scale traffic does not erase the performance continuation gate |
| N4. NVFP4 scale factorization/global fold | Small-medium | Load-time group/channel/global factors reconstruct the declared W4A8 math for fused widths, TP partitions, zeros, and maxima |
| N5. Backend selection and safe Marlin fallback | Medium | Opt-in SM90 selection is observable and deterministic; every unsupported case chooses W4A16 before any irreversible transform |
| N6. Reference, tests, and benchmark evidence | Large but routine | Machine-readable correctness/memory/performance results cover every declared shape and M bucket |

Within the contingency, N2 and N3 are the only genuinely novel kernel
mechanisms. N1 is a new
preprocessing implementation assembled from existing layout machinery; N4 and
N5 are new integration glue. N6 is required evidence rather than new arithmetic.
The critical path is N2 plus N3, not activation quantization or WGMMA itself.

### Contingency implementation order

1. Freeze the target vLLM/CUTLASS commits and inventory every target dense
   `(N,K)` shape after TP partitioning.
2. Implement the deterministic W4A8 reference and load-time scale factorization
   tests before compiling a custom kernel.
3. Implement N1 and prove raw E2M1 reorder round trips independently of GEMM.
4. Implement N2 using the existing eight-way group-scale pack as a
   **correctness-only spike**. This isolates datatype conversion from compact
   scale transport and must not be used for memory/performance claims.
5. Replace the temporary scale feed with N3 and rerun correctness plus persistent
   byte accounting. Stop if compact scale staging cannot meet the memory gate.
6. Add N5 selection/fallback plumbing only after the standalone op passes its
   shape matrix; then run INT4 W4A8 regression tests to guard shared machinery.
7. Produce the dense microbenchmark evidence. Only a passing dense result may
   proceed to full serving integration or a separate MoE design.

## Numerical contract

For activation row `m`, output channel `n`, and weight group `g`, the external
contract for the Humming group-16 specialization is:

```text
A8[m, k] = Q_E4M3(A[m, k] / A_scale[m])
B8[n, k] = Q_E4M3(E2M1(B4[n, k])
                   * float(B_group_scale[n, group(k)]) / 8)
C[m, n] = BF16(A_scale[m] * (8 * B_global_scale)
                * AccFP32_k(A8[m, k] * B8[n, k]))
```

Raw E2M1 values are exactly representable in E4M3, but the product with an
arbitrary E4M3 group scale may not be. The fixed `/8` headroom keeps the maximum
finite product `6 * 448 / 8 = 336` within E4M3 range; multiplying the existing
global scale by 8 restores the factor outside the GEMM. The implementation may
perform the division as an exponent adjustment while converting registers, but
must produce the same declared E4M3 rounding, saturation, signed-zero, and
subnormal behavior. It must not modify the checkpoint on disk, double-apply the
group scale on accumulators, or enable the policy for non-E2M1 weights.

The first specialization accepts only one scalar `B_global_scale` shared by all
logical projections fused into a physical layer. If a checkpoint supplies
different fused global scales, preflight selects Marlin before Humming mutates
the tensors. Supporting a per-output-channel compensation vector is a possible
follow-up, not implicit first-scope behavior.

This path is not expected to be bit-identical to W4A16 Marlin because both
activations and scaled weight operands enter FP8 WGMMA. H2/H4 compare it to the
declared W4A8 reference; Marlin is a compatibility and model-quality baseline.

## Components and boundaries

### NVFP4 loader adapter

Use vLLM `HummingConfig` plus Humming's compressed-tensors schemas for checkpoint
recognition, tensor names, TP slicing, group/global scale interpretation, and
shape metadata. A local adapter is written only if the target fork lacks the
inspected upstream integration.

### Activation quantizer

Use Humming's dynamic per-token E4M3 quantizer selected by the A4-to-A8 platform
fallback. A later project may expose/fuse this quantization through vLLM's graph
only after the standalone path is qualified.

### Dense SM90 W4A8 kernel

Use the existing Humming dense kernel for packed-weight loading, compact group-
scale loading, register conversion, WGMMA, accumulation, and BF16 output. The
custom CUTLASS kernel is a contingency, not a component of the first candidate.

### Kernel selector and fallback

Select the entire experimental arm with `--quantization humming` only after a
preflight confirms the exact versions, SM90, NVFP4 schema, dense shapes, BF16
output, FP32 accumulation policy, and Humming availability. The comparison arm
uses normal NVFP4 selection and W4A16 Marlin. Selection occurs before loading;
there is no runtime retry after tensors have been transformed.

### Reference and benchmark harness

Owns deterministic operand generation, reference W4A8 semantics, correctness
metrics, persistent-memory accounting, kernel latency, and model-serving
measurements. Raw measurements remain separate from the decision report.

## Staged execution and continuation gates

### Stage 0: upstream and version freeze

The broad search found Humming, so Stage 0 now verifies adoption feasibility.
Record the exact target vLLM fork commit, whether `HummingConfig` contains the
compressed-tensors weight **and input** schema route, the installed/pinnable
Humming version, CUDA, PyTorch, NVCC, driver, and H100/H200 details. Record the
resolved CLI/config and generated Humming layer metadata. If the target lacks
the route, prefer backporting current upstream vLLM plus Humming `0.1.10` over
writing a kernel.

### Stage 1: deterministic reference and exact-combination proof

Implement H2 against the pinned Humming version using small deterministic
tensors. Cover all E2M1 codes, group-16 E4M3 scales, a shared global scale,
rejection of differing fused global scales, dynamic per-token A8, BF16 output,
zeros/maxima, fused logical widths, and FP32
accumulation. Confirm that post-transform weight and scale tensor sizes remain
compact and that the generated H100/H200 kernel is WGMMA. This stage does not
claim model-level quality or performance.

### Stage 2: dense Humming proof of concept

Run dense BF16-output Humming linears on SM90 for all aligned rank-local
projection shapes found in the target model configuration. Benchmark every
unique shape at `M = 1, 8, 32, 128, 512, 2048`, covering decode through prefill
behavior.

Correctness gates for every supported shape and `M`:

- no NaN or infinity in output or scales;
- cosine similarity to the deterministic W4A8 reference at least `0.999`;
- relative L2 error at most `0.02`;
- deterministic repeat results under the same inputs and configuration;
- unsupported or misaligned shapes select W4A16 before the first request.

The normal Marlin arm and the Humming arm are separate preflight-selected model
loads. A Humming JIT/shape failure is evidence, not permission to mutate an
already loaded layer into Marlin form.

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

Integrate the qualified Humming arm behind an opt-in serving setting and compare
W4A8, W4A16 Marlin, and load-expanded W8A8 under the same model, prompts,
topology, scheduler settings, and request mix. Measure throughput, inter-token
latency, time to first token, peak HBM, and startup time.

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

### Write the CUTLASS datatype variant immediately

This remains technically feasible and the contingency matrix below is detailed
enough to plan it. It is rejected as the first implementation because Humming
already supplies the expensive GEMM machinery we thought was new: register
E2M1-to-E4M3 conversion, compact scale storage/loading infrastructure, and SM90
WGMMA/TMA execution. It does not supply the exact two-g16-scales-per-K32
mapping; that bounded extension is H2. Rebuilding the surrounding kernel
framework before attempting H2 would violate this repository's prime directive.

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
- Known unsupported cases select the separate W4A16 launch arm with a recorded
  reason before loading.
- A failed scale/layout invariant stops loading rather than falling through after
  partial transformation.
- Runtime non-finite detection belongs in tests and diagnostic builds; production
  selection relies on preflight qualification rather than per-token scans.
- The original checkpoint is never modified.
- Evidence runs use fresh result roots and follow `PLANNER_EXECUTOR_PROTOCOL.md`.

## Estimated effort

- target audit, deterministic reference, and proof of the current A8/g16
  rejection: approximately `2–5 engineering days`;
- Humming A8/g16 register-scale specialization and focused CUDA tests:
  approximately `2–6 weeks` for an engineer familiar with its JIT kernel,
  depending on whether the existing scale loader already exposes both group-16
  values in the required register mapping;
- dense target-shape and memory/performance qualification: approximately
  `1–3 weeks` after the specialized one-layer test passes;
- optional current-upstream Humming backport/selector integration: add roughly
  `1–3 weeks`, depending on divergence in the target vLLM fork;
- only if Humming fails for a fundamental reason, the contingency dense SM90
  CUTLASS/WGMMA proof of concept is `4–8 weeks` for an experienced
  CUDA/CUTLASS engineer if the pinned CUTLASS mixed-input collective can be
  specialized for compact group-16 scales; allow `8–12 weeks` if it requires a
  custom scale-staging collective rather than a local specialization;
- dense model-serving and quality qualification after a passing kernel result:
  approximately `1–3 additional weeks` using the existing harness;
- fused MoE, TP/EP hardening, and broad-model production coverage: a separate
  `3–5 person-month` project if the continuation gate passes.

These are planning ranges, not schedule commitments. The large custom-kernel
estimate is now a contingency, not the expected chosen-path cost.

## Acceptance criteria

- The repository records packed NVFP4 W4A8 as a benchmark-gated long-term goal.
- The implementation plan starts by extending and testing Humming `0.1.10` at
  the isolated A8/g16 boundary, not by recreating its kernel or starting fused
  MoE work.
- The selected dense backend retains packed four-bit persistent weights and uses
  SM90 FP8 WGMMA with FP32 accumulation and BF16 output.
- W4A16 Marlin remains the default fallback until qualification completes.
- Correctness, memory, and performance gates are machine-readable and evaluated
  before MoE work is authorized.
- Cluster quality evaluation is fail-closed under the repository harness
  contract and never conflates paired internal evidence with a public score.
- Failure to pass the continuation gate stops the project without displacing the
  working W4A16 release-day path.
