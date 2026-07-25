# MiniMax-M3 Native Humming W4A8 Backend Design

- Date: 2026-07-25
- Status: APPROVED — 2026-07-25
- Workflow state: PLANNER_ANALYSIS
- Owner: planner
- Initial checkpoint scope: in-house GPTQ W4A8 only
- Performance target: production agentic/high-concurrency behavior
- Secondary guardrail: concurrency-1 behavior

## Goal

Determine whether Humming can execute the existing MiniMax-M3 GPTQ checkpoint's
native W4A8 format faster than vLLM's current CUTLASS W4A8 backend on Hopper,
without changing the checkpoint, quantization recipe, serving topology, or
benchmark contract.

The target format is:

- symmetric packed INT4 routed-expert weights;
- group size 128 weight scales;
- dynamic per-token E4M3 activation quantization;
- BF16 output and normal FP32 accumulation unless Humming explicitly documents
  and qualifies a different policy;
- experts-only quantization, with ignored attention, router, shared-expert, dense,
  and other checkpoint-declared modules remaining BF16.

The primary decision question is:

> On the existing TP8 plus expert-parallel MiniMax-M3 serving topology, does
> Humming improve the repository's existing production concurrency benchmark
> relative to CUTLASS W4A8 while preserving correctness, stability, and an
> acceptable concurrency-1 result?

## Decision summary

Use Humming's existing native GPTQ W4A8 path. Do not write a new CUDA kernel and
do not create another performance harness.

The inspected versions already provide almost all required machinery:

- vLLM `0.24.0` contains `HummingConfig`, `HummingLinearMethod`, and
  `HummingMoEMethod`;
- vLLM selects that path explicitly with `--quantization humming`;
- Humming `0.1.10` supports dynamic E4M3 input quantization, packed unsigned
  four-bit GEMM operands, group-128 weight scales, dense GEMM, and indexed and
  grouped MoE GEMM;
- Humming provides GPTQ and AWQ checkpoint schema conversion, including packed
  weight preparation and scale preparation;
- vLLM's Humming MoE implementation supports TP/EP configurations used here,
  excluding only unrelated FlashInfer NVLink-specialized prepare/finalize modes.

The one confirmed MiniMax-M3 integration gap is activation admission:
`HummingExpertsBase._supports_activation()` in the inspected vLLM `0.24.0`
source does not include `MoEActivation.SWIGLUOAI_UNINTERLEAVE`. Humming already
calls vLLM's generic activation implementation between its two MoE GEMMs, so the
gap is Python selection/plumbing rather than missing GEMM arithmetic. The
repository's existing MiniMax-M3 patch already supplies the required
SwiGLU-OAI defaults (`limit=7.0`, `alpha=1.702`, `beta=1.0`) when the call site
does not pass them.

## Reputable sources inspected

Exact source inspection, rather than version inference, produced this design.

| Source | Pin or location | Relevant finding |
| --- | --- | --- |
| vLLM | tag `v0.24.0`, commit `ee0da84ab9e04ac7610e28580af62c365e898389` | Full Humming linear and MoE quantization integration exists |
| Humming | tag `v0.1.10`, commit `4351af3a8fcdce1a8dee50104ba49566af2427fb` | Exact dense E4M3-per-token plus UINT4-group-128 cross-product is covered; native MoE GEMMs and GPTQ conversion exist |
| Current repository | `pipeline/slurm/patch_vllm_m3_serve.py` | Persistent MiniMax-M3 activation, CUDA-graph, routing, and shared-expert fixes already exist |
| Current repository | `pipeline/slurm/run_vllm_http_serve_smoke.sh` | Stable TP8/EP HTTP serving entry point and dry-run observability already exist |
| Current repository | `pipeline/slurm/perf_eval_arm.sh` and existing performance controllers | Production benchmark workflow already exists and must be reused unchanged |
| Current repository | `pipeline/m3_serve_abi.py` and MiniMax-M3 diagnostics | Checkpoint ABI and serving diagnostics should be extended or reused, not replaced |

Primary upstream references:

- <https://github.com/inclusionAI/humming>
- <https://github.com/vllm-project/vllm/blob/v0.24.0/vllm/model_executor/layers/quantization/humming.py>
- <https://github.com/vllm-project/vllm/tree/v0.24.0/vllm/model_executor/layers/fused_moe/experts>

## Scope

### In scope

1. Add an explicit, fail-closed Humming backend choice to the existing
   MiniMax-M3 HTTP serving path.
2. Extend the persistent vLLM MiniMax-M3 patch so Humming admits
   `SWIGLUOAI_UNINTERLEAVE`.
3. Reuse Humming's GPTQ load-time conversion, packing, scale layout, dynamic
   input quantization, and native MoE kernels.
4. Attest at runtime that the requested backend was actually selected.
5. Validate checkpoint recognition, transformed-weight semantics, activation
   behavior, TP8/EP loading, CUDA graphs, finite output, and repeated serving.
6. Feed the qualified Humming serve into the existing performance benchmark
   without changing the workload or scoring.
7. Keep CUTLASS W4A8 as the default and rollback path until Humming qualifies.

### Out of scope

- a new performance benchmark, workload, analyzer, or score;
- changes to calibration, GPTQ quantization, checkpoint weights, or on-disk
  checkpoint format;
- AWQ enablement in the first vertical slice;
- NVFP4-to-W4A8 fallback work;
- attention, KV-cache, shared-expert, or non-expert quantization changes;
- replacing CUTLASS as the production default before qualification;
- writing or modifying CUDA kernel arithmetic without a demonstrated,
  localized defect in the existing Humming path;
- broad Humming tuning before the default indexed path is measured.

## Existing baseline

The current in-house GPTQ checkpoint is served through compressed-tensors and
vLLM's modular `CutlassExpertsW4A8Fp8` backend. It is stable and preserves
dynamic FP8 activations, but prior performance evidence shows:

- CUTLASS W4A8 is slower than Marlin W4A16 at concurrency 1;
- W4A8 scales better under load and has better high-concurrency TTFT behavior;
- the desired comparison is therefore Humming W4A8 versus CUTLASS W4A8 on the
  same logical checkpoint, with Marlin W4A16 retained only as contextual
  evidence.

This design does not compare different checkpoints as if they were a kernel-only
comparison. The Humming and CUTLASS arms must use the same in-house GPTQ
checkpoint.

## Runtime architecture

### Backend interface

The serving launcher gains one narrow user-facing selector:

```text
M3_W4A8_BACKEND=cutlass   # default; current behavior
M3_W4A8_BACKEND=humming   # explicit experimental path
```

The launcher must reject every other value.

For `cutlass`, no new vLLM argument is added. For `humming`, the launcher adds
`--quantization humming` through a structured argument path. It must not depend
on callers manually embedding the selector in an opaque
`EXTRA_VLLM_ARGS` string.

The experimental selector must be visible in:

- `PRINT_EFFECTIVE_CONFIG=1` output;
- the shell-escaped effective vLLM command;
- server startup metadata;
- the executor evidence packet.

### Humming data path

```text
compressed-tensors GPTQ checkpoint
  -> HummingConfig reads weight and input schemes plus ignored modules
  -> GPTQWeightSchema converts packed checkpoint tensors to Humming tensors
  -> Humming transforms packed INT4 weights and group-128 scales at load time
  -> BF16 token input is dynamically quantized to per-token E4M3
  -> Humming indexed MoE GEMM 1 computes gate/up projections
  -> generic SWIGLUOAI_UNINTERLEAVE applies M3 clamp/alpha/beta
  -> activation output is dynamically quantized to per-token E4M3
  -> Humming indexed MoE GEMM 2 computes the down projection
  -> existing routing weight application, reduction, shared-expert path,
     TP8/EP communication, and HTTP serving continue unchanged
```

Humming's default indexed MoE GEMM is the first target. The grouped-contiguous
alternative controlled by `VLLM_HUMMING_MOE_GEMM_TYPE` is a later, separately
authorized optimization only if the indexed path is correct but insufficient.

### Weight and activation invariants

The implementation must preserve these invariants:

1. The downloaded checkpoint remains packed four-bit GPTQ.
2. Humming may repack/reorder weights in GPU memory during loading, but it must
   not rewrite checkpoint files.
3. Logical signed symmetric INT4 values must be represented through Humming's
   unsigned packed operand plus symmetric scale convention exactly as its GPTQ
   schema defines.
4. Each weight scale continues to apply to its original group of 128 K
   elements.
5. Each runtime activation scale remains per token (`input group size 0` in
   Humming terminology), not group-128.
6. `w13` remains two logical stacks: all gate outputs followed by all up
   outputs. It must not be silently treated as alternating gate/up channels.
7. The activation is `SWIGLUOAI_UNINTERLEAVE` with MiniMax-M3's resolved
   constants.
8. Ignored modules remain BF16 and must not be online-requantized by Humming.
9. `VLLM_HUMMING_USE_F16_ACCUM` remains disabled for the first qualification
   unless a later experiment explicitly changes the precision contract.

## Reuse versus new work

| Component | Reuse from CUTLASS path | Reuse from Humming | New work in this repository |
| --- | --- | --- | --- |
| Checkpoint selection | Existing in-house GPTQ path and config preparation | Compressed-tensors config parser | Humming-specific preflight and attestation |
| GPTQ loading | Existing vLLM expert weight loader conventions | `GPTQWeightSchema.convert_humming` | Exact checkpoint compatibility test |
| Packed weight handling | Logical INT4/scales reference | Humming pack/repack and transform | No new repacker unless a proven defect appears |
| Dynamic FP8 inputs | Existing checkpoint activation contract | Humming `may_quant_input` / E4M3 quantization | Contract assertions |
| MoE routing | Existing modular router, expert map, TP8/EP topology | Indexed/grouped Humming MoE implementations | None initially |
| GEMM 1 and GEMM 2 | CUTLASS correctness/performance control | Native Humming kernels | None initially |
| Gate/up activation | Existing M3 layout and generic activation semantics | Humming calls generic activation between GEMMs | Add `SWIGLUOAI_UNINTERLEAVE` to Humming's support gate |
| Clamp parameters | Existing persistent activation patch | Generic vLLM activation callback | Verify the patch covers the Humming call site |
| CUDA graphs | Existing M3 capture/routing/shared-stream fixes | Humming launch path | Capture qualification and backend-specific stop evidence |
| Serving | Existing HTTP launcher and readiness loop | `--quantization humming` | Explicit backend selector and metadata |
| Performance | Existing paired aiperf workflow | Humming serve endpoint | Add an arm invocation only; no new benchmark logic |

## Persistent vLLM patch

`pipeline/slurm/patch_vllm_m3_serve.py` remains the single persistent
site-packages patcher for MiniMax-M3 serving.

The Humming addition must:

1. locate `fused_moe/experts/fused_humming_moe.py`;
2. add `MoEActivation.SWIGLUOAI_UNINTERLEAVE` to
   `HummingExpertsBase._supports_activation`;
3. use a distinct marker and remain idempotent;
4. fail loudly if `M3_W4A8_BACKEND=humming` is requested and the expected file,
   class, or activation list cannot be found;
5. make `--check` report the Humming patch independently;
6. leave CUTLASS-only serving usable if Humming is absent and Humming was not
   requested.

The existing generic activation patch must remain the source of default
SwiGLU-OAI constants. The Humming patch must not duplicate the activation
formula or introduce another implementation of those constants.

## Preflight and backend attestation

### Before launch

The Humming path must fail closed unless all of the following hold:

- vLLM is the pinned `0.24.0`-class target whose patch layout was verified;
- `humming-kernels==0.1.10` imports successfully;
- the GPU capability is Hopper-compatible for the target run;
- the checkpoint declares packed compressed-tensors GPTQ;
- routed-expert weights are four-bit, symmetric, and group size 128;
- input activations are dynamic E4M3 per token;
- ignored-module metadata is present and passes the existing ABI checks;
- the Humming activation patch and all already-required MiniMax-M3 patches
  report installed.

### After server start

Readiness alone is insufficient. The arm must preserve and classify server logs
and require positive Humming evidence. The exact markers will be pinned during
implementation from the target vLLM/Humming versions, but the policy is:

- require a Humming quantization-method marker;
- require a Humming MoE GEMM selection marker;
- record indexed versus grouped GEMM selection;
- reject CUTLASS W4A8 expert-selection markers in the Humming arm;
- reject Marlin fallback markers;
- reject unquantized routed-expert fallback;
- record Humming JIT compilation and cache behavior;
- preserve the effective package versions and command line.

An ambiguous log is a failed attestation, not permission to benchmark.

## Error handling and rollback

| Condition | Required behavior |
| --- | --- |
| Unsupported backend value | Exit before patching or launching |
| Missing/wrong Humming version | Exit before GPU allocation where possible |
| Checkpoint scheme mismatch | Exit before server launch |
| Humming patch target changed | Exit and return the exact missing pattern/file |
| Weight conversion/load failure | Preserve first traceback and server log; do not retry with a different scheme |
| Unsupported activation selection | Preserve oracle/selection reason; do not switch to CUTLASS silently |
| CUDA graph failure | Preserve capture index, first failing operation, and package/kernel cache metadata; no automatic eager fallback |
| Humming JIT failure | Preserve generated-config identifier and compiler output; no clean-sheet kernel work |
| Backend attestation failure | Stop before performance benchmark |
| Correctness/stability failure | Keep CUTLASS default and end the Humming arm |
| Performance loss | Record result; keep CUTLASS default |

Rollback is selecting `M3_W4A8_BACKEND=cutlass` or omitting the variable. The
experimental work must not mutate the checkpoint, so rollback requires no model
conversion or artifact restoration.

## Validation strategy

### Local, CPU-capable verification

New tests should cover only new integration logic:

1. patch recognition, application, idempotence, and `--check` behavior using a
   minimal fake vLLM tree;
2. unchanged CUTLASS behavior when the selector is absent;
3. accepted and rejected backend selector values;
4. `PRINT_EFFECTIVE_CONFIG=1` includes `--quantization humming` only for the
   Humming arm;
5. fail-closed version and checkpoint-metadata classification;
6. backend-attestation log classification;
7. no accidental staging or dependency on the user's untracked `AGENTS.md`.

The existing launcher, ABI, and patch tests should be extended rather than
creating parallel test frameworks.

### Cluster qualification

Cluster work follows `PLANNER_EXECUTOR_PROTOCOL.md` and remains unauthorized
until a complete `READY_FOR_EXECUTOR` packet is committed. It must use a
persistent controller with top-level `srun`, never `sbatch`.

Qualification proceeds in this order:

1. **Dry run and preflight**
   - verify revision, checkpoint, environment, package pins, selector, and
     expected vLLM command;
   - no GPU launch on failure.
2. **Load and backend attestation**
   - TP8 plus EP, same checkpoint as CUTLASS;
   - require Humming transformation and MoE selection markers.
3. **Correctness smoke**
   - reuse the existing serving smoke and MiniMax-M3 diagnostics;
   - require finite, non-empty, non-garbage output;
   - verify resolved activation constants and routed/shared expert activity;
   - use existing layer-boundary diagnostics if output behavior diverges.
4. **CUDA-graph and stability qualification**
   - exercise the normal graphs-on production configuration;
   - preserve capture evidence and repeated serving results;
   - no eager-mode result may substitute for a graphs-on qualification.
5. **Existing performance benchmark**
   - same checkpoint, profile, TP8/EP topology, model length, KV-cache dtype,
     request workload, timestamp grouping, and serving defaults as the paired
     CUTLASS control;
   - change only the requested W4A8 backend;
   - use the existing benchmark's artifacts and scoring.

No model-quality evaluation or re-quantization is authorized by this design.
The checkpoint and mathematical quantization contract are unchanged; bounded
correctness diagnostics protect against layout or runtime integration errors.

## Performance decision

The existing performance benchmark remains authoritative. This design does not
alter its workload or invent a replacement score.

Interpretation priorities are:

1. production agentic/high-concurrency throughput and TTFT;
2. stability and tail behavior across the benchmark's concurrency sweep;
3. concurrency-1 throughput as a secondary guardrail;
4. startup/JIT time and memory as operational evidence, not substitutes for
   steady-state serving performance.

The planner will compare Humming and CUTLASS arms from the same paired run.
Humming becomes a production candidate only if it wins the benchmark's primary
production decision without failing correctness, graphs-on stability, backend
attestation, or the concurrency-1 guardrail. Otherwise CUTLASS remains the
default and the result is still a valid negative finding.

Marlin W4A16 remains contextual evidence. Its different activation precision
means it is not the kernel-only control for this goal.

## Expected implementation surface

The implementation plan is expected to modify a small set of existing files:

- `pipeline/slurm/patch_vllm_m3_serve.py`;
- `pipeline/slurm/run_vllm_http_serve_smoke.sh`;
- `pipeline/tests/test_patch_vllm_m3_serve.py`;
- the existing launcher/preflight tests most directly responsible for effective
  serving configuration;
- a task-specific planner/executor handoff after local implementation passes.

A small shared helper for Humming preflight or backend-attestation parsing is
permitted only if placing that logic in the launcher would make the shell
interface brittle. It must be a deep, fail-closed module reused by both dry-run
tests and the executor packet, not a new framework.

The benchmark repository and its workload scripts are not part of the expected
implementation diff.

## Effort and escalation boundary

Expected effort is two to five engineering days if existing Humming conversion,
loading, and CUDA graphs work on the target H100 environment:

- local selector, patch, preflight, and tests;
- one bounded TP8/EP integration qualification;
- one paired invocation of the existing performance benchmark.

A Humming-specific loader or CUDA-graph defect may extend this to one or two
weeks.

CUDA logic becomes eligible for design only after all of the following evidence
exists:

1. the failure reproduces with the exact pinned versions and target checkpoint;
2. checkpoint metadata, vLLM selection, weight conversion, activation layout,
   routing, and launch configuration have been ruled out;
3. the failure is localized to an existing Humming kernel operation;
4. upstream Humming issues, commits, branches, and releases have been searched;
5. adapting an existing upstream implementation is insufficient;
6. a separate design is approved.

Until then, no custom kernel is authorized.

## Acceptance criteria for implementation readiness

Before the design can transition to a written implementation plan:

- this specification is reviewed and approved;
- the implementation plan names exact files, tests, and verification commands;
- CUTLASS remains the default in every planned change;
- Humming selection and fallback behavior are fail-closed;
- no benchmark-harness construction appears in the plan;
- no CUDA arithmetic appears in the plan;
- the future executor packet preserves the existing benchmark contract and
  follows the repository protocol.
