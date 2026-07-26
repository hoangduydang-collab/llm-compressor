# MiniMax-M3 native Humming W4A8 qualification — result

**Status:** QUALIFIED (r3 passed all gates)

**Date:** 2026-07-25

**Configuration:** full-stack agent (`FULL_STACK_AGENT_PROTOCOL.md`)

**Supersedes the procedure in** [`M3_HUMMING_W4A8_HANDOFF.md`](M3_HUMMING_W4A8_HANDOFF.md),
which is now `HISTORICAL`. Related kernel analysis:
[`M3_HOPPER_W4A8_KERNEL_INVESTIGATION.md`](M3_HOPPER_W4A8_KERNEL_INVESTIGATION.md).

## Decision question and answer

> Does the pinned Humming path load the in-house GPTQ W4A8 checkpoint on TP8
> plus expert parallelism, positively attest indexed Humming MoE selection, and
> complete ten repeated graphs-on HTTP correctness smokes without non-finite,
> empty, degenerate, or failed output?

**Yes — after fixing two real defects.** Humming was not usable with this
checkpoint out of the box; neither defect was in the checkpoint or the numerical
contract.

This answers compatibility, backend identity, bounded correctness and stability
only. It says **nothing** about whether Humming is faster than CUTLASS.

## Runs

| Run | Node | Died / finished at | Cause | Outcome |
| --- | --- | --- | --- | --- |
| r1 | gpu-h101 | model init, ~40 s | `pack-quantized` rejected by Humming's compressed-tensors *input* schema | FAILED, fixed in `c6757d74` |
| r2 | gpu-h101 | weight repack, ~7 min | NVRTC could not open `libnvrtc-builtins.so.13.0` | FAILED, fixed in `05af0851` |
| r3 | gpu-h101 | completed | — | **PASSED, all 7 gates** |

Result roots under `/mnt/nfs/hoangduy/results/m3-humming-w4a8-qualification/`;
committed small artifacts under `evidence/m3-humming-w4a8/<run-id>/`.

## Defect 1 — `pack-quantized` activations were unreachable by construction

`humming/schema/compressed_tensors.py`, `CompressedTensorsInputSchema.__post_init__`
asserts the checkpoint format is one of `int-quantized`, `float-quantized`,
`naive-quantized`, `nvfp4-pack-quantized`, `mxfp4-pack-quantized`. A
compressed-tensors GPTQ W4 checkpoint is always `pack-quantized`, so **every**
compressed-tensors W4A8 checkpoint raises `AssertionError` while the first
`Linear` is constructed. r1 failed on the vision tower's `qkv_proj` purely
because that is the first `Linear` built; any layer would do it.

Evidence that this is an omission rather than a policy:

- the *weight* schema in the same file whitelists `pack-quantized` in 0.1.6,
  0.1.10 **and** 0.1.11;
- inside the input schema, `format` is used for exactly one thing —
  `input_scale_key = "input_global_scale" if "nvfp4" in format else "input_scale"`;
- `get_tensors_attrs` requests an input-scale tensor only when the activation is
  **not** dynamic (`dynamic is False` or `== "local"`).

Verified on CPU against the real checkpoint config: both schemas build,
`weight_key=weight_packed`, `actorder=static` accepted, activation `bits=8`,
`dynamic=True`, and **required input tensors `== {}`** — so the added whitelist
entry is inert beyond admission for this contract.

Applied by `pipeline/slurm/patch_humming_ct_input_format.py` to an **isolated
side-install**, never the shared `quant` venv.

**This is worth reporting upstream to `inclusionAI/humming`.** Not yet filed.

## Defect 2 — NVRTC builtins unreachable despite being installed

All eight TP workers died in `process_weights_after_loading` →
`transform_humming_layer` → `prepare_humming_weight` → `repack_weight` →
`NVRTCCompiler`:

```
nvrtc: error: failed to open libnvrtc-builtins.so.13.0
nvrtc_compile: compile failed: NVRTC_ERROR_BUILTIN_OPERATION_FAILURE
```

Humming's discovery is *correct*: `libnvrtc.so.13` and
`libnvrtc-builtins.so.13.0` sit in the same `nvidia/cu13/lib`, and its helper
binary links libnvrtc with `-Wl,-rpath,<lib_dir>`. But NVRTC `dlopen`s the
builtins at runtime by plain soname, and `dlopen` does not consult the
*executable's* rpath for a library requested by a dependency.

Verified with zero GPU cost by replaying the cached `cmdline.json` against the
cached `kernel.cu` (NVRTC compilation is host-only): `rc=1` without the dir on
`LD_LIBRARY_PATH`, `rc=0` with it, compiling
`weight_repack_nk<kNumBitsB=4U, kNumBitsA=8U, kPackedInput=true, ...>` — the
exact packed-W4/A8 contract.

Fixed in `pipeline/slurm/run_vllm_http_serve_smoke.sh`: the dir is derived from
the installed nvidia wheel (tracks the CUDA major rather than hardcoding 13.0),
exported as `LD_LIBRARY_PATH`, printed as `HUMMING_NVRTC_LIB_DIR` for
attestation, and Humming selected with no builtins found is now a **hard launch
failure** instead of a multi-minute cluster failure. CUTLASS path untouched.

## r3 evidence

```
srun.rc=0   node.rc=0   qualification-summary.rc=0   first-failure: absent
preflight     valid=true  backend=humming  gemm_type=indexed  group_size=128
                          activation_strategy=token  abi_valid=true
attestation   valid=true  backend=humming  gemm_type=indexed
              indexed_marker=true  grouped_marker=false  quantization_marker=true
              reason_codes=[]
```

Backend selection was positively observed, not inferred:
`fused_humming_moe.py:62 Using indexed gemm for humming moe`. No CUTLASS,
Marlin, grouped-contiguous, or unquantized fallback marker appeared.

CUDA graphs: 51 PIECEWISE (mixed prefill-decode) + 51 FULL (decode),
`Graph capturing finished in 37 secs, took 1.40 GiB`. No IMA in the 43–48/51
window that previously broke shared-stream capture.

Ten smokes, all `rc=0` and all content gates `valid`. Example (smoke-01):

```
content   : "2+2 = 4."
reasoning : "The user is asking a simple math question: What is 2+2? ..."
usage     : prompt_tokens=186 completion_tokens=32 total_tokens=218
gate      : content_chars=90  eight_gram_diversity=1.0
```

Server-log markers: `M3_MOE_PROBE#` ×48, `M3_LOAD_AUDIT#` ×16;
`M3_MOE_PROBE_NONFINITE` ×0, `illegal memory access` ×0, `CUDA error` ×0.

**One artifact requires an honest note.** `serve.log` contains a `Traceback` and
an `EngineDeadError` — these are **teardown**, not failure. Ordering in the log:
last smoke `POST /v1/chat/completions` at line 314, our own cleanup
`shutdown triggered` at line 315, first `Traceback` at line 331. Exactly 10
POSTs, all `200 OK`. The node's fatal-marker grep ran before cleanup, so the
pass is not an artifact of grep timing.

## Environment — read this before reproducing

| Item | Value |
| --- | --- |
| vLLM | 0.24.0 |
| humming-kernels | **0.1.10, side-installed** at `/mnt/nfs/hoangduy/venvs/humming-0.1.10-site` |
| torch / CUDA | 2.11.0+cu130 / 13.0 |
| GPUs | 1 exclusive node, 8×H100, capability (9, 0) |
| Topology | TP8 + `--enable-expert-parallel`, CUDA graphs on |
| Humming policy | `VLLM_HUMMING_MOE_GEMM_TYPE=indexed`, `VLLM_HUMMING_USE_F16_ACCUM=0` |
| JIT cache | `/mnt/nfs/hoangduy/.humming/cache-m3-gptq-w4a8-v1` |
| Checkpoint | `artifacts/m3-awq-gptq-prepared/gptq-checkpoint-vllm-w123-abi-overlay` |

The shared `quant` venv still carries pristine `humming-kernels==0.1.6`. Humming
runs **must** prepend the side-install on `PYTHONPATH`; the node script asserts
`humming.__file__` resolves there.

**The 0.1.10 pin is load-bearing** — not for defect 1 (that whitelist gap is
identical in 0.1.6/0.1.10/0.1.11) but because 0.1.10's weight schema admits
`actorder="static"` where 0.1.6 accepted only `None`/`"weight"`. Our weights are
`actorder: static`, and the checkpoint carries **zero `g_idx` tensors**, so no
runtime reindexing is required.

## Integrity accounting

The qualified stack contains **one declared modification to a third-party
kernel library**, so "Humming qualified" means "Humming 0.1.10 plus a one-line
`pack-quantized` admission". This is recorded, not hidden:

- `DECLARED_PATCH_SHA256` in `pipeline/m3_humming_w4a8.py` pins the post-patch
  SHA-256 (`8e2ab300b595e98f9b66d76096c6a03272ffe948e11dd29844af701c1f6474c3`);
- integrity reports `record-matched-with-declared-patch` **and names the file**
  in `humming_declared_patches`;
- undeclared paths fail closed (`HUMMING_UNDECLARED_PATCH`), and a declared path
  with unexpected content is still a `mismatch`;
- the NVFP4 overlay marker scan still runs, including over bytecode payloads.

A separate earlier fix (`6e747d4b`) stopped the integrity gate from reading
0.1.10's 58 unhashed `__pycache__` RECORD entries as source tampering; unhashed
*source* files still fail closed and the skip count is reported as
`humming_unhashed_bytecode`.

## What this does NOT establish

- **No performance claim.** Humming vs CUTLASS was not measured. Nothing here
  says Humming is faster, slower, or equal.
- **No quality claim.** Ten fixed 2+2 smokes are a liveness and
  non-degeneracy gate, not an evaluation. No benchmark score is implied and no
  result here is comparable to any public leaderboard recipe.
- **Indexed only.** Grouped/automatic Humming scheduling remains untested and is
  still gated behind indexed correctness per the investigation's arm order.

## Recommended next step

The kernel investigation's arm 2 gate ("loader, correctness, backend
attestation, then performance") is now satisfied for correctness and
attestation. The next step is the **paired CUTLASS vs Humming-indexed serving
benchmark** at the target concurrency points, same model / topology / request
set / measurement protocol, with `HUMMING_NVRTC_LIB_DIR`, the declared-patch
attestation, and the effective argv recorded per arm. That is a new experiment
and needs design sign-off on concurrency points and pass/fail thresholds before
it spends cluster time.

---

## Addendum (2026-07-25, later the same day) — grouped_contiguous and three more kernel defects

The grouped arm work that followed this qualification found **three additional
Humming kernel defects**, all patched, SHA-declared, and evidenced. This
supersedes the "Indexed only" limitation above and the single-patch integrity
accounting: the side-install now carries **four declared patches**, and
`DECLARED_PATCH_SHA256` holds a tuple of allowed post-patch hashes per file
(0.1.10 and, where upstream content changed, 0.1.11).

| # | Defect | Patch | Evidence |
|---|---|---|---|
| 3 | grouped_contiguous derives the **last expert's row count from `a.size(0)`**, which vLLM oversizes to `(M·topk, K)` — tail experts corrupted (100% of experts 13–15's rows wrong in the probe) | `patch_humming_grouped_expert_bounds.py` | `evidence/m3-arm3-grouped-bounds/` |
| 4 | **Missing `fence.proxy.async.shared::cta`** before TMA C stores (PTX-required; empirically *not* the observed corruption on its own) | `patch_humming_tma_store_fence.py` | `evidence/m3-arm3-tma-commit/` (fence-only run: 11/48 still bad) |
| 5 | **`cp.async.bulk.commit_group` never called**, so every `tma_wait_store_group` was a no-op — producer overwrote the union-aliased epilogue smem mid-store (intermittent whole-tile garbage; early-EOS at serving scale) and stream-K released tile locks early (nondeterminism at BM=8/16) | `patch_humming_tma_store_commit.py` | `evidence/m3-arm3-tma-commit/` (pre-fix 10/48 bad; post-fix **0/96**, clean sweep, determinism restored) |

None of the four defects (incl. defect 1's schema gap) are fixed in upstream
0.1.11; none are filed upstream yet (outward-facing — needs sign-off). With all
four patches, grouped_contiguous re-qualified and completed the three-arm perf
window: results in [`docs/m3-w4a8-three-arm-perf.md`](docs/m3-w4a8-three-arm-perf.md)
(which also covers the paired benchmark recommended above, extended to three
arms). A patched 0.1.11 side-install exists for the packed-K dequant layout
adoption (`run_humming_0111_packedk_qual_srun.sh`, qualification pending).
