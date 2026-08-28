# MiniMax-M3 AWQ quality: offset RMSNorm smoothing (investigating)

The 20260712 layer-boundary matrix localizes the first catastrophic value to
layer 8 after attention and before MoE: candidate MoE input norm is about
176,845 versus reference 177. MiniMax-M3 uses MiniMaxM3VLRMSNorm, whose
forward multiplies by 1 + weight. The shared offset-norm calibration registry
handled Gemma and Qwen aliases but not the MiniMax class, so AWQ could smooth
the raw zero-centered parameter incorrectly. The repair registers the exact
class and tests a fresh AWQ checkpoint against both a no-MLP-smoothing control
and the repaired GPTQ checkpoint. Validation covers sparse layers 3-59 and
canonical HTTP serving; CUDA graphs remain out of scope until quality passes.

# Bugs and fixes (llm-compressor pipeline)

## Pre-quantization gate: meta-device MoE linearization offload (fixed, 2026-07-13)

**Symptom:** The first real MiniMax-M3 CLI run of the pre-quantization
compatibility gate (`python -m pipeline.prequant_compatibility`) died with
`NotImplementedError: Offload of type meta and distributed=False has not been
implemented` while constructing the meta model. No GPU, calibration data, or
checkpoint weights were involved.

**Root cause:** The gate builds a disposable model under
`accelerate.init_empty_weights` and calls `linearize_moe`, which reaches
`LinearExperts2D.from_experts_module`. That method unconditionally initialized
runtime offload for every submodule via `compressed_tensors.offload.offload_module`.
With a meta source, `get_cache_init_kwargs` resolves `offload_device` to `meta`,
and `compressed-tensors` intentionally has no `meta` offload backend, so
`OffloadCache.cls_from_device("meta")` raises. The synthetic gate tests stubbed the
analyzer and the existing offload tests are GPU-only, so the meta boundary was
uncovered.

**Fix:** In `src/llmcompressor/modeling/moe/linear_experts.py`, skip the offload
loop when the resolved `offload_device` is `meta`, leaving the linearized modules
meta-only. The guard keys off the exact device `offload_module` would reject; CPU,
CUDA, and disk offload paths are unchanged. Covered by a new CPU-only regression,
`tests/llmcompressor/modeling/test_linearize_meta.py`, which reproduces the crash
under `init_empty_weights` before the fix.

## Static quantization serving preflight (MiniMax-M3 proven; generalization deferred)

The CPU-only serving ABI gate correctly rejected the original in-house GPTQ
checkpoint with 228 plain router/shared-expert modules whose Transformers
ignore rules did not match vLLM runtime names. A metadata-only alias overlay
then passed with a byte-identical Safetensors index, and repaired GPTQ produced
coherent results in two independent smoke runs. This closes the catastrophic
GPTQ config/namespace mismatch at smoke level.

The current checker is deliberately MiniMax-M3-specific; it must not be
treated as a generic AWQ/GPTQ/FP8 validator yet. See
`docs/quantization-static-serving-preflight-status-and-roadmap.md` for the
proven contract, limitations, and the future adapter-based design required
before making this gate mandatory for every newly quantized model.

## Active priority: repair MiniMax-M3 shared-expert loading before CUDA-graph RCA (2026-07-11)

Canonical matrix `20260711-135100-canonical-chat` proved that cyankiwi passes
offline and HTTP while the candidate fails identically through both interfaces.
Routed matrix `20260711-144120-routed-diagnostics` then localized the failure.
Both candidate schemes see 171 shared-expert checkpoint tensors but leave exactly
171 unmatched on every rank. vLLM constructs zero packed shared parameters; all
48 candidate probes have `shared_norm=0` and `dropped=true`. The reference loads
BF16 shared weights and all 48 shared outputs are nonzero. Candidate W4A8 and
W4A16 first-MoE inputs match on all ranks, while LM-head hashes match reference.

Root cause: the checkpoint correctly stores BF16 shared experts and persists the
Transformers ignore path `mlp.shared_experts`, but vLLM constructs them as
`block_sparse_moe.shared_experts`. Compressed Tensors therefore applies the
quantized scheme at runtime and creates packed placeholders that cannot accept
the BF16 tensors. Active next step: the config-only alias repair and three-node
`srun` validation in `MINIMAX_M3_QUALITY_RUNBOOK.md`.

Do not re-quantize, rewrite tensor shards, or resume CUDA-graph RCA until W4A8
passes canonical offline and HTTP quality with healthy shared-expert evidence.

## HTTP async cudagraph IMA — RCA matrix protocol (cyankiwi, 2026-07-10)

**Symptom:** HTTP `vllm serve` for `cyankiwi/MiniMax-M3-AWQ-INT4` dies during
PIECEWISE/FULL graph capture with async-reported
`CUDA error: an illegal memory access` at `breakable_cudagraph._capture` →
`empty_cache()`. Offline `LLM()` serve-verify on the same ckpt can PASS graphs-on
at `max_model_len=8192`.

**A/B already observed on h119 (same ckpt, patches 4/4, flashinfer 0.6.12,
language-model-only, 8192/0.9, clean GPUs):**

| Launch | Result |
|--------|--------|
| `DEBUG_CUDAGRAPH=1` (`CUDA_LAUNCH_BLOCKING=1` + DSA) | **51/51 capture PASS**, server up |
| Async CUDA (no `CUDA_LAUNCH_BLOCKING`) | **IMA** at `empty_cache()` during capture |

**Evidence rule:** a named faulting device kernel (or a classified matrix
verdict stronger than `graph_ima_unclassified`) is required before claiming a
root cause. `CUDA_LAUNCH_BLOCKING=1` / `DEBUG_CUDAGRAPH=1` is a **masked_pass**,
never a fix. Do not treat sync-CUDA success as root-cause closure.

**Working hypotheses (discriminate with the matrix — do not assume):**

1. **MoE routing/finalize** — padded-token / NaN→dup-experts class
   ([vLLM #39391](https://github.com/vllm-project/vllm/pull/39391)); patches 1–4
   + flashinfer ≥0.6.10 were already live on both PASS and FAIL, so this is
   *unlikely* unless a different MoE symbol still faults.
2. **Breakable cudagraph + captured NCCL/collective** — MiniMax fused AR path
   under `breakable_cudagraph` with default `capture_error_mode=global`
   ([vLLM #46253](https://github.com/vllm-project/vllm/issues/46253)); sync CUDA
   masks cross-thread / rank-phase invalidation.
3. **Graph memory lifetime** — addresses baked into graphs then freed/realloc’d
   (empty_cache / workspace; [vLLM #45487](https://github.com/vllm-project/vllm/pull/45487)).

**Confounder for earlier W4AFP8 “2048 envelope PASS”:**
`debug_cudagraph_ima.sh` **always** exports `CUDA_LAUNCH_BLOCKING=1`. That PASS
may have been sync-CUDA, not the smaller memory envelope.

**Production default (HTTP + offline MiniMax-M3, 2026-07-10):** async CUDA
(`DEBUG_CUDAGRAPH=0`) with `VLLM_DISABLE_SHARED_EXPERTS_STREAM=1`. Validated by
the h125 stream-disabled matrix (3/3 ready+chat). `DEBUG_CUDAGRAPH=1` is
diagnostic-only (`masked_pass`). Escape hatch: `ENFORCE_EAGER=1`. Other models
keep standard vLLM defaults (stream feature left enabled).

### RCA matrix (no vLLM site-packages edits)

Harness:

- Classifier: `pipeline/m3_cudagraph_evidence.py` (+ unit tests)
- Runner: `pipeline/slurm/test_m3_http_cudagraph_matrix.sh`
- Launcher observability: `PRINT_EFFECTIVE_CONFIG=1` on
  `run_vllm_http_serve_smoke.sh`

Cases (one variable per trial; unique port/log/pid):

| Case | Knobs | Expected use |
|------|-------|----------------|
| `async_baseline_{1,2,3}` | `ENFORCE_EAGER=0 DEBUG_CUDAGRAPH=0` | Determinism of async IMA |
| `graphs_off` | `ENFORCE_EAGER=1` | Must PASS or reject “graphs-only” claim |
| `blocking_mask` | `DEBUG_CUDAGRAPH=1` | Record `masked_pass` only |
| `breakable_off` | `VLLM_USE_BREAKABLE_CUDAGRAPH=0` | Implicate breakable path if async PASS |
| `async_coredump` | async + CUDA core dump env | Name faulting kernel via `cuda-gdb` |

Verdicts: `server_ready`, `masked_pass`, `graph_ima_moe`,
`graph_ima_collective`, `graph_ima_memory_lifetime`, `graph_ima_unclassified`,
`graphs_off_failed`, `inconclusive`.

**How to run:**

```bash
# local dry-run (no GPUs / no nohup):
DRY_RUN=1 bash pipeline/slurm/test_m3_http_cudagraph_matrix.sh

# cluster (free 8-GPU node, e.g. h119):
bash pipeline/slurm/free_gpus.sh
bash pipeline/slurm/test_m3_http_cudagraph_matrix.sh
# results under /mnt/nfs/hoangduy/logs/m3-cudagraph-rca/<run_id>/summary.json

# single case:
MATRIX_CASES=async_baseline_1 bash pipeline/slurm/test_m3_http_cudagraph_matrix.sh
```

**Next branch after classified result:**

| Verdict | Next |
|---------|------|
| `graph_ima_moe` | Compare installed route vs #39391 / FlashInfer finalize |
| `graph_ima_collective` | Separately approved `capture_error_mode=thread_local` / rank barriers (#46253) |
| `graph_ima_memory_lifetime` | Address-lifetime instrumentation (#45487) — follow-up plan |
| `graph_ima_unclassified` | Keep tactical mask; escalate to `compute-sanitizer` before code changes |

**Matrix result (h119, 20260710-051009):**
[`summary.json`](/mnt/nfs/hoangduy/logs/m3-cudagraph-rca/20260710-051009/summary.json).
`async_baseline_1` and `_3` were `graph_ima_collective`; `_2` was
`server_ready` with chat, establishing an async-flaky failure. `graphs_off`,
`breakable_off`, and `blocking_mask` reached ready + chat; the latter is only
`masked_pass`. The coredump's first device kernel was
`at::native::vectorized_elementwise_kernel<8, CUDAFunctor_add<BFloat16>>`,
which is not a named MoE or collective kernel, so
[`result_with_kernel.json`](/mnt/nfs/hoangduy/logs/m3-cudagraph-rca/20260710-051009/async_coredump/result_with_kernel.json)
is `graph_ima_unclassified`. Next: retain the tactical mask and run a
single-failure `compute-sanitizer` follow-up before changing capture or
collective code. `breakable_off` passing makes the breakable-cudagraph path the
highest-priority experiment target, not established root cause.

**Shared-expert stream RCA (h125, 20260710-072629):**
[`comparison.json`](/mnt/nfs/hoangduy/logs/m3-cudagraph-shared-stream/20260710-072629-comparison.json)
recorded IMA in 2/3 threshold-256 controls at 43/51 capture progress, while
disabling only `VLLM_DISABLE_SHARED_EXPERTS_STREAM` passed ready + chat in all
3/3 trials. Threshold 128 still failed at 43/51, 43/51, and 45/51 rather than
the predicted 31–33/51 shift. The shared stream is therefore a strong
workaround signal, but the missing-main-stream-join hypothesis is **narrowed,
not strongly confirmed**: do not patch it yet. Next: independently verify the
fork's effective per-worker threshold behavior before changing stream
synchronization.

**MiniMax-M3-only serving workaround (2026-07-10):** Until the source-level
cause is confirmed and fixed, serve **MiniMax-M3** with
`VLLM_DISABLE_SHARED_EXPERTS_STREAM=1`. This disables only the auxiliary CUDA
stream used to overlap shared-expert work; it does not disable shared experts or
otherwise change the model. It is a reliability-first workaround for this
MiniMax-M3 CUDA-graph failure, not a general vLLM recommendation: all other
models should follow standard vLLM serving practice and leave the feature at its
default unless their own validated issue requires a change.

**Wired as the MiniMax-M3 production default (2026-07-10):** HTTP and offline
launchers now default to async CUDA (`DEBUG_CUDAGRAPH=0` / no
`CUDA_LAUNCH_BLOCKING`) **plus** `VLLM_DISABLE_SHARED_EXPERTS_STREAM=1`. The
h125 stream-disabled matrix already validated that exact envelope (3/3
`server_ready` + chat with graphs and breakable graphs on). Paths covered:

- `pipeline/slurm/run_vllm_http_serve_smoke.sh`
- `pipeline/slurm/serve_minimax_m3.sbatch` / `run_serve_minimax_m3_detached.sh`
- `pipeline/serve_verify.py` (`apply_minimax_m3_serve_env`, M3-gated only)

`DEBUG_CUDAGRAPH=1` remains a diagnostic opt-in (masks the race; classifier
`masked_pass`). RCA A/B can still force `VLLM_DISABLE_SHARED_EXPERTS_STREAM=0`.

**Retained (still required / not proven removable):** W4A8 SwiGLU site-packages
patches 1–2, FlashInfer `FLASHINFER_USE_CUDA_NORM=1`, checkpoint/config
preflight, `disable_custom_all_reduce=true` (h118 2×4 topology deadlock), and
persistent cudagraph patches 3–4 (fused-AR NCCL fallback + router
`nan_to_num`). **Removed as proven-dead only:** unwired in-process duplicates
`patch_vllm_m3_fused_ar_for_cudagraph` / `patch_vllm_m3_moe_router_for_cudagraph`
in `pipeline/vllm_m3_patches.py` (workers never saw them; site-packages edits
remain the live path).

This may cost performance only for scheduled batches at or below the default
256-token shared-expert-stream threshold. Comparable upstream TP+EP tests show
low-single-digit gains from the overlap, but MiniMax-M3 has not been benchmarked
under the production workload, so the cost must not be assumed negligible.
Future work: run a production-shaped A/B benchmark on MiniMax-M3 measuring
output throughput, TTFT, and TPOT with the stream enabled versus disabled.
Keep the stream disabled in production unless that benchmark and a corrected
source-level implementation demonstrate safe, worthwhile re-enablement.

## HTTP `vllm serve` vs offline `LLM()` (cyankiwi, 2026-07-09)

**Earlier hypothesis (partially superseded by the A/B above):** first HTTP smoke
also copied Nemotron knobs and skipped offline preflight. Those footguns are
still real and still fixed in the script (config/VL preflight,
`--language-model-only`, no forced Nemotron batching). They are **necessary but
not sufficient** — after those were fixed, the async race remained.

| Factor | Offline `LLM()` | Early HTTP smoke |
|--------|-----------------|------------------|
| Preflight config + VL processor | Always | Was missing → now applied |
| `--language-model-only` | N/A | Was missing → now default |
| Nemotron `max_num_seqs` / batched_tokens | Unset | Was forced → now omitted |
| `CUDA_LAUNCH_BLOCKING` | Usually unset (PASS) | Was forced to 1 (masked); now default async with stream disabled |

## MiniMax-M3 vLLM serve stage chronicle (h118, 2026-07-07/08)

Reference run: smoke AWQ **W4AFP8** checkpoint
``artifacts/MiniMax-M3-awq-W4AFP8/20260707-082218/checkpoint`` (8 calibration samples),
**vLLM 0.24.0** (``toncao/vllm@minimax-m3-compressed-tensors``), **TP=8 + EP**,
``block_size=128``, ``kv_cache_dtype=fp8``, ``max_model_len=8192``, node **h118**
(8×H100 80GB, 2×4 NV18 + cross-socket SYS).

### What works (verified)

| Configuration | Outcome | Notes |
|---------------|---------|-------|
| **``ENFORCE_EAGER=1``** (CUDA graphs off) | **PASS** — ``overall_ok=True`` | Load, KV init, generation complete. First token ~2–3 min (FlashInfer trtllm workspace + Triton autotune). Output garbage expected (smoke quant). |
| Persistent patches 1–2 (W4A8 SwiGLU-OAI **uninterleaved**) | **Resolved** | ``llm-compressor`` saves MoE ``w13`` as ``[all gates; all ups]`` (natural ``MergedColumnParallelLinear`` layout). vLLM's W4A8 CUTLASS path expected **interleaved** SwiGLU-OAI and omitted ``SWIGLUOAI_UNINTERLEAVE`` from ``_supports_activation``. Fixed by patches 1–2 — **not** a stock-vLLM vs toncao issue (toncao's branch has the same gap; reinstall alone does not help). |
| ``FLASHINFER_USE_CUDA_NORM=1`` | Required on Hopper | Without: CuTe-DSL ``gemma_rmsnorm`` nanobind abort at KV init. |
| ``hidden_act=swigluoai`` in checkpoint | Required | Transformers coerces to ``silu``; ``serve_verify`` restores from ``model.id``. |
| ``toncao/vllm@minimax-m3-compressed-tensors`` | Team serve build | M3 compressed-tensors layout (e.g. bf16 MSA indexer separate from quant q/k/v, per-expert linearized weights). Separate from the uninterleaved w13 / SwiGLU-OAI patch above. |

### Prerequisites fixed (to reach graph capture at all)

Each was a **hard blocker** before CUDA-graph capture could start:

1. **``hidden_act`` → ``swigluoai``** — ``ensure_minimax_m3_vllm_serve_config()`` (no re-quant).
2. **W4A8 uninterleaved w13 + ``SWIGLUOAI_UNINTERLEAVE``** — patches 1–2 (weight layout / activation enum plumbing; **resolved**).
3. **FlashInfer CuTe-DSL RMSNorm JIT** — ``FLASHINFER_USE_CUDA_NORM=1`` in launchers + ``serve_verify``.
4. **Vision ``img_token_compression_config``** — restored from ``model.id`` into checkpoint ``config.json``.
5. **VL processor artifacts** — copied into checkpoint dir before ``LLM()``.
6. **Orphaned ``Worker_TP*`` processes** — kill via ``nvidia-smi`` before restart (else OOM / Broken pipe noise).

### CUDA graph capture (``enforce_eager=false``) — status update 2026-07-09

**Earlier (2026-07-07/08):** PIECEWISE capture progressed to **16/51**, then
``CUDA illegal memory access`` (async report at ``empty_cache()``). Attempts A–G
below did not move that failure under detached-script defaults.

**Update (2026-07-09):** with ``debug_cudagraph_ima.sh`` settings
(``MAX_MODEL_LEN=2048``, ``GPU_UTIL=0.85``, ``CUDA_LAUNCH_BLOCKING=1``,
``TORCH_USE_CUDA_DSA=1``, ``disable_custom_all_reduce=true``) and patches 1–4
applied, **graphs-on serve-verify PASSes** on both smoke
(``20260707-082218``) and full-calib (``20260708-093642``) checkpoints.
Detached defaults (``MAX_MODEL_LEN=8192``, ``GPU_UTIL=0.9``) still reproduced
IMA @ 16/51 on a re-run.

**Later A/B (HTTP cyankiwi, same day):** sync CUDA PASS @ 8192 vs async IMA —
so the W4AFP8 “2048 PASS” is **confounded** by ``CUDA_LAUNCH_BLOCKING=1`` in
the debug script. Prefer treating that as a **sync-vs-async capture** signal
until 8192 is re-validated **without** ``CUDA_LAUNCH_BLOCKING``. See
**“HTTP async cudagraph race”** above. Quality remains garbage; see
**“MiniMax-M3 full-calib AWQ garbage output”** below.

Historical attempt table (kept for the 16/51 investigation):

| # | Attempt | Hypothesis | Result on h118 |
|---|---------|------------|----------------|
| A | ``disable_custom_all_reduce=true`` | FlashInfer / custom AR incompatible with 2×4 topology | **Wrong lever** — does not gate M3's ``fused_allreduce_gemma_rms_norm`` ([vLLM #45800](https://github.com/vllm-project/vllm/issues/45800)). Logs still show ``Auto-selected flashinfer allreduce backend: trtllm``. Removed as default. |
| B | ``ENFORCE_EAGER=1`` | Skip graph capture entirely | **WORKS** — full serve-verify pass. Not a graph fix. |
| C | Fused-AR patch (#46253) — skip FlashInfer in ``_can_use_flashinfer`` when graphs on (patch 3) | FlashInfer fused AR+RMSNorm not graph-capturable on TP8 | **Still IMA at 16/51** after persistent patch applied. |
| D | MoE ``nan_to_num(router_logits)`` in ``MoERunner._apply_quant_method`` (old patch 4, [vLLM #39288](https://github.com/vllm-project/vllm/issues/39288) class) | Padding tokens → NaN logits → duplicate expert IDs → W4A8 CUTLASS OOB | **Still IMA at 16/51** — **root cause of the non-fix found (2026-07-08): wrong code path.** See H. |
| E | Runtime monkeypatches in ``serve_verify`` / ``vllm_m3_patches.py`` | In-process hook before ``LLM()`` | **Ineffective for capture** — ``Worker_TP*`` are **spawned subprocesses**; they load site-packages only. Documented; launcher now auto-runs persistent patch script. |
| F | ``fuse_allreduce_rms=false`` (compile config) | Disable fused AR via inductor | **No effect** — M3 calls fused path directly in ``model.py`` forward. |
| G | ``SERVE_PERF=1`` / re-enable FlashInfer fused AR | Official recipe perf path | Not tested for graphs; eager serve works with FlashInfer trtllm (slow init). |
| **H** | **Repoint router ``nan_to_num`` from ``MoERunner._apply_quant_method`` → ``RouterBase._select_experts`` (new patch 4)** | **Old patch 4 was dead code**: the migrate commit [#42680](https://github.com/vllm-project/vllm/commit/4438b6e7dc480dd59e5edabfcc939c15321a129a) moved W4A8 to the **modular** ``FusedMoEModularKernel``; in vLLM 0.24.0 routing lives in ``fused_moe/router/`` and funnels through ``RouterBase._select_experts`` → ``_compute_routing`` — it **never calls** ``MoERunner``. So the NaN clamp verified as applied but never ran on the logits feeding the CUTLASS grouped GEMM, which is exactly why 16/51 never moved. | **To verify on cluster** (patch relocated 2026-07-08). |

**Root-cause diagnosis (2026-07-08): the two prior "cudagraph" patches were both misdirected.**

1. **Patch 4 was on a dead path** (attempt H above). Fixed by moving the ``nan_to_num`` to ``RouterBase._select_experts`` (``fused_moe/router/base_router.py``) — the template method every router funnels through, and the routing entry vLLM maintainers pointed to for the #39288 IMA class. (vLLM 0.24.0 refactored routing out of ``layer.py`` into the ``router/`` package.)
2. **Patch 3 (fused-AR NCCL fallback) targets a cross-node-only failure.** [#46253](https://github.com/vllm-project/vllm/issues/46253) states single-node NVLink fused AR **is** graph-capturable; h118 is single-node. That patch not moving 16/51 confirms the fused AR is **not** the faulting kernel here. Kept (harmless, needed for future multi-node) but not the h118 fix.

**Secondary suspect (separate mechanism): flashinfer finalize bounds check.**
[#35706](https://github.com/vllm-project/vllm/issues/35706) / [#42906](https://github.com/vllm-project/vllm/issues/42906) — flashinfer **v0.5.3 dropped the bounds check** in ``finalizeMoeRoutingKernel`` (``expanded_permuted_row < 0 || >= expanded_rows``); padding tokens then index out-of-bounds during capture → IMA. Restored in **flashinfer ≥ 0.6.10** (flashinfer#2762; vLLM v0.22.0+ bundles ``0.6.11.post2``). The **quant venv had vLLM force-reinstalled while keeping existing deps**, so its flashinfer may be stale. This affects the *flashinfer-backed* MoE finalize, distinct from vLLM's native W4A8 grouped GEMM — hence it must be confirmed with the sync trace before bumping. ``patch_vllm_m3_serve.py`` now prints the flashinfer version and warns if ``< 0.6.10``.

**Decisive next step (was "not yet run"; now scripted):**

```bash
bash pipeline/slurm/debug_cudagraph_ima.sh 2>&1 | tee /mnt/nfs/hoangduy/logs/m3-cudagraph-debug.log
```

Sets ``CUDA_LAUNCH_BLOCKING=1 TORCH_USE_CUDA_DSA=1`` so the crash names the **real** faulting kernel (the async ``empty_cache()`` traceback hides it). Then:

- fault in **MoE routing/finalize** (``topk_softmax`` / ``moe_unpermute`` / ``finalizeMoeRoutingKernel`` / cutlass grouped GEMM) → the relocated router ``nan_to_num`` (patch 4/H) and/or flashinfer ``>= 0.6.11.post2`` is the fix;
- fault in **``fused_allreduce_gemma_rms_norm``** → revisit patch 3 / [#46253](https://github.com/vllm-project/vllm/issues/46253) (segment ``breakable_cudagraph`` at the collective).

**Other candidate fixes (apply guided by the sync trace):**

- Patch ``csrc/moe/topk_softmax_kernels.cu`` NaN clamp ([#39391](https://github.com/vllm-project/vllm/pull/39391)) if the Python ``nan_to_num`` in ``select_experts`` is still insufficient (fuses the clamp into the kernel).
- Zero-init **padding hidden states** at MoE layer entry ([#40047](https://github.com/vllm-project/vllm/issues/40047) class) if router logits are clean but upstream activations are garbage/NaN (per #35706, stale padding produces *finite* garbage experts too, not just NaN).
- Segment ``breakable_cudagraph`` at NCCL collectives (upstream [#46253](https://github.com/vllm-project/vllm/issues/46253) long-term fix).

### Commits (``duy-branch``, graph-capture work)

| Commit | Summary |
|--------|---------|
| ``3fd893ed`` | Fused-AR NCCL fallback when graphs on (patch 3) |
| ``39c3d6e5`` | MoE router ``nan_to_num`` (patch 4) |
| ``0199a48e`` | ``ensure_vllm_m3_patches()`` — workers need site-packages edits |
| ``f45667b5`` | Launcher/sbatch auto-run ``patch_vllm_m3_serve.py`` |

### Current status (2026-07-08)

- **Serve with graphs:** **root cause of the persistent 16/51 non-fix found** — the
  prior router patch (patch 4) sanitized ``MoERunner._apply_quant_method``, which the
  W4A8 **modular** kernel never calls (dead path); patch 3 (fused AR) targets a
  cross-node-only failure absent on single-node h118. **Fix applied in repo:** router
  ``nan_to_num`` relocated to ``RouterBase._select_experts``; flashinfer-version guard
  (#42906) + ``CUDA_LAUNCH_BLOCKING`` diagnostic added. **Pending cluster re-run** on h118.
- **Serve without graphs:** **OK** — ``ENFORCE_EAGER=1``.
- **Production path:** re-run ``debug_cudagraph_ima.sh`` to confirm the faulting kernel,
  then the relocated patch 4 (and/or flashinfer ``>= 0.6.11.post2``) should clear capture;
  do not ship with ``enforce_eager`` as the only solution if graph perf is required.

---

## MiniMax-M3 vLLM serve: CUDA graph capture (`enforce_eager`)

**Symptom (graphs on, `enforce_eager=false`):** KV init / graph capture dies with
worker ``CUDA error: an illegal memory access was encountered``; EngineCore
cascades with ``TCPStore Broken pipe`` on all ranks (shutdown noise, not root cause).
Often fails at a **fixed graph index** (e.g. 16/51) — batch-size-dependent.

**Root causes (two layers, both must be fixed for graphs on):**

1. **Fused all-reduce + RMSNorm in graph** — M3's forward calls
   ``fused_allreduce_gemma_rms_norm`` directly (``model.py``). With
   ``VLLM_USE_BREAKABLE_CUDAGRAPH=1`` (auto-enabled for M3), FlashInfer's fused
   all-reduce+RMSNorm kernel is captured inside the CUDA graph. On TP8 H100 it is
   **not graph-capturable** → illegal memory access at ``capture_end``
   ([vLLM #46253](https://github.com/vllm-project/vllm/issues/46253)).

2. **MoE padding NaNs during capture** — PIECEWISE graph dummy runs pad to fixed
   batch sizes. Padding tokens produce **NaN router logits**; top-k returns
   **duplicate expert IDs**; W4A8 CUTLASS MoE finalize dereferences bad rows →
   IMA at a deterministic capture step (e.g. 16/51)
   ([vLLM #39288](https://github.com/vllm-project/vllm/issues/39288),
   fixed upstream in [#39391](https://github.com/vllm-project/vllm/pull/39391) —
   our toncao 0.24.0 build may lack it).

``fuse_allreduce_rms=false`` in compile config does **not** gate the fused AR path
([vLLM #45800](https://github.com/vllm-project/vllm/issues/45800)).

**Fix:**

1. **Fused AR:** ``fused_allreduce_gemma_rms_norm`` NCCL fallback when
   ``enforce_eager=false`` (skip FlashInfer in ``_can_use_flashinfer``).

2. **MoE router:** ``torch.nan_to_num(router_logits)`` in
   ``RouterBase._select_experts`` (``fused_moe/router/base_router.py``, before
   ``_compute_routing``) — the routing entry every subclass funnels through (same
   mechanism as #39391). **Corrected 2026-07-08** from the earlier dead target
   ``MoERunner._apply_quant_method`` (the modular W4A8 path never calls
   ``MoERunner``; see chronicle attempt H).

- **Persistent:** ``pipeline/slurm/patch_vllm_m3_serve.py`` (4 edits); **required**
  for ``Worker_TP*`` subprocesses (spawned fresh — runtime monkeypatches in
  ``serve_verify`` do not apply). Launcher auto-runs this script; re-run after
  any vLLM reinstall. ``--check`` also reports the flashinfer version (finalize
  bounds-check suspect #42906).
- **Runtime:** removed for cudagraph (ineffective on workers). ``ensure_vllm_m3_patches()``
  in ``serve_verify`` verifies site-packages before ``LLM()``.

**Verify on cluster:**

```bash
python pipeline/slurm/patch_vllm_m3_serve.py --check   # all 4 patches + flashinfer version
python pipeline/slurm/patch_vllm_m3_serve.py            # apply if needed
grep -rl 'llmc M3' "$(python -c 'import vllm, pathlib; print(pathlib.Path(vllm.__file__).parent)')"
# confirm the router patch is now in router/base_router.py (NOT moe_runner.py):
grep -n 'nan_to_num router_logits in _select_experts' \
  "$(python -c 'import vllm, pathlib; print(pathlib.Path(vllm.__file__).parent)')/model_executor/layers/fused_moe/router/base_router.py"

# 1) get the REAL faulting kernel (decisive):
bash pipeline/slurm/debug_cudagraph_ima.sh 2>&1 | tee /mnt/nfs/hoangduy/logs/m3-cudagraph-debug.log
# 2) then a normal capture run:
ENFORCE_EAGER=0 bash pipeline/slurm/run_serve_minimax_m3_detached.sh
# success = 51/51 capture
```

**Status (2026-07-08):** **Root cause of the persistent 16/51 non-fix identified** —
old patch 4 sanitized ``MoERunner._apply_quant_method``, a **dead path** for the
W4A8 modular kernel; patch 3 (fused AR) addresses a cross-node-only failure not
present on single-node h118. **Fix applied:** router ``nan_to_num`` relocated to
``RouterBase._select_experts`` (``fused_moe/router/base_router.py``, the live routing
entry in vLLM 0.24.0), plus a flashinfer-version
guard (#42906) and a ``CUDA_LAUNCH_BLOCKING`` diagnostic to confirm the faulting
kernel. **Awaiting cluster verification** (needs h118). Patches 1–2 +
``ENFORCE_EAGER=1`` remain the known-good serve path meanwhile.

**Escape hatch:** ``ENFORCE_EAGER=1`` disables graphs entirely (confirmed working
2026-07-08). Use only while debugging if the fused-AR patch is insufficient
(e.g. separate W4A8 MoE padding bug — run with ``CUDA_LAUNCH_BLOCKING=1``).

**Long-term fix:** upstream vLLM ``breakable_cudagraph`` segments at
``fused_allreduce_gemma_rms_norm`` (compute–comm–compute), FlashInfer makes
the fused kernel graph-safe, and #39391-class ``topk_softmax`` NaN clamp ships in
the toncao/M3 serve build. **Removal criteria:** delete patches 3–4 once stock
vLLM serves M3 W4AFP8 with graphs and no IMA on h118.

## MiniMax-M3 vLLM serve hangs: FlashInfer fused all-reduce (`shm_broadcast` timeout)

**Symptom:** Model loads, KV profiling completes, then generation hangs with repeating:

```
[shm_broadcast.py:705] No available shared memory broadcast block found in 60 seconds.
```

Often preceded by:

```
[flashinfer_all_reduce.py:111] Auto-selected flashinfer allreduce backend: trtllm
```

(or `mnnvl` on multi-node). `enforce_eager=true` does **not** fix it — M3 calls
`fused_allreduce_gemma_rms_norm` directly in `model.py` forward, not via CUDA graphs
or inductor `fuse_allreduce_rms`. See [vLLM #45604](https://github.com/vllm-project/vllm/issues/45604),
[vLLM #45800](https://github.com/vllm-project/vllm/issues/45800).

**Root cause:** FlashInfer auto-selects a fused all-reduce + Gemma RMSNorm backend
incompatible with the deployment topology (e.g. cross-NUMA SYS links on 2×4 H100,
non-MNNVL multi-node). The rendezvous deadlocks; `shm_broadcast` timeouts are a
symptom, not the root cause.

**Optional escape hatch (partial for M3):** `disable_custom_all_reduce=True` disables
vLLM's **custom** all-reduce kernel only. It does **not** disable M3's FlashInfer
`fused_allreduce_gemma_rms_norm` forward path or `fuse_allreduce_rms` compile fusion
— logs may still show `Auto-selected flashinfer allreduce backend: trtllm`.

**Root-cause / long-term fix:** topology-aware FlashInfer backend selection in vLLM;
M3 model should fall back cleanly when `_can_use_flashinfer()` fails.

### Update (2026-07-08): the *decode* hang after the IMA fix IS the custom all-reduce deadlock

Once the CUDA-graph IMA was fixed (router `nan_to_num` + flashinfer 0.6.12), the run
progressed to generation and then hung with the same `shm_broadcast` timeout. Capturing
live worker stacks (`dump_hang_stacks.sh` → `SIGABRT` + `PYTHONFAULTHANDLER=1`, because
`py-spy` is blocked by the node's `ptrace_scope`) gave the decisive signal. **All 8
workers** were parked at the identical frame:

```
vocab_parallel_embedding.py:491   forward                  (input embedding)
communication_op.py:14            tensor_model_parallel_all_reduce
parallel_state.py:642/649         all_reduce / _all_reduce_out_place
cuda_communicator.py:289          all_reduce
custom_all_reduce.py:280          custom_all_reduce        <-- vLLM CUSTOM all-reduce
custom_all_reduce.py:259          all_reduce               <-- HUNG
  via minimax_m3/nvidia/model.py:955/798  embed_input_ids
  via gpu_model_runner.py:3497 _preprocess -> execute_model
```

This is **not** FlashInfer, **not** NCCL, **not** MoE. The hang is the **first TP
collective** of the forward — the vocab-embedding all-reduce — inside vLLM's custom
P2P/IPC all-reduce, which deadlocks on the h118 2×4 topology (`disable_custom_all_reduce=False`
in the engine config). Because the stack is literally in `custom_all_reduce.py`, the
`disable_custom_all_reduce` flag (previously dismissed for the *FlashInfer* hang) is now
the **correct, targeted fix**: it routes these collectives through NCCL.

**Fix (enabled by default for M3):** `serve.disable_custom_all_reduce: true` in
`pipeline/configs/minimax_m3.yaml`, and `--set serve.disable_custom_all_reduce=true` in
`debug_cudagraph_ima.sh`, `run_serve_minimax_m3_detached.sh`, `serve_minimax_m3.sbatch`.
For raw `vllm serve`: `--disable-custom-all-reduce`.

**Removal criteria:** drop the flag once the node has full P2P (NVLink/NVSwitch all-to-all)
or vLLM's custom all-reduce P2P capability check correctly disables itself on this topology.

**Validated (2026-07-08, h118):** with `disable_custom_all_reduce=true` and graphs on
(`enforce_eager=false`), serve-verify no longer stalls — generation completes at
~17.8 tok/s, `overall_ok=True`, `rc=0`. The `TCPStore ... Broken pipe` /
`recvValue failed` lines during shutdown are benign teardown noise after `SIGTERM`.

> Note: the smoke checkpoint (`MiniMax-M3-awq-W4AFP8/20260707-082218/…`) still emits
> degenerate text (`"arringarring…"` repetition). That is a **quantization-quality**
> artifact, not an infra bug — the serve path (W4A8 MoE, CUDA graphs, TP all-reduce)
> is healthy. **Update 2026-07-09:** the full-calibration checkpoint
> (`20260708-093642`) also emits the same `"arringarring…"` garbage under a known-good
> graphs-on serve. See **“MiniMax-M3 full-calib AWQ garbage output (quality ablation)”**
> below — FP8 activations have been ruled out; recipe / weight-scope work remains.

## MiniMax-M3 vLLM serve aborts: FlashInfer `gemma_rmsnorm` CuTe-DSL JIT (nanobind "Expected an MLIR object")

**Symptom:** After the W4A8-MoE and `hidden_act` fixes, the model loads all shards and reaches KV-cache profiling, then **all 8 workers die simultaneously** with a bare C++ abort (no Python traceback):

```
terminate called after throwing an instance of 'nanobind::builtin_exception'
  what():  Expected an MLIR object (got <cutlass._mlir._mlir_libs._cutlass_ir._mlir.ir.OpResultList object at 0x...>).
```

The engine then reports the misleading downstream `RuntimeError: cancelled` / `Engine core initialization failed` from `determine_available_memory`.

**Root cause (a FlashInfer × cutlass-dsl JIT gap — NOT flash attention, NOT the checkpoint):** `PYTHONFAULTHANDLER=1` reveals the real stack:

```
vllm/models/minimax_m3/nvidia/model.py  forward   (MiniMAXGemmaRMSNorm)
  -> flashinfer/norm/__init__.py  gemma_rmsnorm -> _gemma_rmsnorm_impl
    -> flashinfer/norm/kernels/rmsnorm.py  rmsnorm_cute -> _get_compiled_rmsnorm_kernel
      -> nvidia_cutlass_dsl/.../base_dsl/compiler.py  generate_mlir   <-- nanobind abort
(under gpu_worker.determine_available_memory -> profile_run -> _dummy_run)
```

M3 uses Gemma-style RMSNorm everywhere (`MiniMAXGemmaRMSNorm.forward` hardcodes `from flashinfer.norm import gemma_rmsnorm, gemma_fused_add_rmsnorm`). This build routes that to FlashInfer's **CuTe-DSL** kernel (`rmsnorm_cute`), which fails to JIT-compile against the pinned `nvidia-cutlass-dsl==4.5.2` on Hopper (SM90). This is the same class of failure as [vllm #45392](https://github.com/vllm-project/vllm/issues/45392) (a CuTe-DSL JIT abort validated-fixed only on Blackwell), but the aborting kernel here is FlashInfer RMSNorm, not FA4 — the flash-attention path is FA3 the whole time (`head_dim=128`, no FA3->FA4 upgrade; ViT `head_dim=80`).

**Fix (preferred — FlashInfer's own documented fallback, no source patch):** set `FLASHINFER_USE_CUDA_NORM=1` so FlashInfer uses its **CUDA-JIT** norm kernels instead of CuTe-DSL (`flashinfer/norm/__init__.py` reads it at import time; the same flag is auto-enabled when cutlass-dsl is absent/incompatible). Numerically identical Gemma RMSNorm, just a different backend.

Wired in three places so every launch path is covered:
- `pipeline/serve_verify.py` — `os.environ.setdefault("FLASHINFER_USE_CUDA_NORM", "1")` at import (before vLLM/FlashInfer import); covers `python -m pipeline.run --stage serve`.
- `pipeline/slurm/serve_minimax_m3.sbatch` and `run_serve_minimax_m3_detached.sh` — `export FLASHINFER_USE_CUDA_NORM=1` (plus `PYTHONFAULTHANDLER=1` so future kernel aborts self-diagnose).

For a raw `vllm serve`, export `FLASHINFER_USE_CUDA_NORM=1` before launch.

**Root-cause / long-term fix:** upgrade FlashInfer / `nvidia-cutlass-dsl` to a combination whose CuTe-DSL RMSNorm JIT-compiles on Hopper, or an M3 model rev that doesn't hard-depend on FlashInfer's cute norm. **Removal criteria:** drop the env toggle once FlashInfer's CuTe-DSL RMSNorm compiles cleanly with the pinned DSL on SM90 (re-test by unsetting `FLASHINFER_USE_CUDA_NORM` and serving).

## MiniMax-M3 OOM during `linearize_moe` (fp32 expert materialization)

**Symptom:** Quantization (GPTQ/AWQ) dies around MoE layer 45–47 of 57 on 2 TB RAM nodes. Process RSS ~1980 GiB at `linearize_moe` start; `mem_avail` drops ~27 GiB per layer with no recovery. OS OOM-kill, no Python traceback.

**Root cause:** MiniMax-M3 `config.json` sets `torch_dtype: bfloat16` only at the top level; `text_config` / `vision_config` have no `dtype`. `coerce_minimax_m3_vl_config()` rebuilds `MiniMaxM3VLTextConfig` from a dict without dtype, so sub-configs default to `float32`. During post-load `linearize_moe`, `MoEConfig.from_config()` reads `text_config.dtype` and `LinearExperts2D.from_experts_module()` allocates new `nn.Linear` experts in fp32 (~2× memory vs bf16). The 428B backbone plus per-layer fp32 copies exceed ~2 TB.

**Long-term fix (preferred):** Register a 2D load conversion mapping for `minimax_m3_vl` in `src/llmcompressor/modeling/moe/conversion_mappings.py` so weights load directly in linearized 2D form (avoids 3D load + post-load conversion peak). See [add-moe-support.md](docs/developer-tutorials/add-moe-support.md).

**Fix applied (2026-07-06):**

1. `pipeline/minimax_m3_config.py` — propagate top-level `dtype` (fallback `torch.bfloat16`) to `text_config` and `vision_config` after coercion.
2. `pipeline/configs/minimax_m3.yaml` — set `model.dtype: bfloat16` explicitly for `from_pretrained`.
3. `pipeline/quantize.py` — log `text_config.dtype` and a sample expert weight dtype after load.

**Removal criteria:** None; dtype propagation is the correct contract for VL configs with nested sub-configs.

**Verify on cluster:**

```bash
source /mnt/nfs/hoangduy/env.sh
source /mnt/nfs/hoangduy/venvs/quant/bin/activate
cd /mnt/nfs/hoangduy/projects/llm-compressor

python -c "
from pipeline.minimax_m3_config import load_minimax_m3_vl_config
c = load_minimax_m3_vl_config('/mnt/nfs/hoangduy/hf_assets/MiniMaxAI/MiniMax-M3')
print(c.dtype, c.text_config.dtype, c.vision_config.dtype)
"
# Expect: torch.bfloat16 for all three

# Re-run quantize (detached script activates venv automatically):
# METHOD=awq bash pipeline/slurm/run_quantize_minimax_m3_detached.sh

grep -E 'backbone dtype|linearize_moe start' /mnt/nfs/hoangduy/logs/quantize-m3-*.log
# Expect: text_config.dtype=torch.bfloat16; rss ~600–850 GiB (not ~1980)
```

**Related upstream issues:** [llm-compressor #2669](https://github.com/vllm-project/llm-compressor/issues/2669) (progressive MoE replacement OOM on other models; different code path but similar symptom).

## MiniMax-M3 AWQ full calibration OOM during `pin_memory` (VMA exhaustion)

**Symptom:** Full-calibration AWQ (`pipeline/configs/minimax_m3_full_calib.yaml`, 512 samples × 2048 seq) dies during sequential calibration, typically layer 5/61 around sample 451/512, with:

`torch.AcceleratorError: CUDA error: out of memory` at `IntermediatesCache._offload_value` → `offloaded.pin_memory()` (`cache.py:331`), inside `AWQModifier.cache_parent_kwargs_hook` while caching MoE expert `down_proj` parent inputs. Host RSS ~1960 GiB with hundreds of GiB `mem_avail` — not a GPU VRAM or ordinary RAM shortage.

**Root cause:** Upstream prefetch PR [#2392](https://github.com/vllm-project/llm-compressor/pull/2392) added unconditional `pin_memory()` in `IntermediatesCache._offload_value()` when offloading to CPU. Each `pin_memory()` → `cudaMallocHost()` → `mmap()` consumes one process VMA. AWQ caches every parent-module input per calibration sample; with `moe_calibrate_all_experts: true` on MiniMax-M3 (57 routed experts × many mappings × 512 samples), VMA count exceeds the kernel per-process `vm.max_map_count` default (65536). `mmap` returns ENOMEM; the CUDA driver surfaces it as `CUDA_ERROR_OUT_OF_MEMORY`. Confirmed upstream in [llm-compressor #2790](https://github.com/vllm-project/llm-compressor/issues/2790).

**Long-term fix (preferred):** Upgrade to an llm-compressor release containing [PR #2813](https://github.com/vllm-project/llm-compressor/pull/2813) once merged. Upstream's final form removes offload-path pinning and may re-pin per-parent only immediately before AWQ smoothing (grid-search reuse).

**Fix applied (2026-07-08):** Remove the `pin_memory()` block from `_offload_value()` in `src/llmcompressor/pipelines/cache.py` (equivalent to PR #2813 HEAD, "drop pinning entirely"). `_onload_value` / `iter_prefetch` fall back to blocking H2D copies (`is_pinned()` is false). Trade-off: <5% calibration throughput on small models (upstream benchmark); **no numerical impact** on AWQ scales or quantized weights.

**Removal criteria:** Drop this local patch once the quant venv upgrades to an llm-compressor release that includes #2813 (or equivalent upstream fix).

**Optional non-code mitigation (node admin):** `sudo sysctl -w vm.max_map_count=1048576` raises the VMA ceiling; masks the bug, does not persist across reboot without `/etc/sysctl.d/`. Not a substitute for the code fix.

**Verify on cluster:**

```bash
source /mnt/nfs/hoangduy/env.sh
source /mnt/nfs/hoangduy/venvs/quant/bin/activate
cd /mnt/nfs/hoangduy/projects/llm-compressor

# Confirm patch is present (no pin_memory in offload path):
grep -n pin_memory src/llmcompressor/pipelines/cache.py
# Expect: only iter_prefetch / _onload_value comments or is_pinned checks — NOT pin_memory() in _offload_value

# Unit tests (fast):
python -m pytest tests/llmcompressor/pipelines/test_cache.py -q

# Re-run full calibration (detached):
CONFIG=pipeline/configs/minimax_m3_full_calib.yaml METHOD=awq \
  bash pipeline/slurm/run_quantize_minimax_m3_full_calib.sh

# Expect: passes layer 5/61 calibration; no pin_memory OOM in log
grep -E 'pin_memory|CUDA error: out of memory|\(5/61\): Calibrating' \
  /mnt/nfs/hoangduy/logs/quantize-m3-full-*.log
```

**Related upstream issues:** [#2790](https://github.com/vllm-project/llm-compressor/issues/2790), [#2813](https://github.com/vllm-project/llm-compressor/pull/2813).

**Validated (2026-07-09):** after the `pin_memory` removal (`e3dff4ed` on `duy-branch`), full calibration completed on h118:

- Compressing model: **22116/22116** modules (matches expected sparse-layer Linear count)
- Checkpoint: `artifacts/MiniMax-M3-awq-W4AFP8/20260708-093642/checkpoint`
- Config patches on save: vision `img_token_compression_config` restored; `hidden_act` forced `swigluoai`; recipe `ignore` patterns persisted

## MiniMax-M3 DDP full-calib AWQ: weights-side VMA exhaustion (2026-07-17)

**Symptom:** Distributed full-calibration AWQ (`minimax_m3_distributed_awq_full.yaml`,
8-rank torchrun, run `20260717T064357Z-m3-ddp-awq-full-r1`, job 12987 on h97) dies with
no checkpoint after 3h40m. Rank 0 logs the compressed-tensors remediation warning
(`CPU offloading ran out of host RAM or mmap descriptors`) at the 40-minute mark, right
after model load, during AWQ `on_initialize`'s 355 offset-norm conversions. Ranks 1–7
then sit in a NCCL `BROADCAST` (NumelIn=3, the offload handle exchange) for exactly the
3h PG timeout with GPUs at 100% (NCCL spin-wait, not work), the rank-1 watchdog dies
with `MemoryError: <EMPTY MESSAGE>` → SIGABRT, torchrun SIGTERMs the rest.
`quant_metrics.*.jsonl` all 0 bytes — calibration never started.

**Not a capacity problem:** at failure MemAvailable = 1168 GB and `/dev/shm` held
805 GB of 1082 GB (277 GB headroom).

**Root cause (weights-side sibling of the `pin_memory` VMA incident above):**
`DistributedCPUCache.offload()` creates **one `/dev/shm` file per tensor**
(`_share_filename_cpu_`) and every rank mmaps all of them. The dead run left
**63,122 `torch_*` segments** on h97 (post-mortem census) against the kernel default
`vm.max_map_count = 65530` (verified identical on h97 and h103). Note the checkpoint
index has only 23,416 tensors — MoE linearization expands the offloaded module tree
~2.7×, which is why the VMA gate estimates with a 3× factor. 63k weight maps +
CUDA/allocator/library maps ≈ the cap; rank 0 (source rank, most maps) hit ENOMEM
first on the +355 norm-conversion offloads. The r9 smoke passed the same load on h103
with a few hundred maps to spare — luck at the cliff edge, not a safer node; a full
run's calibration activation cache (each >128 KiB cpu tensor is its own mmap) would
have pushed it over even if load had survived. Matches the well-documented failure
class: [pytorch#60626](https://github.com/pytorch/pytorch/issues/60626) (file-backed
shared tensors vs `max_map_count`); Red Hat
[max_map_count](https://access.redhat.com/solutions/99913).

**Fix applied (2026-07-17):** stop putting the bulk of the weights in per-tensor shm
segments; use compressed-tensors' first-class disk offload instead (upstream-supported
for DDP since [llm-compressor v0.10](https://github.com/vllm-project/llm-compressor/releases/tag/0.10.0)):

1. `minimax_m3_distributed_awq_full.yaml`: `model.max_memory.cpu` 1e12 → **32e9**, so
   `device_map=auto_offload` overflows the weights to `DistributedDiskCache` (reads
   layers straight from the checkpoint safetensors via page cache; writes only updated
   tensors to `offload_folder`). VMA demand drops from ~63k to ~1k.
2. Fail-closed gate `assert_vma_budget_for_shared_offload` in `pipeline/quantize.py`:
   before load, in distributed `auto_offload` mode, estimates planned shm segments from
   the checkpoint index vs `/proc/sys/vm/max_map_count` and refuses doomed plans before
   GPU spend (`M3_SKIP_VMA_GUARD=1` to bypass).
3. Node preflight (`run_m3_distributed_quant_smoke_srun.sh`) records
   `vm_max_map_count` as evidence.

**Cleanup:** the dead run's 63,122 stale shm files (801 GB) were removed from h97 on
2026-07-17 (the launcher preflight also removes unmapped `torch_*` files on the next
run).

**Fleet-level alternative (admin request, parallel track):** raise
`vm.max_map_count` to ≥ 1048576 in `/etc/sysctl.d/` on the H100 nodes (standard
practice for map-hungry workloads; no performance downside per Red Hat). That restores
the faster all-shm mode with ~16× headroom; until then the disk-offload config is the
safe default.

**Removal criteria:** none for the gate (cheap, correct); the 32e9 cpu budget can be
raised back toward shm-resident once nodes run with a raised `max_map_count`.

### Follow-on: `DistributedDiskCache.update_offload` write race (smoke r10, 2026-07-18)

The disk-offload smoke (`20260718T044923Z-m3-ddp-quant-smoke-r10-diskoffload`) validated
the VMA fix — gate OK on all ranks, dispatch 30,882 modules in 40 s (vs ~23 min to shm),
Shmem flat at ~55 GB, grid search ran with synchronized stats (identical errors on all
ranks) — then died in `_apply_smoothing` → `update_offload_parameter` with
`FileNotFoundError` on a shared `ct_disk_cache_*.safetensors`.

**Root cause (upstream bug, unfixed as of compressed-tensors 0.17.2a20260707 and main
2026-07-18):** `DistributedDiskCache` overrides `offload`/`__delitem__` with source-rank
gating but inherits the non-distributed `DiskCache.update_offload`, so every rank
concurrently `os.unlink`s the shared symlink and rewrites the same file. Racy crash at
best; silent concurrent-write corruption at worst.

**Fix applied (2026-07-18):** runtime patch
`install_distributed_disk_update_offload_patch` in `pipeline/quantize.py` (installed for
distributed runs before load): source rank performs `DiskCache.update_offload`, all
ranks barrier (writers-before-readers). Mirrors upstream's own `offload()` sync design;
correct because smoothed data is identical across ranks. The patch self-disables once
upstream defines `update_offload` on `DistributedDiskCache` — drop it then. Worth
filing upstream.

### Follow-on: offloaded-save tie detection crashes on buffers (smoke r11, 2026-07-18)

Smoke r11 (`20260718T052330Z-m3-ddp-quant-smoke-r11-diskoffload`, job 13008) got past
r10's race — patch installed, VMA gate OK, calibrate + smooth + `Compressing model`
(144/144 per rank, 25 s) all clean, quant metrics captured — then produced a
**3-hour false "slow save"**: stdout/err silent from 05:55:31, rank 0 at 0 % GPU,
ranks 1–7 pinned at 100 %, zero shards, until the NCCL watchdog killed everything at
08:55:36 (BROADCAST SeqNum=108879, `Timeout(ms)=10800000`).

**Root cause (upstream transformers 5.12.1, fixed on upstream main):** with disk
offload, `save_pretrained`'s state dict holds **meta tensors** (that is by design —
data lives in the disk-cache index). `remove_tied_weights_from_state_dict`
(`modeling_utils.py:474`) resolves each meta entry with `model.get_parameter(name)`,
which **raises `AttributeError` for registered buffers** — first hit: M3's router
buffer `e_score_correction_bias`. Rank 0 crashed ~1 s into the save; the exception
unwound into a blocking PG teardown while ranks 1–7 waited in the collective-save
BROADCAST, so the whole job hung the full 3 h watchdog window doing nothing
(`MemAvailable` flat at ~1974 GB throughout — the proof no gather was running).
r9-save never hit this because shm offload yields real CPU tensors, not meta.

**Fixes applied (2026-07-18):**
- `_tied_weights_meta_buffer_compat` (`pipeline/quantize.py`): backports upstream
  main's own fix (`get_parameter` → `get_parameter_or_buffer` semantics) as an
  instance-scoped shadow during `save_pretrained`; self-disables once the installed
  transformers carries the fix. Repro test pins the 5.12.1 failure mode.
- `_save_heartbeat` + `prewarm_offload_page_cache` (`pipeline/quantize.py`): 60 s
  save-phase progress lines (shards/GB written + `/proc/self/io` read-back GB) and a
  16-thread page-cache prefetch of the offload files. A heartbeat showing "0 GB read"
  would have exposed r11's dead rank in minutes instead of hours.
- `_DISTRIBUTED_TIMEOUT` 3 h → 8 h (`pipeline/distributed.py`): the save-phase
  collective wait must survive a genuinely long serial gather on the full model.

**Diagnostic lesson:** "rank 0 at 0 % GPU during save" is consistent with BOTH the
documented healthy save pattern AND a crashed-then-hung rank. Distinguish them with
`MemAvailable`/page-cache trend and file mtimes, not GPU utilization.

### Follow-on: offloaded-save revert renames meta tensors before materialization (smoke r12, 2026-07-18)

Smoke r12 (`20260718T093241Z-m3-ddp-quant-smoke-r12-diskoffload`, job 13013) proved the
r11 tie-detection fix (save got past `remove_tied_weights_from_state_dict`, computed the
17-shard split, printed `Writing model shards 0/17`, prewarm streamed 898 GB at
11.3 GB/s) — then froze ~90 s into the save with the exact r11 zombie signature:
one heartbeat, zero shards, `MemAvailable` dead flat, rank 0 at 0 % GPU, ranks 1–7
NCCL-spinning at 100 %. Killed after 3 h (traceback lost to SIGKILL, as predicted).

**Root cause (upstream transformers 5.12.1, fixed on upstream main):**
`save_pretrained` runs `revert_weight_conversion` on the **whole state dict before the
shard loop** (`modeling_utils.py:3511`). With disk offload the entries are meta, so
after the revert every offloaded tensor carries its **checkpoint-format name**
(M3: `language_model.model.*.block_sparse_moe.*`) while
`load_offloaded_parameter` (`integrations/accelerate.py:516`) resolves names against
the **runtime module tree** (`model.language_model.*.mlp.*`) → the first offloaded
tensor raises and the save dies seconds in. `WeightConverter` entries (M3 dense-layer
`mlp.gate_up_proj` split, shared experts) are worse: reverting on meta chunk/concats
into brand-new meta tensors nothing can materialize. Upstream main fixed it by
skipping the early revert when offloaded and reverting **per shard after
materialization** (`modeling_utils.py` ~3649–3676 on main). r9's shm save never hit
this: real CPU tensors never take the `load_offloaded_parameter` branch.

**Why 3 h of silence instead of a traceback (r11 + r12 shared mechanism):** rank 0's
exception exits `suspend_distributed_timeout` (all ranks clear its barriers), then
ranks 1–7 proceed into `from_accelerate`'s `broadcast_object_list([device_map,
offload_dir])` — the parked `BROADCAST NumelIn=2` from r11's watchdog dump — which the
unwinding rank 0 never joins. Rank 0 blocks in PG teardown, the traceback only flushes
if the process dies by signal-able means; SIGKILL (walltime/scancel escalation) eats it.

**Fixes applied (2026-07-18):**
- `_deferred_weight_conversion_compat` (`pipeline/quantize.py`): backports upstream's
  per-shard revert — `revert_weight_conversion` becomes a passthrough when the dict
  holds meta tensors; `safe_save_file` applies the real revert to each fully
  materialized shard. `rebuild_safetensors_index` then rewrites
  `model.safetensors.index.json` from the actual shard headers (the index
  `save_pretrained` wrote maps pre-revert runtime names). Self-disables once the
  installed transformers does per-shard reverts. Repro test pins the failure with a
  real accelerate-offloaded tiny model (rename + converter split), and verifies the
  shimmed save is byte-identical to the non-offloaded baseline.
- `coordinate_collective_save` (`src/llmcompressor/.../compressed_tensors_utils.py`):
  replaces the bare `suspend_distributed_timeout` wait in `save_pretrained_wrapper`.
  The source rank catches its save exception and broadcasts the outcome over a
  dedicated gloo group (CPU wait, no GPU spin) before anyone proceeds; on failure all
  ranks raise within seconds with the real error. Also raises the save-wait ceiling
  3 h → 9 h (`_SAVE_WAIT_TIMEOUT`) — the old hardcoded 3 h gloo timeout would have
  killed any legitimate >3 h full-calib save regardless of the 8 h PG timeout.
  2-rank gloo test proves failure propagation.

**Diagnostic lesson:** a silent save-phase heartbeat is itself a signal — the
heartbeat context exits when `save_pretrained` raises, so "heartbeat stopped, no
shards, flat memory, non-source GPUs spinning" means *the save already crashed and
the job is a zombie*, not that the save is slow. Don't wait for the watchdog: the
traceback will not appear (SIGKILL), and the failure choreography above explains
every observable.

### Transformers 5.14.1 upgrade assessment (2026-07-18, after r12)

Evaluated replacing the two save shims with an upgrade. Verdict: **feasible, low
effort, recommended as a follow-up after the current smoke-gated run** — not
mid-flight. Evidence (all verified, not inferred):

- Installed 5.12.1 is **stock PyPI** (every file matches the wheel RECORD hashes;
  `conversion_mapping.py` byte-identical to the upstream v5.12.1 tag). M3-VL
  support is genuinely upstream, not a local patch.
- Target must be **5.14.x**: tie-detection fix shipped in v5.13.1, but the
  per-shard revert fix (upstream PR #47018 — exactly the r12 bug) only in v5.14.0.
- M3-critical surface stable: `minimax_m3_vl` conversion-mapping entry is
  **identical** 5.12.1 → 5.14.1; `MiniMaxM3VLExperts` intact, so
  `modeling/moe/conversion_mappings.py` linearization is unaffected.
- Full 572-test suite in a trial venv (`/mnt/nfs/hoangduy/venvs/quant-tf514-trial`,
  transformers==5.14.1): **567 pass**; all 5 failures understood — 4 are the
  crash-pin tests that assert the 5.12.1 bugs exist (gate with the same
  `fixed_upstream` source check the shims use), 1 is
  `GraniteMoeParallelExperts` → `GraniteMoeExperts` rename
  (`src/llmcompressor/modeling/moe/granitemoe.py:4`, one line, not on the M3 path).
- Both shims (`_tied_weights_meta_buffer_compat`,
  `_deferred_weight_conversion_compat`) verified to **self-disable** under 5.14.1.
- No dependency blockers: vllm 0.24.0 needs only `transformers>=5.5.3`;
  compressed-tensors `>=4.45.0`; serve/eval venvs are separate. The
  `<=5.12.1` cap in `setup.py` is upstream llm-compressor's release pin (their
  dev builds accept `>=5.9.0`) — bump ours.
- **Real cost is revalidation, not code**: M3 forward numerics changed
  (v5.13.0 "EP router contract corrected"; sparse-attention indexer moved from
  per-query to per-head block selection in `modeling_minimax_m3_vl.py`).
  Calibration statistics will shift, so pre/post-upgrade checkpoints are not
  comparable — rerun the disk-offload smoke + scale audit after upgrading
  (~2-3h GPU), same gate as any save-path change.

Estimated effort: ~1-2h local (venv bump, granite rename, gate 4 pin tests,
setup.py pin, commit) + one smoke-gate cluster run.

**Executed 2026-07-18** (user-approved, after smoke r13 passed on 5.12.1+shims):
target 5.14.1 (= PyPI latest, released 2026-07-16). Local changes:
- `granitemoe.py`: guarded import (5.14 renamed/fused the experts class);
  `from_experts_module` fails fast with `NotImplementedError` on the fused
  layout until granite linearization is ported (outside M3 scope).
- Crash-pin tests made version-adaptive via the same `fixed_upstream` source
  checks the shims use (`test_tied_weights_meta_buffer_compat`,
  `test_deferred_weight_conversion` assert `deferred is (not fixed)`).
- `tests/llmcompressor/modeling/test_linearize.py`: 5.14 removed the
  `*NaiveMoe` fixture classes (deepseek_v3, glm4_moe, glm4_moe_lite,
  glm_moe_dsa) and `GraniteMoeParallelExperts`; imports guarded and the
  affected params/tests skip with explicit reasons until ported. Note:
  `test_linearize_moe[DeepseekV4…]` and `[HYV3…]` already failed on clean HEAD
  under 5.12.1 (pre-existing, GPU-marked; not part of the standard suite).
- `setup.py` release pin bumped to `<=5.14.1`.
Both shims verified self-disabling; r13's smoke checkpoint passed the serving
ABI gate (valid, 0 errors, 1152 quantized routed-expert Linears) — which also
exercised the rebuilt safetensors index end to end.

**NEW upstream bug found during the upgrade (caught by our repro test, zero GPU
spent):** 5.14's own per-shard-revert fix ships a broken bookkeeping line in
`save_pretrained` (5.14.1 `modeling_utils.py:3675`, still present on upstream
main as of 2026-07-18):

```python
weight_map.update({k: os.path.basename(shard_file)} for k in shard_state_dict.keys())
```

`dict.update` over a generator of 1-element dicts raises
`ValueError: dictionary update sequence element #0 has length 1`, which the
surrounding broad `except Exception` re-raises as the misleading "unlucky
sharding" RuntimeError. The branch runs for **every sharded + offloaded +
original-format save** — exactly the M3 production path — so stock 5.14.1
crashes at the end of shard 1. A naive upgrade would have burned another
smoke run discovering this on cluster.

**Resolution:** one-line venv hotfix (dict comprehension instead of the
generator, marked `# llm-compressor hotfix`), plus a fail-closed preflight —
`assert_transformers_offloaded_save_healthy()` in `pipeline/quantize.py`,
wired next to the VMA gate — that classifies the installed save path as
`shimmed` (pre-5.14: our save shims own it) / `healthy` (5.14 + hotfix) /
`broken` (5.14 stock → refuse to start). This catches a venv rebuild that
silently drops the hotfix. With the hotfix, the previously-failing
`test_linearize_offload` tests also pass — they were the same bug.
Recommend filing the one-liner upstream (huggingface/transformers; bug is in
the #47018 follow-up code).

**Final validation (2026-07-18):** standard suite (pipeline/tests +
test_save_coordination + test_compress_tensor_utils) green in both venvs with
the version-adaptive tests: 5.12.1 quant venv **575 passed**; hotfixed 5.14.1
trial venv **574 passed, 1 skipped** (the skip is the shim-restoration test,
skipif'd where upstream is fixed). One earlier trial-run failure did not
reproduce across a full rerun (transient; name lost to an output filter —
lesson: never pipe a suite's only output through `tail`/`grep`, keep the full
log). Quant venv swapped to 5.14.1 + hotfix via
`upgrade_quant_venv_tf5141.sh` after r13 completed, gated by
`assert_transformers_offloaded_save_healthy()`; smoke r14 re-gates the save
path under 5.14.1 before the full 512-sample calibration.

**Smoke r14 (2026-07-18, job 13016-adjacent, run
`20260718T150509Z-m3-ddp-quant-smoke-r14-tf514`): PASSED.** First save through
transformers 5.14.1's native per-shard revert + the weight_map hotfix (shims
correctly self-disabled; gate printed "healthy" on all 8 ranks). 17 shards /
821.8 GB in ~12.5 min — same throughput as r13's shimmed path. Serving ABI
gate: `valid: true`, 0 errors, 1152 quantized routed-expert Linears (exact
match with r13). Full-calibration AWQ relaunched immediately after as
`20260718T160612Z-m3-ddp-awq-full-r2-tf514` (job 13016; prior full r1 of
2026-07-17 predated the VMA + save fixes and failed).

**Cross-goal conflict check (2026-07-18, per PROJECT_GOALS.md):**

- *Goal 1 (fast parallel quant)* — direct target; covered above. The r-series
  save-path validation history is on 5.12.1, so the upgrade resets that gate:
  one extra smoke run, which is already the standard gate for any save change.
- *Goal 2 (eval pipeline)* — **no invalidation of existing evidence.** All eval
  arms execute in the `serve` venv (vLLM 0.24 + transformers 5.12.1) and
  `sglang-eval` (5.8.1), which the quant-venv upgrade does not touch; completed
  results (BF16, MXFP8, paired r4, GPTQ replay) are immutable files. Two real
  interactions, both bounded:
  1. **Provenance confound for paired method claims**: a future in-house AWQ
     checkpoint quantized under 5.14.1 vs the existing in-house GPTQ checkpoint
     (2026-07-12, 5.12.1-era) differ in quant-time forward numerics (per-head
     indexer, corrected router contract). Method-vs-method claims should either
     requantize the GPTQ arm under the same version (one 4–8h distributed run —
     exactly what goal 1 makes cheap) or scope claims as
     checkpoint-vs-checkpoint with recorded provenance (the harness contract
     already mandates recording). Effort: bookkeeping now, optional one quant
     run later.
  2. **Serving compat of 5.14-written checkpoints under serve-venv 5.12.1**:
     verified low-risk — the M3 config-class diff 5.12.1→5.14.1 is a class
     constant only (`base_model_ep_plan`, never serialized to config.json);
     `PretrainedConfig` tolerates unknown keys; the audit tools
     (`verify_quant_checkpoint.py`, `m3_checkpoint_scale_audit.py`) read shards
     via `safe_open` with no transformers dependency. Residual risk is covered
     by the TP8 serving smoke already in the post-quant gate chain. Effort: 0.
  New checkpoints' tokenizer/chat-template hashes may differ under 5.14
  serialization — the fail-closed harness records and verifies hashes per run,
  so this is expected drift, not a conflict.
- *Goal 3 (working AWQ)* — same path as goal 1; finish the current 5.12.1+shims
  smoke first, then upgrade.
- *Goals 4/5 (generalization)* — net positive: 5.14 carries the corrected
  EP/router contracts and both save fixes upstream, shrinking our shim surface;
  the granite rename fix is goal-4 surface anyway.
- **Do not upgrade the `serve` venv as part of this** — vLLM 0.24 pins a
  compressed-tensors prerelease and its transformers floor is 5.5.3; nothing in
  the quant-venv upgrade requires touching it.

## AWQ smoothing fold lost under disk offload (full r2 post-mortem, 2026-07-19)

**Symptom:** the first full-calibration AWQ run with disk offload
(`20260718T160612Z-m3-ddp-awq-full-r2-tf514`, ~6.8h calibration, 5 shards /
240.9 GB, controller rc=0, serving-ABI gate valid with 21,888 quantized
Linears) failed the checkpoint scale audit: on layers 3/31/59 the router and
shared-expert weights carry a per-input-channel smoothing multiply (verified
rank-1 column fit, residual ≈ 2e-3, scale range 0.75–1.34) while the
post-attention norms are **byte-identical to base** — norm-implied scale
exactly 1.0, router relative-L2 0.09–0.27. The checkpoint is numerically
inconsistent: expert inputs are effectively scaled twice. r13 (5.12.1+shims)
and r14 (5.14.1 native) smoke checkpoints show byte-identical signatures, so
the transformers upgrade is exonerated; r9 (2026-07-17, CPU placement, no
disk offload) shows the consistent folded signature (norm delta 0.46–0.89,
compensation residual ≈ 3e-3). The regression entered with disk offload
(r10+, `max_memory cpu=32GB` + `offload_folder`).

**Root cause:** `CalibrationOffsetNorm.restore`
(`src/llmcompressor/modeling/offset_norm.py`) wrote the folded norm back with
a raw `original.weight.data = ...` assignment. With compressed-tensors
offload, `module._parameters` is an `OffloadCache`: attribute reads onload a
fresh tensor from offloaded storage, so raw `.data` writes mutate a temporary
view and never reach the disk copy that `save_pretrained` reads. The AWQ
balance path writes via `update_offload_parameter` (cache write-through),
which is why the router/expert multiplies persisted while the norm divide
vanished — precisely the inconsistent half-fold.

**Fixes:**
1. `CalibrationOffsetNorm.restore` now writes through
   `update_offload_parameter` (one-line mechanism change).
2. Regression tests `tests/llmcompressor/modeling/test_offset_norm_offload.py`:
   fold must survive a disk-offloaded norm end to end (verified to fail on the
   old code), plus an environment pin that raw `.data` writes do not persist
   (if that ever changes upstream, restore can be simplified).
3. **Fail-closed post-save gate** `assert_smooth_fold_consistency` in
   `pipeline/quantize.py`, wired on the source rank after the post-save
   barrier (so a gate failure cannot strand other ranks in a collective): it
   re-derives the norm-implied scale from the saved checkpoint and requires it
   to explain the router and shared-expert changes (threshold 0.02 vs ≈3e-3
   for a consistent fold and 0.09–0.27 for the bug). Validated to pass on r9
   and fail on r2. The static serving-ABI gate cannot catch numerics drift —
   this closes that gap for smoothing folds.

**Disposition of affected checkpoints:** r13/r14 smoke checkpoints were
save-path gates only (no quality claims). The full r2 checkpoint must not be
served or evaluated; rerun full calibration after the fix (smoke r15 first —
standard gate for any calibration-path change).

**Validation (2026-07-19):** smoke r15
(`20260719T032045Z-m3-ddp-quant-smoke-r15-foldfix`) PASSED end to end with the
fix: controller/torchrun rc=0, in-line `smooth-fold gate OK` on layers
3/31/59, and the independent scale audit shows the consistent folded
signature — router/shared compensation relative-L2 2.5e-3–4.1e-3 with
norm-implied scale means 0.77–0.91 (vs exactly 1.0 and 0.09–0.27 for broken
r2; matches r9's ≈3e-3 reference). Serving-ABI gate valid, 1,152 quantized
modules. Full calibration relaunched as r3
(`20260719T040748Z-m3-ddp-awq-full-r3-foldfix`, slurm 13027, 8×H100); the
smooth-fold gate now runs in-line after every save, so a recurrence fails the
run instead of producing another silently broken checkpoint.

**Full r3 PASSED (2026-07-19):** ~8h wall clock end to end, controller and
torchrun rc=0, 5 shards / 225 GB. In-line gates: offloaded-save OK,
`smooth-fold gate OK` on layers 3/31/59. Independent 5-layer audit (3, 15,
31, 45, 59) shows the consistent folded signature everywhere:
router/shared compensation relative-L2 2.3e-3–4.1e-3, norm-implied scale
means 0.73–0.91 (audit JSON at the run root:
`scale_audit_r3_vs_base.json`). Serving-ABI gate valid with all 21,888
quantized Linears. r3 replaces quarantined r2 as the in-house full AWQ
checkpoint; TP8 HTTP serving smoke launched next
(`20260719T121905Z-m3-awq-full-r3-serve-smoke`, single
`async_baseline_1` production-envelope case via the cudagraph matrix
harness).

**Serving-smoke lesson (2026-07-19): raw checkpoints must be re-exported
before vLLM serving.** The first r3 smoke served the RAW checkpoint and
reproduced the classic `"arringarring…"` collapse (server healthy, graphs
captured 51/51, HTTP 200) — because raw transformers exports name routed
experts `gate_proj/down_proj/up_proj` while the vLLM M3 loader expects
`w1/w2/w3`, so all 21,888 quantized expert tensors silently miss at load
(the 2026-07-09 wiring class, NOT quant quality; calibration data cannot fix
it). The serve-ABI gate passes raw checkpoints **by design** — its
`transformers_alias` treats `w1↔gate_proj` as equivalent because the rename
is the prepare step's job (`python -m pipeline.reexport_minimax_m3_vllm SRC
DST`, header-only rename + byte-identical payloads, ~6 min for 225 GB, no
re-quant — same transform that produced
`artifacts/m3-awq-gptq-prepared/gptq-checkpoint-vllm-w123`). r3 portable
re-export verified (5 shards, 65,664 routed keys renamed) at
`…/20260719-040810/checkpoint-vllm-w123`; smoke relaunched against it
(`20260719T130731Z`). Also: the matrix harness classifier calls bare
`python` — srun controllers must export
`PATH=/mnt/nfs/hoangduy/venvs/quant/bin:$PATH` or the case dies rc=127
after chat succeeds.

## Distributed qparam broadcast lost under disk offload (full r3 post-mortem, 2026-07-19)

**Symptom:** full r3 (`20260719T040748Z-m3-ddp-awq-full-r3-foldfix`) passed
every gate then live — offloaded-save, smooth-fold, serving ABI (21,888
modules) — and its portable re-export served TP8 with graphs captured and
HTTP 200, but chat returned 64 NUL characters (` …`, NaN logits →
argmax token 0). `pipeline.verify_quant_checkpoint --check-tensors` found
**non-finite `weight_scale` in 18/20 sampled modules**, and the bad tensors
are uninitialized junk (±2.8e38 near-bf16-max rows, NaN/inf) with
`weight_packed` frozen at constant `0x88888888` (weights quantized against
garbage scales). The same-stack A/B — in-house GPTQ portable
(single-process-era quant) through the identical harness — answered "The
capital of France is **Paris**.", clearing the serve stack entirely.

**Bisection:** r9 (pure CPU placement) PASSES the tensor check; r2, r15, r3
(disk offload) all FAIL with the same 18/20 pattern. 18/20 ≈ 7/8 = the
fraction of modules each rank does NOT own in the 8-rank distributed
calibration.

**Root cause (same class as the offset-norm fold loss, different site):**
in `QuantizationModifier._broadcast_qparam_onloads`
(`src/llmcompressor/modifiers/quantization/quantization/base.py`) each rank
observes weight qparams only for its greedy-binned module subset, then
`dist.broadcast` shares them. With dict/disk offload,
`getattr(module, qparam_name)` mints a FRESH onload tensor per call, so the
broadcast fills a temporary that `save_pretrained` never sees — non-owner
ranks (including the saving rank 0) keep uninitialized bytes for ~7/8 of
modules. GPTQ's `_broadcast_quantized_params`
(`src/llmcompressor/modifiers/gptq/base.py`) has the identical pattern (and
also broadcasts the quantized `weight` itself). r9 passed because
non-offloaded params alias real storage, so in-place broadcast persists.

**Fixes:**
1. Both broadcast sites now write received values back through
   `update_offload_parameter` (owner rank skipped — it already persisted via
   `update_qparams`/`compress_module_list`).
2. Regression tests
   `tests/llmcompressor/modifiers/quantization/test_qparam_broadcast_offload.py`
   (fake `dist.broadcast` fills the onload temp; disk copy must reflect it;
   verified to fail on the old code) + an owner-rank no-rewrite guard.
3. **Fail-closed post-save gate** `assert_quant_checkpoint_verified` in
   `pipeline/quantize.py` (after the smooth-fold gate, source rank, post
   barrier): runs `verify_quant_checkpoint.verify(ckpt, check_tensors=True)`
   — structure plus sampled scale/packed finiteness. Non-owner corruption
   hits ~87.5% of modules, so a 20-module sample cannot miss it. ABI,
   smooth-fold and save-health gates all passed on r3 because none of them
   read quantized tensors; this closes that gap.

**Why the smoke gates missed it:** r15's smooth-fold audit reads only BF16
norm/router/shared tensors — those were correct (the fold fix works). The
scale corruption lives in the quantized expert tensors no existing gate
sampled.

**Disposition:** r2, r3 (raw + `checkpoint-vllm-w123` portable), r13, r14,
r15 checkpoints all carry garbage scales — none may be served or evaluated.
The GPTQ portable artifact predates the distributed pipeline and is clean
(fresh serving smoke PASS on the current stack, 2026-07-19). Rerun: smoke
r16 with the fix, then full r4. Note distributed GPTQ runs under disk
offload were equally affected in principle — any such checkpoint must pass
the new gate before use.

**Gate-chain upgrade (2026-07-19):** two additions close the "no gate reads
the quantized tensors" blind spot at different costs:

1. **Smoke + full, in-line (CPU, ~1 min): dequant-vs-base value gate.**
   `verify_quant_checkpoint --dequant-base <base>` (wired into
   `assert_quant_checkpoint_verified`) dequantizes sampled packed modules
   and requires them to match base weights up to a fitted per-input-column
   scale (absorbs AWQ smoothing; ≈1.0 for GPTQ) with residual ≤ 0.25.
   Calibrated: consistent W4 (r9) resid 0.09–0.12, scales 0.7–1.4;
   garbage-scale checkpoints (r15/r3) NaN. Catches garbage scales, corrupt
   packed values, and lost transforms — both July-19 bugs would have failed
   a 45-min smoke here. A 2-node TP16 serve of the ~820 GB smoke checkpoint
   was considered and rejected: unvalidated topology (BF16 TP16 ray worked,
   quantized cross-node never tested) would confound checkpoint bugs with
   serve bugs, exactly the ambiguity gates exist to avoid.
2. **Full runs only (+~40 min GPU): serve + chat probe.** The 225 GB full
   checkpoint fits the proven 1-node TP8 envelope, so the r4 chain ends
   with portable re-export → HTTP serve (matrix harness, production case) →
   France-prompt probe (must contain "Paris"; NUL/repetition/empty fails
   the chain). This is the end-to-end wiring check (expert aliasing, swiglu
   clamp, loader) that CPU gates cannot see. Catastrophic-failure detector,
   not a quality measurement — real quality comparison remains the goal-2
   eval pipeline.

Smoke checkpoints (~820 GB each) are deleted at the end of a PASSING smoke
chain (evidence/logs kept); broken smoke checkpoints r13/r14/r15 were
deleted 2026-07-19 (~2.4 TB reclaimed, NFS was at 95%).

## AWQ smoothing-scale degeneracy on dead norm channels (full r4 post-mortem, 2026-07-20)

**Symptom:** Full run r4 (first with the qparam-broadcast fix) quantized for
7h00m, then failed the new dequant-vs-base gate: `dequant mismatch in
language_model.model.layers.12...experts.81.up_proj: resid=12.979`. Census:
1,423/21,888 routed-expert modules bad — exactly the gate+up projections of
layers 8, 10, 11, 12, 13 (down_proj clean; all other layers clean). Weight
*scales* were uniformly ~118× too large while the packed int4 values were
healthy (cosine ≈ 0.99 vs base after refitting); the saved post-attention
norms of those layers had mean ≈ −0.992 (effective gain ~1/128).

**Root cause (in `AWQModifier._compute_best_scale`, identical upstream):**
The M3 base model's post-attention norms on layers 8/10/11/12/13 each carry a
channel whose weight is EXACTLY −1.0. MiniMaxM3VLRMSNorm is Gemma-style
(`y = x̂ · (1 + w)`), so that channel's gain is 0 — its output is always
zero, hence its observed `x_mean` is exactly 0. The grid-search scale formula
`x_mean.pow(ratio).clamp(min=1e-4)` floors the dead channel at 1e-4, and the
geometric normalization `scales / (scales.max()·scales.min()).sqrt()` then
divides EVERY channel by `sqrt(max·1e-4)` ≈ inflating all scales ~×100
uniformly. The fold pushes the norm weights to ≈ −0.992, where bf16 cannot
resolve per-channel gains (spacing ~2⁻⁹ near 1), so the norm÷ / balance×
inverse pair no longer cancels and the layer is numerically destroyed.
Corroboration: r3 and r4 grid-search error logs are bit-identical (predates
the broadcast fix); the July-8 single-process full checkpoint shows the same
disease (layer-8 norm min −188 pre-clamping era); layers 9/14 (norm min
−0.9961, small-but-alive channel) are only mildly affected; upstream
llm-compressor `main` has the identical unprotected formula (verified
2026-07-19), so there was no existing fix to adopt.

**Fix (`src/llmcompressor/modifiers/transform/awq/base.py`):** scale
computation refactored into module-level `_grid_search_scales(x_mean, w_mean,
ratio)`: channels with `x_mean <= max(x_mean)·1e-6` are classified dead,
EXCLUDED from the geometric normalization, and pinned to scale 1.0 (any
scale on an always-zero channel is a functional no-op, but a floored one
poisons the normalization). Final scales are additionally hard-clamped to
[1e-2, 1e2] as a backstop against any unbounded fold. Regression tests:
`tests/llmcompressor/modifiers/transform/awq/test_grid_search_scales.py`
(includes a test documenting the old formula reproducing the ~×100
inflation).

**Gate upgrades (would have caught r4 in a 45-min smoke):**

1. `assert_smooth_fold_consistency` now audits ALL M3 MoE layers 3–59 by
   default (was spot-check 3/31/59, which is why r4's layers 8–13 sailed
   through); unsmoothed layers audit as scale = 1 and pass trivially, so the
   full sweep is safe for partial-layer smokes.
2. The gate also bounds fold MAGNITUDE: per-layer norm-implied scale mean
   must lie in [0.05, 20] (healthy ≈ 0.7–0.9; r4 disease ≈ 110–118). A fold
   can be perfectly self-consistent and still fatal via bf16 precision loss.
3. NaN-robustness: rel_l2 compares with `not (err <= threshold)` so NaN/inf
   fails closed, and the audit (`pipeline/m3_checkpoint_scale_audit.py`)
   treats both-gains-zero dead channels as scale 1 instead of 0/0 = NaN —
   otherwise a HEALTHY post-fix checkpoint would fail on layers 8/10–13.
4. Smoke config now quantizes layer 8 (dead-channel layer) alongside
   3/31/59 so every smoke exercises the degeneracy path end to end.

Validated against real checkpoints: corrupt r4 FAILS on exactly layers
8/10/11/12/13 (rel_l2 inf / scale mean ~117 / non-finite scale); known-good
r9 PASSES all 57 layers. Corrupt r4 checkpoint preserved at
`/mnt/nfs/hoangduy/results/m3-distributed-awq-full/20260719T162052Z-m3-ddp-awq-full-r4-qparamfix/...`
until a good full checkpoint exists. Open question: grid-search error still
grows monotonically with depth (3.74 at layer 59, 64 partially-corrupt
modules there in r4) — watch whether the dead-channel fix also cures it in
smoke/full r5.

## MiniMax-M3 full-calib AWQ garbage output (quality ablation, 2026-07-09)

**Symptom:** After a successful graphs-on serve-verify, both smoke and full-calib AWQ
W4AFP8 checkpoints answer the smoke prompt with the same degenerate repetition:

```
prompt: 'The capital of France is'
output: 'arringarringarringarring…'
```

Serve itself is healthy (`overall_ok=True`, `rc=0`, CUDA graphs capture, no IMA).
This is a **quantization-quality** failure, not load/serve infra.

### Checkpoints under test

| Label | Path | Calib | Scheme (as quantized) |
|-------|------|-------|------------------------|
| Smoke | `artifacts/MiniMax-M3-awq-W4AFP8/20260707-082218/checkpoint` | 8 × 512 (`minimax_m3.yaml`) | W4AFP8 |
| Full-calib | `artifacts/MiniMax-M3-awq-W4AFP8/20260708-093642/checkpoint` | 512 × 2048 (`minimax_m3_full_calib.yaml`) | W4AFP8 |
| Full-calib W4A16-serve | `…/20260708-093642/checkpoint-w4a16-serve` | same weights as full-calib | config-only: `input_activations: null` |

Smoke and full-calib share the **same ignore list** (dense 0–2, vision, projector,
patch_merge, MoE gate, shared experts, MSA indexer, `lm_head`). Full-calib only
increases sample count / seq length; it does **not** change which modules are INT4.

### Serve path that works (infra baseline)

Earlier chronicle entries assumed graphs-on was still broken at 16/51. On 2026-07-09
that was clarified:

| Attempt | Command / settings | Smoke | Full-calib |
|---------|--------------------|-------|------------|
| Detached defaults | `run_serve_minimax_m3_detached.sh` (`MAX_MODEL_LEN=8192`, `GPU_UTIL=0.9`) | IMA @ 16/51 (reproduced) | IMA @ 16/51 |
| Known-good graphs-on | `debug_cudagraph_ima.sh` (`MAX_MODEL_LEN=2048`, `GPU_UTIL=0.85`, `CUDA_LAUNCH_BLOCKING=1`, `TORCH_USE_CUDA_DSA=1`, `disable_custom_all_reduce=true`) | **PASS** (`overall_ok=True`) | **PASS** (`overall_ok=True`) |

So: **graphs + patches + custom-AR disable work on both checkpoints** under the
debug-script memory envelope. The remaining failure is **text quality**.

> Practical note: prefer `debug_cudagraph_ima.sh` (or the same
> `MAX_MODEL_LEN` / `GPU_UTIL`) when comparing quality across checkpoints, so
> graph-capture OOMs do not get confused with quant quality.

### Structural audit (rules out the usual MiniMax garbage modes)

`python -m pipeline.verify_quant_checkpoint` on the full-calib checkpoint
(**PASS**). Same ignore shape as smoke (`True True` + matching patterns).

| Hypothesis | Status | Evidence |
|------------|--------|----------|
| Shared expert dropped / zero-loaded ([aquaman164 MiniMax-M3 AutoRound](https://huggingface.co/aquaman164/MiniMax-M3-AutoRound-3.2bit-longctx)) | **Ruled out** | `shared_experts: none quantized`; `re:.*mlp[.]shared_experts[.].*` in saved `ignore` |
| MoE gate pruned from saved `ignore` (Qwen3 class; `pipeline/README.md`) | **Ruled out** | `re:.*mlp[.]gate$` present; `moe_router_gate: none quantized` |
| Wrong expert count / layout | **Ruled out** | Exactly **22116** = 57 × (128×3 + 4 attn) |
| MSA indexer accidentally INT4 | **Ruled out** | `msa_indexer: none quantized` |
| Dense layers 0–2 / vision quantized | **Ruled out** | ignore patterns match; verify PASS |
| “Full calib will fix smoke garbage” | **Falsified** | Full-calib still `"arringarring…"` |

Log note from quantize: **“57 mappings were skipped due to incompatible shapes.”**
That is expected for the MSA indexer / non-matching AWQ balance sets on this
architecture (see `pipeline/probe_awq_mappings.py` / correctness audit); it is
**not** evidence that shared experts or gates were dropped. Do not treat it as
the garbage root cause without a separate mapping-coverage check.

### Ablation 1 — serve INT4 weights with A16 activations (no re-quant) — DONE

**Hypothesis:** W4AFP8’s dynamic per-token FP8 activations on attention + experts
are too aggressive vs community W4A16 weight-only
([cyankiwi/MiniMax-M3-AWQ-INT4](https://huggingface.co/cyankiwi/MiniMax-M3-AWQ-INT4));
nulling `input_activations` should switch vLLM from
`CompressedTensorsW4A8Fp8` → `CompressedTensorsWNA16` while keeping the same
packed INT4 weights.

**How (NFS-safe — do not `cp -a` the full shard tree):**

```bash
SRC=artifacts/MiniMax-M3-awq-W4AFP8/20260708-093642/checkpoint
DST=artifacts/MiniMax-M3-awq-W4AFP8/20260708-093642/checkpoint-w4a16-serve

mkdir -p "$DST"
for f in "$SRC"/*; do
  base=$(basename "$f")
  [[ "$base" == config.json ]] && continue
  ln -sfn "$(realpath "$f")" "$DST/$base"
done
cp -a "$SRC/config.json" "$DST/config.json"

python - <<'PY'
import json
from pathlib import Path
p = Path("artifacts/MiniMax-M3-awq-W4AFP8/20260708-093642/checkpoint-w4a16-serve/config.json")
cfg = json.loads(p.read_text())
qc = cfg["quantization_config"]
for g in qc.get("config_groups", {}).values():
    g["input_activations"] = None
p.write_text(json.dumps(cfg, indent=2) + "\n")
for name, g in qc["config_groups"].items():
    print(name, "weights bits=", g["weights"]["num_bits"], "acts=", g.get("input_activations"))
PY
# Expect: group_0 weights bits= 4 acts= None
# Sanity: du -sh "$DST" is tiny (~132K); *.safetensors are symlinks into $SRC
```

**Serve:**

```bash
CHECKPOINT=artifacts/MiniMax-M3-awq-W4AFP8/20260708-093642/checkpoint-w4a16-serve \
OUT_DIR=serves/m3-awq-w4a16-serve-full \
  bash pipeline/slurm/debug_cudagraph_ima.sh
```

**Result (2026-07-09, h118):**

| Check | Outcome |
|-------|---------|
| Engine load | OK — `quantization=compressed-tensors`, `dtype=torch.bfloat16`, `quant: pack-quantized` |
| CUDA graphs | OK — capture completed; no IMA |
| Serve-verify | `overall_ok=True`, `rc=0` |
| Prompt output | **Still garbage** — `"arringarringarring…"` (same class as W4AFP8) |
| Config sanity | `group_0 weights bits= 4 acts= None`; DST ~132K with safetensor symlinks |

**Conclusion:** FP8 activations are **not** the cause of the `"arring"` collapse.
The packed INT4 **weights** (and/or **which Linears were quantized**) are the
problem. Re-serving the same checkpoint as W4A16 does not recover quality.

### Root-cause update (2026-07-09): online sources point to a SERVE-side load/wiring bug, not quant accuracy

Web search of reputable community M3-quant efforts on the **same pipeline shape**
(a *transformers* MiniMax-M3-VL export loaded into the official/`toncao` vLLM M3
backbone) turned up three checkpoints that hit the **identical garbled/repetition
failure**, and one that **bisected it to root cause**:

- **[aquaman164/MiniMax-M3-AutoRound-3.2bit-longctx](https://huggingface.co/aquaman164/MiniMax-M3-AutoRound-3.2bit-longctx)**
  (`m3_official_loader.py`, `M3_FIX_SHARED`) — proved with a standalone HF
  reference (`dbg_moe_ref_official.py`) that the official `MiniMaxM3MoE` passes
  `shared_experts` into `FusedMoE` expecting the **quant method** to fuse the
  shared-expert output; when the quant path returns `shared_output=None`, the
  **shared expert is dropped in every MoE layer 3–59**: `moe_out == routed*2`,
  with the shared expert (**norm 338 of 540**) entirely missing. The residual
  stream loses that contribution each layer → washes out to garbage. Two
  independent triggers for the same drop:
  1. **Name mismatch** — checkpoint keys shared experts as `mlp.shared_experts.*`
     (our transformers-VL naming, same as ours — see `minimax_m3.yaml` ignore
     `re:.*mlp[.]shared_experts[.].*` and `verify_quant_checkpoint.py`), but the
     official vLLM model looks them up as `block_sparse_moe.shared_experts.*`
     (`startswith` match) → lookup **misses** → weight loads as **zero**.
  2. **`n_shared_experts` not seen** — official `MiniMaxM3MoE` only *builds* the
     shared expert when `config.n_shared_experts` is truthy; when it's nested
     under `text_config` and the text-only arch reads top-level config, the module
     is never built. aquaman fixed it at the **source config** (force
     `n_shared_experts=1`), which builds the module and lets the runner add it.
  Also flagged in the same loader: **`lm_head` naming** — VL export stores
  `language_model.lm_head.weight`, text-only CausalLM expects top-level
  `lm_head.weight`, and `tie_word_embeddings=False`, so a miss = **random
  logits = garbage** (independent of any MoE issue).
- **[CosmicRaisins/minimax-m3-awq-gb10](https://github.com/CosmicRaisins/minimax-m3-awq-gb10)**
  / **[toncao/vllm minimax-m3-compressed-tensors](https://github.com/toncao/vllm/tree/minimax-m3-compressed-tensors)**
  — the bf16 **MSA lightning indexer** gets fused into the INT4 q/k/v GEMM
  (`MinimaxM3QKVParallelLinearWithIndexer`); a single quantized linear can't mix
  precisions, so the indexer is effectively quantized → mis-selected KV blocks →
  garbled/context-bleed. Their fix **de-quants q/k/v to bf16 at load** (not just
  "un-fuse"), feeding all 5 shards bf16.

**Why this beats the previous "attention INT4 too aggressive" ranking:** two of our
own observations are hard to explain with an *accuracy* hypothesis but are the
textbook signature of a **structural serve-side drop**:

- **Ablation 1 (W4A16 re-serve) still garbage** — changing the activation dtype
  cannot fix a shared-expert that is never *added*, or an `lm_head` that is never
  *loaded*.
- **Smoke and full-calib emit the identical `"arring"` loop** — a quant/calibration
  error would move with 64× more calibration data; a deterministic collapse that
  is invariant to calibration is a wiring/load bug, not a scale-fidelity bug.

Our earlier "shared expert dropped / zero-loaded → **Ruled out**" line checked the
**checkpoint** (`shared_experts: none quantized`, ignore pattern persisted). That
only proves the shared expert was kept bf16 *in the checkpoint*; aquaman's drop
happens **downstream at the vLLM loader/runner**, which `verify_quant_checkpoint`
(metadata-only, no model load, no runtime norms) does not exercise. So it was
ruled out at the wrong stage.

### Remaining hypotheses (re-ranked after the online root-cause finding)

| Rank | Hypothesis | Why it fits | How to test next |
|------|------------|-------------|------------------|
| **1** | **Shared expert silently dropped / zero-loaded at serve** in every MoE layer 3–59 (name mismatch `mlp.shared_experts.*` vs `block_sparse_moe.shared_experts.*`, and/or `shared_output=None` in the MoE runner, and/or `n_shared_experts` unset) | **Proven on the identical pipeline** by aquaman164 (`moe_out==routed*2`, shared norm 338/540 missing). Explains W4A16-still-garbage and calib-invariance. **Not a re-quant problem.** | **Runtime probe (no re-quant):** log per-MoE-layer shared-expert output norm and whether `moe_out ≈ routed*2`; log loaded `lm_head` param norm (0 ⇒ never loaded). Reuse aquaman's `M3_MOE_DIAG` / `M3_SHARED_PARAM` hook pattern. |
| **2** | **`lm_head` not loaded** (`language_model.lm_head.weight` vs top-level; `tie_word_embeddings=False`) → random logits | Same class of VL-export naming mismatch; short-prompt garbage fits. | Print the serve-loaded `lm_head.weight` norm; compare to checkpoint tensor. Cheap; do it in the same probe as #1. |
| **3** | **q/k/v INT4 fused with bf16 MSA indexer** corrupts token selection | Real (CosmicRaisins/toncao). We *assume* toncao's branch un-fuses; verify it also **de-quants q/k/v to bf16** at load, not just un-fuses. | With #1/#2 clean, run aquaman's `M3_FULL_ATTN=1` / `M3_FA_COMPARE=1` bisection or confirm the toncao loader de-quants q/k/v. |
| **4** | Attention/experts INT4 accuracy genuinely too aggressive | Only after 1–3 are cleared; community keeps self-attn bf16 (experts-only W4A16). | Re-quant experts-only W4A16 (~21888 Linears), serve with `debug_cudagraph_ima.sh`. |
| **5** | AWQ mapping coverage / calib length | Least likely (calib-invariant symptom). | Revisit only if 1–4 fail. |

### What we are **not** doing next (and why)

- **Re-quantizing first** — the top-2 suspects are **serve-side load/wiring** bugs a
  re-quant cannot fix; confirm with a runtime probe before spending a ~hours-long
  full-calib run. (Previous plan led with an experts-only re-quant; deferred to Rank 4.)
- **Another config-only activation tweak** — Ablation 1 already nulls FP8 acts.
- **Blaming graphs / custom AR / patch 4** — both checkpoints generate under the
  known-good graphs-on path; infra is cleared for this symptom.
- **Full `cp -a` of checkpoint shards on NFS** — hangs; use symlink + config copy.

### Next concrete step (decisive, no re-quant) — 2026-07-09

Two diagnostics are wired into the serve path (both cheap; the runtime probe is a
no-op unless enabled). **No re-quant** — they confirm/rule out the serve-side
shared-expert drop (Rank 1) and `lm_head` (Rank 2) before spending a full-calib run.

**1. Static checkpoint audit (main process, always on for M3).** `serve_verify.py`
now calls `audit_m3_checkpoint()`, which reads `config.json` + the safetensors
index (no tensor load) and prints an `M3 CHECKPOINT QUALITY AUDIT` block:
`tie_word_embeddings`, `n_shared_experts` (top-level **vs** `text_config`), the
`lm_head` keys, and the **shared-expert key style** (`mlp.shared_experts.*` vs
`block_sparse_moe.shared_experts.*`). A `mlp.shared_experts` style, a missing
`lm_head`, or `n_shared_experts` only under `text_config` is flagged as a likely
trigger. Run it standalone:

```bash
python -c "
from pathlib import Path
from pipeline.serve_verify import audit_m3_checkpoint, _read_model_config
ck = Path('artifacts/MiniMax-M3-awq-W4AFP8/20260708-093642/checkpoint')
audit_m3_checkpoint(ck, _read_model_config(ck))
"
```

**2. Runtime MoE probe (Worker_TP*, env-gated).** Injected into vLLM
**site-packages** (so it reaches the spawned workers — in-process hooks do not)
by `patch_vllm_m3_serve.py::ensure_m3_moe_probe()`; `serve_verify` applies it
automatically. Dormant unless `M3_MOE_PROBE=1`. For the first few real-prefill MoE
forwards it logs the shared-expert output norm and combined `moe_out` norm:

```bash
python pipeline/slurm/patch_vllm_m3_serve.py --probe   # inject once (idempotent)
# then serve with the known-good graphs-on envelope + probe on:
M3_MOE_PROBE=1 CHECKPOINT=artifacts/MiniMax-M3-awq-W4AFP8/20260708-093642/checkpoint \
  bash pipeline/slurm/debug_cudagraph_ima.sh 2>&1 | tee /mnt/nfs/hoangduy/logs/m3-moe-probe.log
grep 'M3_MOE_PROBE#' /mnt/nfs/hoangduy/logs/m3-moe-probe.log
# M3_MOE_PROBE_RECOMPUTE=1 also logs routed-only norm + out/routed ratio (heavier;
# recomputes self.experts, so opt-in). moe_out ~= routed*2 => shared dropped.
```

Read: **`shared_present=False`**, **`shared_norm≈0`**, or **`out/routed≈2.0`**
⇒ shared expert dropped in every MoE layer (Rank 1 confirmed).

If Rank 1 confirmed, the fix is **serve-side, not a re-quant**: correct the
shared-expert key mapping (`mlp.shared_experts.*` → `block_sparse_moe.shared_experts.*`),
ensure `n_shared_experts` is set at the level the vLLM M3 arch reads (so the module
is built), and ensure the MoE forward adds the shared-expert output exactly once
(aquaman's note: with the module present the FusedMoE runner already adds it — do
**not** double-add). Only if 1–3 are clean do we fall back to the experts-only
W4A16 re-quant (Rank 4). Document PASS/FAIL of the `"arring"` prompt here when the
probe run finishes.

## MiniMax-M3 sequential calibration fails on `image_token_id`

**Symptom:** After `linearize_moe` completes, `SequentialPipeline` FX tracing fails with:

`AttributeError: 'MiniMaxM3VLConfig' object has no attribute 'image_token_id'. Did you mean: 'image_token_index'?`

**Root cause:** `modeling_minimax_m3_vl.get_placeholder_mask` reads `config.image_token_id` / `config.video_token_id`. The public checkpoint defines `image_token_index` / `video_token_index` only. Transformers' `attribute_map` alias does not satisfy strict `__getattribute__` during FX tracing.

**Fix applied (2026-07-07):** `pipeline/minimax_m3_config.py` — `_ensure_token_id_aliases()` sets `image_token_id` and `video_token_id` explicitly after coercion.

**Verify:**

```bash
source /mnt/nfs/hoangduy/env.sh
source /mnt/nfs/hoangduy/venvs/quant/bin/activate
cd /mnt/nfs/hoangduy/projects/llm-compressor
python -c "
from pipeline.minimax_m3_config import load_minimax_m3_vl_config
c = load_minimax_m3_vl_config('/mnt/nfs/hoangduy/hf_assets/MiniMaxAI/MiniMax-M3')
print(c.image_token_index, c.image_token_id, c.video_token_index, c.video_token_id)
"
# Expect: 200025 200025 200026 200026
```

## MiniMax-M3 AWQ/GPTQ FX trace fails on `image_features.numel()`

**Symptom:** After `linearize_moe`, sequential AWQ/GPTQ tracing fails with:

`'NoneType' object has no attribute 'numel'` in `get_placeholder_mask` when `image_features` is absent (text-only calibration).

**Root cause:** Text-only calibration passes no `pixel_values`. FX tracing still runs the VL forward; absent `image_features` / `video_features` are represented as non-`None` proxies, so `get_placeholder_mask` enters the feature-count validation and calls `.numel()` on `None`.

**Fix applied (2026-07-07):** `pipeline/minimax_m3_config.py` — `patch_minimax_m3_for_text_calibration()` coerces non-tensor features to `None` before delegating to the original method. Called from `pipeline/quantize.py` after model load.

**Long-term fix:** Multimodal calibration dataset (images + processor) like `examples/multimodal_vision/qwen3_vl_example.py`, if vision-tower quantization behavior must be validated under real inputs.

## MiniMax-M3 AWQ fails on default smooth-layer mappings

**Symptom:** After FX tracing succeeds, AWQ calibration fails with:

`ValueError: AWQ needs to match a single smoothlayer for each mapping but got [layers.0..59 input_layernorm] for mapping: AWQMapping(smooth_layer='re:.*input_layernorm$', ...)`

**Root cause:** `MiniMaxM3SparseForConditionalGeneration` is not in llm-compressor's `AWQ_MAPPING_REGISTRY`, so `AWQModifier` falls back to `default_mappings`. Those break on M3 because:

1. Sparse layers add `self_attn.indexer.{q,k}_proj` — generic `re:.*q_proj$` / `re:.*k_proj$` match both attention and indexer, so `match_modules_set` cannot close per-layer groups.
2. MLP is fused (`mlp.gate_up_proj` on dense layers) and mixed dense (layers 0-2) vs MoE (3-59) with `mlp.shared_experts` and per-expert `gate_proj`/`up_proj` after `linearize_moe`.

**Community reference:** `cyankiwi/MiniMax-M3-AWQ-INT4` uses a keep-bf16 ignore list (dense layers 0-2, indexer, router, shared experts, vision) and W4A16 weights; module names differ on transformers 5.13 (`block_sparse_moe.*` vs our `mlp.*`).

**Second root cause (vision-tower name collision):** An initial scoped mapping still collapsed layers 3-59 into one group. A meta-device probe (`pipeline/probe_awq_mappings.py`) showed the balance targets `re:.*layers[.]<n>[.]self_attn[.]{q,k,v}_proj` matched **86** modules, not 57: `model.vision_tower.layers.N.self_attn.{q,k,v}_proj` also match. The vision layers have no matching `input_layernorm` (smooth matched only the 57 language layers), so `match_modules_set` never closes a per-layer group and collapses all smooth layers. Fix: anchor every pattern to `.*language_model[.]layers[.]` so the vision tower is excluded. The indexer was NOT the problem (present on all 57 sparse layers).

**Fix applied (2026-07-07):**

1. `pipeline/minimax_m3_config.py` — `get_minimax_m3_awq_mappings()` + `register_minimax_m3_awq_mappings()` for sparse layers 3-59, anchored to `language_model.layers` (excludes vision tower), `self_attn.*`, and `mlp.experts.N.*`.
2. `pipeline/quantize.py` — register mappings after M3 text-calibration patch.
3. `pipeline/configs/minimax_m3.yaml` — translated keep-bf16 `quantization.ignore` list.
4. `pipeline/probe_awq_mappings.py` — meta-device probe that replicates `match_modules_set` grouping (indexer coverage, per-target match counts, groups yielded) to diagnose mapping issues in seconds without a full model load. Routed-expert (split) patterns only resolve after `linearize_moe`, so mappings 3/4 are validated in the smoke run, not on meta.

**KV cache:** Leave `kv_cache_scheme` unset in the recipe (`null` in saved config). fp8 KV is applied at serve time via `serve.kv_cache_dtype: fp8` / vLLM `--kv-cache-dtype fp8` (same as community recipe).

**Serve follow-up:** Stock vLLM may need the `toncao/vllm` `minimax-m3-compressed-tensors` branch to un-fuse the bf16 MSA indexer from quantized q/k/v projections for correct long-context output. W4AFP8 weight scheme is independent; validate on H100 with patched vLLM before production serve.

**Venv (do not mix):** M3 quantize + vLLM serve use `venvs/quant`. The separate `venvs/sglang-eval` is only for SGLang-backed eval (e.g. GLM-5.2): SGLang's DeepGEMM JIT needs a real **nvcc >= 12.9** (system 12.4 is too old), and `pip install -e .` in that venv upgrades torch and breaks FlashInfer. Launchers: `pipeline/slurm/run_serve_minimax_m3_detached.sh` (vLLM) vs `run_eval_glm52_sglang_detached.sh` (SGLang).

**Verify:**

```bash
source /mnt/nfs/hoangduy/env.sh
source /mnt/nfs/hoangduy/venvs/quant/bin/activate
cd /mnt/nfs/hoangduy/projects/llm-compressor
python -m pytest pipeline/tests/test_minimax_m3_config.py -v

METHOD=awq EXTRA='--set calibration.num_samples=8 --set calibration.max_seq_length=512' \
  bash pipeline/slurm/run_quantize_minimax_m3_detached.sh
# Expect: no "single smoothlayer" error; AWQ resolves mappings per sparse layer
```

## MiniMax-M3 calibration fails: `LinearExperts2D` has no attribute `swiglu_limit`

**Symptom:** AWQ mappings resolve and calibration begins, but around sequential layer 5 the forward crashes:

```
File ".../transformers/models/minimax_m3_vl/modeling_minimax_m3_vl.py", line 212, in _apply_gate
    gate = gate.clamp(max=self.swiglu_limit)
AttributeError: 'LinearExperts2D' object has no attribute 'swiglu_limit'
```

**Root cause:** `MiniMaxM3VLExperts._apply_gate` reads `self.swiglu_limit` / `self.swiglu_alpha` (config-derived scalars the original fused module sets in its `__init__`). M3 has no registered 2D load mapping, so it takes the post-load `linearize_moe` path via the generic `LinearExperts2D`. That path **reuses M3's `_apply_gate` verbatim** (bound to the new `LinearExperts2D` instance) but `MoEConfig` only captures the generic `limit`/`alpha` names — `swiglu_limit`/`swiglu_alpha` are never set on the linearized module, so the reused method raises at calibration forward time. (`limit`/`alpha` are set but M3's method doesn't read them.)

**Long-term fix (applied 2026-07-07):** `src/llmcompressor/modeling/moe/linear_experts.py` — `LinearExperts2D.from_experts_module()` now calls `_carry_over_gate_scalars(self, experts)`, copying plain bool/int/float attributes from the source experts module onto the linearized module when not already set. This keeps the reused `_apply_gate` numerically identical to the fused experts and is general for any model whose custom gate reads config-derived scalars (not M3-specific). Structural fields and parameters/buffers/submodules are left untouched (skips `_`-prefixed and existing attrs).

**Removal criteria:** None; carrying over the scalars the reused `_apply_gate` depends on is the correct contract. Registering a dedicated `MiniMaxM3LinearExperts` subclass (like `Llama4LinearExperts`) with a 2D load mapping would supersede this path but is a larger change tracked separately.

**Verify:**

```bash
METHOD=awq EXTRA='--set calibration.num_samples=8 --set calibration.max_seq_length=512' \
  bash pipeline/slurm/run_quantize_minimax_m3_detached.sh
# Expect: calibration proceeds past layer 5 through all 61 subgraphs -> checkpoint save
```

## MiniMax-M3 post-quant `SAMPLE GENERATION` appears to hang (offloaded generation gates the save)

**Symptom:** After `Compression lifecycle finalized`, the run prints `SAMPLE GENERATION` then `dispatch_model | WARNING - Forced to offload modules due to insufficient gpu resources` and sits for many minutes with no further output.

**Root cause:** Two issues. (1) `_sample_generation()` runs `model.generate(max_new_tokens=64)` on the 428B model, which is dispatched with CPU/disk offload — each token streams expert weights off CPU/disk, so 64 tokens take tens of minutes to hours. It is slow, not hung. (2) Worse, the sanity generation ran *before* `model.save_pretrained`, so interrupting it lost the quantized weights.

**Fix applied (2026-07-07):**

1. `pipeline/quantize.py` — reordered `run_quantize()` to `save_pretrained` (+ `_persist_ignore_to_config`, `write_recipe`) **before** the sanity generation, so a slow/interrupted generation can never lose the checkpoint. Generation is now safe to Ctrl-C.
2. `pipeline/config.py` — added `QuantizationConfig.sample_generation: bool = True`.
3. `pipeline/configs/minimax_m3.yaml` — `quantization.sample_generation: false` (validate via `serve_verify.py` on vLLM/H100 instead of offloaded HF generate).

**Removal criteria:** None; saving before an optional sanity check is the correct ordering. Re-enable `sample_generation` for small models that fit on GPU.

## Detached launcher `kill $(cat PID_FILE)` misses the worker (stale `$!`)

**Symptom:** `kill "$(cat …/quantize-m3-awq.pid)"` returns success but the run keeps going; relaunch reports `Quantize already running (pid=…)`. `Ctrl-C` only kills the foreground `tail`.

**Root cause:** `run_quantize_minimax_m3_detached.sh` (and `run_eval_glm52_sglang_detached.sh`) recorded `$!` — the PID of the `nohup setsid …` launcher — into `PID_FILE`. `setsid` may fork a new session leader for the actual python, so the recorded PID is a stale/parent process; `kill` then targets the wrong PID.

**Fix applied (2026-07-07):** both detached scripts now `echo $$ > PID_FILE` **inside** the generated run script, just before `exec python …`. Because that bash `exec`s into python (same PID), `PID_FILE` holds the real worker PID. Stop hints updated to include `kill -9 -"$(cat PID_FILE)"` (process-group hard kill) and `pkill -9 -f 'pipeline\.run .*--stage quantize'`.

**Removal criteria:** None.

## MiniMax-M3 quantization correctness audit (Qwen3-informed)

Cross-checked the M3 W4AFP8 run against the documented Qwen3 MoE failure modes
(ModelOpt `BUGS_AND_FIXES.md` bug #1: wrong expert layout -> ~56k vs ~13k
quantizers; `pipeline/README.md`: MoE gate pruned from saved `ignore` -> vLLM
garbage; W4A8 MoE expert width multiple-of-256).

**Findings (static audit):**

- **Scheme:** `W4AFP8` = INT4 weights, GROUP `group_size=128`, symmetric + FP8
  (E4M3) dynamic per-token activations. Note this is **more aggressive than the
  community `cyankiwi/MiniMax-M3-AWQ-INT4` (W4A16, weight-only)** — FP8 activations
  on attention + experts are an added accuracy risk to watch in eval.
- **AWQ scale invariance (correct):** the `post_attention_layernorm` balance set
  MUST include *every* consumer of that norm — router (`mlp.gate`), `shared_experts`,
  and routed `experts.*.{gate,up}_proj` — else the smoothing scale is not invariant
  and routing/shared output is corrupted. Our mappings include all of them (even the
  bf16-ignored router/shared, which get the scale folded into their bf16 weights).
  Same for `input_layernorm` -> q/k/v + indexer q/k. This is why the router/shared
  appear as AWQ balance layers despite being in `ignore`.
- **Expected quantized Linears:** sparse layers 3-59 (57) x (128 experts x 3 +
  4 attn) = **22,116**. Dense layers 0-2, router, shared_experts, indexer, vision,
  projector, patch_merge, lm_head stay bf16.
- **Geometry:** routed-expert `intermediate_size=3072 = 12 x 256` and `= 24 x 128`
  -> satisfies the vLLM CUTLASS W4A8 256 constraint **only at full/EP width**. Serve
  with `enable_expert_parallel: true` (config does). Plain TP would divide 3072 and
  can break the multiple-of-256 rule.
- **`ignore` persistence:** `_persist_ignore_to_config()` re-adds all recipe ignore
  patterns to saved `config.json` (fixes the Qwen3 gate-pruning bug).

**Open risk (serve-side, not quantization):** M3 has **no 2D save-conversion
mapping** (`conversion_mappings.py` only covers deepseek_v4 / qwen2_moe / qwen3_moe),
so the checkpoint is saved with **per-expert linearized** experts
(`mlp.experts.N.{gate,up,down}_proj` + `weight_packed`/`weight_scale`), not the fused
3D `mlp.experts.gate_up_proj` the HF modeling expects. Stock vLLM M3 support must be
able to load this per-expert compressed layout (see the `toncao/vllm`
`minimax-m3-compressed-tensors` branch note); validate load before trusting eval.

## MiniMax-M3 vLLM serve fails: missing `preprocessor_config.json`

**Symptom:** vLLM init crashes before weight load:

```
OSError: Can't load image processor for '.../checkpoint'. ...
make sure '.../checkpoint' is the correct path to a directory containing a preprocessor_config.json file
```

**Root cause:** Quantize saved `model` + `tokenizer` only. vLLM still boots the VL multimodal stack (encoder budget / image processor) even for a text-only smoke prompt, and needs image-processor configs that `tokenizer.save_pretrained` does not write.

**Fix (long-term):** `pipeline/quantize.py` now calls `ensure_vl_processor_artifacts()` after save for `AutoModelForImageTextToText` models.

**Fix (existing checkpoints):** `serve_verify.py` auto-copies processor files from `model.id` (original HF path) into the checkpoint before vLLM load. Re-run serve with:

```bash
MODEL_ID=/mnt/nfs/hoangduy/hf_assets/MiniMaxAI/MiniMax-M3 \
  bash pipeline/slurm/run_serve_minimax_m3_detached.sh
```

Or one-shot manual copy:

```bash
python -c "
from pathlib import Path
from pipeline.vl_artifacts import ensure_vl_processor_artifacts
ckpt = Path('artifacts/MiniMax-M3-awq-W4AFP8/20260707-082218/checkpoint')
src = '/mnt/nfs/hoangduy/hf_assets/MiniMaxAI/MiniMax-M3'
print(ensure_vl_processor_artifacts(ckpt, src, trust_remote_code=True))
"
```

## MiniMax-M3 vLLM worker init fails (`EngineCore failed to start`)

**Symptom:** vLLM gets past config/processor init, then worker ranks die:

```
Exception: WorkerProc initialization failed due to an exception in a background process.
RuntimeError: EngineCore initialization failed.
WARNING: destroy_process_group() was not called before program exit
```

**Root cause (usual):** Stock vLLM tries to load sparse layers with a **fused q/k/v + indexer** packed GEMM, but our checkpoint **quantizes q/k/v and keeps the MSA indexer in bf16** (`ignore: re:.*self_attn[.]indexer[.].*`). MoE weights are also saved as **per-expert linearized** `block_sparse_moe.experts.N.{gate,up,down}_proj` pack-quantized tensors. Ton Cao's vLLM branch un-fuses the indexer and plumbs M3 SwiGLU clamp params:

```bash
bash pipeline/slurm/install_vllm_m3_serve.sh   # toncao/vllm minimax-m3-compressed-tensors
```

**Diagnose:** the parent traceback hides the worker error. Grep the serve log:

```bash
grep -E 'Worker|ERROR|Error|OOM|out of memory|KeyError|size mismatch' serves/m3-awq-w4afp8/run.log | tail -40
```

**If OOM:** retry with `MAX_MODEL_LEN=2048 GPU_UTIL=0.85` (and optionally `--set serve.enforce_eager=true`).

## MiniMax-M3 vLLM serve fails: `img_token_compression_config` missing

**Symptom:** Worker dies during vision tower init (before weight load):

```
AttributeError: 'PreTrainedConfig' object has no attribute 'img_token_compression_config'
```

**Root cause:** `coerce_minimax_m3_vl_config()` (quantize load path) hoists compression fields onto `MiniMaxM3VLVisionConfig` and drops the nested `img_token_compression_config` dict. vLLM's `MiniMaxVLVisionModel` still reads that attribute. Vision weights are bf16/unchanged, so restoring the source model's `vision_config` is safe.

**Fix:** `ensure_minimax_m3_vllm_serve_config()` in `serve_verify.py` (and `quantize.py` on save) copies `vision_config` from `model.id` back into the checkpoint `config.json`.

```bash
python -c "
from pathlib import Path
from pipeline.minimax_m3_config import ensure_minimax_m3_vllm_serve_config
print(ensure_minimax_m3_vllm_serve_config(
    Path('artifacts/MiniMax-M3-awq-W4AFP8/20260707-082218/checkpoint'),
    '/mnt/nfs/hoangduy/hf_assets/MiniMaxAI/MiniMax-M3',
))
"
```

Then rerun serve. Ton Cao's vLLM branch may still be needed **after** this for quantized q/k/v + indexer un-fuse.

## MiniMax-M3 vLLM serve fails: `Unsupported activation: silu. Only swigluoai is supported.`

**Symptom:** All TP workers die during language-model layer init (before weight load completes):

```
ValueError: Unsupported activation: silu. Only swigluoai is supported.
  File ".../vllm/models/minimax_m3/nvidia/model.py", line 168, in MiniMaxM3MLP.__init__
```

**Root cause:** The public HF checkpoint declares `text_config.hidden_act: "swigluoai"`, but `MiniMaxM3VLTextConfig.__post_init__` in transformers 5.12+ rewrites it to `"silu"` (ACT2FN fallback; the real SwiGLU-OAI gate uses `swiglu_alpha` / `swiglu_limit`). Quantize load uses that coercion and `save_pretrained` persists `hidden_act: "silu"`. vLLM's `MiniMaxM3MLP` only wires `SiluAndMulWithClamp` when `hidden_act == "swigluoai"`.

**Fix:** `ensure_minimax_m3_vllm_serve_config()` forces `text_config.hidden_act = "swigluoai"` for any `minimax_m3_vl` checkpoint and (re)copies `swiglu_*` scalars. `serve_verify.py` applies this automatically before `LLM()`. No re-quantize needed.

**Why the first attempt silently failed (real root cause of the re-occurrence):** the original patch only restored `hidden_act` when it read the literal string `"swigluoai"` from the source, and it resolved a hub `model.id` via `AutoConfig.from_pretrained(...).to_dict()` — which runs the same `__post_init__` coercion and returns `"silu"`. So `src_act == "swigluoai"` was never true, the `hidden_act` line was skipped, and the checkpoint kept `silu` (only the plain `swiglu_*` scalars copied). The patch appeared to "run" (it printed changes for the scalars) but never fixed the activation. The patch now forces `swigluoai` unconditionally instead of trusting the coercion-prone source string. Regression test: `test_ensure_vllm_serve_config_forces_hidden_act_even_if_source_coerced`.

**Verify after patching** (must show `swigluoai`):
```bash
venvs/quant/bin/python -c "import json;print(json.load(open('<ckpt>/config.json'))['text_config']['hidden_act'])"
```
If serve *still* aborts with `Unsupported activation: silu` after `config.json` shows `swigluoai`, then the transformers in `venvs/quant` re-coerces `hidden_act` on vLLM's own config load, and a JSON patch cannot fix it — escalate to a transformers-side fix (register `swigluoai` in `ACT2FN`, or override `MiniMaxM3VLTextConfig.__post_init__`) or a vLLM config hook.

**Note:** Ton Cao's `minimax-m3-compressed-tensors` branch has the same `hidden_act` check — installing it alone does **not** fix this. You still need the config patch **and** that branch for indexer un-fuse + linearized MoE pack-quantized load.

## MiniMax-M3 vLLM serve fails: `W4A8 MoE backend CUTLASS does not support ... SWIGLUOAI_UNINTERLEAVE`

**Symptom:** After the `hidden_act` fix, all TP workers die during MoE layer init:

```
NotImplementedError: W4A8 MoE backend CUTLASS does not support the deployment
configuration: kernel does not support MoEActivation.SWIGLUOAI_UNINTERLEAVE activation.
  File ".../fused_moe/oracle/w4a8.py", line 76, in select_w4a8_moe_backend
```

**Root cause (a real vLLM gap, NOT a checkpoint problem):** M3's routed experts store `w13` as `[all gates; all ups]` (the `MergedColumnParallelLinear` layout `llm-compressor` produces), so the M3 model requests `MoEActivation.SWIGLUOAI_UNINTERLEAVE`. The W4A8 CUTLASS MoE path applies the activation *generically* via `apply_moe_activation` **after** GEMM1 (not fused in the epilogue), and `apply_moe_activation` **already implements** `SWIGLUOAI_UNINTERLEAVE` (its docstring literally names MiniMax-M3). Two plumbing gaps block it in **every** build checked (stock 0.24.0, the NVIDIA `vllm/models/minimax_m3/nvidia` build, and toncao's branch):

1. `CutlassExpertsW4A8Fp8._supports_activation` omits `SWIGLUOAI_UNINTERLEAVE` (the gate that raises).
2. The W4A8 call site `apply_moe_activation(activation, act_out, mm1_out)` passes no `clamp_limit`/`alpha`/`beta`, but `SWIGLUOAI_UNINTERLEAVE` asserts `clamp_limit is not None`.

Reinstalling toncao's branch does **not** help (verified: its `_supports_activation` is identical). The only W4A8 MoE backend is CUTLASS — there is no Triton/Marlin fallback to route to.

**Fix — two options, same two edits:**

- **In-process (offline serve-verify):** `pipeline/vllm_m3_patches.py::patch_vllm_w4a8_swigluoai_uninterleave()` (1) adds `SWIGLUOAI_UNINTERLEAVE` to `CutlassExpertsW4A8Fp8._supports_activation` and (2) wraps `cutlass_moe.apply_moe_activation` to inject `clamp_limit`/`alpha`/`beta` read from the checkpoint's **resolved** config (`read_swiglu_params`). Applied automatically by `serve_verify.py` before `LLM()`. Only affects the `serve_verify` process — **not** a standalone `vllm serve`.
- **Persistent (production `vllm serve`):** `python pipeline/slurm/patch_vllm_m3_serve.py` edits the two installed vLLM source files once (idempotent), so every launch path works with no runtime hook. `install_vllm_m3_serve.sh` re-applies it after a (re)install. Use `--check` to verify status.

Both make the MoE path numerically identical to the dense `MiniMaxM3MLP` (`SiluAndMulWithClamp`: `gate*sigmoid(alpha*gate)*(up+beta)`, clamped). M3 uses the gpt-oss constants `alpha=1.702`, `limit=7.0`, `beta=1.0` (`swiglu_beta` is `null` in raw json but transformers resolves it to a float at load; the dense MLP relies on the same).

**Root-cause / long-term fix (preferred):** upstream the two-line capability into vLLM — add the enum to `CutlassExpertsW4A8Fp8._supports_activation` and thread the swiglu scalars into the W4A8 `apply_moe_activation` call. **Removal criteria:** delete `pipeline/vllm_m3_patches.py` and the `serve_verify` hook once a vLLM release serves M3 W4A8 (SwiGLU-OAI uninterleaved) natively.

**Tooling:** `pipeline/verify_quant_checkpoint.py` runs all of the above structural
checks on checkpoint metadata (fast, no model load); `--check-tensors` adds sampled
finiteness + group-scale-shape checks:

```bash
python -m pipeline.verify_quant_checkpoint --ckpt artifacts/MiniMax-M3-awq-W4AFP8/<ts>/checkpoint
python -m pipeline.verify_quant_checkpoint --ckpt <dir> --check-tensors   # opens shards
```


## Pre-quantization compatibility gate (original model + recipe)

Long calibration runs must now be preceded by the planner-only gate:

```bash
python -m pipeline.prequant_compatibility \
  --config pipeline/configs/minimax_m3.yaml \
  --output artifacts/preflight/minimax-m3-awq.json
```

The command builds a disposable meta model, mirrors MoE linearization, constructs the
exact pipeline recipe, and invokes llm-compressor's real quantization initialization,
target matching, group-divisibility checks, dynamic AWQ mappings, and AWQ mapping
resolver. It never loads checkpoint tensors or calibration data, installs hooks, runs
a forward, or allocates a GPU. Exit status is `0` for structural compatibility and `2`
for a persisted incompatibility report.

For MiniMax-M3 AWQ, the report verifies that every resolved
`MiniMaxM3VLRMSNorm` smooth layer is backed by `CalibrationOffsetNorm`; removing that
adapter is a hard `missing_offset_norm_adapter` failure before calibration. The report
also preserves resolved targets, ignores, quantized module names, AWQ smooth/balance
mappings, failures, warnings, and properties that remain unverified.

This does not replace the representative-layer canary, post-quantization serving ABI
gate, runtime smoke, or quality evaluation. The required order is: pre-quantization
gate, representative canary for expensive/new recipes, full quantization, serving ABI
gate, runtime smoke, then quality evaluation. Version one supports GPTQ and AWQ; other
methods fail as unsupported rather than receiving a guessed pass.

## AWQ up->down smoothing fold is not function-preserving on MiniMax-M3 (root-cause candidate for the AWQ non-termination pathology, 2026-07-23)

**Symptom (long-standing):** in-house AWQ W4AFP8 (r5) loses 24 pts on GPQA /
6 pts on IFEval purely through reasoning non-termination (runaway `<think>`
loops, budget exhaustion), while GPTQ on the identical scheme/pipeline is
clean (`M3_OFFICIAL_QUALITY_RESULTS.html`). GLM-5.2's AWQ-produced W4AFP8 and
Qwen3-30B-A3B (our own pipeline) are also clean — a model x method
interaction no prior hypothesis explained.

**Root cause (recipe-level, `get_minimax_m3_awq_mappings` mapping 4):** the
per-expert `experts.N.up_proj -> experts.N.down_proj` AWQ mapping folds
`up_rows /= s_r`, `down_cols *= s_r`. That fold is a pure reparameterization
only if down's input is LINEAR in up's output. M3's expert activation is
gpt-oss style — `h = (clamp(up, ±7) + 1.0) * glu` (`swiglu_beta = 1.0`,
`swiglu_limit = 7.0`; `MiniMaxM3VLExperts._apply_gate`) — affine and clamped
in `up`. After folding, the model effectively computes `(up + s_r·1.0) * glu`
with the up-clamp moved to `±7·s_r`: a per-channel FUNCTION change on every
token. Recovered from the shipped r5 checkpoint (base row-group maxes vs
stored `weight_scale`), the fold scales are far from 1 — median `s_r` 0.90 /
1.66 / 1.39 at layers 3 / 30 / 59 — and a gaussian estimate puts the
perturbation at ~5% / ~33% / ~14% RMS of the true down input, ~10x the int4
rounding error itself. It is a coherent per-channel drift (fixed sign per
channel), which matches the observed phenotype: capability intact, long-
horizon termination behavior broken, partially rescued by sampling.

**Why every probe missed it:** the AWQ grid-search loss evaluates
`Q(W·s)/s` on unmodified modules — the scale round-trips inside the weights
and never crosses the activation, so the selection loss cannot see the
damage the later fold causes. The fold-consistency and dequant-vs-base gates
verify WEIGHT algebra (which is perfectly consistent); the dequant gate's
fitted per-column scales absorb exactly this rescaling. The linearized
calibration experts reuse `_apply_gate`, so calibration matched serve — the
bug is purely the fold's linearity assumption.

**Why it is M3-specific and AWQ-specific:** Qwen3/GLM use plain SwiGLU
(`h = u·glu`, beta 0, no clamp) — the identical mapping is exactly legal
there, hence their clean results. GPTQ never rescales weights, hence its
clean result on M3. AutoAWQ's Mixtral-family mapping performs the same
w3 -> w2 fold, consistent with the community cyankiwi quant failing worse.
Upstream llm-compressor has NO registered mappings for gpt-oss-family
activations; the M3 mapping was hand-authored here (mirroring cyankiwi).

**Supporting evidence that removal is cheap:** the mapping's measured
benefit is small — the r5 run's own telemetry (identity included in the
grid) shows median 6% down-weight-MSE reduction; both M3's expert-input
activation landscape (recovered from production calibration statistics) and
its weights are flat/incoherent (random-rotation test: 2.9% gain), so
salience-protection has little to protect on M3.

**Fix (r6):** mapping removed in `get_minimax_m3_awq_mappings` with a
regression guard (`tests/pipeline/test_minimax_m3_awq_mappings.py`) that
fails if any fold crosses the expert activation boundary. General rule
encoded there: a fold may only pass through an activation factor in which it
is homogeneous. The post-attention-norm MoE-input mapping is unaffected
(purely linear boundary, consumer set verified complete: router + shared +
expert gate/up; r5 fold audit clean at bf16 rounding).

**Planned follow-up (r7, optional):** a function-preserving replacement that
keeps the down-side group reshaping by folding through the gate path's
homogeneous factor — `gate_rows /= s_r`, per-channel `alpha_r = 1.702·s_r`
AND per-channel gate clamp `limit_r = 7/s_r`, `down_cols *= s_r` — exact for
every input (`glu' = glu/s`). Design note:
`docs/superpowers/plans/2026-07-23-m3-awq-gate-alpha-fold.md`. Decision gate:
only pursue if the r6 eval shows AWQ still materially behind GPTQ.

**Verification path:** requantize r6 (same 512x2048 contract), then the
stuck-item GPQA probe (`pipeline/sampling_probe.py` item set from the tok64k
run) before any full eval: prediction is r6 non-termination collapses toward
GPTQ's rate. Packet: `M3_AWQ_R6_REQUANT_HANDOFF.md`.

## r7 gate-alpha fold: `gate_smooth_scale` lost under disk offload (fixed, 2026-07-23)

**Symptom:** First r7 smoke (RUN_ID `20260723T092246Z-m3-ddp-awq-smoke-r7-gatealpha`)
completed RC=0 with all markers healthy (fold prepared on 7,296 experts,
`gate_smooth_scale` persisted for every expert, AWQ_LANDSCAPE telemetry
present), but `pipeline.m3_verify_no_updown_fold --mode r7` FAILED (exit 4):
down_proj columns scaled r5-style (relerr med 0.13–0.25 vs base) while every
stored `gate_smooth_scale` was exactly 1.0 (`fold_nontrivial=false`). Direct
weight probes confirmed both fold sides applied (gate rows ÷ s, down cols × s,
implied s med ≈0.81, mutually consistent) — the *weights* were folded, the
*scales* were not persisted.

**Root cause:** Production M3 runs set `offload_folder`, so every module's
`_parameters`/`_buffers` is a compressed-tensors `DiskCache`. The fold
consumer mutated the buffer in place (`buf.mul_(s)`), which only touches the
transient onloaded copy; the disk store keeps the original ones and
`save_pretrained` reads the store. AWQModifier's own weight folds persist
because `_smooth` goes through `update_offload_parameter` — the sanctioned
write-back API. Note the asymmetry: CPU offload shares storage (raw in-place
writes DO persist), so the loss reproduces **only with the disk cache** —
which is why nothing in the pre-smoke unit suite caught it.

**Consequences of the broken artifact:** calibration itself stayed
function-preserving (the derived `swiglu_alpha_vec`/`swiglu_limit_vec` are
plain in-memory attributes computed from the correct product), but the saved
checkpoint is r5-class broken at serve time: folded weights with no way to
reconstruct the per-channel alpha/limit. The r7-mode fold-consistency gate
exists precisely for this class and fail-closed correctly.

**Fix:** `_make_fold_consumer` now writes the composed, clamped scale via
`update_offload_parameter(expert, GATE_SMOOTH_SCALE_NAME, updated)` (works for
offloaded and plain modules) and derives the alpha/limit vectors from the same
persisted value. Regression test
`test_consumer_persists_through_offload_cache` reproduces the loss with a real
`DiskCache` (mechanism guard asserts raw writes still get dropped, so we learn
if upstream semantics change) and asserts the fixed path persists.

**Lesson:** any state an llm-compressor modifier must persist on an
offload-managed module — parameters OR buffers — must go through
`update_offload_parameter`. In-place tensor mutation is silently discarded by
the disk cache, and only functional/persistence gates catch it.

## r8 smoke: NCCL deadlock from rank-sharded FP8 weight-qparam updates (fixed, 2026-07-23)

**Symptom:** First distributed GPTQ + FP8_DYNAMIC mixed-recipe smoke
(`20260723T105742Z-m3-ddp-gptq-smoke-r8-fp8rest`) froze after subgraph 5
(layer 3): all 8 ranks' last log lines at the same instant (11:17:10), GPUs
pinned at 100% for 90+ min with zero output. Per-rank metrics showed GPTQ
compress COMPLETED evenly (48 modules/rank = 384/8) — the hang was after it.

**Root cause:** `QuantizationModifier.on_sequential_epoch_end`'s distributed
branch rank-sharded the weight-qparam update
(`update_qparams(rank_to_modules[rank], "weight")` via greedy_bin_packing).
Each update goes through the patched `DistributedDiskCache.update_offload`,
which ends in `dist.barrier()`. Layer 3 has 6 FP8 modules (4 attention + 2
shared-expert) over 8 ranks → six ranks emitted per-update barriers, two
emitted none → mismatched collective sequences → permanent NCCL spin (100%
GPU = busy-poll, not compute). Never seen before because (a) pure-GPTQ runs
shard 384 M3 expert modules = always divisible by 8, and (b) the
ACTIVATION_OBS qparam path just above already updates all modules on all
ranks.

**Fix:** weights are replicated across DDP ranks, so weight observation is
deterministic — every rank now runs `observe`/`update_qparams` over ALL
modules (identical values; the patched disk cache's source-rank gating
deduplicates the writes). The sharding + `_broadcast_qparam_onloads` helper
were removed. Regression test:
`test_weight_qparam_update_is_rank_aligned`.

**Latent risk noted:** GPTQ's `compress_modules` retains rank-sharded
`update_offload_parameter` calls (uneven bin packing would deadlock the same
way); M3's per-layer module counts keep it even today. If a future model's
expert count isn't divisible by world_size, apply the same treatment (or
make the patched update_offload barrier-free).

**Lesson:** with a collective inside `update_offload`, every code path that
persists parameters must make an IDENTICAL sequence of update calls on every
rank. Audit any rank-conditional `update_offload_parameter` under
distributed disk offload.

## r8 smoke v2: global pack-quantized override corrupts FP8 group at save (fixed, 2026-07-23)

**Symptom:** With the deadlock fixed, r8 smoke v2 ran to completion but the
post-save verification gate failed: dense/shared/attention FP8 modules were
saved as `weight_packed I32` (int4-style packing of 8-bit values, e.g. q_proj
[8192, 1536] = 4-per-int32) instead of `weight F8_E4M3 + weight_scale`.
vLLM's FP8 path cannot load that. Both config groups showed
`format: pack-quantized`.

**Root cause (two-sided trap):** the pipeline forces
`quantization_format="pack-quantized"` at save for W4-family schemes — with
mixed recipes that force stamps EVERY config group. But simply dropping the
override is wrong the other way: compressed-tensors' per-module inference
maps the W4AFP8 expert group to 'int-quantized' (naive int8-style), which
vLLM's W4A8 CUTLASS path cannot load either (that is why the force existed).

**Fix:** `_stamp_mixed_precision_formats(model)` sets `scheme.format` per
group before save (4-bit int weights -> pack-quantized; 8-bit float weights
-> float-quantized); `infer_model_format` respects per-scheme formats and
flattens the model-level format to 'mixed-precision'. vLLM 0.24 resolves
formats per config group (verified in `_quantization_scheme_map_from_config`),
falling back to the global format only when a group has none.
`verify_quant_checkpoint` is now mixed-recipe aware: fp8 group must carry
format='float-quantized'; shared/dense may be fp8 (never int4); fp8 coverage
(attention/shared/dense present) and fp8 weight dtype are fail-closed;
untouched-tensor sampling excludes any module with a weight_scale sibling.

**Lesson:** never apply a global compression-format override to a
multi-config-group model; formats are per-scheme state.

## AWQ grid search hijacked by FP8-schemed balance layers (fixed, 2026-07-23)

**Symptom:** r8a smoke (AWQ int4 experts + FP8_DYNAMIC attention/shared/dense,
`minimax_m3_distributed_r8a_awq_smoke.yaml`) failed the in-pipeline verifier:
"implausible column scales" on layer-3 experts — 27% of significant columns
outside (0.2, 5.0), smoothing-scale median 0.33 with range 0.02–8.5, versus
0.9 tight in the r6 full run. Only the FP8-carrying layers (3, 8) were
affected; smoke layers 31/59 reproduced r6-full scales almost exactly, ruling
out the 8-sample calibration. The smoke also grid-searched
`input_layernorm` on layers 3/8 — a mapping r6 never smoothed.

**Root cause:** all recipe modifiers attach quantization schemes at
initialization, so when AWQ ran, the FP8 modules already carried
`quantization_scheme`. AWQ used bare `hasattr(module, "quantization_scheme")`
to decide (a) whether a mapping is "targeted" at all and (b) which balance
layers join the grid-search pseudo-quant loss and duo-scaling weight means.
FP8 weight quantization error is tiny and nearly scale-invariant, so
including fp8 modules optimizes the smoothing ratio against a meaningless
objective: the fp8 shared expert distorted the post-attention search, and
fp8 q/k/v activated the previously-skipped input_layernorm mapping outright.

**Fix:** `_is_grid_search_targeted` in
`src/llmcompressor/modifiers/transform/awq/base.py`: a module counts as an
AWQ target only if its weight scheme exists AND is not float-typed. Used for
mapping eligibility (`any_targeted`), the grid-search patch list, and
duo-scaling `w_mean`. Float-schemed balance layers still receive the
apply-time compensation fold (verified by the smooth-fold gates), exactly
like unquantized ones did in r6. Regression:
`tests/llmcompressor/modifiers/transform/awq/test_fp8_mixed_recipe.py`.

**Lesson:** "has a quantization scheme" is not "wants AWQ smoothing" — in
mixed-precision recipes the scheme type must gate participation in
smoothing objectives. The checkpoint verifier's column-scale plausibility
band caught this before any eval GPU time.

## r7 gate-alpha serve shim crashed CUDA graph capture; bind markers invisible (fixed, 2026-07-24)

**Symptom:** the r7 serve ABI smoke died during engine init with "Cannot
copy between CPU and CUDA tensors during CUDA graph capture unless the CPU
tensor is pinned" from the patched `apply_moe_activation`. After fixing
that, the relaunch became READY but the fail-closed bind-marker grep still
failed with zero "M3 gate-alpha" lines in the serve log.

**Root causes:** (1) the sidecar scale table stayed a CPU fp32 tensor and
the activation hot path did `ctx["table"].to(input.device)`; the same path
also ran a dynamic-shape `torch.nonzero` for the global->local expert
remap. Both are illegal under CUDA graph capture. (2) The shim logged via
`logging.getLogger("llmc.m3_gate_alpha")`; vLLM's logging config only
wires handlers for the `vllm.*` tree, so worker INFO records were dropped
and a correctly bound server looked unbound to the marker check.

**Fix:** move the table to the weight's device AND reorder rows to
local-expert ids once at bind time (`RoutedExperts.expert_map` is available
in the loader-level `process_weights_after_loading` hook); the hot path is
now pure static-shape GPU ops with a fail-closed device check. Logger
renamed under the vllm namespace (`vllm.llmc_m3_gate_alpha`). Validated:
"M3 gate-alpha: bound 57/57 MoE layers" on workers, coherent 512-token
generation, ABI_SMOKE_RC=0.

**Lesson:** anything a serve shim touches per-forward must be device-
resident and static-shape BEFORE capture: do moves/remaps at bind time. And
a fail-closed marker is only fail-closed if its log path is proven — use
vLLM's own logger namespace inside patched vLLM modules.

## r8 mixed int4+FP8 checkpoint served garbage with a passing smoke (fixed, 2026-07-24)

**Symptom:** the r8 (GPTQ int4 experts + FP8 rest) serve ABI smoke returned
RC=0 while both probes emitted pure repetition garbage ("omensomens...").

**Root causes (stacked):** (1) the smoke only checked HTTP success, not
content. (2) The saved `quantization_config.ignore` carries the GPTQ
recipe's broad quant-layout regexes (`re:.*self_attn[.].*`,
`re:.*(mlp|block_sparse_moe)[.]shared_experts[.].*`,
`re:.*layers[.][0-2][.].*`) — vLLM checks ignore BEFORE targets, so every
FP8 module served as "unquantized" and its raw fp8 bits were cast into
bf16 params with the scales silently dropped (the tolerant M3 loader skips
unknown keys). Verified against the shipped config with vLLM's own
`should_ignore_layer`. (3) The float group's targets are quant-layout
(`language_model.layers...`, `mlp.shared_experts.gate_up_proj`) and can
never match serve/disk names. (4) Architectural: vLLM's M3 plugin fuses
q/k/v with the deliberately-bf16 indexer projections into ONE GEMM weight
(`MinimaxM3QKVParallelLinearWithIndexer`) on sparse layers, so fp8 qkv
cannot serve on this model at all.

**Fix:** `pipeline/reexport_minimax_m3_vllm.py --fp8-serve-fix` dequantizes
attention q/k/v back to BF16 (dropping scales, recomputing offsets),
rewrites the float targets to serve layout (o_proj + shared experts +
dense 0-2), replaces the broad ignores with precise serve-layout entries,
and runs a fail-closed storage-vs-scheme audit using vLLM's own matcher.
The ABI smoke now parses each completion and enforces keyword + 8-gram
diversity gates. Tests: `tests/pipeline/test_reexport_fp8_serve.py`.

**Lesson:** a quantization_config is a serving contract written in the
SERVE model's namespace — every rename between quant layout and disk
layout must rewrite targets and ignores too, and an exported checkpoint
should be audited storage-vs-scheme before GPU time. Smoke tests must
assert on output content, never transport success.

## MiniMax-M3 shared-experts aux-stream CUDA-graph capture IMA (root-caused, fixed 2026-07-24)

**Symptom:** With the shared-experts overlap stream enabled
(`VLLM_DISABLE_SHARED_EXPERTS_STREAM=0`), M3 TP8/EP serves crash ~1/3 of
launches with an illegal memory access during the decode CUDA-graph capture
ladder (trials died at 43-48 of 51), always inside
`torch._C._accelerator_emptyCache()`. Production has run with the stream
disabled since the 2026-07-10 RCA, giving up shared/routed-expert overlap
(~2-5% TPOT).

**False leads (each falsified by a dedicated matrix arm, session
20260724-130708-fixmatrix, 3-6 trials/arm, cyankiwi ckpt, TP8/EP,
graphs+breakable):** the FlashInfer trtllm fused AR+RMSNorm (its workspace
init fires ~1s before the typical IMA and its size gate matches the 43/51
failure point — but NCCL-under-capture 5/6, PDL-off 1/6, and fused-AR fully
OFF still 2/6 clean with the failure merely moved to 48/51: an amplifier,
not the cause); `Tensor.record_stream` deferred-free bookkeeping (skipping
it under capture still failed 1/3 clean).

**Root cause:** the fork's `BreakableCUDAGraph._capture`
(`vllm/compilation/breakable_cudagraph.py`) replicates
`torch.cuda.graph.__enter__`'s pre-capture cleanup (`gc.collect()` +
`empty_cache()`) but drops the `torch.cuda.synchronize()` that upstream runs
FIRST (torch/cuda/graphs.py:239). Each capture is preceded by an eager
warmup pass that runs the shared-experts overlap on the aux stream; without
the sync, those kernels can still be in flight when `empty_cache()` returns
memory to the driver. Under `expandable_segments:True` the release is
`cuMemUnmap`, which — unlike `cudaFree` — does not implicitly synchronize
the device, so the aux stream races the unmap → IMA. This also explains why
`expandable_segments:False` alone fixed it (12/12 clean): `cudaFree`
supplies the missing sync as a side effect.

**Fix:** `pipeline/slurm/patch_vllm_m3_serve.py` target "breakable-capture
pre-cleanup sync" (`LLMC_M3_CAPTURE_SYNC=sync`) restores the upstream
ordering with a `torch.cuda.synchronize()` before the pre-capture cleanup.
Startup-only cost (~51 syncs). Validation: arm H (stream ON,
expandable_segments:True, fused AR legacy) — see tally below. Fallback fix
if ever needed: `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:False`
(arm G, 12/12 + arm FG 3/3), at the price of losing expandable segments'
fragmentation resistance on long serves.

**Validation tally (stream ON unless noted):** streamOFF-legacy 3/3 clean
(prod baseline); legacy 2/3; nccl_graphs 5/6; pdl_off 1/6; fused-AR-off
2/6; record_stream-skip 1/3; **G expSegOff 12/12; FG 3/3; H capture-sync
12/12**. Legacy stream-on failure rate ≈1/3 → 12 clean trials ≈ <1%
false-pass; both fixes clear that bar independently.

**Not yet done before any production default flip:** conc-1 TPOT A/B
(stream-on+H vs stream-off prod) and a paired quality smoke — a capture
race that IMAs when caught could in principle bake a stale pointer into a
graph silently, so correctness must be asserted on outputs, not on
crash-free startup.

**Lesson:** when porting a torch context manager's body into bespoke code,
port its ORDERING, not just its calls — the dropped `synchronize()` was
load-bearing precisely because `cudaFree`'s implicit sync usually hides its
absence. And when a fix works for reasons you can't state (expSegOff), keep
digging: the mechanism (`cudaFree` vs `cuMemUnmap`) pointed straight at the
one-line root cause.

## GLM-5.2 distributed GPTQ: `offload_hessians` defeated by the reduce (fixed, 2026-08-27)

First GLM-5.2 incident, and an **upstream bug** — present verbatim in
`vllm-project/llm-compressor` main when found, with no matching issue.

**Symptom.** 4×H100, world_size 4, GPTQ smoke with `gptq_offload_hessians: true`.
Calibration of the first MoE layer (subgraph 5/79) completed with Hessians on
CPU — 3:46 for 8 samples, the expected PCIe cost — then:

```
gptq/base.py:414 in _maybe_onload_hessian
    self._hessians[module] = self._hessians[module].to(device=device)
torch.OutOfMemoryError: Tried to allocate 144.00 MiB. GPU 3 has a total
capacity of 79.18 GiB of which 4.12 MiB is free. ... 77.36 GiB is allocated
by PyTorch
```

77.36 GiB is the **full unoffloaded Hessian footprint**: 256 routed experts ×
(gate 144 + up 144 + down 16 MiB) = 76.0 GiB, plus attention and shared expert.
The offloading had been undone.

**Root cause.** `_reduce_hessian_to_target_rank` issued one *async*
`dist.reduce` per module and called `wait_for_comms` only after the loop. An
onloaded Hessian stays resident until its reduce is waited on: NCCL holds an
internal reference, so neither `self._hessians.pop(module)` nor
`_maybe_onload_hessian`'s move-back to CPU can release it. Both operations
rebind or drop a *Python* reference to a tensor the collective still owns. So
the loop re-materialized every Hessian in the layer on the accelerator, and
`offload_hessians` bought nothing in the one mode that needs it.

Note that the flag is *not* broken in the non-distributed path, and not broken
during calibration — only where the async reduce pins what the flag just
offloaded. That is why it presented as "offloading works, then OOMs anyway".

**Fix.** Chunk the reduce (`_HESSIAN_REDUCE_ONLOAD_BYTES`, 4 GiB) and release
the accelerator copies only *after* `wait_for_comms`. Chunk boundaries derive
from `module_list` and the Hessian shapes, both identical across ranks, so every
rank issues its collectives in the same order — a rank that chunked differently
would deadlock against its peers. The non-offloaded path keeps the original
single-batch behaviour exactly: its Hessians are already resident, so batching
costs no memory and preserves maximum comm overlap. That is the path MiniMax-M3
was validated on, and it is unchanged.

**Verification** (`tests/.../gptq/test_hessian_reduce_offload.py`, 12 tests, no
GPU): peak resident **108 GiB → 3.94 GiB** across 28 windows on GLM-5.2's 768
expert Linears. The harness was run against the original implementation to
confirm it fails there — a memory test that passes both ways proves nothing.
Tests also pin the ordering invariant, the after-wait release, and that the
non-offloaded path still uses one window.

**Still open: `offload_hessians` is not the answer for the full run.** Layer 5
calibration took 3:46 at 8 samples; that is linear in samples, so ~4 h per MoE
layer at 512 samples, across 75 MoE layers. It unblocks a smoke and nothing
more. Candidate real fixes, in order of preference:
1. `sequential_targets_per_subgraph` (`args/dataset_arguments.py:252`) with
   expert-level `sequential_targets` — an **existing upstream mechanism** whose
   own help text is "Higher values use more VRAM but are faster to calibrate".
   Targeting experts at ~32/subgraph bounds Hessians at ~10 GiB.

   **Investigated 2026-08-27. Mechanically viable, NOT numerically equivalent.**
   The blocker was not the MoE at all: `GlmMoeDsaMoE.forward` wraps the experts
   call inside an untraceable starred `.view(*orig_shape)`, so the AST
   autowrapper swallows `self.experts(...)` into an opaque `torch.fx.wrap` and
   the experts get **zero** graph nodes regardless of `sequential_targets`.
   Splitting that one statement exposes all experts (1 subgraph -> 17; 16 expert
   nodes) and is **provably neutral** — 0.000e+00 across all 104 Linears with
   targets held fixed.

   But partitioning at expert granularity changes GPTQ's output, because the
   greedy topological partitioner **interleaves layers**: measured, 12 layer-3
   modules were compressed *after* layer 4 had already been calibrated (0 in the
   decoder-layer baseline), because `shared_experts(residuals)` depends only on
   the layer input and can be scheduled early. Layer 4 is then fitted against a
   partly-full-precision upstream. Layer 3's own weights are bit-identical; only
   downstream moves. Held-out quality is close (mse/rel/KL within 0.4-0.8%,
   top-1 agreement -0.59pp) but consistently worse. Not inherent — constraining
   the partition so no layer *k+1* module precedes layer *k*'s completion should
   restore exactness — but that is an upstream partitioner change.
   Also note the `"Expected N subgraphs, but only traced M"` warning ignores
   `targets_per_subgraph`, so a correct run using this knob looks broken.
2. Expert-parallel calibration (shard experts, all-gather activations).
   ~19 GiB/rank at EP=4, and **preserves the decoder-layer partition**, so the
   compression schedule — and therefore the result — is identical by
   construction. Chosen over option 1 for that reason. Core landed 2026-08-27
   (`modeling/moe/expert_parallel.py`, bitwise identical to the unsharded
   forward at world sizes 2/4/8); config plumbing and GPU memory verification
   still outstanding.

**AWQ does NOT share this blocker — do not port the EP work to it.** AWQ
accumulates no Hessian. Its cross-rank statistic is a *vector*: `x_sum`
`[in_features]` fp32 plus a scalar count, accumulated **on CPU**
(`modifiers/transform/awq/base.py:604`, `.float().sum(dim=0).cpu()`). That is
~24 KiB per smooth point against GPTQ's 144 MiB per module, and on the host
rather than the accelerator — so there is nothing to shard. Distributed
aggregation is already implemented and complete: `_allreduce_data_sum([x_sum,
count])` for the smoothing scale, and the same for the grid-search `loss` /
`num_elements`. `offload_device` auto-defaults to CPU when a MoE model is
detected, keeping the per-parent activation caches off the card. Validated at
full scale on MiniMax-M3 across r2/r5/r6/r7 (8-rank torchrun; r2 alone ~6.8 h
calibration, 21,888 quantized Linears). AWQ's scaling risks are a different
class entirely — VMA/shm exhaustion, and silent smoothing-fold loss under disk
offload — not GPU memory.

**Lesson.** An async collective is a reference holder. Any memory-management
code that drops a reference to free memory must first prove nothing else owns
the tensor — and "I popped it from the dict" is not that proof.

## Environment reproducibility: `pip freeze` does not record source patches (2026-08-27)

Immediate recurrence of the 5.14.1 `weight_map` bug above, from a new direction,
and the more transferable of the two lessons.

GLM-5.2 was being quantized on the sglang image's transformers 5.12.1 while
MiniMax-M3 was validated on 5.14.1. Pinning the stack to
`envs/m3-quant-freeze.txt` (the verbatim freeze of the M3 quant venv) fixed the
version skew — and *introduced* the save bug, because the M3 venv carried the
one-line repair as an **in-place edit to the installed `modeling_utils.py`**.
A `pip freeze` records versions, not edits. The manifest faithfully reproduced
`transformers==5.14.1` and silently dropped the patch. Worse, the image's older
5.12.1 had been *safe* — our save shims cover that path — so pinning "closer to
M3" moved the environment backwards on this axis.

`assert_transformers_offloaded_save_healthy()` caught it 3 minutes into a 4-GPU
job, before any calibration spend. The gate did exactly the job it was written
for; the note above ("catches a venv rebuild that silently drops the hotfix")
predicted this precisely.

The repair script named in the 2026-07-18 entry, `upgrade_quant_venv_tf5141.sh`,
**was never committed** — it existed only on the retired cluster and went away
with it. So the fix had to be re-derived from the gate's error message and the
notes here.

**Resolution.** `envs/hotfix-transformers-sharded-save.py`: idempotent, verifies
by re-reading the file from disk rather than trusting the write, and fail-closed
— if the marker is absent or appears more than once it writes nothing and exits
non-zero rather than pattern-matching a changed upstream line. Wired into both
`pipeline/k8s/quantize-glm52.yaml.tmpl` and `envs/setup-m3-quant-venv.sh`, so
neither a fresh pod nor a rebuilt venv can lose it. Covered by
`pipeline/tests/test_sharded_save_hotfix.py` (15 tests with the existing gate
tests), which also pins the two copies of the `shimmed`/`healthy`/`broken`
classifier against drift — the classifier is duplicated because the hotfix must
run before the repo is installed and cannot import `pipeline.quantize`.

**Lessons.**
- A freeze manifest is not an environment. Any repair applied *inside*
  site-packages must live in version control and be re-applied by a script, or
  it is lost at the next rebuild — silently, because the manifest still matches.
- A repair script kept only on the cluster it was used on is a repair that will
  be re-derived from scratch. Environment fixes belong in the repo, not the box.
- "Pin to the validated environment" is not automatically safer than the status
  quo. Here it removed a shim that was protecting us, so the pin needed its own
  gate — which, fortunately, already existed.

## GLM-5.2 distributed PTQ: `max_memory.cpu` and the ~3.7 h setup (2026-08-28)

**Symptom.** GLM-5.2 AWQ on an exclusive 8xH100 node spent ~3.7 h before
calibration even started, against MiniMax-M3's 71 min for a comparable run
*including the save*. No error, no OOM — just slow.

```
weight load                                          66 min
"Dispatching model", 78460 tensors at 8-9.5 it/s     ETA 2h34m
```

**Root cause: `model.max_memory.cpu`, raised from M3's 32e9 to 900e9 earlier in
this work.** That knob decides how many bytes the dispatch phase *copies*.
Everything inside the budget is memcpy'd into `/dev/shm` by
`DistributedCPUCache.offload` (`share_memory_()` + `broadcast_object_list` +
`barrier`, per tensor); everything beyond it becomes a near-free symlink to the
original HF blob. 900e9 copies ~838 GiB; 32e9 copies ~30 GiB.

Copy-bound, not per-tensor-bound: 6197 tensors dispatched corresponded to
100 GiB of shm (~16.5 MB/tensor against a 19 MB average tensor). A purely
per-tensor cost at the observed 8.15 it/s would have made M3's ~40k tensors take
82 min on their own, which its 71 min total rules out.

And the copy buys almost nothing, because `DiskCache.onload` reads straight from
the symlinked original through the page cache:

```python
with safe_open(weight_info["safetensors_file"], framework="pt", device=device) as file:
    onloaded = file.get_tensor(weight_info["weight_name"])
```

A disk-placed unmodified weight served from cache is about as fast as an shm
copy. The copy upgrades *best-effort* residency to *guaranteed* residency,
nothing more.

**Fix.** `cpu: 32000000000` on both GLM-5.2 smoke arms (`a098fb56`), plus
`stop_after_last_target` to skip the untargeted tail of the sequential walk
(`3eaad356`, smoke-only, fail-closed — see below).

**Four things that were measured and are worth not re-deriving.**

1. **`auto_offload` is not `auto`; GPUs contribute ZERO residency.**
   `compressed_tensors/offload/load.py` intercepts it: "same as `auto`, but only
   cpu/disk are visible", rewriting `max_memory` to a cpu-only dict. Measured in
   situ: 1509 MiB per GPU (bare CUDA context) for the entire load on all 8 GPUs.
   Never size this budget as "VRAM + CPU". Going 4 -> 8 GPUs buys no residency.

2. **The offload folder is not a copy of the model.** 36838 of its ~45k entries
   are symlinks straight to
   `/mnt/cephfs/.hf-cache/models--zai-org--GLM-5.2/blobs/`; only genuinely
   modified tensors become real files. Do not size `max_memory.cpu` to "avoid a
   spill" — there is no spill to avoid. (Beware stale residue when diagnosing:
   the 39.4 GB of real files present during the 2026-08-28 run were all
   timestamped 2026-08-27 19:21-23:10, from the previous day's failed runs, and
   were briefly misread as evidence of the live run copying weights.)

3. **Page cache is charged to the pod's memory cgroup.** `memory.current`
   697 GiB broke down as anon 12 / file 683 / shmem 15.6. So shm and page cache
   are *substitutes*, not additions — total residency is bounded by model size +
   working set (~1500 GiB) however the budget splits it, and the POD MEMORY LIMIT
   is what actually bounds it. Peak over the whole run was 980 GiB against a
   1700 GiB limit, so eviction was never the risk.

4. **CPU is NOT the bottleneck**, despite the pod having a 48-core quota against
   M3's 192: `cpu.stat` showed `nr_throttled 12` of 54566 periods (0.02%, 26 s
   total) at load average 9.36. The ranks wait on I/O and collectives.

**The real regression is storage, and no `max_memory` value fixes it.** cephfs
measures 31 MB/s single-stream with a **28 ms** TCP RTT to the monitor — it is
latency-bound, not bandwidth-bound — against 135-260 MB/s aggregate across 8
ranks. That is roughly 4x worse than the retired cluster's `/mnt/nfs`, and it is
why M3 finished in 71 min and GLM-5.2 does not. The sequential walk is
storage-bound and therefore *sample-independent*: 86 min for 35 untargeted
layers is 2.5 min/layer = 19 GB/layer at ~127 MB/s, independently reconfirmed on
2026-08-28 at **120-131 s per untargeted MoE layer**, timed directly from
subgraph transitions on the 8-GPU run.

**Staging onto flash is NOT available. Do not pursue it.** The cluster does have
an all-flash `sc-file-nfs-vast-my` class with exactly the mount options this
workload wants (`nconnect=16`, 1 MiB rsize/wsize, nfsvers=4.1 — 16 parallel TCP
streams would defeat the single-stream latency limit). But it is region-locked:

```yaml
allowedTopologies:
- matchLabelExpressions:
  - key: topology.kubernetes.io/region
    values: [my]
```

and every GPU node is in `ca-van3`:

```
aicloud-infermesh-test-ca-gpu01..06    ca-van3   ca-van3-a
aicloud-infermesh-test-my-cpuworker*   my        my-a
```

So a VAST PVC cannot bind to a pod on any GPU node — which is also why port 2049
was unreachable from the GPU pod: no route, not a network policy. The existing
`benchmark-datasets` and `registry-cache` PVCs on that class are reachable only
from the Malaysian CPU workers. `sc-file-ceph-ca` is the only storage GPU work
can use here, so 31 MB/s single-stream is the floor.

What IS available, ordered by how much is ours to control:
1. **Read concurrency — and specifically NOT "hide I/O behind compute".**
   Measured on the 8-GPU run mid-walk: GPU utilisation was 0-2% in 9 of 10
   samples over 20 s (one sample at ~41%), i.e. **eight H100s idle ~90% of the
   time waiting on cephfs**. There is essentially no compute on an untargeted
   layer to overlap a prefetch with, so framing the fix as "prefetch the next
   layer during this layer's compute" is wrong.

   The lever is simply MORE REQUESTS IN FLIGHT, because cephfs is latency-bound
   (28 ms RTT) and scales with concurrent readers: 31 MB/s single-stream ->
   135-260 MB/s at 8 streams. Today `DiskCache.onload` issues one
   `safe_open`/`get_tensor` at a time, so 8 ranks = 8 streams = the 152 MB/s we
   observe (19 GB/layer in ~125 s). A layer holds ~768 expert tensors; reading
   them through a small per-rank thread pool multiplies in-flight requests by the
   pool size. Prefetching the next layer only doubles concurrency, so
   parallelising WITHIN a layer is the larger win.

   Note `sequential_prefetch` already exists but covers the ACTIVATION cache, not
   weights.

   About the page cache, stated carefully because an earlier revision of this
   entry got it wrong in a way that would cause harm if acted on. It said the
   cache "is useless for this access pattern" and that each layer is "touched
   exactly once". The second half is false: all 8 ranks work the SAME layer
   simultaneously (each needs the whole layer in its own GPU because each holds
   different calibration samples), so every layer is requested EIGHT times. The
   first request pulls from cephfs and **the other seven are page-cache hits.**
   The arithmetic confirms it: 19 GB of unique bytes per ~150 s is ~127 MB/s,
   the cephfs rate, whereas 8 x 19 GB in 150 s would demand ~1 GB/s.

   So DO NOT try to bypass or aggressively drop the page cache — losing the
   cross-rank sharing would multiply cephfs traffic by 8. What is genuinely
   useless is the ACCUMULATION: the cache sat at 757 GiB, its ceiling, evicting
   as fast as it filled, and nothing ever revisits layer 5 once the walk is on
   layer 20. Only the current layer (and usefully the next) needs to be resident
   — tens of GB, not hundreds. A larger pod memory limit therefore does not speed
   the walk up, but a much smaller one would.

   Corollary for the concurrency fix above: because 7 of 8 reads per layer are
   cache hits, the ranks are NOT 8 independent streams pulling unique bytes. The
   unique-byte fetch achieves ~127 MB/s against 31 MB/s single-stream, i.e. only
   ~4x effective concurrency, which is why aggregate throughput sits well below
   the 260 MB/s seen elsewhere. The headroom is real.
2. **`stop_after_last_target`** for anything layer-restricted — measured worth
   35 x 131 s = ~76 min on this smoke.
3. **Asking cluster admins for a `ca`-region flash class.** A conversation, not a
   config change.

**Do not oversell the revert.** A small budget *shifts* cost from dispatch to
the walk rather than removing it. Projected after load: `32e9` -> ~5 + ~190 =
~195 min; `900e9` -> ~155 + ~82 = ~237 min. Worth ~45 min, not hours. The large
win came from `stop_after_last_target`, not from the budget.

**`stop_after_last_target` (smoke only).** A layer-restricted smoke quantizes 2
of 78 layers but still propagates through all 78; trailing subgraphs exist only
to feed subgraphs after them. Skipping layers 43-77 saves ~87 min per run. It is
**fail-closed** because the failure mode is silent — stopping one subgraph early
leaves a layer that should have been quantized in BF16, and the checkpoint still
saves and still loads. `last_subgraph_with_targets()` returns `None` (walk
everything) unless every module carrying a `quantization_scheme` is attributable
to a subgraph at or before the cut. It reads the graph's `call_module` targets as
name prefixes rather than `subgraph.submodules()`, because
`self.experts(...).view(*orig_shape)` cannot be traced (fx does not handle
`*args` unpacking) so the AST autowrapper wraps the whole expression and GLM's
expert modules get zero graph nodes. Covered by
`tests/llmcompressor/pipelines/test_stop_after_last_target.py` (10 cases,
mutation-tested: removing the coverage invariant reproduces the
silent-unquantized-layer bug).

**Lessons.**
- Measure the cost of the fix, not just the cost of the problem. Raising
  `max_memory.cpu` was justified by a real measurement (86 min of storage-bound
  propagation) but the remedy's own price — 2h34m of dispatch — went unpriced
  for two runs.
- `sum(VmRSS)` is not the memory a cgroup limit enforces. Shared pages are
  counted once by the cgroup and up to N times across N ranks; the first sampler
  here read 68 GiB while `memory.current` was 697 GiB, and its growth ratio
  would have projected a false eviction.
- Timestamp the artifacts before drawing conclusions from a shared scratch
  directory. Stale residue from a failed run looks exactly like live output.

### Follow-on: the scale audit had the wrong norm semantics for GLM-5.2 (fixed 2026-08-28, `727bf6f0`)

Caught by inspection before the run reached it, not by a failure.
`compensation_error()` in `pipeline/m3_checkpoint_scale_audit.py` hardcoded the
Gemma/MiniMax-M3 gain form:

```python
# MiniMaxM3VLRMSNorm is Gemma-style: smooth the effective 1 + weight.
base_gain = 1.0 + base_norm
cand_gain = 1.0 + candidate_norm
```

GLM-5.2's `GlmMoeDsaRMSNorm` applies plain `self.weight * hidden_states` — that
is asserted in `KNOWN_ORDINARY_NORM_CLASSES`, verified by reading its forward on
2026-08-27. Auditing a plain norm with the offset form does not error; it derives
a **wrong implied smoothing scale**, so a perfectly consistent fold reports a
large relative L2 and the fail-closed post-save gate rejects a healthy run at
the very end, after all the calibration has been paid for.

Measured on a synthetic consistent plain-norm fold:

```
audited correctly (offset 0.0):  rel_l2 = 0.000e+00
audited as offset (offset 1.0):  rel_l2 = 1.694e-01
gate threshold 2.0e-02        ->  FAILS
```

`1.694e-01` sits inside the M3 **"lost fold" reference band (0.09–0.27)**, so
this would not merely have failed — it would have looked exactly like the
catastrophic r2 lost-smoothing signature and sent someone chasing a numerics bug
that did not exist. The GLM-5.2 smoke does reach this gate: it runs inside
`if save_checkpoint:` over `range(3, 60)`, covering both targeted layers.

**Fix.** `norm_gain_offset` is now a required keyword-only argument on
`compensation_error()` and `audit_checkpoint()` (no default — a new family must
state the form), a required `--norm-gain-offset` on the CLI, and is recorded in
the output JSON (`schema_version` 2). `resolve_norm_gain_offset(model)` in
`pipeline/quantize.py` derives it from `KNOWN_OFFSET_NORM_CLASSES` /
`KNOWN_ORDINARY_NORM_CLASSES` — the existing single source of truth, where each
entry is an assertion that someone read the class's forward — and returns `None`
for an unclassified norm *or* a model mixing both forms, in which case the gate
skips with a printed reason instead of guessing. Guessing is unsafe in both
directions. Tests: `pipeline/tests/test_scale_audit_norm_gain_form.py` (15).

**For GLM-5.2, run the audit with `--norm-gain-offset 0.0`.** A rel_l2 in
0.09–0.27 *with that flag set* is a genuine lost fold; ~3e-3 is a consistent one
(M3 r9 reference).

**Lesson.** A verification tool inherits the assumptions of the model it was
written for, and a verification tool that is wrong is worse than none: it either
fails good work or blesses bad work, and both cost more than having no gate. When
reusing an `m3_*` gate on a new family, check every architectural constant in it
— the suffix table (already once "had never successfully run against these
checkpoints") and the norm semantics were both M3-specific here.

### The ~53 min "Loading weights" phase reads every tensor and throws almost all of them away (2026-08-28)

**Not filed upstream.** Needs sign-off, and the fix is not a two-line patch — see
"Is it fixable" below.

**Symptom.** GLM-5.2 spends ~53 min in transformers' `Loading weights` phase
(58,794 tensors) before calibration. During it, GPUs hold 1509 MiB — bare CUDA
context, nothing loaded — and the page cache fills at 135-260 MB/s, i.e. several
hundred GB is genuinely read. The end state is meta tensors plus an index of
symlinks pointing back at the original HF blobs. The sequential walk then re-reads
every layer from those same files anyway (measured 120-150 s/layer).

**What the load legitimately has to produce**, and it is all cheap:
1. the module tree (built on meta, no data),
2. `disk_offload_index`: `{param -> (safetensors_file, weight_name, dtype)}`,
3. real bytes for the small non-offloaded minority (~4 GB in shm at `cpu=32e9`,
   plus embeddings/norms).

Item 2 costs essentially nothing —
`transformers/integrations/accelerate.py::accelerate_disk_offload` builds it by
dict manipulation over `model.safetensors.index.json`:

```python
weight_map = {k: os.path.join(folder, v) for k, v in sharded_metadata["weight_map"].items()}
...
disk_offload_index = {target_name: {"safetensors_file": weight_map[source_name], ...}}
```

Zero weight reads. And its docstring documents the intended fast path:

> If reading from a safetensors file, parameters which do not need any special
> WeightConverter operation during loading (i.e. they are used as-is, or only
> renamed) will be mapped to where they already reside on disk. Otherwise, the
> parameters will be resaved inside `disk_offload_folder` during loading.

**We are on that fast path.** Verified three independent ways:
- `ARCH_TO_2D_MAPPINGS` entries are `WeightRenaming`, the class transformers
  exempts (`renamings = [e for e in weight_mapping if isinstance(e, WeightRenaming)]`).
- our own guard in `modeling/moe/conversion_mappings.py` that warns
  `"Linearized model performs a weight conversion during loading"` never fired —
  grep count 0 across the whole run log.
- the offload folder holds 119k symlinks and ~0 new real bytes, which is what the
  no-resave path looks like.

So nothing is converted and nothing is resaved. **The waste is that the read
happens upstream of the decision not to use it.**
`transformers/core_model_loading.py`:

```python
for first_param_name, mapping in tqdm(param_name_to_load.items(), desc="Loading weights"):
    realized_value = mapping.convert(...)             # materializes the tensor
    for target_name, param in realized_value.items():
        param_device = get_device(device_map, target_name)
        if param_device == "disk" and (...):
            disk_offload_index = offload_and_maybe_resave_param(
                target_name, param, ..., disk_offload_index, mapping)
```

and inside that (`core_model_loading.py:1376`):

```python
if target_name not in disk_offload_index or isinstance(applied_ops, WeightConverter):
    disk_offload_index = offload_weight(param, target_name, disk_offload_folder, disk_offload_index)
return disk_offload_index      # already indexed and no converter -> do nothing
```

`mapping.convert()` realizes the value for EVERY parameter; only afterwards does
the loop discover the parameter is disk-destined and already covered by the index,
and then does nothing with what it just read. For GLM-5.2 that is ~99% of 58,794
iterations.

**Supporting evidence that this is per-tensor, not byte-driven** (and a warning
against a wrong hypothesis): load time barely moved between `cpu=900e9` (66 min)
and `cpu=32e9` (53 min) despite 26x less CPU placement. 58,794 tensors in 53 min
is 60 ms/tensor, ~2x the measured 28 ms cephfs RTT. An earlier hypothesis
attributed the ~700 GB of page cache to kernel readahead on header reads; that
arithmetic happened to fit but the code trace above is the real answer — the bytes
are genuinely read tensor data, faulted in by `mapping.convert` and dropped.

**Is it fixable?** In principle yes and cleanly, because the test already exists —
it is simply on the wrong side of the read. Hoisting
`param_device == "disk" and target_name in disk_offload_index and not
isinstance(applied_ops, WeightConverter)` above `mapping.convert()` would skip
materialization entirely for those params. But `mapping.convert()` may have side
effects the loop depends on — populating `loading_info` (missing/unexpected keys),
resolving tied weights, per-tensor dtype validation — so this needs reading the
whole `convert` path before touching it. Do not attempt as a quick patch.

**Worth, here:** ~53 min per run, on every run including production. Worse for
larger models or higher-latency storage.

**Lesson.** "Necessary phase" and "necessarily expensive phase" are different
claims. The load IS required — the module tree and the offload index are what the
whole run reads through — but what it must produce is metadata plus a few GB. Cost
attribution needs the code path, not the wall clock: three separate hypotheses
here (MoE weight conversion forcing a resave, kernel readahead, byte-driven
placement copying) all fit the timing and all were wrong.

### Tooling hazard: this shell collapses `\\` to `\` inside heredocs, and it silently corrupts shell templates (2026-08-28)

**Environment fact.** In this development shell, a `python - <<'PY' ... PY` heredoc
does **not** deliver backslashes literally: `\\` arrives as `\`. A single-quoted
heredoc is supposed to be literal, so this violates the reasonable expectation
and there is no warning.

**Cost incurred, in order:**
1. Three string-anchor edits matched zero times. I attributed this to CRLF and
   then to invisible characters before finding the real cause.
2. A `sed` continuation in `pipeline/k8s/launch-quant-glm52.sh` became a literal
   `"\n"`. Caught before launch.
3. **A wasted 6-GPU launch.** The same class of edit put a literal two-character
   `\n` into `pipeline/k8s/quantize-glm52.yaml.tmpl` where a line continuation
   belonged:

   ```
   --set "quantization.method=$METHOD" \n                $EVIDENCE_FLAG
   ```

   Bash parsed `\n` as an escaped `n` and passed a bare positional `n` to
   argparse. All 6 ranks died at startup with
   `run.py: error: unrecognized arguments: n`, killing the calibration-scaling
   probe (`quant-glm52-awq-20260828t121642z`) two minutes in — before it read a
   single weight. Fixed in `9c34c110`; relaunched as
   `quant-glm52-awq-20260828t122858z`.

**Rule.** Any edit whose text contains a backslash goes in a file (Write/Edit),
never through a heredoc. This was already known when (3) happened: the launcher
edit in (2) had deliberately been routed through a file
(`scratchpad/add_sed_line.py`, whose docstring records the reason), and then the
template edit went through the buggy path anyway.

**Detection, since the rule will be broken again.** `grep '\\n'` cannot find this
— grep's own BRE reduces `\n` to `n` and matches every line containing a letter
`n`. That false-clean result nearly closed the investigation. Use a bracket
expression, `[\]n`, or `grep -F`. Better: after editing any rendered template,
diff the *rendered* output rather than trusting the source, which is how this was
finally confirmed — the old and new manifests sat side by side in
`.k8s-rendered/`, one with the literal `\n` and one with a real continuation.

**Lesson about the gate that did not catch it.** The launcher has a fail-closed
check for unsubstituted `@@TOKEN@@` placeholders, and it passed: substitution was
never the problem. A template can render to *valid YAML containing a broken
shell script*, and nothing between the edit and 6 allocated GPUs looked at the
shell. A `bash -n` on the extracted container command would have caught it in
milliseconds.

**Fixed by gate 0** in `pipeline/k8s/launch-quant-glm52.sh`: the rendered
container script is now extracted from the block scalar (awk, not a YAML parser —
the only python on PATH in this dev shell is the Windows Store stub, and a launch
gate must not depend on an interpreter that may be absent) and checked two ways
before `kubectl apply`:

1. `bash -n`, for genuine syntax errors;
2. `grep '[\]n'`, because **a bare `n` argument is syntactically legal bash** and
   `bash -n` alone would have passed the exact manifest that wasted the launch.

Mutation-tested in both directions: the good template prints
`gate 0: rendered container script parses as bash OK`; reintroducing the literal
`\n` and, separately, an unterminated double quote each abort with exit 1 and a
message naming the extracted body file. The template was then restored and
verified byte-identical to the committed version.

### Second hardcoded-M3-assumption gate: verify_quant_checkpoint rejects healthy GLM-5.2 checkpoints (2026-08-28)

**Found by inspection while the GLM-5.2 AWQ smoke was still saving**, i.e. before
the gate ran. Same defect class as the scale audit's hardcoded norm-gain form,
found the same way, in a different file — which is the reason this entry exists
rather than a one-line fix note.

**The bug.** `pipeline/verify_quant_checkpoint.py::_EXPECTED_IGNORE_SUBSTR` is a
hardcoded MiniMax-M3 keep-bf16 list, asserted unconditionally:

```python
for sub in _EXPECTED_IGNORE_SUBSTR:
    if not any(sub in p for p in ignore):
        _fail(f"expected ignore pattern containing '{sub}' missing from config", errors)
```

Five of its nine entries name modules that **do not exist in GLM-5.2** —
`vision_tower`, `multi_modal_projector`, `patch_merge`, `block_sparse_moe`,
`indexer`. A healthy GLM-5.2 checkpoint therefore collects five `[FAIL]` lines.

**Why that is not cosmetic.** `pipeline/quantize.py:1278` calls this from
`assert_quant_checkpoint_verified`, which raises on a non-zero return:

```python
rc = verify(Path(ckpt), check_tensors=True, dequant_base=dequant_base)
if rc != 0:
    raise RuntimeError("quant checkpoint verification gate FAILED ... do not serve or evaluate this checkpoint")
```

and the branch guard is `if save_checkpoint:` (:1173) — **not** a partial-layer
check, despite the `else` arm's comment about smokes. So any GLM-5.2 run that
saves a checkpoint ends with a fail-closed gate telling the operator the
checkpoint is corrupt and must not be served. The checkpoint is fine; the gate is
wrong about which model it is looking at.

**What was NOT wrong**, checked before changing anything, because the cheap
assumption was that the whole file was M3-shaped:
- The expected-quantized coverage section derives `sparse_layers`, `n_experts`,
  `proj_names` and `attn_names` from the checkpoint itself (`:289`), so it is
  model-agnostic by construction. GLM-5.2 ignores attention wholesale, which
  makes `attn_names` empty and the per-layer attention assertion vacuous rather
  than false — correct behaviour.
- `_NORM_MAX_DELTA = 5.0` is a loose corruption bound, not a fold computation, so
  unlike the audit script this file does **not** encode the `1 + w` norm form. Its
  comment asserts the offset-norm rewrite happens universally, which is an
  M3-only claim, but nothing computes on it.
- The `"sparse layers 3-59"` heading was a stale print label only.

**Fix.** `verify(..., expect_ignore=None)` plus `--expect-ignore-preset {m3,glm52}`
and repeatable `--expect-ignore`. `None` keeps the M3 list, so every existing
caller is unaffected. An explicitly empty list raises rather than passing
vacuously — the failure mode to avoid here is someone silencing a wrong-model gate
into a gate that checks nothing. 15 tests in
`pipeline/tests/test_verify_expect_ignore.py`, including a characterization test
asserting the M3 preset still fails 5x on GLM-5.2 (if that stops failing, the
presets have been conflated) and drop-one mutation tests proving the GLM preset is
a real gate.

**Post-hoc verification of the in-flight smoke** must therefore use:

```
python -m pipeline.verify_quant_checkpoint --ckpt <ckpt> --check-tensors \
    --expect-ignore-preset glm52
```

**Lesson.** Fixing one hardcoded-M3 gate is evidence that the others exist, not
that they don't. `m3_checkpoint_scale_audit.py` was corrected earlier today for
the norm-gain form; the same afternoon's read of a *different* gate found the same
class of defect. Both were caught by inspection before the run reached them, and
both would have presented as a catastrophic numerics/corruption failure on a
healthy checkpoint. The remaining M3-shaped gates in `pipeline/` deserve the same
pass before the full GLM-5.2 run, not after it fails.

### GLM-5.2 AWQ leaves the MoE router uncompensated, so smoothing changes expert routing (2026-08-28)

**Status: real defect, root-caused, NOT yet fixed. Blocks the production run.**
Found by post-hoc verification of the AWQ smoke
(`quant-glm52-awq-20260828t070917z`), whose checkpoint saved completely and whose
own gates then failed.

**Measurement.** `audit_checkpoint` on the saved checkpoint vs the base snapshot,
run under BOTH norm-gain forms to discriminate a wrong-form artifact from a real
inconsistency:

| component | offset 0.0 (ordinary, correct for GLM) | offset 1.0 (M3 offset form) |
|---|---|---|
| layer 3 `shared_gate_up` | **2.06e-2** | 2.32e-1 |
| layer 42 `shared_gate_up` | **2.15e-3** | 8.19e-2 |
| layer 3 `router` | 2.42e-1 | 6.75e-3 |
| layer 42 `router` | 1.08e-1 | 2.66e-2 |
| implied scale mean (layer 3) | 0.779 | 0.9956 |

`GlmMoeDsaRMSNorm` applies plain `output * weight`, so 0.0 is the correct form
(it is registered in `KNOWN_ORDINARY_NORM_CLASSES`). Under it, the shared experts
match the norm-implied fold at **2.15e-3** at layer 42 -- a textbook consistent
fold, against a 0.02 threshold -- while the router is off by 1.08e-1 to 2.42e-1.

The offset-1.0 column is a trap and worth understanding: it makes the router look
*healthy* (6.75e-3). That is because the wrong form collapses the implied scale to
0.9956 ~= 1, so the check degenerates into "candidate router ~= base router" --
which is trivially true precisely BECAUSE the router was never touched. A gate run
only under the wrong form would have reported the router fine and the shared
experts broken, i.e. exactly backwards.

**Root cause, and it is documented in our own tree.**
`src/llmcompressor/modifiers/transform/awq/mappings.py` registers
`_mla_mixed_dense_moe_mappings` for `GlmMoeDsaForCausalLM`, and its comment says:

> The dense layers have no mlp.gate router, so mlp.gate must NOT appear in the
> balance layers: match_modules_set groups a mapping per layer only when every
> balance pattern matches inside that layer [...] **Dropping mlp.gate costs
> nothing: the router is never quantized (it is in every recipe's ignore list),
> so it was never a legitimate balance layer to begin with.**

The constraint is real -- including `mlp.gate` globally does break resolution,
because dense layers 0-2 have no router and `match_modules_set` never closes the
set. **The justification for the workaround is wrong.** It conflates two
different requirements:

* whether the router must be **quantized** -- it must not, and the ignore list is
  correct;
* whether the router must be **compensated** -- it must, and quantization has
  nothing to do with it.

AWQ smoothing divides the smooth layer (`post_attention_layernorm`) by a
per-channel scale `s` and multiplies every consumer by `s`, so that
`(x/s) @ (W*s)^T == x @ W^T`. The router consumes that norm's output exactly like
the experts and shared experts do. Leaving `W_router` untouched while the norm is
divided by `s` means the router sees inputs scaled by `1/s` **per channel**, so
its logits change non-uniformly, so **top-k expert selection changes**. An
unquantized consumer of a rescaled input still needs the compensation; being
exempt from quantization does not exempt it from algebra.

Evidence that the norm really was rescaled and this is the AWQ fold rather than
some unrelated drift: the shared experts were multiplied by exactly the
norm-implied 0.779 (residual 2e-3). Same `s`, same layer, one consumer
compensated and one not.

**Why the shared experts are compensated but not the router:** the mapping's
balance list contains the expert and shared-expert projections and omits only
`mlp.gate`.

**Proposed fix** (not applied yet; it changes quantization numerics and needs a
validation run): keep `mlp.gate` out of the mapping that must also match dense
layers, and add a **MoE-layer-scoped** mapping that includes it -- the technique
`dynamic_mappings.py` already uses, building `re:.*layers\.({moe_re})\.` from
`first_k_dense_replace`. Note `dynamic_mappings.py`'s existing MoE entry cannot be
reused as-is: its balance patterns are Qwen-style (`moe.gate`,
`share_expert.gate_proj`) and do not match GLM's `mlp.gate` /
`mlp.shared_experts.gate_proj`.

**Also flagged by the same comment, still unfixed:** `DeepseekV3ForCausalLM` has
`first_k_dense_replace=3` and is still pointed at `_deepseek_mappings`, which does
include `mlp.gate`, so it is expected to fail resolution the same way GLM-5.2
originally did.

**What this does NOT explain**, kept separate because the smoke used only 32
samples x 512 tokens and low sample count is a legitimate alternative explanation
for numeric-margin failures:

* `dequant mismatch model.layers.3.mlp.experts.50.up_proj: resid=0.323` against
  `_DEQUANT_MAX_RESID = 0.25`. That threshold was calibrated on **MiniMax-M3 r9
  full calibration**, where a healthy checkpoint fits at 0.09-0.12. Our run is a
  different model at 1/64 the calibration tokens, and AWQ scales fitted from
  16,384 tokens are noisier, which raises W4 reconstruction residual. Not evidence
  of corruption on its own; re-check on a higher-sample run before treating it as
  a defect.
* `shared_gate_up` at layer 3 reading 2.06e-2 against a 0.02 threshold -- 3% over,
  while layer 42 reads 2.15e-3. That marginality is also consistent with noisier
  scales at low sample count.

**What passed cleanly and is unaffected by sample count:** 12 sampled identity
tensors and 12/12 norm tensors match the base bitwise/allclose, so there is no
stray-write or offload-corruption problem; expert coverage, packed weights and
finite scales are all structurally sound.

**Lesson.** A workaround's justification needs to be checked against the algebra
it is waving away, not just against the error it silences. "The router is never
quantized" is true and irrelevant; it made a function-preservation bug look like a
free simplification, and the bug then survived into a checkpoint. Related: run
consistency gates under the *correct* architectural form -- the wrong form here
did not merely weaken the gate, it inverted which component looked broken.

**Confirmed from the saved checkpoint: the router is NOT quantized, which is
correct.** Read directly from the index and config:

```
quant_method: compressed-tensors   format: mixed-precision
model.layers.3.mlp.gate.weight                 <- plain bf16, packed=0
model.layers.42.mlp.gate.weight                <- plain bf16, packed=0
model.layers.3.mlp.experts.0.gate_proj.weight_packed / _scale / _shape
```

Quantizing the router WOULD be wrong -- it produces the top-k selection and 4-bit
logits would perturb routing directly -- and we do not. The ignore list (expanded
to 56,817 explicit module names, which is why config.json is 2.8 MB) covers it.

This sharpens the defect rather than softening it: the router is correctly exempt
from quantization and incorrectly exempt from compensation. Two different lists,
two different purposes, and only one of them is right.

**Side note relevant to serving:** the saved format is literally
`format: mixed-precision`. Published reports are that SGLang's compressed-tensors
integration does not properly support mixed-precision checkpoints, which is direct
evidence for the earlier conclusion that vLLM is the only validated path for this
artifact.

**M3 hit this exact situation and solved it correctly; GLM regressed, and the
regression is TEST-LOCKED.**

`pipeline/minimax_m3_config.py::get_minimax_m3_awq_mappings` balances the
post-attention norm against its **complete** consumer set, router included:

```python
AWQMapping(
    rf"re:{lm}{s}[.]post_attention_layernorm$",
    [
        rf"re:{lm}{s}[.]mlp[.]gate$",                      # <- THE ROUTER
        rf"re:{lm}{s}[.]mlp[.]shared_experts[.]gate_up_proj$",
        rf"re:{lm}{s}[.]mlp[.]experts[.][0-9]+[.]gate_proj$",
        rf"re:{lm}{s}[.]mlp[.]experts[.][0-9]+[.]up_proj$",
    ],
)
```

It avoids the dense-layer resolution failure not by dropping the router but by
**scoping every pattern to the sparse layers**:
`_M3_SPARSE_LAYER = r"(?:[3-9]|[1-5][0-9])"` (layers 3-59), with
`_M3_LM = r".*language_model[.]layers[."` additionally excluding the vision tower.
Dense layers 0-2 never participate, so `match_modules_set` closes cleanly *with*
the router present.

And it is a deliberate, tested invariant --
`tests/pipeline/test_minimax_m3_awq_mappings.py::test_moe_input_mapping_present_and_complete`:

```python
"""... must keep its complete consumer set: router + shared experts + expert gate/up."""
assert any(_matches(b, EXAMPLE_ROUTER) for b in balances), "router must be balanced"
```

That is also why `m3_checkpoint_scale_audit.py` has a `router_compensation`
component at all: on M3 it was a checked invariant that passed. The GLM smoke is
the first time it fired, and it fired correctly.

**The GLM path asserts the opposite.**
`tests/llmcompressor/modifiers/awq/test_mixed_dense_moe_mappings.py::test_router_is_never_a_balance_layer`
requires the router to be absent, repeating the same wrong justification: *"It is
also never quantized, so it was never a legitimate balance layer."* So the tree
currently holds **two tests asserting contradictory invariants about the same
algebra**, both green because they inspect different mapping sets. Only the M3 one
is right.

The GLM test does guard a real failure -- an UNSCOPED router pattern resolves zero
mappings -- but the guard is over-broad: it forbids the router everywhere instead
of forbidding an unscoped router pattern.

**Fix (now a port of a proven in-tree design, not a guess):**
1. add a MoE-layer-scoped mapping for GLM including `mlp.gate`, built from
   `first_k_dense_replace` exactly as M3 does with `_M3_SPARSE_LAYER`;
2. narrow `test_router_is_never_a_balance_layer` to forbid only *unscoped* router
   patterns, keeping its real protection;
3. add the M3-style positive assertion ("router must be balanced") for GLM, which
   is the test whose absence let this ship.

---

## 2026-08-28 — no gate compared our quantization SCOPE to the vendor's; the AWQ arm had no full-scope config at all

**Symptom (latent, found by asking rather than by failing).** A W4A8 checkpoint is
supposed to differ from the vendor's FP8 release in exactly one way: routed-expert
weights carry 4 bits instead of 8. Nothing in the pipeline checked that. The
recipe's `ignore` list and `fp8_dynamic_targets` are hand-written regexes over a
59,487-module tree, and the failure mode is silent in every existing gate — an
over-broad `ignore` leaves a whole component in BF16 while the int4 side is
untouched, the fold audit only inspects what was folded, and a quality eval absorbs
a few percent without naming a cause. That is the r8 class of failure, and the
router defect of the previous day was the same shape: a scope list nobody diffed.

Worse, the arm we intend to publish had nothing to check. Every AWQ config on
GLM-5.2 (`..._awq_smoke`, `glm52_awq_nvme_probe`, `glm52_awq_calib_scaling_probe`,
`glm52_awq_routerfix_validate`) carries a negative-lookahead pattern restricting the
run to one or two sampled layers. The GPTQ arm had a full config; the AWQ arm did
not, so "are we quantizing the right components" had no answer to give.

**Fix.** `pipeline/compare_upstream_quant_scope.py`, plus
`pipeline/configs/glm52_distributed_w4afp8_awq_full.yaml` and
`glm53_distributed_w4afp8_awq_full.yaml` (scope-identical to the GPTQ full config,
pinned by a test).

Four decisions in the gate worth keeping:

1. **Upstream ground truth is the released weight index, not `config.json`.** A
   module carrying `.weight_scale_inv` was block-FP8 quantized; one carrying only
   `.weight` was not. `modules_to_not_convert` is a statement of intent that can
   disagree with the artifact — GLM-5.3 does not list
   `self_attn.indexer.weights_proj` yet ships it in BF16, because its shape is
   `[32, 6144]` and a `[128,128]` block grid does not tile 32. Reading the index
   avoids modelling the vendor quantizer's skip rules at all.
2. **The matcher is `re.match`, copied from `compressed_tensors.utils.match`**, not
   `re.fullmatch`. A gate that judged the recipe by different rules than the
   pipeline applies would be worse than no gate.
3. **`fp8_dynamic_targets` outranks `ignore`**, because `pipeline/recipe.py` builds
   `QuantizationModifier(targets=fp8_dynamic_targets)` with no ignore list. Applying
   `ignore` first would report the shared experts as BF16 — they are in both lists
   by design.
4. **Partial recipes are reported, not failed.** A recipe targeting 2 of 78 layers
   makes no scope claim to check; `partial` is derived from layer coverage rather
   than a flag, because a flag is exactly what goes stale when a smoke config is
   copied into a production one.

**A defect in the gate itself, caught before it shipped.** The first revision
treated `fp8 -> int4` as the intended difference for *any* component. But dropping
the MLA entry from `fp8_dynamic_targets` does not send those projections to BF16 —
`ignore` is what keeps them out of the int4 modifier, so they fall *through* to
int4, and the gate would have called that intended. Narrowed to
`INTENDED_INT4_COMPONENTS` (the three routed-expert projections only) and pinned by
`test_int4_outside_the_routed_experts_fails`. This is the second time this week that
a check written to catch a scope error had a scope error of its own.

**Result on the production recipes** (against `zai-org/GLM-5.3`, whose `config.json`
is byte-identical to `zai-org/GLM-5.2-FP8`'s apart from `transformers_version`, so
one comparison serves both): all 78 MLA projection sets, all 75 shared-expert
triples and all three dense MLPs FP8 exactly as upstream; all 57,600 routed-expert
projections int4 where upstream is FP8; router, `lm_head`, `embed_tokens`,
`model.norm` and all 333 norms at source precision exactly as upstream. Two declared
divergences (`indexer.wq_b`, `indexer.wk`) and the MTP head, which transformers does
not build.

**The router answer, stated plainly since it was the previous day's defect.**
Upstream lists `mlp.gate` and `mlp.gate.e_score_correction_bias` in
`modules_to_not_convert` and ships them unscaled; we keep them at source precision
too. We agree, for two different reasons — `GlmMoeDsaTopkRouter` holds `gate.weight`
as a bare Parameter so `targets="Linear"` never reaches it, and the `ignore` entry
is belt-and-braces. Not quantizing the router is a different thing from not
*compensating* it; the router must still be an AWQ balance layer.

**On the indexer divergence — two corrections to what I first wrote here.**

*First*, I explained upstream's skip of `weights_proj [32,6144]` as a `[128,128]`
block grid failing to tile 32 rows, and concluded their indexer decision carried no
quality judgement. **That is wrong.** `kv_a_proj_with_mqa` is `[576, 6144]` and
ships a `[5, 48]` scale — `ceil(576/128)` — so partial blocks are supported. The
skip needs another explanation (a size threshold, or a deliberate choice) and the
artifact does not say which. Read as evidence, the corrected picture points the
other way from my original claim: zai-org quantized `wq_b` and `wk` deliberately,
and they designed DSA.

*Second*, the recipe justified our BF16 indexer by "our own recorded long-context
retrieval failure from touching indexer precision." **No such measurement exists in
this repo.** What exists is a mechanism argument (wq_b/wk feed the index scores, so
error changes WHICH tokens attend — a discrete selection effect like the router),
generic literature not about indexers, and M3's paired arms `r8-fp8rest` vs
`r8-uniformqkv` — one quant run, two exports, plus a paired GPQA config — whose
quality eval was never run (`docs/m3-benchmark-arms.md`: "Perf-only. No quality eval
of either export exists").

The decision stands on cost asymmetry rather than evidence: the divergence is
21 layers x (wq_b + wk) = 192.7 M params = **0.048% of a ~399 GB W4A8 checkpoint and
0.69% of the 26.1 GiB per-token activated weight bytes**, while the failure it
guards against is one our eval suite could not currently detect (RULER is listed as
"later" in `QUANT_REGRESSION_METRICS_SURVEY.md` and has never been run). Keeping it
is right; calling it validated was not. Provenance corrected in all three configs
and in the gate.
