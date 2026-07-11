# Bugs and fixes (llm-compressor pipeline)

## Active priority: MiniMax-M3 quality before CUDA-graph RCA (2026-07-11)

Original AWQ/GPTQ and the portable routed-key re-export load but generate
repetitive garbage. Renaming routed `gate_proj/down_proj/up_proj` tensors to
the vLLM `w1/w2/w3` contract fixed a real mismatch but did not restore
quality.

Next: run `MINIMAX_M3_QUALITY_RUNBOOK.md`, which compares cyankiwi against
the portable W4A8 checkpoint in eager mode and returns semantic quality,
loader mappings, loaded-parameter fingerprints, shared-expert contribution,
environment provenance, and full-log hashes through Git.

Do not resume CUDA-graph RCA, re-quantize, or apply another loader fix until
this comparison identifies the first failing quality boundary.

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
