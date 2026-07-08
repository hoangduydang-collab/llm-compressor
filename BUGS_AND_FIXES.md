# Bugs and fixes (llm-compressor pipeline)

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

### CUDA graph capture (``enforce_eager=false``) — still failing

**Symptom:** PIECEWISE capture progresses to **16/51** (deterministic), then
``CUDA illegal memory access`` in ``breakable_cudagraph._capture`` (reported at
``empty_cache()`` — async). EngineCore dies → **``TCPStore Broken pipe``** on all
ranks (shutdown cascade, **not** root cause).

| # | Attempt | Hypothesis | Result on h118 |
|---|---------|------------|----------------|
| A | ``disable_custom_all_reduce=true`` | FlashInfer / custom AR incompatible with 2×4 topology | **Wrong lever** — does not gate M3's ``fused_allreduce_gemma_rms_norm`` ([vLLM #45800](https://github.com/vllm-project/vllm/issues/45800)). Logs still show ``Auto-selected flashinfer allreduce backend: trtllm``. Removed as default. |
| B | ``ENFORCE_EAGER=1`` | Skip graph capture entirely | **WORKS** — full serve-verify pass. Not a graph fix. |
| C | Fused-AR patch (#46253) — skip FlashInfer in ``_can_use_flashinfer`` when graphs on (patch 3) | FlashInfer fused AR+RMSNorm not graph-capturable on TP8 | **Still IMA at 16/51** after persistent patch applied. |
| D | MoE ``nan_to_num(router_logits)`` (patch 4, [vLLM #39288](https://github.com/vllm-project/vllm/issues/39288) class) | Padding tokens → NaN logits → duplicate expert IDs → W4A8 CUTLASS OOB | **Still IMA at 16/51** with ``patch_vllm_m3_serve.py --check`` → **STATUS: patched** (all 4). |
| E | Runtime monkeypatches in ``serve_verify`` / ``vllm_m3_patches.py`` | In-process hook before ``LLM()`` | **Ineffective for capture** — ``Worker_TP*`` are **spawned subprocesses**; they load site-packages only. Documented; launcher now auto-runs persistent patch script. |
| F | ``fuse_allreduce_rms=false`` (compile config) | Disable fused AR via inductor | **No effect** — M3 calls fused path directly in ``model.py`` forward. |
| G | ``SERVE_PERF=1`` / re-enable FlashInfer fused AR | Official recipe perf path | Not tested for graphs; eager serve works with FlashInfer trtllm (slow init). |

**Not yet run on cluster (next debug steps):**

- ``CUDA_LAUNCH_BLOCKING=1 ENFORCE_EAGER=0`` — get synchronous stack trace for the kernel that actually faults (before EngineCore cascade).
- Map **graph index 16 → token batch size** in ``gpu_model_runner._capture_cudagraphs`` (likely a specific padded batch triggers W4A8 CUTLASS or sparse-attn kernel bug).
- Confirm patch 4 body in site-packages (not just marker): ``grep -A3 'nan_to_num router_logits' …/moe_runner.py``.
- Patch ``csrc/moe/topk_softmax_kernels.cu`` NaN clamp ([#39391](https://github.com/vllm-project/vllm/pull/39391)) if toncao 0.24.0 lacks it — Python ``nan_to_num`` may be insufficient for **monolithic** W4A8 path where top-k runs inside CUTLASS.
- Zero-init **padding hidden states** at MoE layer entry ([#40047](https://github.com/vllm-project/vllm/issues/40047) class) if router logits are clean but upstream activations are NaN.
- Segment ``breakable_cudagraph`` at NCCL collectives (upstream [#46253](https://github.com/vllm-project/vllm/issues/46253) long-term fix).

### Commits (``duy-branch``, graph-capture work)

| Commit | Summary |
|--------|---------|
| ``3fd893ed`` | Fused-AR NCCL fallback when graphs on (patch 3) |
| ``39c3d6e5`` | MoE router ``nan_to_num`` (patch 4) |
| ``0199a48e`` | ``ensure_vllm_m3_patches()`` — workers need site-packages edits |
| ``f45667b5`` | Launcher/sbatch auto-run ``patch_vllm_m3_serve.py`` |

### Current status (2026-07-08)

- **Serve with graphs:** **BLOCKED** — IMA at capture **16/51** despite all 4 persistent patches verified.
- **Serve without graphs:** **OK** — ``ENFORCE_EAGER=1``.
- **Production path:** fix graph capture (above next steps) or upstream vLLM/FlashInfer fixes; do not ship with ``enforce_eager`` as the only solution if graph perf is required.

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
   ``MoERunner._apply_quant_method`` before routing (same mechanism as #39391).

- **Persistent:** ``pipeline/slurm/patch_vllm_m3_serve.py`` (4 edits); **required**
  for ``Worker_TP*`` subprocesses (spawned fresh — runtime monkeypatches in
  ``serve_verify`` do not apply). Launcher auto-runs this script; re-run after
  any vLLM reinstall.
- **Runtime:** removed for cudagraph (ineffective on workers). ``ensure_vllm_m3_patches()``
  in ``serve_verify`` verifies site-packages before ``LLM()``.

**Verify on cluster:**

```bash
python pipeline/slurm/patch_vllm_m3_serve.py --check   # all 4 patches
python pipeline/slurm/patch_vllm_m3_serve.py            # apply if needed
grep -r 'llmc M3' "$(python -c 'import vllm, pathlib; print(pathlib.Path(vllm.__file__).parent)')"
ENFORCE_EAGER=0 bash pipeline/slurm/run_serve_minimax_m3_detached.sh
# success = 51/51 capture; as of 2026-07-08 still fails at 16/51 with patches applied
```

**Status (2026-07-08):** Patches 3–4 apply and verify, but **IMA at 16/51 persists**.
Patches 1–2 + ``ENFORCE_EAGER=1`` remain the known-good serve path. See chronicle
above for full attempt log.

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
— logs may still show `Auto-selected flashinfer allreduce backend: trtllm`. On h118
serve-verify succeeded without setting this flag; **not enabled by default**.

To try explicitly: `--set serve.disable_custom_all_reduce=true` or
`ServeConfig.disable_custom_all_reduce` in yaml.

For raw `vllm serve`: `--disable-custom-all-reduce` (and keep
`FLASHINFER_USE_CUDA_NORM=1` on Hopper). Unlikely to change M3 behavior.

**Root-cause / long-term fix:** topology-aware FlashInfer backend selection in vLLM;
M3 model should fall back cleanly when `_can_use_flashinfer()` fails.

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
