# MiniMax-M3 serving recipe (reproducible)

**Purpose.** One authoritative record of *how MiniMax-M3 is served* in this repo, so
the endpoint can be reproduced — including as a pinned Docker/enroot image consumed by
the external evaluation pipeline (`AICloud/benchmarks`, which talks to an
OpenAI-compatible endpoint over HTTP).

Last verified against repo evidence: **2026-07-29** (patch inventory re-derived from
`pipeline/slurm/patch_vllm_m3_serve.py`; the earlier four-edit table was stale).

## TL;DR

**Serving base = released vLLM `0.24.0` (stable) + a small in-place Python patch
overlay. It is NOT a custom fork build.**

The overlay is applied by `pipeline/slurm/patch_vllm_m3_serve.py` (idempotent,
fail-loud, `--check` supported). It is pure Python — **no CUDA recompilation** — layered
on top of the released wheel's precompiled binaries.

> **Important:** the two W4A8 activation gaps this overlay fixes are present in
> **stock vLLM, the NVIDIA build, *and* `toncao/vllm@minimax-m3-compressed-tensors`**
> (see the docstring in `pipeline/vllm_m3_patches.py`). So the patch is required on
> *any* 0.24.0-class base. Serving M3 W4A8 was never "the fork" — it is "0.24.0 + this
> overlay." The `toncao` fork is the reference tree for the separate Goal-6 Hopper
> NVFP4 / Humming track, not the general M3 eval/production serving base.

## The patch overlay

`python pipeline/slurm/patch_vllm_m3_serve.py` applies **eight required edits** to the
installed vLLM, plus one release-conditional edit, plus (only on Humming arms) two
Humming edits, plus several opt-in overlays that are **not** needed for production
serving. The authoritative list is `_patch_targets()` in that script — if this table and
the code disagree, the code wins.

### Required (`_patch_targets`) — the overlay fails closed if any anchor is missing

| # | File | Edit | Why |
|---|------|------|-----|
| 1 | `fused_moe/experts/cutlass_moe.py` | Add `MoEActivation.SWIGLUOAI_UNINTERLEAVE` to `CutlassExpertsW4A8Fp8._supports_activation` | Otherwise kernel selection raises `NotImplementedError` for M3's activation |
| 2 | `fused_moe/activation.py` | Default the SwiGLU-OAI clamp scalars (`limit=7.0, alpha=1.702, beta=1.0`) in the `SWIGLUOAI_UNINTERLEAVE` branch when the W4A8 call site passes none | W4A8 call site passes no scalars; branch otherwise asserts |
| 3 | `fused_allreduce_gemma_rms_norm.py` | Skip FlashInfer fused all-reduce in `_can_use_flashinfer` when CUDA graphs are on (NCCL fallback, graph-capturable) | Avoids a CUDA-graph capture failure |
| 4 | `fused_allreduce_gemma_rms_norm.py` | `LLMC_M3_FI_AR_MODE=off` disables the fused AR entirely, **including eager warmup** | Banning it only from capture left a residual capture IMA; warmup was still initialising the FlashInfer workspace |
| 5 | `fused_moe/runner/shared_experts.py` | `LLMC_M3_SHARED_RS_MODE=skip_capture` skips `record_stream` while capturing | `record_stream` under multi-stream capture is a known-fragile torch path (pytorch #155398 / #175560); the graph pool + `wait_stream` edges already order usage |
| 6 | `compilation/breakable_cudagraph.py` | `LLMC_M3_CAPTURE_SYNC=sync` restores the device `synchronize()` before `_capture`'s gc + `empty_cache` | The fork's `BreakableCUDAGraph._capture` dropped the synchronize that upstream `torch.cuda.graph.__enter__` performs. With `expandable_segments:True` the unmap is `cuMemUnmap` (no implicit sync) → capture-ladder IMA in `_accelerator_emptyCache` |
| 7 | `fused_moe/router/base_router.py` | `nan_to_num` on `router_logits` in `RouterBase._select_experts` | Padding NaNs → duplicate/OOB expert IDs → W4A8 MoE illegal-memory-access (vLLM #39288 / #39391). One edit covers fused_topk / grouped_topk / bias / custom routers |
| 8 | `models/minimax_m3/nvidia/model.py` | Restore head-major `topk_indices_buffer` allocation off SM100 | Root cause of the vLLM **0.26.0** IMA that killed every M3 serve on Hopper under concurrency, including plain k=0. A clean no-op on 0.24.0, which already allocates head-major — see [`m3-026-topk-buffer-layout.md`](m3-026-topk-buffer-layout.md) |

### Conditional and arm-specific

| Scope | Edit | When |
|---|---|---|
| `_optional_patch_targets` | `QuantKey.__str__` ScalarType fallback (`quantization/utils/quant_utils.py`) | Diagnostics-only; anchor absent before 0.26.0, so a missing anchor must **not** fail the overlay |
| `_humming_patch_targets` | Humming W4A8 SWIGLU support + W4AFP8 quant-scheme admit (`fused_moe/experts/fused_humming_moe.py`) | Humming MoE-backend arms only |
| `ensure_m3_gate_alpha()` | Gate-alpha fold support | **Required to serve the AWQ `r7` checkpoint** — see `docs/m3-benchmark-arms.md` |
| `ensure_m3_moe_probe()` / `_load_audit()` / `_layer_boundary()` | Diagnostic instrumentation, env-gated (`M3_MOE_PROBE=1`, `M3_LAYER_BOUNDARY=1`) | Never for production serving |

Check status without applying: `python pipeline/slurm/patch_vllm_m3_serve.py --check`
(exit 1 if unpatched). A serving run prints `STATUS: patched` and a
`system_fingerprint` like `vllm-0.24.0-tp8-ep-<hash>`.

**Which edits matter when.** Edits 1–2 are required for the **W4A8** kernel path
regardless of runtime mode. Edits 3–7 matter when **CUDA graphs are enabled**; the paired
quality eval serves with `enforce_eager: true` (graphs off), so it exercises 1–2 in all
cases. Edit 8 is required on **0.26.0** and inert on 0.24.0. Edits 4–6 are env-gated
knobs whose non-default values were needed to close the graphs-on capture IMA — the
overlay installs the capability; the launcher chooses the mode.
`pipeline/vllm_m3_patches.py` is the in-process equivalent used by `serve_verify`
(workers need the persistent file patch above for the HTTP `vllm serve` path).

**Removal criteria:** delete the overlay once a vLLM release serves M3 W4A8
(SwiGLU-OAI uninterleaved) natively.

## Per-checkpoint serving on 8×H100 (evidenced)

| Checkpoint | Precision | Topology on H100 | Needs the W4A8 overlay? |
|---|---|---|---|
| In-house GPTQ (`…gptq-checkpoint-vllm-w123-abi-overlay`) | W4A8 (W4AFP8) | 1 node, **TP8**, expert-parallel | **Yes** (edits 1–2 essential) |
| In-house AWQ `r6` | W4A8 (W4AFP8) | 1 node, **TP8**, expert-parallel | **Yes** (edits 1–2 essential) |
| In-house AWQ `r7` | W4A8 (W4AFP8) | 1 node, **TP8**, expert-parallel | **Yes**, edits 1–2 **plus** the gate-alpha overlay (`ensure_m3_gate_alpha`) |
| Official MXFP8 (`MiniMaxAI/MiniMax-M3-MXFP8`) | W8A16, Marlin-MXFP8 on Hopper | 1 node, **TP8** | Not the W4A8 SwiGLU path; overlay not required for kernel selection (verify 3–7 if graphs on) |
| cyankiwi AWQ-INT4 (`cyankiwi/MiniMax-M3-AWQ-INT4`) | W4A16 (AWQ Marlin) | 1 node, **TP8** | Different kernel; W4A8 edits N/A |
| Official BF16 (`MiniMaxAI/MiniMax-M3`) | BF16 | **2 nodes, TP16** (Ray) — ~920 GB won't fit 8×80 GB | No |

Per-arm checkpoint paths and recipe provenance live in
[`m3-benchmark-arms.md`](m3-benchmark-arms.md), which owns that table.

(Sources: `results/m3-shared-expert-repair/**/server_start.txt` and
`results/m3-layer-boundary/**` for W4A8 on 8×H100 TP8; the paired-eval matrix configs
`pipeline/configs/minimax_m3_*_reasoning_r4.yaml` for the four-model topologies.)

## Representative `vllm serve` command (OpenAI-compatible)

```bash
# in a venv/image with vLLM 0.24.0 + the overlay applied
vllm serve <checkpoint-path> \
  --served-model-name MiniMaxAI/MiniMax-M3 \
  --tensor-parallel-size 8 \
  --enable-expert-parallel \
  --block-size 128 \
  --kv-cache-dtype fp8 \
  --max-model-len 65536 \
  --gpu-memory-utilization 0.85 \
  --enforce-eager \
  --tool-call-parser minimax_m3 --reasoning-parser minimax_m3 --enable-auto-tool-choice \
  --trust-remote-code \
  --host 0.0.0.0 --port 8000
```
(Parsers `minimax_m3` are vLLM's documented M3 parsers; `--enforce-eager` matches the
paired-eval serving profile. Drop `--enforce-eager` only after confirming the CUDA-graph
edits 3–7 are applied **and** the graphs-on env knobs are set — `LLMC_M3_CAPTURE_SYNC=sync`
in particular; the fix matrix showed the capture IMA survives every other arm without it.)

## Docker / enroot path (for the benchmarks pipeline)

This cluster runs **enroot**, not Docker. To serve M3 W4A8 through the benchmarks repo's
image-based pattern, build a **patched image** once, then import + run under `srun`:

1. Base: `vllm/vllm-openai:v0.24.0` (or verify whether the day-0 `vllm/vllm-openai:minimax-m3`
   tag already carries the edits — do not assume; `--check` against it).
2. Layer the overlay: copy this repo's `pipeline/slurm/patch_vllm_m3_serve.py` into the
   image and run it (Python-only, no compile). Prefer running the script over hand-baking
   the edits, so the image tracks the current `_patch_targets()` list rather than a
   snapshot of it.
3. `enroot import docker://<your-registry>/vllm-openai-m3w4a8:v0.24.0` → `enroot create` →
   run under `srun` on 8×H100 with the serve command above.
4. Point the benchmarks profile (`configs/minimax/minimax-m3.sh`) `BASE_URL` at the endpoint.

This patched image is the "pinned serving runtime" artifact — reproducible, versioned,
and independent of the mutable `quant` dev venv.

## Reproducibility record

Pin and record, per run: base vLLM version (`0.24.0`), overlay `--check` status, the
`system_fingerprint` (`vllm-0.24.0-tp8-ep-<hash>`), and the checkpoint hash. The
quality-eval `run_manifest.json` currently records only `lm_eval_version`; the vLLM
version + patch status live in the serving-diagnostic run dirs
(`results/m3-shared-expert-repair/`, `results/m3-layer-boundary/`). Persisting the full
serving provenance into the eval manifest closes that gap.

## Evidence

- `pipeline/README.md` — "vLLM 0.24.0 stable serves W4AFP8 MoE correctly".
- `pipeline/slurm/patch_vllm_m3_serve.py`, `pipeline/vllm_m3_patches.py` — the overlay (and the note that stock/NVIDIA/fork builds all need it).
- `results/m3-shared-expert-repair/20260711-152808-shared-repair/repaired_w4a8_http/{server_start,patch_status,software_versions}.txt` — W4A8 served on 8×H100, TP8, vLLM 0.24.0, `STATUS: patched`.
- `BUGS_AND_FIXES.md` — "W4A8 MoE … SWIGLUOAI_UNINTERLEAVE" and "CUDA graph capture".
