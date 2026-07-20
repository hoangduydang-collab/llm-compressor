# MiniMax-M3 serving recipe (reproducible)

**Purpose.** One authoritative record of *how MiniMax-M3 is served* in this repo, so
the endpoint can be reproduced — including as a pinned Docker/enroot image consumed by
the external evaluation pipeline (`AICloud/benchmarks`, which talks to an
OpenAI-compatible endpoint over HTTP).

Last verified against repo evidence: 2026-07-20.

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

`python pipeline/slurm/patch_vllm_m3_serve.py` applies four edits to the installed
vLLM (and, env-gated via `M3_MOE_PROBE=1`, an optional diagnostic probe that is **not**
needed for production serving):

| # | File | Edit | Why |
|---|------|------|-----|
| 1 | `model_executor/layers/fused_moe/experts/cutlass_moe.py` | Add `MoEActivation.SWIGLUOAI_UNINTERLEAVE` to `CutlassExpertsW4A8Fp8._supports_activation` | Otherwise kernel selection raises `NotImplementedError` for M3's activation |
| 2 | `model_executor/layers/fused_moe/activation.py` | Default the SwiGLU-OAI clamp scalars (`limit=7.0, alpha=1.702, beta=1.0`) in the `SWIGLUOAI_UNINTERLEAVE` branch when the W4A8 call site passes none | W4A8 call site passes no scalars; branch otherwise asserts |
| 3 | `model_executor/layers/fused_allreduce_gemma_rms_norm.py` | Skip FlashInfer fused all-reduce in `_can_use_flashinfer` when CUDA graphs are on (NCCL fallback, graph-capturable) | Avoids a CUDA-graph capture failure |
| 4 | `model_executor/layers/fused_moe/router/base_router.py` | `nan_to_num` on `router_logits` in `RouterBase._select_experts` | Padding NaNs → duplicate/OOB expert IDs → W4A8 MoE illegal-memory-access (vLLM #39288 / #39391) |

Check status without applying: `python pipeline/slurm/patch_vllm_m3_serve.py --check`
(exit 1 if unpatched). A serving run prints `STATUS: patched` and a
`system_fingerprint` like `vllm-0.24.0-tp8-ep-<hash>`.

Edits 1–2 are required for the **W4A8** kernel path regardless of runtime mode. Edits
3–4 matter when **CUDA graphs are enabled**; the paired quality eval serves with
`enforce_eager: true` (graphs off), so it exercises 1–2 in all cases and 3–4 only when
graphs are turned on. `pipeline/vllm_m3_patches.py` is the in-process equivalent used by
`serve_verify` (workers need the persistent file patch above for the HTTP `vllm serve`
path).

**Removal criteria:** delete the overlay once a vLLM release serves M3 W4A8
(SwiGLU-OAI uninterleaved) natively.

## Per-checkpoint serving on 8×H100 (evidenced)

| Checkpoint | Precision | Topology on H100 | Needs the W4A8 overlay? |
|---|---|---|---|
| In-house GPTQ (`…gptq-checkpoint-vllm-w123-abi-overlay`) | W4A8 (W4AFP8) | 1 node, **TP8**, expert-parallel | **Yes** (edits 1–2 essential) |
| Official MXFP8 (`MiniMaxAI/MiniMax-M3-MXFP8`) | W8A16, Marlin-MXFP8 on Hopper | 1 node, **TP8** | Not the W4A8 SwiGLU path; overlay not required for kernel selection (verify 3–4 if graphs on) |
| cyankiwi AWQ-INT4 (`cyankiwi/MiniMax-M3-AWQ-INT4`) | W4A16 (AWQ Marlin) | 1 node, **TP8** | Different kernel; W4A8 edits N/A |
| Official BF16 (`MiniMaxAI/MiniMax-M3`) | BF16 | **2 nodes, TP16** (Ray) — ~920 GB won't fit 8×80 GB | No |

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
edits 3–4 are applied.)

## Docker / enroot path (for the benchmarks pipeline)

This cluster runs **enroot**, not Docker. To serve M3 W4A8 through the benchmarks repo's
image-based pattern, build a **patched image** once, then import + run under `srun`:

1. Base: `vllm/vllm-openai:v0.24.0` (or verify whether the day-0 `vllm/vllm-openai:minimax-m3`
   tag already carries the four edits — do not assume; `--check` against it).
2. Layer the overlay: copy this repo's `pipeline/slurm/patch_vllm_m3_serve.py` into the
   image and run it (Python-only, no compile), or bake the four edits directly.
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
