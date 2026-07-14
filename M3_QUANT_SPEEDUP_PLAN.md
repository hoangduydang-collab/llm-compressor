# MiniMax-M3 Quantization Speed-Up Plan (MoE-Quant–inspired)

Status: **design / Phase 1 runnable.** Speed engineering is still gated on a
quality-verified recipe (see §0), but the Phase-1 premise/concurrency benchmark
(`pipeline/bench_expert_scatter.py`) has no such gate and runs now. Planner
document; the executor owns cluster/GPU runs.

> **2026-07-14 decision log.** The two in-house `safe-*` AWQ full-calibration
> runs were cancelled: at ~half done after 15h they projected ~27–34h against a
> ~9h-remaining wall-clock, so they could not finish, and a partially quantized
> checkpoint is unusable. They are **not** in the paired eval (that compares our
> GPTQ vs cyankiwi's external AWQ), so cancelling cost the eval nothing and freed
> the GPUs for Phase 1. The paired production eval keeps running as the §0 gate.

## 0. Precondition (do not skip)

Do **not** start this work until the in-house recipe we intend to accelerate is
confirmed non-garbage at eval scale. As of 2026-07-14 the repaired **GPTQ**
checkpoint passes the smoke suite (coherent output, GSM8K 2/2, MMLU-Pro 12/14 —
see `M3_3MODEL_GPTQ_AWQ_FINAL_REPORT.md`); the paired GPTQ-vs-cyankiwi full eval
(`pipeline/configs/minimax_m3_paired_gptq_awq.yaml`) must pass before we invest
in calibration-speed engineering. A 10× faster calibration of a bad recipe is
worthless. **Target the recipe that passes that eval** (GPTQ is the current
front-runner and, conveniently, is exactly what MoE-Quant accelerates).

## 1. Motivation

Full-calibration AWQ/GPTQ of MiniMax-M3 is slow, and the slowness is almost
entirely wasted hardware, not necessary compute:

- **Only 1 of 8 GPUs is used.** The production config runs a single
  `python -m pipeline.run` process with `device_map: null`
  (`pipeline/configs/minimax_m3_full_calib.yaml`); `tensor_parallel_size: 8` is
  serve-only. Calibration streams one layer to one GPU while 7 sit idle.
- **Experts are quantized serially, one at a time.** MiniMax-M3 stores experts
  as fused 3D tensors
  (`MiniMaxM3VLExperts.gate_up_proj[num_experts, 2·inter, hidden]`,
  `down_proj[num_experts, hidden, inter]`); llm-compressor's `linearize_moe`
  un-fuses them into 128 per-expert `nn.Linear` modules, and AWQ/GPTQ then loop
  over those 128 mappings serially (`_apply_smoothing` in
  `src/llmcompressor/modifiers/transform/awq/base.py`; the per-layer loop in the
  sequential pipeline). Each narrow expert GEMM underutilizes an H100 — the
  fine-grained-MoE small-op pathology.
- The experts in the up→down (and, for AWQ, per-expert) mappings are
  **mathematically independent** (each sees only its own routed tokens), so this
  serialization is an implementation choice, not an algorithmic requirement.

### Model facts (drive the memory math)

From `configuration_minimax_m3_vl.py`: `hidden=6144`, per-expert
`intermediate=3072`, `num_local_experts=128`, `num_hidden_layers=60` (sparse
layers 3–59, i.e. 57 MoE layers).

- Per expert ≈ `2·3072·6144 (gate_up) + 6144·3072 (down)` ≈ **56.6M params**.
- Per MoE layer ≈ `128 × 56.6M` ≈ **7.25B params** ≈ **14.5 GB** in bf16.
- Whole model ≈ **~450–460B params** ≈ **~920 GB** bf16 (consistent with a
  single load straining the ~2000 GiB host RAM after `linearize_moe`/accelerate
  staging spikes).

**Key consequence:** one MoE layer's full expert set (~14.5 GB) fits on a single
H100 with room to spare. We therefore do **not** need model-wide expert
parallelism or all-to-all routing — we only need to (a) stream one layer at a
time (already done) and (b) parallelize the per-expert quantization of that one
layer across the 8 GPUs.

## 2. Prior art / citations

- **MoE-Quant** (IST-DASLab — the GPTQ authors), `github.com/IST-DASLab/MoE-Quant`.
  Quantizes DeepSeek-V3/R1 (671B, 256 experts) with GPTQ in **~2 h on 8×H100**
  (512 seqs × 4096). Design (verified by reading `quant.py`, `src/gptq.py`):
  - **Expert sharding at construction** via `config.ep_size = world_size` +
    `init_empty_weights()`; each rank materializes only `1/world_size` experts.
  - **Data parallelism**: calibration data sliced across ranks
    (`calibration_dataset[rank*n : (rank+1)*n]`).
  - **Block-by-block streaming**: rank 0 streams safetensor shards through a
    sliding-window buffer, dequantizes FP8→bf16 per block, `send/recv`s each rank
    its expert slice, offloads the block to `meta` afterward — the full model is
    never materialized anywhere.
  - **Shared (non-expert) layers**: Hessian all-reduced (AVG) across ranks
    (`gptq.py`: `dist.all_reduce(self.H, op=AVG)` when `is_distributed=True`),
    quantized on rank 0, broadcast back.
  - **Expert layers**: `is_distributed=False` → each rank quantizes its experts
    **fully independently, no communication** — the embarrassingly-parallel core.
  - **~10× custom Triton GPTQ kernel** vs default torch; tied gate/up Hessians
    (`--tie_gptq_handles`); activation-order + MSE-scale options.
- **GPTQ** (Frantar et al., arXiv:2210.17323): lazy batch-updates (B=128 columns)
  give ~10× on large models; 175B in ~4 GPU-hours. Orthogonal, stacks.
- **AWQ** (Lin et al., arXiv:2306.00978, MLSys 2024): per-channel smoothing-scale
  search; no backprop.
- **NVIDIA modelopt**: `awq_lite` (`alpha_step` coarsens the ~20-pt grid);
  data-parallel calibration (DistributedSampler + amax max-reduce); a fused
  Triton NVFP4 scale-search fast path reported **~34×** on one 8192×4096 weight.
- **llm-compressor**: distributed GPTQ (v0.10) — per-module rank assignment,
  ~3.8× on 4 GPUs (3.9 h→1 h, Qwen3-30B-A3B); batched calibration (v0.9.0,
  `batch_size=32`) — up to 3× on AWQ, ~15% on GPTQ.
- **MxMoE** (arXiv:2505.05799): grouped-GEMM parallel-across-experts — but an
  *inference* kernel, not a calibration accelerator.

**Novelty note (from verified deep-research, 2026-07-14):** distributed expert
sharding is published only for GPTQ and only as model-per-rank data/expert
parallelism (MoE-Quant, GPTQModel, llm-compressor v0.10). No source implements
per-block, single-copy, expert-scatter quantization for a fused-expert MoE, nor
combines batched per-expert scale search with expert sharding. The design below
is an adaptation of MoE-Quant's *ideas* to MiniMax-M3's fused layout; it is not a
port (see §5).

## 3. Why MoE-Quant cannot be copied verbatim

| MoE-Quant assumes | MiniMax-M3 reality | Resolution |
|---|---|---|
| Per-expert `nn.Linear` (`.*mlp.experts.\d+.(gate\|up\|down)_proj`) | Fused 3D tensors (`gate_up_proj`, `down_proj`) | We already run `linearize_moe`, which un-fuses to per-expert Linears — matches after linearization |
| `config.ep_size` expert-parallel forward (all-to-all) | No EP support in modeling code | **Not needed** — one layer fits on one GPU; shard only the *quantization*, not the forward |
| GPTQ, CausalLM, DeepSeek-hardcoded (`assert architectures==["DeepseekV3ForCausalLM"]`, 163-shard names, `first_k_dense_replace`) | AWQ+GPTQ, VL wrapper (`MiniMaxM3SparseForConditionalGeneration`) | Reimplement the ideas inside llm-compressor's pipeline; target `model.model.language_model.layers.N` |

## 4. Proposed design

Adopt MoE-Quant's three load-bearing ideas, adapted to a **single process** (our
model fits in host RAM exactly once; multi-process would need 8 CPU copies ≈
7.4 TB → infeasible):

1. **Block streaming + offload** (already provided by llm-compressor's sequential
   pipeline): one decoder layer onloaded to GPU at a time, model otherwise on CPU.
2. **Per-block expert-scatter quantization** (new): for the current layer, place
   its 128 (linearized) experts across the 8 GPUs (~16/GPU) and run their
   per-expert scale/Hessian-quantization **concurrently**. CUDA ops release the
   GIL, so a small thread pool (or per-device CUDA streams) over the 8 devices
   yields real parallelism. Gather the quantized weights back.
3. **Shared / non-expert layers** (attention, router gate, shared expert, and —
   for AWQ — the single shared-input smoothing mapping) stay on the existing
   single-GPU path; for the AWQ shared-input mapping, keep the current
   data-parallel-style stats aggregation. These are O(1) per layer, not O(128).

### GPTQ-first (recommended)

Because GPTQ is the currently-verified recipe and its per-expert quantization is
fully independent (a local Hessian per expert; no cross-expert scale), it maps
onto expert-scatter with **zero** inter-expert communication — identical in
spirit to MoE-Quant's `is_distributed=False` expert path. Plan:

- Collect each expert's Hessian during one block forward on the (single) GPU
  holding the block, capturing per-expert inputs (the sequential pipeline already
  hooks these).
- Scatter `{expert_weight, expert_hessian}` for the 128 experts across 8 GPUs.
- Each GPU runs `quantize_weight` (llm-compressor's existing GPTQ core, or a
  Triton kernel à la MoE-Quant for the ~10×) on its ~16 experts concurrently.
- Gather `{qweight, scale, zero}`; write back; propagate activations; offload.

### AWQ variant (only if AWQ becomes the chosen recipe)

Same expert-scatter for the per-expert up→down mappings. The one shared-input
mapping (post-attn-norm → all experts' gate/up, a single shared scale) is **not**
expert-parallel; run it once data-parallel (all-reduce stats) as today. Expect a
smaller win than GPTQ because that shared mapping's full-MoE forward is not
sharded.

### Expected speed-up (honest ceilings, not promises)

- Expert-scatter across 8 GPUs: up to ~8× on the per-expert portion, minus
  load-imbalance and scatter/gather overhead → realistically ~4–6×.
- Optional Triton GPTQ kernel (MoE-Quant ~10× per-layer microbench) stacks on top.
- Amdahl: attention/shared/router layers and the calibration forwards don't get
  the expert-scatter win, so end-to-end speed-up is below the per-axis product.
  Measure, don't assume.

## 5. Implementation phases

1. **Validate premise + de-risk single-process concurrency (do this first).**
   Runnable now: `python -m pipeline.bench_expert_scatter --experts 128` (on a
   freed 8-GPU node). It measures serial-1-GPU vs thread-pool-8-GPU wall-clock on
   a GPTQ-shaped per-expert proxy (real MiniMax-M3 expert dims), reports the
   python-setup/CUDA split, and prints a go/no-go verdict on plan §6's top risk
   (does the GIL + column-loop launch overhead eat the scatter win?). Also run
   `nvidia-smi dmon -s u` during its serial phase to confirm per-expert SM
   underutilization. **Do not write Phase-2 production scatter until this shows
   real (≥~0.5×ceiling) parallelism.** Needs no model download, no chosen recipe,
   and no eval verdict — so it proceeds in parallel with the paired eval.
2. **Expert-scatter GPTQ, single node.** Add a MoE-aware fast path to the GPTQ
   modifier / sequential pipeline that, per decoder layer, dispatches the
   linearized experts' `quantize_weight` calls across `cuda:0..7` concurrently
   and gathers results. Keep it behind a flag (e.g. `M3_EXPERT_SCATTER=1`);
   default off. Correctness gate: bit-identical (or within numerical tolerance)
   quantized weights vs the serial path on a tiny model.

   **Done (2026-07-14): device-agnostic orchestration core** in
   `pipeline/expert_scatter.py` — `assign_devices` (largest-Hessian-first greedy
   balance), `serial_quantize`/`scatter_quantize` (thread-pool dispatch, results
   gathered by expert name, scheduling-order-independent), and
   `default_gptq_quantize_fn` (lazy adapter to the real `quantize_weight`). CPU
   bit-parity gate in `pipeline/tests/test_expert_scatter.py` (5 tests) proves
   scatter == serial per expert across 1/4/8 workers and that there is no
   cross-expert contamination. Rests on GPTQ's per-expert independence, so it
   changes no quantization math.

   **Remaining (GPU-gated, after the Phase-1 bench verdict): modifier wiring.**
   Relocate each expert onto its assigned device inside
   `GPTQModifier.compress_module_list` using accelerate onload/offload, call the
   scatter core, write params back with `update_offload_parameter`. This is the
   part that touches offload accounting (cf. the FSDP2 reshard bug class) and
   must be validated on GPU with a real small-model serial-vs-scatter parity run.
3. **Optional Triton GPTQ kernel** port (MoE-Quant `src/gptq_loop.py` +
   `linalg_utils`) for the per-expert inner loop if step 2's gather isn't enough.
4. **AWQ variant** (only if AWQ is chosen): expert-scatter the per-expert
   mappings; keep the shared-input mapping data-parallel.
5. **Verify quality unchanged.** Re-run the paired eval (§0 harness) on the
   fast-path checkpoint; it must match the serial-path checkpoint within noise.

## 6. Risks / open questions

- **Single-process 8-GPU concurrency**: GIL is released during CUDA kernels, but
  Python-side per-expert setup (observer, packing) is serial. If setup dominates,
  use per-device CUDA streams or a `ProcessPoolExecutor` sharing CPU-pinned
  weights. Prototype early.
- **Numerical parity**: expert-scatter must not change results vs serial. Gate on
  a small-model bit-parity test before trusting a full run.
- **Sequential-pipeline coupling**: the fast path must slot into the existing
  onload/offload + activation-propagation loop without breaking offload
  accounting (cf. the FSDP2 reshard bug class we've hit before).
- **linearize_moe cost**: un-fusing 128 experts per layer itself has overhead; a
  further optimization is to quantize the *fused* 3D tensors directly (batched
  bmm over the expert dim), skipping linearization entirely — larger change,
  deferred.
- **Scope discipline**: this is a speed project. It must not alter the
  quantization math/recipe that passed the quality eval.

## 7. Non-goals

- No custom expert-parallel forward / all-to-all (unnecessary here).
- No multi-process/torchrun replication (host RAM can't hold 8 copies).
- No change to the quantization recipe, scheme, or calibration data.
