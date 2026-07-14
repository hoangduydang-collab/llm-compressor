# MiniMax-M3 Quantization Speed-Up Plan (MoE-Quant–inspired)

Status: **CONCLUSION REACHED (2026-07-14) — use the built-in distributed
calibration; stop bespoke work.** Multi-GPU calibration for **both AWQ and GPTQ**
already exists in this fork and was simply never switched on (we launched
single-process). See the handoff section immediately below; §§3–6 are the
now-superseded archaeology of how we got here (kept for context). Speed
engineering is still gated on a quality-verified recipe (§0). Planner document;
the executor owns cluster/GPU runs.

---

## HANDOFF (2026-07-14) — read this first

**A different planner agent is continuing. This section is the current truth; the
older sections below (§§4–6 EP/scatter design) are superseded — do not implement
them.**

### The conclusion

The 30-hour `safe-*` calibration was slow for ONE reason: it launched as a single
process (`python -m pipeline.run`), so `is_distributed()` was `False` and every
distributed path sat dormant. **This fork already has multi-GPU distributed
calibration for both recipes:**

- **AWQ** — data-parallel: `_compute_best_scale` all-reduces activation stats
  across ranks when `is_distributed()`
  (`src/llmcompressor/modifiers/transform/awq/base.py:645`, loss at `:806`,
  `_allreduce_data_sum` at `:1047`).
- **GPTQ** — module-parallel: `compress_modules` bin-packs modules across ranks,
  reduces Hessians to owner, quantizes, broadcasts
  (`src/llmcompressor/modifiers/gptq/base.py:284`).
- **MoE coverage**: `MoECalibrationModule` exists
  (`src/llmcompressor/modeling/moe/context.py`), ensuring all experts get data.
- **Memory**: distributed CT offload keeps **one shared CPU copy** in `/dev/shm`
  across ranks (`DistributedCPUCache`; memory scales with model size, NOT rank
  count — verified from compressed-tensors offload docs). The entrypoint already
  converts accelerate→CT offload (`entrypoints/utils.py:92`). So 8-rank DDP of the
  ~920 GB model fits in ~2 TB RAM (one shared copy + per-rank GPU working set +
  transient copy-on-write of the current layer). The "8×920 GB = 7.4 TB" fear that
  drove the bespoke design was WRONG.

Reputable prior art confirms this is solved, not novel: llm-compressor v0.10 ships
DDP for AWQ (2.9–3.2×/4GPU) and GPTQ (3.8×/4GPU); GPTQModel v6.1 has both for MoE
(80%+ time reduction). cyankiwi already made a working MiniMax-M3-AWQ-INT4 with the
standard tooling.

### Next steps (in order)

1. **Verify the launch wiring** (small). The current calibration launches
   single-process (`pipeline/slurm/run_quantize_minimax_m3_{local,detached}.sh`):
   ```
   python -m pipeline.run --config "$CONFIG" --stage quantize ...
   ```
   The change to try:
   ```
   torchrun --nproc_per_node=8 -m pipeline.run --config "$CONFIG" --stage quantize ...
   ```
   Then confirm: (a) `pipeline.run` reaches llm-compressor's `oneshot`, whose
   `pre_process` already branches on `is_distributed()` and converts to CT offload
   (`entrypoints/utils.py:57,92`); (b) the calibration data is **sharded** across
   ranks (look for rank-partitioning in the pipeline's data loading — if it
   replicates, forwards won't parallelize and you must add a `DistributedSampler`-
   style split); (c) the model is loaded with an offload `device_map` so CT
   distributed offload engages (one shared `/dev/shm` copy) rather than 8 full
   per-rank loads. This is the one real integration unknown — config/launch, not
   new algorithm code.
2. **2–3-layer smoke on 8 GPUs**, one AWQ config and one GPTQ config, watching host
   RAM. Success = RAM ≈ one shared copy (~920 GB), not a multiple, and per-layer
   wall-clock drops toward ~1/N. Decisive and cheap.
3. **Full multi-GPU calibration** of the recipe that passes the §0 quality gate.

### What is shelved (do NOT continue)

`pipeline/bench_expert_scatter.py`, `pipeline/expert_scatter.py`, `pipeline/ep_moe.py`
(+ their tests) were bespoke work that re-solved a solved problem. They are correct
and CPU-tested but are NOT the path forward. Keep for reference only. The
`ep_moe.py` dispatch/combine core would only matter if we ever needed a custom EP
forward, which the built-in offload-based DDP makes unnecessary.

### Corrections banked this session (so the next agent doesn't repeat them)

- The installed version string `0.1.dev3131+...` is a **setuptools-scm fallback**
  (fork has no version tags), NOT evidence of being behind upstream. This fork is
  v0.12-era (has `MoECalibrationModule`).
- The AWQ modifier is at `modifiers/**transform**/awq/base.py`, not `modifiers/awq/`.
- Single-process thread-pool scatter is dead (GIL-bound per-column GPTQ loop →
  0.55–0.88× measured). Not the mechanism.
- Distributed CT **CPU** offload = one shared copy across ranks (not per-rank).

---

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

## 4. Proposed design (revised 2026-07-14: load-time sharding, not scatter)

**Correction to the earlier design.** The Phase-1 bench (§5) killed the
single-process thread-pool scatter: GPTQ's per-expert cost is a Python-driven
per-column loop, and one interpreter's GIL serializes its kernel launches, so a
thread pool over 8 GPUs gave 0.88× (slower than serial). The fix is not CUDA
streams (they don't cure a GIL-bound launch loop) — it is **one OS process per
GPU, each with its own interpreter**, which is exactly what MoE-Quant does.

**The "7.4 TB / 8 CPU copies" objection that pushed the single-process design was
wrong.** It assumed each process replicates the *full* model. MoE-Quant proves
you never do that: each rank materializes only its **disjoint 1/N expert shard**,
streamed from disk, and the full model is never resident anywhere. So the
multi-process path is both feasible and the proven one.

### How MoE-Quant loads a 671B model without holding it in one process

Verified by reading `quant.py` + `loading_utils.py` (`scratchpad/moequant/`):

1. **Empty skeleton.** `config.ep_size = world_size`, then
   `with init_empty_weights(): AutoModelForCausalLM.from_config(...)` — the whole
   model is built on the `meta` device (zero real weight bytes). Because of
   `ep_size`, each rank's expert submodules are sized to only `1/world_size` of
   the experts, so a rank's `block.state_dict()` keys *define its shard*.
2. **Only rank 0 touches disk.** It streams safetensor shard files into a CPU
   `param_buffer` dict (`safe_open` + `get_tensor`), pulling the *next* shard only
   when the current block's keys aren't all present yet (a sliding window, not the
   whole model).
3. **Per block** (`for block_idx, block in enumerate(model.model.layers)`):
   - rank 0 assembles the block's state dict from `param_buffer`, dequantizes
     FP8→bf16 (DeepSeek-specific; **we skip this — MiniMax-M3 is already bf16**);
   - `block.to_empty(device=cuda:rank)` materializes just this block's storage;
   - **dense block** (`block_idx < first_k_dense_replace`): rank 0 loads it and
     `broadcast`s to all ranks (replicated — these layers are small);
   - **MoE block**: rank 0 slices the expert state dict per rank and `dist.send`s
     each rank only its expert keys; each rank `recv`s and loads its shard.
4. **Quantize.** Shared layers: Hessian `all_reduce`d, quantized on rank 0,
   broadcast. Experts (`is_distributed=False`): each rank quantizes its own
   resident experts with **zero communication** — the embarrassingly-parallel core.
5. **Free and advance.** `block.to("meta")` + pop the block's keys from
   `param_buffer` → both GPU and CPU RAM released; activations are offloaded to
   CPU between blocks.

Peak memory per rank ≈ one block's shard on GPU + a few shard files in rank-0 CPU
+ calibration activations. That is how 671B fits on 8×H100 with modest host RAM.

### What we adopt vs. adapt for MiniMax-M3

| MoE-Quant | MiniMax-M3 adaptation |
|---|---|
| `init_empty_weights` + `from_config` on meta | Same, via the VL wrapper; target `model.model.language_model.layers.N` |
| Rank-0 sliding `param_buffer` shard streaming | Same, but discover shard filenames from the safetensors **index json** (not a hardcoded `-of-000163` pattern) |
| FP8 dequant per block | **Drop** — MiniMax-M3 weights are bf16 (simpler) |
| `first_k_dense_replace` dense/replicated path | Dense layers are **0–2** → `block_idx < 3` uses the replicated path |
| `ep_size` gives per-rank 1/N **per-expert Linears** | **De-risked (verified from HF `model.safetensors.index.json`).** The MiniMax-M3 *checkpoint* already stores experts per-expert — `…layers.N.block_sparse_moe.experts.{i}.w{1,2,3}.weight` (w1=gate, w3=up, w2=down) — **not** fused on disk (the fused `gate_up_proj[128,…]` is only a runtime module form). So the loader shards by expert index `i` over per-expert keys, exactly like MoE-Quant. Mechanical, not a fused-tensor problem. 59 shards, `model-XXXXX-of-00059.safetensors`. |
| `ep_size` also enables an **expert-parallel forward (all-to-all)** so each rank's calibration tokens reach the expert that owns them | **The load-bearing task, now chosen: implement the EP all-to-all forward.** With experts sharded, a rank cannot run `block(inputs)` alone — it lacks 7/8 of the experts needed to (a) collect each expert's full Hessian and (b) rebuild the full MoE output for activation propagation. **Dispatch/combine core built + CPU-tested** (`pipeline/ep_moe.py`; parity vs dense across world_size 1/2/4/8, top-k 1–4, with routing bias + shared expert). Remaining: swap the simulated per-rank loop for real `all_to_all_single` under `torchrun`/nccl, and wire in MiniMax's exact `block_sparse_moe` forward (on-cluster modeling file). |
| Custom Triton GPTQ kernel (~10×) | Optional later; start with llm-compressor's `quantize_weight` |

MiniMax-M3 MoE config (HF `config.json`, verified): 60 layers (dense 0–2, MoE
3–59), `hidden=6144`, expert `intermediate=3072`, 128 experts, **top-4**,
`scoring_func=sigmoid` + `use_routing_bias` (`e_score_correction_bias`),
`routed_scaling_factor=2.0`, **1 always-on shared expert**, `swigluoai`
(`alpha=1.702`, `limit=7.0`). The shared expert is replicated on every rank (a
local dense add, no all-to-all).

### GPTQ-first (recommended)

GPTQ is the currently-verified recipe and its per-expert *quantization step* is
fully independent (a local Hessian per expert; no cross-expert scale), so it is
exactly MoE-Quant's `is_distributed=False` expert path: each rank quantizes its 16
resident experts with no inter-rank comm. The comm is confined to the *forward*
(Hessian collection + activation propagation) per the crux in §6, not the solve.
Shared/attention/router layers take the replicated + `all_reduce` path.

### AWQ variant (only if AWQ becomes the chosen recipe)

Per-expert up→down mappings shard the same way. The single shared-input mapping
(post-attn-norm → all experts' gate/up, one shared scale) is not expert-parallel;
its full-MoE forward runs once with `all_reduce`d stats. Expect a smaller win
than GPTQ.

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
   freed 8-GPU node). It persists a JSON record incrementally to
   `results/m3-expert-scatter-bench/expert_scatter_bench.json` (override with
   `--out`) and also dumps the final JSON to stdout, so the result survives even
   if srun stdout is not captured. It measures serial-1-GPU vs thread-pool-8-GPU
   wall-clock on
   a GPTQ-shaped per-expert proxy (real MiniMax-M3 expert dims), reports the
   python-setup/CUDA split, and prints a go/no-go verdict on plan §6's top risk
   (does the GIL + column-loop launch overhead eat the scatter win?). Also run
   `nvidia-smi dmon -s u` during its serial phase to confirm per-expert SM
   underutilization. **Do not write Phase-2 production scatter until this shows
   real (≥~0.5×ceiling) parallelism.** Needs no model download, no chosen recipe,
   and no eval verdict — so it proceeds in parallel with the paired eval.

   **Phase-1 result (2026-07-14): thread-pool scatter is a no-go.** The
   GPTQ-shaped MiniMax-M3 proxy was run on `gpu-h125` with 8× H100 GPUs,
   128 experts, bf16 weights, and blocksize 128. The serial one-GPU baseline
   took **116.98 s**; the one-process thread-pool scatter took **132.97 s**,
   for **0.88× speedup** against an 8× ceiling. Python setup accounted for
   **45%** of the measured serial work. The benchmark therefore fails the
   `>=~0.5×` ceiling gate: do not wire the thread-pool implementation into
   production. The next viable experiment is the plan-6 fallback using
   per-device CUDA streams or a process-based design.

   Raw artifacts:
   `results/m3-expert-scatter-bench/expert_scatter_bench.json` and
   `results/m3-expert-scatter-bench-20260714T0829Z.out`.

   **Bench v2 (schema_version=2): representativeness fix + process mode.** The v1
   number was partly an artifact — 45% of the timed serial work was the bench
   *fabricating* synthetic weights/Hessians on CPU, which the real pipeline never
   does at quant time (weights are already on-GPU; the Hessian is accumulated in
   the calibration forward). v2 moves all data generation OUTSIDE the timed region
   (times only GPU-resident quant compute) and measures three modes:
   `serial` / `threads` / `processes` (`--mode`). The **`processes`** mode
   (one OS process per GPU, own interpreter → no shared GIL) is the mechanism that
   can actually work and mirrors MoE-Quant. Re-run:
   `python -m pipeline.bench_expert_scatter --experts 128 --mode all`. Decision
   rule unchanged: proceed only if the best real-parallel mode ≥ ~0.5×ceiling.
   **Bench v2 result (2026-07-14): process measurement blocked by runner bug.**
   The compute-only schema-v2 rerun on `gpu-h125` successfully recorded the
   serial baseline (**63.88 s**) and thread mode (**115.88 s**, **0.55×**),
   but then exited before process mode with
   `NameError: name 'ProcessPoolExecutor' is not defined`. This is a
   benchmark-runner import defect, not evidence about process-per-GPU
   performance. The JSON remains `status: "error"` and must not be used as a
   Phase-2 go/no-go verdict. Add the missing import and rerun `--mode all`
   before implementing the distributed loader/EP path.

   Raw v2 artifacts:
   `results/m3-expert-scatter-bench/expert_scatter_bench_v2.json` and
   `results/m3-expert-scatter-bench-20260714T0905Z-v2.out`.
2. **MoE-Quant-style sharded loader + EP forward + per-rank GPTQ (the real design;
   see §4).** `torchrun`-launched, one process per GPU. Adopt MoE-Quant's loader
   almost verbatim (empty meta skeleton, rank-0 sliding shard streaming, per-block
   `to_empty` → shard-send → quantize-local → `to("meta")`); sharding is by
   per-expert checkpoint key `experts.{i}.w{1,2,3}` (de-risked — not fused on disk).

   **EP all-to-all forward — dispatch/combine core DONE + CPU-tested**
   (`pipeline/ep_moe.py`, `pipeline/tests/test_ep_moe.py`): `route` (MiniMax
   sigmoid + routing-bias + scaling), `plan_dispatch` (per-rank split sizes +
   permutation for `all_to_all_single`), `ep_moe_simulated` (route → dispatch →
   local experts → combine), matched bit-for-bit against a dense reference across
   world_size 1/2/4/8 including the shared expert. Remaining (GPU): replace the
   simulated per-rank loop with real `all_to_all_single`/nccl under `torchrun`,
   and bind to MiniMax's on-cluster `block_sparse_moe` forward. Then each rank runs
   llm-compressor's `quantize_weight` on its ~16 resident experts; shared/attention
   /router layers take the replicated + `all_reduce` path.

   *Reusable from the abandoned single-process scatter:* `pipeline/expert_scatter.py`
   still applies as each rank's **local** per-expert quantization loop (its
   `serial_quantize` + the CPU bit-parity test carry over unchanged — within a
   rank there is no GIL contention because there is nothing to parallelize
   further). The thread-pool `scatter_quantize` path is retired.

   Correctness gate: on a tiny 2-layer MoE config, the sharded multi-rank run must
   produce quantized weights bit-identical (within GPTQ's numerical tolerance) to
   a single-rank serial run.
3. **Optional Triton GPTQ kernel** port (MoE-Quant `src/gptq_loop.py` +
   `linalg_utils`) for the per-expert inner loop if step 2 isn't enough.
4. **AWQ variant** (only if AWQ is chosen): shard the per-expert mappings the same
   way; keep the shared-input mapping replicated + `all_reduce`d.
5. **Verify quality unchanged.** Re-run the paired eval (§0 harness) on the
   fast-path checkpoint; it must match the serial-path checkpoint within noise.

## 6. Risks / open questions

- **RESOLVED — single-process 8-GPU concurrency**: the Phase-1 bench settled it.
  A shared-interpreter thread pool cannot parallelize GPTQ's GIL-bound per-column
  launch loop (0.88× measured). Decision: one OS process per GPU (MoE-Quant
  model). CUDA streams were considered and rejected (don't fix a GIL-bound loop).
- **Sharded forward mechanism (chosen: EP all-to-all; core done)**: the
  dispatch/combine bookkeeping — the part that gets EP forwards wrong — is built
  and CPU-parity-tested in `pipeline/ep_moe.py`. Remaining risk is in the GPU
  integration: real `all_to_all_single` split-size handling under nccl, and
  binding to MiniMax's exact `block_sparse_moe` forward from the on-cluster
  modeling file (which I cannot access locally — the HF repo ships only the
  config, no `modeling_*.py`). Validate on a tiny 2-layer config before any full
  run. Until the GPU integration lands, the end-to-end speedup is unproven.
- **Fused-expert tensor slicing — de-risked**: the checkpoint stores experts
  per-expert (`experts.{i}.w{1,2,3}`), so sharding is by expert index like
  MoE-Quant; the fused runtime tensor is not involved at load.
- **Router-weight normalization**: `ep_moe.route` uses a DeepSeek-V3-style
  normalize-then-scale; confirm against MiniMax's modeling forward (config gives
  sigmoid + bias + `routed_scaling_factor` but not the exact norm). Dispatch parity
  is independent of this, but the final numbers are not.
- **Numerical parity**: sharded multi-rank output must match single-rank serial
  within GPTQ tolerance. Gate on the tiny-model parity test before any full run.
- **Distributed loader correctness**: rank-0 shard streaming + send/recv of expert
  slices must deliver exactly each rank's keys; a missing/misrouted key corrupts a
  rank silently. Assert key-set coverage per rank (MoE-Quant does this via
  `send_object_list` of keys).
- **Scope discipline**: this is a speed project. It must not alter the
  quantization math/recipe that passed the quality eval.

## 7. Non-goals

- No custom expert-parallel forward / all-to-all (unnecessary here).
- No multi-process/torchrun replication (host RAM can't hold 8 copies).
- No change to the quantization recipe, scheme, or calibration data.
