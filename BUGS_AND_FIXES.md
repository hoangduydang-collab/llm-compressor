# Bugs and fixes (llm-compressor pipeline)

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

**Tooling:** `pipeline/verify_quant_checkpoint.py` runs all of the above structural
checks on checkpoint metadata (fast, no model load); `--check-tensors` adds sampled
finiteness + group-scale-shape checks:

```bash
python -m pipeline.verify_quant_checkpoint --ckpt artifacts/MiniMax-M3-awq-W4AFP8/<ts>/checkpoint
python -m pipeline.verify_quant_checkpoint --ckpt <dir> --check-tensors   # opens shards
```
