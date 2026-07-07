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

**Fix applied (2026-07-07):**

1. `pipeline/minimax_m3_config.py` — `get_minimax_m3_awq_mappings()` + `register_minimax_m3_awq_mappings()` for sparse layers 3-59 only, anchored to `self_attn.*` and `mlp.experts.N.*`.
2. `pipeline/quantize.py` — register mappings after M3 text-calibration patch.
3. `pipeline/configs/minimax_m3.yaml` — translated keep-bf16 `quantization.ignore` list.

**KV cache:** Leave `kv_cache_scheme` unset in the recipe (`null` in saved config). fp8 KV is applied at serve time via `serve.kv_cache_dtype: fp8` / vLLM `--kv-cache-dtype fp8` (same as community recipe).

**Serve follow-up:** Stock vLLM may need the `toncao/vllm` `minimax-m3-compressed-tensors` branch to un-fuse the bf16 MSA indexer from quantized q/k/v projections for correct long-context output. W4AFP8 weight scheme is independent; validate on H100 with patched vLLM before production serve.

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
