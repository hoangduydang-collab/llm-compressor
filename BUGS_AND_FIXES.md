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
