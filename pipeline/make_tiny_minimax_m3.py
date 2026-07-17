"""Build a miniature random-weight MiniMax-M3 VL checkpoint for CPU/2-GPU
end-to-end rehearsals of the distributed quantization path.

The tiny model mirrors every structural property the pipeline depends on:

- 4 decoder layers: dense MLP + full attention on 0-2, sparse MoE + sparse
  attention (lightning indexer) on 3 -- so the production ignore regexes
  (dense ``layers.[0-2]`` exclusion, layer-3 target, ``self_attn`` ignore) and
  the registered AWQ mappings (which expect ``self_attn.indexer.{q,k}_proj`` on
  sparse layers) resolve exactly as they do on the real 60-layer model.
- 8 per-expert 2D checkpoint keys (``experts.{i}.w{1,2,3}.weight``), so the
  load-time linearize mapping (``load_quantizable_moe``) engages identically.
- The real MiniMax-M3 tokenizer files are copied in so ultrachat calibration
  tokenization matches production.

Rehearsal (mirrors pipeline/slurm/run_m3_distributed_quant_smoke_srun.sh's
worker on 2 local GPUs; ~2 min/method on H100s):

    python -m pipeline.make_tiny_minimax_m3 --out /tmp/tiny-m3 \
        --tokenizer-from /mnt/nfs/hoangduy/hf_assets/MiniMaxAI/MiniMax-M3
    torchrun --nproc_per_node=2 -m pipeline.run \
        --config pipeline/configs/tiny_m3_distributed_e2e.yaml \
        --stage quantize --evidence-only \
        --set quantization.method=gptq \
        --set model.id=/tmp/tiny-m3 \
        --set model.offload_folder=/tmp/tiny-m3-offload \
        --set output_dir=/tmp/tiny-m3-out-gptq
    # then the same with quantization.method=awq

This rehearsal exists because every real-cluster iteration of the distributed
smoke costs 2-4 GPU-hours to surface one integration bug (r2-r6). Run it after
any change to the load/calibration/quantization/evidence path and require both
methods to exit 0 with nonempty per-rank ``quant_metrics.rank-*.jsonl`` before
authorizing a cluster launch.
"""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path

_TOKENIZER_FILES = (
    "tokenizer.json",
    "tokenizer_config.json",
    "special_tokens_map.json",
    "added_tokens.json",
    "chat_template.jinja",
)


def build_tiny_minimax_m3(out_dir: Path, tokenizer_from: Path | None) -> Path:
    import torch
    from transformers.models.minimax_m3_vl.configuration_minimax_m3_vl import (
        MiniMaxM3VLConfig,
        MiniMaxM3VLTextConfig,
        MiniMaxM3VLVisionConfig,
    )
    from transformers.models.minimax_m3_vl.modeling_minimax_m3_vl import (
        MiniMaxM3SparseForConditionalGeneration,
    )

    text = MiniMaxM3VLTextConfig(
        hidden_size=256,
        num_hidden_layers=4,
        # real model: dense MLP layers 0-2, sparse MoE 3-59
        mlp_layer_types=["dense", "dense", "dense", "sparse"],
        # real model: full attention 0-2, sparse attention (indexer) 3-59
        layer_types=["full_attention"] * 3 + ["minimax_m3_sparse"],
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=32,
        rotary_dim=16,
        index_head_dim=32,
        index_n_heads=2,
        num_local_experts=8,
        num_experts_per_tok=4,
        intermediate_size=128,
        dense_intermediate_size=256,
        shared_intermediate_size=128,
        # keep the real vocab so the real tokenizer can be reused verbatim
        vocab_size=200064,
    )
    vision = MiniMaxM3VLVisionConfig(
        hidden_size=64,
        intermediate_size=128,
        num_hidden_layers=2,
        num_attention_heads=2,
        image_size=112,
        patch_size=14,
    )
    config = MiniMaxM3VLConfig(text_config=text, vision_config=vision)
    config.architectures = ["MiniMaxM3SparseForConditionalGeneration"]

    torch.manual_seed(0)
    model = MiniMaxM3SparseForConditionalGeneration(config).to(torch.bfloat16)
    assert any("indexer" in name for name, _ in model.named_modules()), (
        "tiny model must carry the sparse-attention indexer so AWQ mappings "
        "resolve like they do on the real model"
    )

    out_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(str(out_dir))

    if tokenizer_from is not None:
        for name in _TOKENIZER_FILES:
            source = tokenizer_from / name
            if source.is_file():
                shutil.copy2(source, out_dir / name)

    return out_dir


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument(
        "--tokenizer-from",
        type=Path,
        default=Path("/mnt/nfs/hoangduy/hf_assets/MiniMaxAI/MiniMax-M3"),
        help="checkpoint directory whose tokenizer files are copied in",
    )
    args = parser.parse_args(argv)
    path = build_tiny_minimax_m3(args.out, args.tokenizer_from)
    print(f"tiny MiniMax-M3 checkpoint written to {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
