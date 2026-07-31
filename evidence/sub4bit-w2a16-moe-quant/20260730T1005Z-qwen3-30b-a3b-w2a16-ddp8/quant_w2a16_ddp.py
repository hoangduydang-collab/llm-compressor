"""In-house sub-4-bit quant: W2A16 g128 sym, compressed-tensors pack-quantized.

DDP AutoRound on upstream llm-compressor main (0.12.1.dev92+g8cec0acc) +
auto-round 0.14.1 (fixed gradient sync: setup_ddp_if_needed_ returns
(block, sync_fn); grads all-reduced before every SignSGD step).

Launch:  torchrun --nproc_per_node=$NPROC quant_w2a16_ddp.py
Env:     MODEL, SAVE_DIR (required); NSAMPLES, ITERS, SEQLEN (optional)
Venv:    /mnt/nfs/hoangduy/venvs/quant-sub4
"""

import os

import torch.distributed as dist
from compressed_tensors.offload import init_dist
from transformers import AutoModelForCausalLM, AutoTokenizer

from llmcompressor import oneshot
from llmcompressor.utils import load_context

MODEL = os.environ["MODEL"]
SAVE_DIR = os.environ["SAVE_DIR"]
NSAMPLES = int(os.environ.get("NSAMPLES", "512"))
ITERS = int(os.environ.get("ITERS", "200"))
SEQLEN = int(os.environ.get("SEQLEN", "2048"))

init_dist()
with load_context():
    model = AutoModelForCausalLM.from_pretrained(MODEL, device_map="auto_offload")
tokenizer = AutoTokenizer.from_pretrained(MODEL)

# Upstream example requires the model to be loaded before auto-round imports.
from auto_round.calib_dataset import get_dataset  # noqa: E402

from llmcompressor.modifiers.autoround import AutoRoundModifier  # noqa: E402

ds = get_dataset(tokenizer=tokenizer, seqlen=SEQLEN, nsamples=NSAMPLES)

recipe = AutoRoundModifier(
    targets="Linear",
    # mlp.gate = MoE router (kept in BF16, same as the Yi30 reference checkpoint);
    # harmless no-op on dense smoke models.
    ignore=["lm_head", "re:.*mlp.gate$"],
    iters=ITERS,
    enable_torch_compile=False,
    config_groups={
        "group_0": {
            "targets": ["Linear"],
            "weights": {
                "num_bits": 2,
                "type": "int",
                "symmetric": True,
                "strategy": "group",
                "group_size": 128,
            },
        }
    },
)

oneshot(
    model=model,
    dataset=ds,
    recipe=recipe,
    max_seq_length=SEQLEN,
    num_calibration_samples=NSAMPLES,
    shuffle_calibration_samples=False,
)

rank = dist.get_rank()
# save_pretrained is rank-guarded internally (is_source_process); every rank
# must enter it because compression uses collectives.
model.save_pretrained(SAVE_DIR, save_compressed=True)
if rank == 0:
    tokenizer.save_pretrained(SAVE_DIR)
print(f"[rank {rank}] QUANT_DONE", flush=True)
dist.destroy_process_group()
