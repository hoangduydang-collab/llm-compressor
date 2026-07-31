#!/usr/bin/env bash
# DDP AutoRound W2A16 quant launcher (runs on the allocated node via srun).
# Parameterized by env: MODEL, SAVE_DIR, NPROC (default 8), NSAMPLES, ITERS.
# Fail-closed: aborts on GPU residue; verifies the saved artifact is a 2-bit
# pack-quantized compressed-tensors checkpoint before declaring PASS.
set -euo pipefail

DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV=/mnt/nfs/hoangduy/venvs/quant-sub4
export PATH="$VENV/bin:$PATH"
export HF_HOME=/mnt/nfs/hoangduy/hf_assets/xet
export HF_HUB_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false

export MODEL="${MODEL:?set MODEL}"
export SAVE_DIR="${SAVE_DIR:?set SAVE_DIR}"
export NSAMPLES="${NSAMPLES:-512}"
export ITERS="${ITERS:-200}"
NPROC="${NPROC:-8}"

echo "=== node: $(hostname)"
echo "=== model: $MODEL -> $SAVE_DIR (nproc=$NPROC nsamples=$NSAMPLES iters=$ITERS)"
echo "=== versions: $("$VENV/bin/python" -c 'from importlib.metadata import version as v; print("llmc", v("llmcompressor"), "| auto-round", v("auto-round"), "| ct", v("compressed-tensors"))')"

echo "=== GPU state before launch:"
nvidia-smi --query-gpu=index,memory.used --format=csv,noheader
busy=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | awk '$1 > 2000' | wc -l)
if [ "$busy" -gt 0 ]; then
  echo "QUANT_RESULT: FAIL (residue on $busy GPUs)"
  exit 1
fi

"$VENV/bin/torchrun" --nproc_per_node="$NPROC" --master_port=29613 \
  "$DIR/quant_w2a16_ddp.py"

echo "=== artifact gate:"
"$VENV/bin/python" - <<'EOF'
import glob
import json
import os
import sys

d = os.environ["SAVE_DIR"]
cfg = json.load(open(os.path.join(d, "config.json")))
q = cfg.get("quantization_config", {})
fmt = q.get("format")
g = q.get("config_groups", {}).get("group_0", {}).get("weights", {})
shards = glob.glob(os.path.join(d, "*.safetensors"))
ok = (
    fmt == "pack-quantized"
    and g.get("num_bits") == 2
    and g.get("group_size") == 128
    and g.get("symmetric") is True
    and len(shards) > 0
)
print(f"format={fmt} num_bits={g.get('num_bits')} group={g.get('group_size')} "
      f"sym={g.get('symmetric')} shards={len(shards)}")
print("QUANT_RESULT: " + ("PASS" if ok else "FAIL"))
sys.exit(0 if ok else 1)
EOF
