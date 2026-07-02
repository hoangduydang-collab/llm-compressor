#!/usr/bin/env bash
# Download MiniMaxAI/MiniMax-M3 weights to NFS for quantization.
#
# Prerequisites:
#   1. Accept the model license on https://huggingface.co/MiniMaxAI/MiniMax-M3
#   2. huggingface-cli login  (or export HF_TOKEN=...)
#   3. ~1 TB free under LOCAL_DIR (428B BF16 safetensors)
#
# Run on any compute node with good network (GPU not required):
#   bash pipeline/slurm/download_minimax_m3.sh
#
# Optional env:
#   LOCAL_DIR  — destination (default: $WORK_ROOT/hf_assets/MiniMaxAI/MiniMax-M3)
#   HF_TOKEN   — if not already logged in

set -euo pipefail

source /mnt/nfs/hoangduy/env.sh
source /mnt/nfs/hoangduy/venvs/quant/bin/activate
export HOME=${WORK_ROOT:-/mnt/nfs/hoangduy}

REPO_ID=MiniMaxAI/MiniMax-M3
LOCAL_DIR=${LOCAL_DIR:-$HOME/hf_assets/MiniMaxAI/MiniMax-M3}
export LOCAL_DIR

echo "host=$(hostname) repo=$REPO_ID dest=$LOCAL_DIR"
df -h "$(dirname "$LOCAL_DIR")" || true

if [[ ! -f "$LOCAL_DIR/config.json" ]]; then
  mkdir -p "$LOCAL_DIR"
  echo "[download] fetching $REPO_ID -> $LOCAL_DIR (resumable)..."
  hf download "$REPO_ID" \
    --local-dir "$LOCAL_DIR" \
    --local-dir-use-symlinks False
else
  echo "[download] $LOCAL_DIR/config.json exists; resuming incomplete shards..."
  hf download "$REPO_ID" \
    --local-dir "$LOCAL_DIR" \
    --local-dir-use-symlinks False
fi

echo "[download] verifying snapshot..."
python - <<'PY'
import json
import sys
from pathlib import Path

local = Path(__import__("os").environ["LOCAL_DIR"])
cfg = json.loads((local / "config.json").read_text())
arch = cfg.get("architectures", [])
shards = sorted(local.glob("*.safetensors"))
print(f"architectures={arch} safetensor_shards={len(shards)}")
if not shards:
    print("ERROR: no .safetensors files found", file=sys.stderr)
    sys.exit(1)
print("OK")
PY

echo "[download] done. Point quantize at the local copy:"
echo "  --set model.id=$LOCAL_DIR"
