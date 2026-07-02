#!/usr/bin/env bash
# Run MiniMax-M3 quantize interactively (no sbatch).
#
# Each method needs a DEDICATED node: ~428B BF16 loads to CPU RAM (~850GB+).
# Do NOT run gptq + awq on the same host.
#
# Parallel GPTQ + AWQ (two idle GPU nodes):
#   Node A:
#     tmux new -s m3-gptq
#     METHOD=gptq bash pipeline/slurm/run_quantize_minimax_m3_local.sh
#   Node B:
#     tmux new -s m3-awq
#     METHOD=awq bash pipeline/slurm/run_quantize_minimax_m3_local.sh
#
# Env:
#   METHOD     gptq | awq (required)
#   MODEL_ID   default: /mnt/nfs/hoangduy/hf_assets/MiniMaxAI/MiniMax-M3
#   CONFIG     default: pipeline/configs/minimax_m3.yaml
#   SCHEME     default: W4AFP8
#   EXTRA      extra --set flags for pipeline.run

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
cd "$REPO_ROOT"

source /mnt/nfs/hoangduy/env.sh
source /mnt/nfs/hoangduy/venvs/quant/bin/activate
export HOME=${WORK_ROOT:-/mnt/nfs/hoangduy}
export FLASHINFER_WORKSPACE_DIR=${FLASHINFER_WORKSPACE_DIR:-$HOME/cache/flashinfer}
mkdir -p "$FLASHINFER_WORKSPACE_DIR" 2>/dev/null || true

export CONFIG="${CONFIG:-pipeline/configs/minimax_m3.yaml}"
export METHOD="${METHOD:?set METHOD=gptq or awq}"
export SCHEME="${SCHEME:-W4AFP8}"
export MODEL_ID="${MODEL_ID:-/mnt/nfs/hoangduy/hf_assets/MiniMaxAI/MiniMax-M3}"
EXTRA="${EXTRA:-}"

if [[ ! -f "$MODEL_ID/config.json" ]]; then
  echo "ERROR: model not found at $MODEL_ID"
  exit 1
fi

mkdir -p /mnt/nfs/hoangduy/logs
LOG="/mnt/nfs/hoangduy/logs/quantize-m3-${METHOD}-$(hostname)-$(date +%Y%m%d-%H%M%S).log"

echo "host=$(hostname) method=$METHOD scheme=$SCHEME"
echo "model=$MODEL_ID"
echo "log=$LOG"
nvidia-smi --query-gpu=index,name,memory.total --format=csv || true
free -g || true
echo ""

set -x
python -m pipeline.run --config "$CONFIG" --stage quantize \
  --set quantization.method="$METHOD" \
  --set quantization.scheme="$SCHEME" \
  --set model.id="$MODEL_ID" \
  $EXTRA \
  2>&1 | tee "$LOG"
rc=${PIPESTATUS[0]}

echo ""
echo "QUANTIZE DONE method=$METHOD rc=$rc"
echo "log: $LOG"
if [[ $rc -eq 0 ]]; then
  echo "checkpoint: artifacts/MiniMax-M3-${METHOD}-${SCHEME}/ (newest timestamp dir)"
fi
exit $rc
