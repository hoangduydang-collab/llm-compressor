#!/bin/bash
#SBATCH --job-name=quantize
#SBATCH --partition=compute
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=32
#SBATCH --mem=0
#SBATCH --time=48:00:00
#SBATCH --output=/mnt/nfs/hoangduy/logs/quantize-%j.out
#
# Generic quantize-only job (sequential onloading -> needs 1 GPU + lots of CPU RAM).
# Sized for very large models (e.g. MiniMax-M3 ~428B): --mem=0 reserves the whole
# node's RAM (the BF16 model is loaded to CPU and onloaded layer-by-layer).
#
#   CONFIG=pipeline/configs/minimax_m3.yaml METHOD=gptq sbatch pipeline/slurm/quantize.sbatch
#   CONFIG=pipeline/configs/minimax_m3.yaml METHOD=awq  sbatch pipeline/slurm/quantize.sbatch
# Optional: SCHEME (default W4AFP8), EXTRA (extra --set overrides).

set -uo pipefail
source /mnt/nfs/hoangduy/env.sh
source /mnt/nfs/hoangduy/venvs/quant/bin/activate
cd /mnt/nfs/hoangduy/projects/llm-compressor
export HOME=${WORK_ROOT:-/mnt/nfs/hoangduy}
export FLASHINFER_WORKSPACE_DIR=${FLASHINFER_WORKSPACE_DIR:-$HOME/cache/flashinfer}

CONFIG=${CONFIG:?set CONFIG=path/to/config.yaml}
METHOD=${METHOD:-gptq}
SCHEME=${SCHEME:-W4AFP8}
EXTRA=${EXTRA:-}

echo "host=$(hostname) job=${SLURM_JOB_ID:-none} config=$CONFIG method=$METHOD scheme=$SCHEME"
nvidia-smi --query-gpu=index,name,memory.total --format=csv || true
free -g || true

python -m pipeline.run --config "$CONFIG" --stage quantize \
  --set quantization.method="$METHOD" \
  --set quantization.scheme="$SCHEME" $EXTRA

echo "QUANTIZE DONE (rc=$?)"
