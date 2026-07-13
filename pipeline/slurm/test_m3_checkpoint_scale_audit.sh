#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
cd "$REPO_ROOT"

ENV_FILE="${ENV_FILE:-/mnt/nfs/hoangduy/env.sh}"
VENV_ACTIVATE="${VENV_ACTIVATE:-/mnt/nfs/hoangduy/venvs/quant/bin/activate}"
BASE_CKPT="${BASE_CKPT:-/mnt/nfs/hoangduy/hf_assets/MiniMaxAI/MiniMax-M3}"
REFERENCE_CKPT="${REFERENCE_CKPT:-/mnt/nfs/hoangduy/hf_assets/cyankiwi/MiniMax-M3-AWQ-INT4}"
AWQ_CKPT="${AWQ_CKPT:-/mnt/nfs/hoangduy/projects/llm-compressor/artifacts/MiniMax-M3-awq-W4AFP8/20260709-064104/checkpoint-vllm-w123}"
GPTQ_CKPT="${GPTQ_CKPT:-/mnt/nfs/hoangduy/projects/llm-compressor/artifacts/m3-awq-gptq-prepared/gptq-checkpoint-vllm-w123}"
OUTPUT="${OUTPUT:?set OUTPUT to checkpoint_scale_audit.json}"
LAYERS="${M3_DIAGNOSTIC_LAYERS:-$(seq -s, 3 59)}"

source "$ENV_FILE"
source "$VENV_ACTIVATE"
export PYTHONPATH="$REPO_ROOT"
python -m pipeline.m3_checkpoint_scale_audit \
  --base "$BASE_CKPT" --reference "$REFERENCE_CKPT" \
  --awq "$AWQ_CKPT" --gptq "$GPTQ_CKPT" \
  --layers "$LAYERS" --output "$OUTPUT"
