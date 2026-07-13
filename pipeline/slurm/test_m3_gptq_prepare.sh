#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
cd "$REPO_ROOT"

ENV_FILE="${ENV_FILE:-/mnt/nfs/hoangduy/env.sh}"
VENV_ACTIVATE="${VENV_ACTIVATE:-/mnt/nfs/hoangduy/venvs/quant/bin/activate}"
GPTQ_SOURCE="${GPTQ_SOURCE:-/mnt/nfs/hoangduy/projects/llm-compressor/artifacts/MiniMax-M3-gptq-W4AFP8/20260709-064842/checkpoint}"
PREPARED_ROOT="${PREPARED_ROOT:-/mnt/nfs/hoangduy/projects/llm-compressor/artifacts/m3-awq-gptq-prepared}"
GPTQ_PORTABLE="$PREPARED_ROOT/gptq-checkpoint-vllm-w123"

source "$ENV_FILE"
source "$VENV_ACTIVATE"
export PYTHONPATH="$REPO_ROOT"
mkdir -p "$PREPARED_ROOT"
if [[ ! -f "$GPTQ_PORTABLE/model.safetensors.index.json" ]]; then
  python -m pipeline.reexport_minimax_m3_vllm "$GPTQ_SOURCE" "$GPTQ_PORTABLE"
fi
echo "$GPTQ_PORTABLE"
