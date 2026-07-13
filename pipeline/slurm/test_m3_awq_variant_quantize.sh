#!/usr/bin/env bash
# Quantize one MiniMax-M3 AWQ repair variant and produce a portable checkpoint.

set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
cd "$REPO_ROOT"

VARIANT="${VARIANT:?set VARIANT=offsetfix or nosmooth}"
ENV_FILE="${ENV_FILE:-/mnt/nfs/hoangduy/env.sh}"
VENV_ACTIVATE="${VENV_ACTIVATE:-/mnt/nfs/hoangduy/venvs/quant/bin/activate}"
MODEL_ID="${MODEL_ID:-/mnt/nfs/hoangduy/hf_assets/MiniMaxAI/MiniMax-M3}"
CONFIG="${CONFIG:-pipeline/configs/minimax_m3_full_calib.yaml}"
PREPARED_ROOT="${PREPARED_ROOT:-/mnt/nfs/hoangduy/projects/llm-compressor/artifacts/m3-awq-gptq-prepared}"
OUTPUT_ROOT="$PREPARED_ROOT/quantized-$VARIANT"
PORTABLE_CKPT="$PREPARED_ROOT/awq-$VARIANT-checkpoint-vllm-w123"

case "$VARIANT" in
  offsetfix) export M3_AWQ_DISABLE_MLP_INPUT_SMOOTH=0 ;;
  nosmooth) export M3_AWQ_DISABLE_MLP_INPUT_SMOOTH=1 ;;
  *) echo "ERROR: unknown VARIANT=$VARIANT" >&2; exit 2 ;;
esac

source "$ENV_FILE"
source "$VENV_ACTIVATE"
export PYTHONPATH="$REPO_ROOT/src:$REPO_ROOT" PYTHONUNBUFFERED=1 TOKENIZERS_PARALLELISM=false
export HF_ENABLE_PARALLEL_LOADING="${HF_ENABLE_PARALLEL_LOADING:-true}"
export HF_PARALLEL_LOADING_WORKERS="${HF_PARALLEL_LOADING_WORKERS:-16}"

if [[ ! -f "$PORTABLE_CKPT/model.safetensors.index.json" ]]; then
  python -m pipeline.run --config "$CONFIG" --stage quantize \
    --set quantization.method=awq --set quantization.scheme=W4AFP8 \
    --set model.id="$MODEL_ID" --set output_dir="$OUTPUT_ROOT"
  SOURCE_CKPT="$(find "$OUTPUT_ROOT" -type d -name checkpoint -printf '%T@ %p\n' \
    | sort -nr | head -1 | cut -d' ' -f2-)"
  [[ -n "$SOURCE_CKPT" ]] || { echo "ERROR: quantized checkpoint not found" >&2; exit 1; }
  python -m pipeline.reexport_minimax_m3_vllm "$SOURCE_CKPT" "$PORTABLE_CKPT"
fi
python -m pipeline.reexport_minimax_m3_vllm \
  --help >/dev/null
echo "$PORTABLE_CKPT"
