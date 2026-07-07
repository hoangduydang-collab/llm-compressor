#!/usr/bin/env bash
# Run MiniMax-M3 vLLM serve-verify in the current shell (no sbatch).
# Use when Slurm submission fails but you have an idle 8-GPU node (salloc / interactive).
#
# VENV: serve_minimax_m3.sbatch activates venvs/quant (vLLM), not sglang-eval.
#
#   bash pipeline/slurm/run_serve_minimax_m3_local.sh
#
# Options: CONFIG, OUT_DIR, CHECKPOINT (same as submit_serve_minimax_m3.sh)

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
cd "$REPO_ROOT"

export CONFIG="${CONFIG:-pipeline/configs/minimax_m3.yaml}"
export OUT_DIR="${OUT_DIR:-serves/m3-awq-w4afp8}"
export CHECKPOINT="${CHECKPOINT:-artifacts/MiniMax-M3-awq-W4AFP8/20260707-082218/checkpoint}"
export MAX_MODEL_LEN="${MAX_MODEL_LEN:-8192}"
export GPU_UTIL="${GPU_UTIL:-0.9}"

echo "host=$(hostname) local MiniMax-M3 vLLM serve-verify (no sbatch)"
nvidia-smi --query-gpu=index,name,memory.total --format=csv || true

bash pipeline/slurm/serve_minimax_m3.sbatch
