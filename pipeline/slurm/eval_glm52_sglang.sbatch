#!/bin/bash
#SBATCH --job-name=glm52-eval
#SBATCH --partition=compute
#SBATCH --gres=gpu:8
#SBATCH --cpus-per-task=64
#SBATCH --mem=0
#SBATCH --time=7-00:00:00
#SBATCH --output=/mnt/nfs/hoangduy/logs/glm52-eval-%j.out
#
# Full static lm-eval (8 tasks) for PhalaCloud/GLM-5.2-W4AFP8 via SGLang (TP=8).
# Requires: sglang-eval venv, CUDA 12.9 in env.sh, 8x Hopper GPUs.
#
# Submit:
#   bash pipeline/slurm/submit_eval_glm52_sglang.sh
#
# Or directly:
#   sbatch pipeline/slurm/eval_glm52_sglang.sbatch
#
# Overrides (injected by submit wrapper or export before sbatch):
#   CONFIG=pipeline/configs/eval_glm52_w4afp8_sglang_h100.yaml
#   OUT_DIR=evals/glm52-w4afp8-phala
#   MODEL_PATH=/mnt/nfs/hoangduy/hf_assets/PhalaCloud/GLM-5.2-W4AFP8

set -uo pipefail

source /mnt/nfs/hoangduy/env.sh
source /mnt/nfs/hoangduy/venvs/sglang-eval/bin/activate
cd /mnt/nfs/hoangduy/projects/llm-compressor

export HOME=${WORK_ROOT:-/mnt/nfs/hoangduy}
export PYTHONPATH=/mnt/nfs/hoangduy/projects/llm-compressor
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export FLASHINFER_WORKSPACE_DIR=${FLASHINFER_WORKSPACE_DIR:-$HOME/cache/flashinfer}
mkdir -p "$FLASHINFER_WORKSPACE_DIR" /mnt/nfs/hoangduy/logs

CONFIG=${CONFIG:-pipeline/configs/eval_glm52_w4afp8_sglang_h100.yaml}
OUT_DIR=${OUT_DIR:-evals/glm52-w4afp8-phala}
MODEL_PATH=${MODEL_PATH:-/mnt/nfs/hoangduy/hf_assets/PhalaCloud/GLM-5.2-W4AFP8}

echo "host=$(hostname) job=${SLURM_JOB_ID:-none}"
echo "config=$CONFIG"
echo "model=$MODEL_PATH"
echo "out=$OUT_DIR"
echo "cuda_home=${CUDA_HOME:-unset} nvcc=${DG_JIT_NVCC_COMPILER:-unset}"
nvidia-smi --query-gpu=index,name,memory.total --format=csv || true
if [[ -n "${DG_JIT_NVCC_COMPILER:-}" && -x "$DG_JIT_NVCC_COMPILER" ]]; then
  "$DG_JIT_NVCC_COMPILER" --version
elif command -v nvcc >/dev/null; then
  nvcc --version
fi

if [[ ! -f "$MODEL_PATH/config.json" ]]; then
  echo "[FAIL] model not found: $MODEL_PATH (missing config.json)"
  exit 1
fi

mkdir -p "$OUT_DIR"
LOG="/mnt/nfs/hoangduy/logs/glm52-eval-${SLURM_JOB_ID:-local}.log"
echo "tee log -> $OUT_DIR/run.log and $LOG"

python -m pipeline.evalsuite.cli run \
  --config "$CONFIG" \
  --model "$MODEL_PATH" \
  --out "$OUT_DIR" \
  2>&1 | tee -a "$OUT_DIR/run.log" "$LOG"
rc=${PIPESTATUS[0]}

echo ""
echo "GLM52 SGLANG EVAL DONE (rc=$rc)"
if [[ -f "$OUT_DIR/aggregate.json" ]]; then
  echo "results: $OUT_DIR/aggregate.json"
else
  echo "[WARN] no aggregate.json — check $OUT_DIR/run.log"
fi
exit $rc
