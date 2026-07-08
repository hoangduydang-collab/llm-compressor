#!/bin/bash
#SBATCH --job-name=m3-serve
#SBATCH --partition=compute
#SBATCH --gres=gpu:8
#SBATCH --cpus-per-task=64
#SBATCH --mem=0
#SBATCH --time=06:00:00
#SBATCH --output=/mnt/nfs/hoangduy/logs/m3-serve-%j.out
#
# vLLM serve-verify for MiniMax-M3 W4AFP8 checkpoint (TP=8, EP, fp8 KV).
# Requires: quant venv with vLLM (NOT sglang-eval — see pipeline/README.md
# "DeepGEMM / NVCC" and "sglang-eval venv" for why the venvs are separate).
# 8x Hopper GPUs.
#
# Submit:
#   bash pipeline/slurm/submit_serve_minimax_m3.sh
#
# Or directly:
#   sbatch pipeline/slurm/serve_minimax_m3.sbatch
#
# Overrides (injected by submit wrapper or export before sbatch):
#   CONFIG=pipeline/configs/minimax_m3.yaml
#   OUT_DIR=serves/m3-awq-w4afp8
#   CHECKPOINT=artifacts/MiniMax-M3-awq-W4AFP8/20260707-082218/checkpoint

set -uo pipefail

source /mnt/nfs/hoangduy/env.sh
source /mnt/nfs/hoangduy/venvs/quant/bin/activate
cd /mnt/nfs/hoangduy/projects/llm-compressor

export HOME=${WORK_ROOT:-/mnt/nfs/hoangduy}
export PYTHONPATH=/mnt/nfs/hoangduy/projects/llm-compressor
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export PYTHONUNBUFFERED=1
# Dump native stacks of all threads on a C++ abort (e.g. CuTe-DSL JIT SIGABRT)
# so kernel crashes are self-diagnosing instead of a bare "RuntimeError: cancelled".
export PYTHONFAULTHANDLER=1
# Force FlashInfer's CUDA-JIT norm kernels; its CuTe-DSL gemma_rmsnorm fails to
# JIT-compile against nvidia-cutlass-dsl 4.5.2 on Hopper. Must be set before
# flashinfer.norm import. See BUGS_AND_FIXES.md "FlashInfer gemma_rmsnorm CuTe-DSL".
export FLASHINFER_USE_CUDA_NORM=1
export FLASHINFER_WORKSPACE_DIR=${FLASHINFER_WORKSPACE_DIR:-$HOME/cache/flashinfer}
export TOKENIZERS_PARALLELISM=false
mkdir -p "$FLASHINFER_WORKSPACE_DIR" /mnt/nfs/hoangduy/logs

CONFIG=${CONFIG:-pipeline/configs/minimax_m3.yaml}
OUT_DIR=${OUT_DIR:-serves/m3-awq-w4afp8}
CHECKPOINT=${CHECKPOINT:-artifacts/MiniMax-M3-awq-W4AFP8/20260707-082218/checkpoint}
MODEL_ID=${MODEL_ID:-/mnt/nfs/hoangduy/hf_assets/MiniMaxAI/MiniMax-M3}
MAX_MODEL_LEN=${MAX_MODEL_LEN:-8192}
GPU_UTIL=${GPU_UTIL:-0.9}
ENFORCE_EAGER=${ENFORCE_EAGER:-0}

echo "host=$(hostname) job=${SLURM_JOB_ID:-none}"
echo "config=$CONFIG"
echo "checkpoint=$CHECKPOINT"
echo "processor_source=$MODEL_ID"
echo "out=$OUT_DIR"
echo "max_model_len=$MAX_MODEL_LEN gpu_util=$GPU_UTIL"
echo "enforce_eager=$([[ "$ENFORCE_EAGER" == "1" || "$ENFORCE_EAGER" == "true" ]] && echo true || echo false)"
python -c "import vllm; print('vllm', vllm.__version__)" 2>/dev/null || echo "vllm: not importable"
nvidia-smi --query-gpu=index,name,memory.total,memory.free --format=csv || true

if [[ ! -d "$CHECKPOINT" || ! -f "$CHECKPOINT/config.json" ]]; then
  echo "[FAIL] checkpoint not found: $CHECKPOINT (missing config.json)"
  exit 1
fi

mkdir -p "$OUT_DIR"
LOG="/mnt/nfs/hoangduy/logs/m3-serve-${SLURM_JOB_ID:-local}.log"
echo "tee log -> $OUT_DIR/run.log and $LOG"

python -m pipeline.run --config "$CONFIG" --stage serve \
  --checkpoint "$CHECKPOINT" \
  --set model.id="$MODEL_ID" \
  --set serve.tensor_parallel_size=8 \
  --set serve.enable_expert_parallel=true \
  --set serve.block_size=128 \
  --set serve.kv_cache_dtype=fp8 \
  --set serve.max_model_len="$MAX_MODEL_LEN" \
  --set serve.gpu_memory_utilization="$GPU_UTIL" \
  --set serve.enforce_eager="$([[ "$ENFORCE_EAGER" == "1" || "$ENFORCE_EAGER" == "true" ]] && echo true || echo false)" \
  --set eval.enabled=false \
  2>&1 | tee -a "$OUT_DIR/run.log" "$LOG"
rc=${PIPESTATUS[0]}

echo ""
echo "M3 VLLM SERVE DONE (rc=$rc)"
SERVE_REPORT="$(dirname "$CHECKPOINT")/serve_report.json"
if [[ -f "$SERVE_REPORT" ]]; then
  echo "serve_report: $SERVE_REPORT"
else
  echo "[WARN] no serve_report.json — check $OUT_DIR/run.log"
fi
exit $rc
