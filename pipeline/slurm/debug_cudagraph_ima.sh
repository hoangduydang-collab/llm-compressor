#!/usr/bin/env bash
# Diagnose the MiniMax-M3 W4AFP8 CUDA-graph capture IMA (deterministic at ~16/51).
#
# Why: the async "illegal memory access ... at empty_cache()" traceback names
# breakable_cudagraph, NOT the kernel that actually faults. This launcher forces
# SYNCHRONOUS kernel launches + device-side assertions so the crash points at the
# real kernel (e.g. cutlass grouped GEMM, moe_unpermute/finalize, topk_softmax,
# or fused_allreduce_gemma_rms_norm). That tells us which fix is required:
#
#   * fault in MoE routing/finalize  -> router NaN patch (patch 4, select_experts)
#                                       and/or flashinfer >= 0.6.11.post2 (#42906)
#   * fault in fused AR              -> fused-AR NCCL fallback (patch 3) / #46253
#
# See BUGS_AND_FIXES.md "CUDA graph capture". Run on an idle 8-GPU node (h118).
#
#   bash pipeline/slurm/debug_cudagraph_ima.sh 2>&1 | tee /mnt/nfs/hoangduy/logs/m3-cudagraph-debug.log
#
# Then inspect the FIRST "illegal memory access" or "device-side assert" line and
# the frames just ABOVE it — that is the faulting kernel.

set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
cd "$REPO_ROOT"

source /mnt/nfs/hoangduy/env.sh
source /mnt/nfs/hoangduy/venvs/quant/bin/activate

CONFIG="${CONFIG:-pipeline/configs/minimax_m3.yaml}"
OUT_DIR="${OUT_DIR:-serves/m3-awq-w4afp8-cgdebug}"
CHECKPOINT="${CHECKPOINT:-artifacts/MiniMax-M3-awq-W4AFP8/20260707-082218/checkpoint}"
MODEL_ID="${MODEL_ID:-/mnt/nfs/hoangduy/hf_assets/MiniMaxAI/MiniMax-M3}"
# Keep it small: capture faults during profiling/dummy runs, not real traffic.
MAX_MODEL_LEN="${MAX_MODEL_LEN:-2048}"
GPU_UTIL="${GPU_UTIL:-0.85}"

export HOME="${WORK_ROOT:-/mnt/nfs/hoangduy}"
export PYTHONPATH="$REPO_ROOT"
export PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM=false
export FLASHINFER_USE_CUDA_NORM=1
export FLASHINFER_WORKSPACE_DIR="${FLASHINFER_WORKSPACE_DIR:-$HOME/cache/flashinfer}"
mkdir -p "$FLASHINFER_WORKSPACE_DIR" "$OUT_DIR" /mnt/nfs/hoangduy/logs

# --- the diagnostic knobs ---------------------------------------------------
# Synchronous launches so the Python frame maps to the faulting kernel.
export CUDA_LAUNCH_BLOCKING=1
# Device-side assertions (arms bounds checks compiled with TORCH_USE_CUDA_DSA).
export TORCH_USE_CUDA_DSA=1
# Native thread stacks on a C++ abort.
export PYTHONFAULTHANDLER=1
# ---------------------------------------------------------------------------

echo "host=$(hostname) cudagraph IMA diagnostic started=$(date -Is)"
echo "checkpoint=$CHECKPOINT max_model_len=$MAX_MODEL_LEN gpu_util=$GPU_UTIL"
python -c "import vllm; print('vllm', vllm.__version__)" 2>/dev/null || echo "vllm: not importable"
# Confirm the persistent patches (incl. the corrected select_experts NaN patch)
# and report the flashinfer version (finalize bounds-check suspect, #42906).
python pipeline/slurm/patch_vllm_m3_serve.py --check || {
  echo "[WARN] persistent patches NOT fully applied; applying now..."
  python pipeline/slurm/patch_vllm_m3_serve.py
}
nvidia-smi --query-gpu=index,name,memory.free --format=csv 2>/dev/null || true

if [[ ! -f "$CHECKPOINT/config.json" ]]; then
  echo "[FAIL] checkpoint not found: $CHECKPOINT"
  exit 1
fi

# enforce_eager=false => graph capture ON (the failing path we want to trace).
python -m pipeline.run --config "$CONFIG" --stage serve \
  --checkpoint "$CHECKPOINT" \
  --set model.id="$MODEL_ID" \
  --set serve.tensor_parallel_size=8 \
  --set serve.enable_expert_parallel=true \
  --set serve.block_size=128 \
  --set serve.kv_cache_dtype=fp8 \
  --set serve.max_model_len="$MAX_MODEL_LEN" \
  --set serve.gpu_memory_utilization="$GPU_UTIL" \
  --set serve.enforce_eager=false \
  --set eval.enabled=false \
  2>&1 | tee "$OUT_DIR/cudagraph-debug.log"
rc=${PIPESTATUS[0]}

echo ""
echo "=== faulting-kernel hints (first CUDA fault frames) ==="
grep -nE 'illegal memory access|device-side assert|CUDA error|finalizeMoeRouting|moe_unpermute|cutlass|grouped_gemm|topk_softmax|fused_allreduce|rms_norm' \
  "$OUT_DIR/cudagraph-debug.log" | head -60 || true
echo ""
echo "DONE rc=$rc  (full log: $OUT_DIR/cudagraph-debug.log)"
exit "$rc"
