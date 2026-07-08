#!/usr/bin/env bash
# Full-calibration MiniMax-M3 quantize (512 samples x 2048 seq).
#
# Defaults to AWQ W4AFP8 — the scheme validated on h118 with CUDA graphs + serve.
# Needs a dedicated node: ~428B BF16 loads to CPU RAM (~850GB+), 1 GPU for
# sequential onloading. Survives SSH disconnect via the detached launcher.
#
#   bash pipeline/slurm/run_quantize_minimax_m3_full_calib.sh
#
# Env:
#   METHOD   awq | gptq (default: awq)
#   SCHEME   W4AFP8 | W4A8 (default: W4AFP8)
#   MODEL_ID default: /mnt/nfs/hoangduy/hf_assets/MiniMaxAI/MiniMax-M3
#   SBATCH=1 submit via Slurm instead of detached (96h wall time)
#   EXTRA    extra --set flags forwarded to pipeline.run
#
# Monitor:
#   tail -f /mnt/nfs/hoangduy/logs/quantize-m3-awq-full.log
#
# After quantize completes, serve-verify the new checkpoint:
#   CHECKPOINT="$(ls -td artifacts/MiniMax-M3-awq-W4AFP8/*/checkpoint | head -1)"
#   CHECKPOINT="$CHECKPOINT" bash pipeline/slurm/run_serve_minimax_m3_detached.sh

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
cd "$REPO_ROOT"

METHOD="${METHOD:-awq}"
SCHEME="${SCHEME:-W4AFP8}"
CONFIG="${CONFIG:-pipeline/configs/minimax_m3_full_calib.yaml}"
MODEL_ID="${MODEL_ID:-/mnt/nfs/hoangduy/hf_assets/MiniMaxAI/MiniMax-M3}"
EXTRA="${EXTRA:-}"
LOG="/mnt/nfs/hoangduy/logs/quantize-m3-${METHOD}-full.log"
PID_FILE="/mnt/nfs/hoangduy/logs/quantize-m3-${METHOD}-full.pid"

if [[ "${SBATCH:-0}" == "1" ]]; then
  echo "Submitting full-calibration quantize via Slurm (96h)"
  CONFIG="$CONFIG" METHOD="$METHOD" SCHEME="$SCHEME" MODEL_ID="$MODEL_ID" \
    LOG="$LOG" EXTRA="$EXTRA" \
    sbatch --time=96:00:00 pipeline/slurm/quantize.sbatch
  exit 0
fi

export CONFIG METHOD SCHEME MODEL_ID EXTRA LOG PID_FILE
exec bash pipeline/slurm/run_quantize_minimax_m3_detached.sh
