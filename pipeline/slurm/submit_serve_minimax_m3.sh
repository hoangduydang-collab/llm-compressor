#!/usr/bin/env bash
# Submit MiniMax-M3 W4AFP8 vLLM serve-verify (TP=8, EP, fp8 KV).
#
# VENV: jobs use venvs/quant (vLLM). Do NOT use sglang-eval here — that venv is
# only for SGLang-backed eval (e.g. GLM-5.2) and requires nvcc >= 12.9 for
# DeepGEMM; see pipeline/README.md.
#
# Usage (from repo root on a Slurm LOGIN/head node — NOT a gpu-h* compute node):
#   bash pipeline/slurm/submit_serve_minimax_m3.sh
#
# Options (env vars):
#   CONFIG      default: pipeline/configs/minimax_m3.yaml
#   OUT_DIR     default: serves/m3-awq-w4afp8
#   CHECKPOINT  default: artifacts/MiniMax-M3-awq-W4AFP8/20260707-082218/checkpoint
#   MODEL_ID    default: /mnt/nfs/hoangduy/hf_assets/MiniMaxAI/MiniMax-M3 (processor source)
#   MAX_MODEL_LEN  default: 8192 (raise to 32768 after smoke passes)
#   ENFORCE_EAGER  default: 0 (set 1 if hang during CUDA graph capture)
#   SERVE_PERF     default: 0 (set 1 to re-enable FlashInfer fused all-reduce)
#   SBATCH_EXTRA  extra sbatch flags, e.g. '--nodelist=gpu-h118'
#
# Examples:
#   bash pipeline/slurm/submit_serve_minimax_m3.sh
#   CHECKPOINT=artifacts/MiniMax-M3-gptq-W4AFP8/<ts>/checkpoint \
#     OUT_DIR=serves/m3-gptq-w4afp8 \
#     bash pipeline/slurm/submit_serve_minimax_m3.sh
#
# If you SSH as ubuntu and must sudo:
#   sudo -u hoangduy env HOME=/mnt/nfs/hoangduy USER=hoangduy bash -lc '
#     cd /mnt/nfs/hoangduy/projects/llm-compressor
#     bash pipeline/slurm/submit_serve_minimax_m3.sh
#   '
# (sudo alone is NOT the usual sbatch failure; submitting FROM a compute node is.)

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
cd "$REPO_ROOT"

export HOME="${HOME:-/mnt/nfs/hoangduy}"
export USER="${USER:-hoangduy}"

CONFIG="${CONFIG:-pipeline/configs/minimax_m3.yaml}"
OUT_DIR="${OUT_DIR:-serves/m3-awq-w4afp8}"
CHECKPOINT="${CHECKPOINT:-artifacts/MiniMax-M3-awq-W4AFP8/20260707-082218/checkpoint}"
MODEL_ID="${MODEL_ID:-/mnt/nfs/hoangduy/hf_assets/MiniMaxAI/MiniMax-M3}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-8192}"
GPU_UTIL="${GPU_UTIL:-0.9}"
SBATCH_EXTRA="${SBATCH_EXTRA:-}"

HOST="$(hostname -s 2>/dev/null || hostname)"
if [[ "$HOST" == gpu-h* || "$HOST" == *-gpu-* ]]; then
  echo "WARN: hostname=$HOST looks like a GPU compute node."
  echo "      sbatch often fails here with:"
  echo "        I/O error writing script/environment to file"
  echo "      Submit from the jump/login head node instead, then let Slurm allocate GPUs."
  echo "      If you are already on an idle 8-GPU node, use:"
  echo "        bash pipeline/slurm/run_serve_minimax_m3_detached.sh"
  echo ""
fi

if [[ ! -f "$CONFIG" ]]; then
  echo "ERROR: config not found: $CONFIG"
  exit 1
fi
if [[ ! -d "$CHECKPOINT" || ! -f "$CHECKPOINT/config.json" ]]; then
  echo "ERROR: checkpoint not found: $CHECKPOINT (missing config.json)"
  exit 1
fi

mkdir -p /mnt/nfs/hoangduy/logs

echo "Submitting MiniMax-M3 W4AFP8 vLLM serve-verify (TP=8, EP)"
echo "  host:       $(hostname)"
echo "  user:       $USER (uid=$(id -u))"
echo "  home:       $HOME"
echo "  config:     $CONFIG"
echo "  checkpoint: $CHECKPOINT"
echo "  processor:  $MODEL_ID"
echo "  out:        $OUT_DIR"
echo "  max_model_len: $MAX_MODEL_LEN"
echo "  gpus:       8 (gres=gpu:8)"
echo "  time:       6 hours (override with SBATCH_EXTRA='--time=...')"

TMP_JOB="$(mktemp /tmp/serve_m3_${USER:-user}_XXXXXX.sbatch)"
cp pipeline/slurm/serve_minimax_m3.sbatch "$TMP_JOB"

inject=$'export CONFIG='"$(printf '%q' "$CONFIG")"$'\n'
inject+=$'export OUT_DIR='"$(printf '%q' "$OUT_DIR")"$'\n'
inject+=$'export CHECKPOINT='"$(printf '%q' "$CHECKPOINT")"$'\n'
inject+=$'export MODEL_ID='"$(printf '%q' "$MODEL_ID")"$'\n'
inject+=$'export MAX_MODEL_LEN='"$(printf '%q' "$MAX_MODEL_LEN")"$'\n'
inject+=$'export GPU_UTIL='"$(printf '%q' "$GPU_UTIL")"$'\n'
inject+=$'export ENFORCE_EAGER='"$(printf '%q' "${ENFORCE_EAGER:-0}")"$'\n'
inject+=$'export SERVE_PERF='"$(printf '%q' "${SERVE_PERF:-0}")"$'\n'
if [[ -n "${DISABLE_CUSTOM_ALL_REDUCE+x}" ]]; then
  inject+=$'export DISABLE_CUSTOM_ALL_REDUCE='"$(printf '%q' "$DISABLE_CUSTOM_ALL_REDUCE")"$'\n'
fi

awk -v inject="$inject" '
  /^set -uo pipefail/ { print; print inject; next }
  { print }
' "$TMP_JOB" > "${TMP_JOB}.new"
mv "${TMP_JOB}.new" "$TMP_JOB"
chmod +x "$TMP_JOB"

submit_job() {
  # shellcheck disable=SC2086
  env -i \
    HOME="$HOME" \
    USER="$USER" \
    PATH="${PATH:-/usr/bin:/bin:/usr/local/sbin:/usr/local/bin:/sbin:/bin}" \
    LANG="${LANG:-C.UTF-8}" \
    sbatch --export=NONE --chdir=/tmp $SBATCH_EXTRA "$TMP_JOB"
}

if ! JOB_LINE="$(submit_job 2>&1)"; then
  echo ""
  echo "sbatch failed:"
  echo "$JOB_LINE"
  echo ""
  echo "Common fixes:"
  echo "  1) Run from jump/login head node (ssh jump.bitdeer.vip), NOT gpu-h118 / h118-gpu-polaris"
  echo "  2) export HOME=/mnt/nfs/hoangduy   # avoid /home/ubuntu when using sudo"
  echo "  sacctmgr show user \$USER          # Slurm account?"
  echo "  sinfo -o '%P %a %l %D %G'           # partitions / GRES?"
  echo "  squeue -u \$USER"
  echo "  df -h /tmp /var/spool/slurm 2>/dev/null || true"
  echo ""
  echo "If GRES gpu:8 is unavailable, pin a known 8-GPU node:"
  echo "  SBATCH_EXTRA='--nodelist=gpu-h118' bash pipeline/slurm/submit_serve_minimax_m3.sh"
  echo ""
  echo "If still failing on a compute node — run detached locally:"
  echo "  bash pipeline/slurm/run_serve_minimax_m3_detached.sh"
  echo "  tail -f $OUT_DIR/run.log"
  echo ""
  echo "Staged script kept at: $TMP_JOB"
  exit 1
fi

JOB_ID="${JOB_LINE##* }"
LOG_PATH="/mnt/nfs/hoangduy/logs/m3-serve-${JOB_ID}.out"

echo ""
echo "$JOB_LINE"
echo "  monitor:  tail -f $LOG_PATH"
echo "  monitor:  tail -f $OUT_DIR/run.log"
echo "  status:   squeue -j $JOB_ID"
echo "  cancel:   scancel $JOB_ID"
echo "  report:   $(dirname "$CHECKPOINT")/serve_report.json"

rm -f "$TMP_JOB" 2>/dev/null || true
