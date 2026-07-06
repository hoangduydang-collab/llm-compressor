#!/usr/bin/env bash
# Submit GLM-5.2-W4AFP8 full static eval (SGLang TP=8, 8 lm-eval tasks).
#
# Usage (from repo root on a Slurm LOGIN/head node — NOT a gpu-h* compute node):
#   bash pipeline/slurm/submit_eval_glm52_sglang.sh
#
# Options (env vars):
#   CONFIG   default: pipeline/configs/eval_glm52_w4afp8_sglang_h100_graphs.yaml
#   OUT_DIR  default: evals/glm52-w4afp8-phala
#   MODEL_PATH  default: /mnt/nfs/hoangduy/hf_assets/PhalaCloud/GLM-5.2-W4AFP8
#   SBATCH_EXTRA  extra sbatch flags, e.g. '--nodelist=gpu-h119'
#
# Examples:
#   bash pipeline/slurm/submit_eval_glm52_sglang.sh
#   CONFIG=pipeline/configs/eval_glm52_w4afp8_sglang_h100_8k.yaml \
#     OUT_DIR=evals/glm52-w4afp8-phala-8k \
#     bash pipeline/slurm/submit_eval_glm52_sglang.sh
#
# If you SSH as ubuntu and must sudo:
#   sudo -u hoangduy env HOME=/mnt/nfs/hoangduy USER=hoangduy bash -lc '
#     cd /mnt/nfs/hoangduy/projects/llm-compressor
#     bash pipeline/slurm/submit_eval_glm52_sglang.sh
#   '
# (sudo alone is NOT the usual sbatch failure; submitting FROM a compute node is.)

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
cd "$REPO_ROOT"

# Slurm maps jobs to the submitting UID; keep HOME on NFS for logs/cache.
export HOME="${HOME:-/mnt/nfs/hoangduy}"
export USER="${USER:-hoangduy}"

CONFIG="${CONFIG:-pipeline/configs/eval_glm52_w4afp8_sglang_h100_graphs.yaml}"
OUT_DIR="${OUT_DIR:-evals/glm52-w4afp8-phala}"
MODEL_PATH="${MODEL_PATH:-/mnt/nfs/hoangduy/hf_assets/PhalaCloud/GLM-5.2-W4AFP8}"
SBATCH_EXTRA="${SBATCH_EXTRA:-}"

HOST="$(hostname -s 2>/dev/null || hostname)"
if [[ "$HOST" == gpu-h* || "$HOST" == *-gpu-* ]]; then
  echo "WARN: hostname=$HOST looks like a GPU compute node."
  echo "      sbatch often fails here with:"
  echo "        I/O error writing script/environment to file"
  echo "      Submit from the jump/login head node instead, then let Slurm allocate GPUs."
  echo "      If you are already on an idle 8-GPU node, use:"
  echo "        bash pipeline/slurm/run_eval_glm52_sglang_detached.sh"
  echo ""
fi

if [[ ! -f "$CONFIG" ]]; then
  echo "ERROR: config not found: $CONFIG"
  exit 1
fi
if [[ ! -f "$MODEL_PATH/config.json" ]]; then
  echo "ERROR: model not found: $MODEL_PATH (missing config.json)"
  exit 1
fi

mkdir -p /mnt/nfs/hoangduy/logs

echo "Submitting GLM-5.2-W4AFP8 SGLang eval (TP=8, 8 tasks)"
echo "  host:    $(hostname)"
echo "  user:    $USER (uid=$(id -u))"
echo "  home:    $HOME"
echo "  config:  $CONFIG"
echo "  model:   $MODEL_PATH"
echo "  out:     $OUT_DIR"
echo "  gpus:    8 (gres=gpu:8)"
echo "  time:    7 days (override with SBATCH_EXTRA='--time=...')"

# Stage on local disk: some Slurm controllers fail to spool batch scripts from NFS.
TMP_JOB="$(mktemp /tmp/eval_glm52_sglang_${USER:-user}_XXXXXX.sbatch)"
cp pipeline/slurm/eval_glm52_sglang.sbatch "$TMP_JOB"

inject=$'export CONFIG='"$(printf '%q' "$CONFIG")"$'\n'
inject+=$'export OUT_DIR='"$(printf '%q' "$OUT_DIR")"$'\n'
inject+=$'export MODEL_PATH='"$(printf '%q' "$MODEL_PATH")"$'\n'

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
  echo "  1) Run from jump/login head node (ssh jump.bitdeer.vip), NOT gpu-h119 / h119-gpu-polaris"
  echo "  2) export HOME=/mnt/nfs/hoangduy   # avoid /home/ubuntu when using sudo"
  echo "  sacctmgr show user \$USER          # Slurm account?"
  echo "  sinfo -o '%P %a %l %D %G'           # partitions / GRES?"
  echo "  squeue -u \$USER"
  echo "  df -h /tmp /var/spool/slurm 2>/dev/null || true"
  echo ""
  echo "If GRES gpu:8 is unavailable, pin a known 8-GPU node:"
  echo "  SBATCH_EXTRA='--nodelist=gpu-h119' bash pipeline/slurm/submit_eval_glm52_sglang.sh"
  echo ""
  echo "If still failing on a compute node — run detached locally:"
  echo "  bash pipeline/slurm/run_eval_glm52_sglang_detached.sh"
  echo "  tail -f $OUT_DIR/run.log"
  echo ""
  echo "Staged script kept at: $TMP_JOB"
  exit 1
fi

JOB_ID="${JOB_LINE##* }"
LOG_PATH="/mnt/nfs/hoangduy/logs/glm52-eval-${JOB_ID}.out"

echo ""
echo "$JOB_LINE"
echo "  monitor:  tail -f $LOG_PATH"
echo "  monitor:  tail -f $OUT_DIR/run.log"
echo "  status:   squeue -j $JOB_ID"
echo "  cancel:   scancel $JOB_ID"

rm -f "$TMP_JOB" 2>/dev/null || true
