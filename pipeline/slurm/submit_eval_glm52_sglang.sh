#!/usr/bin/env bash
# Submit GLM-5.2-W4AFP8 full static eval (SGLang TP=8, 8 lm-eval tasks).
#
# Usage (from repo root on the cluster, as hoangduy):
#   bash pipeline/slurm/submit_eval_glm52_sglang.sh
#
# Options (env vars):
#   CONFIG   default: pipeline/configs/eval_glm52_w4afp8_sglang_h100.yaml
#   OUT_DIR  default: evals/glm52-w4afp8-phala
#   MODEL_PATH  default: /mnt/nfs/hoangduy/hf_assets/PhalaCloud/GLM-5.2-W4AFP8
#   SBATCH_EXTRA  extra sbatch flags, e.g. '--nodelist=h119-gpu-polaris'
#
# Examples:
#   bash pipeline/slurm/submit_eval_glm52_sglang.sh
#   OUT_DIR=evals/glm52-smoke CONFIG=pipeline/configs/eval_glm52_w4afp8_sglang_h100.yaml \
#     SBATCH_EXTRA='--time=12:00:00' bash pipeline/slurm/submit_eval_glm52_sglang.sh

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
cd "$REPO_ROOT"

CONFIG="${CONFIG:-pipeline/configs/eval_glm52_w4afp8_sglang_h100.yaml}"
OUT_DIR="${OUT_DIR:-evals/glm52-w4afp8-phala}"
MODEL_PATH="${MODEL_PATH:-/mnt/nfs/hoangduy/hf_assets/PhalaCloud/GLM-5.2-W4AFP8}"
SBATCH_EXTRA="${SBATCH_EXTRA:-}"

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
    HOME="${HOME:-/mnt/nfs/hoangduy}" \
    USER="${USER:-hoangduy}" \
    PATH="${PATH:-/usr/bin:/bin:/usr/local/bin}" \
    LANG="${LANG:-C.UTF-8}" \
    sbatch --export=NONE $SBATCH_EXTRA "$TMP_JOB"
}

if ! JOB_LINE="$(submit_job 2>&1)"; then
  echo ""
  echo "sbatch failed:"
  echo "$JOB_LINE"
  echo ""
  echo "Common fixes:"
  echo "  sacctmgr show user \$USER          # Slurm account?"
  echo "  sinfo -o '%P %a %l %D %G'           # partitions / GRES?"
  echo "  squeue -u \$USER"
  echo ""
  echo "If GRES gpu:8 is unavailable, try pinning a known 8-GPU node:"
  echo "  SBATCH_EXTRA='--nodelist=h119-gpu-polaris' bash pipeline/slurm/submit_eval_glm52_sglang.sh"
  echo ""
  echo "If NFS spool I/O error, the staged script is at: $TMP_JOB"
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
