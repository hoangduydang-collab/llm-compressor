#!/usr/bin/env bash
# Submit parallel GPTQ + AWQ quantize-only jobs for MiniMax-M3.
#
# Each job needs a full compute node (--mem=0): the ~428B BF16 model loads to
# CPU RAM and onloads one decoder layer at a time onto 1 GPU.
#
# Usage (from repo root on the cluster):
#   bash pipeline/slurm/submit_quantize_minimax_m3.sh
#
# Options (env vars):
#   MODEL_ID   local weights dir (default: /mnt/nfs/hoangduy/hf_assets/MiniMaxAI/MiniMax-M3)
#   CONFIG     pipeline config (default: pipeline/configs/minimax_m3.yaml)
#   SCHEME     W4AFP8 | W4A8 (default W4AFP8)
#   METHODS    space-separated list (default: "gptq awq")
#
# Examples:
#   bash pipeline/slurm/submit_quantize_minimax_m3.sh
#   METHODS=gptq bash pipeline/slurm/submit_quantize_minimax_m3.sh
#   SCHEME=W4A8 bash pipeline/slurm/submit_quantize_minimax_m3.sh

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
cd "$REPO_ROOT"

MODEL_ID="${MODEL_ID:-/mnt/nfs/hoangduy/hf_assets/MiniMaxAI/MiniMax-M3}"
CONFIG="${CONFIG:-pipeline/configs/minimax_m3.yaml}"
SCHEME="${SCHEME:-W4AFP8}"
METHODS="${METHODS:-gptq awq}"

if [[ ! -f "$MODEL_ID/config.json" ]]; then
  echo "ERROR: model not found at $MODEL_ID (missing config.json)"
  exit 1
fi

mkdir -p /mnt/nfs/hoangduy/logs

echo "Submitting MiniMax-M3 quantize jobs (quantize stage only)"
echo "  model:   $MODEL_ID"
echo "  config:  $CONFIG"
echo "  scheme:  $SCHEME"
echo "  methods: $METHODS"
echo "  artifacts -> artifacts/MiniMax-M3-<method>-${SCHEME}/<timestamp>/checkpoint"

submit_method() {
  local method=$1
  local tmp_job
  tmp_job="$(mktemp "/tmp/quantize_m3_${method}_${USER:-user}_XXXXXX.sbatch")"
  cp pipeline/slurm/quantize.sbatch "$tmp_job"

  local inject
  inject=$'export CONFIG='"$(printf '%q' "$CONFIG")"$'\n'
  inject+=$'export METHOD='"$(printf '%q' "$method")"$'\n'
  inject+=$'export SCHEME='"$(printf '%q' "$SCHEME")"$'\n'
  inject+=$'export MODEL_ID='"$(printf '%q' "$MODEL_ID")"$'\n'

  awk -v inject="$inject" '
    /^set -uo pipefail/ { print; print inject; next }
    { print }
  ' "$tmp_job" > "${tmp_job}.new"
  mv "${tmp_job}.new" "$tmp_job"
  chmod +x "$tmp_job"

  local job_line
  if ! job_line="$(env -i \
    HOME="${HOME:-/mnt/nfs/hoangduy}" \
    USER="${USER:-hoangduy}" \
    PATH="${PATH:-/usr/bin:/bin}" \
    LANG="${LANG:-C.UTF-8}" \
    sbatch --export=NONE --job-name="quantize-m3-${method}" "$tmp_job" 2>&1)"; then
    echo ""
    echo "sbatch failed for method=$method:"
    echo "$job_line"
    echo ""
    echo "If you see NFS spool I/O errors, run quantize interactively on a free node:"
    echo "  METHOD=$method MODEL_ID=$MODEL_ID CONFIG=$CONFIG SCHEME=$SCHEME \\"
    echo "    bash -c 'source /mnt/nfs/hoangduy/env.sh && source /mnt/nfs/hoangduy/venvs/quant/bin/activate && cd $REPO_ROOT && export HOME=/mnt/nfs/hoangduy && bash pipeline/slurm/quantize.sbatch'"
    rm -f "$tmp_job" 2>/dev/null || true
    return 1
  fi

  local job_id="${job_line##* }"
  echo "$job_line"
  echo "  log: tail -f /mnt/nfs/hoangduy/logs/quantize-${job_id}.out"
  rm -f "$tmp_job" 2>/dev/null || true
}

failed=0
for method in $METHODS; do
  echo ""
  echo "--- submitting $method ---"
  submit_method "$method" || failed=1
done

echo ""
if [[ $failed -ne 0 ]]; then
  echo "One or more submissions failed."
  exit 1
fi
echo "All jobs submitted. Monitor with: squeue -u \$USER"
echo "When done, checkpoints under:"
for method in $METHODS; do
  echo "  artifacts/MiniMax-M3-${method}-${SCHEME}/"
done
