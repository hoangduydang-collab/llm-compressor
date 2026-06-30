#!/bin/bash
#SBATCH --job-name=serve-eval
#SBATCH --partition=compute
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=200G
#SBATCH --time=04:00:00
#SBATCH --output=/mnt/nfs/hoangduy/logs/serve-eval-%j.out
#
# Serve-verify + eval gate on EXISTING checkpoint(s) - no re-quantize.
#
# Option A: point directly at one or more checkpoint dirs (space-separated):
#   CKPTS="artifacts/Qwen3-30B-A3B-gptq-W4AFP8/20260630-080115/checkpoint \
#          artifacts/Qwen3-30B-A3B-awq-W4AFP8/20260630-080112/checkpoint" \
#     sbatch pipeline/slurm/serve_eval.sbatch
#
# Option B: auto-resolve the newest checkpoint per method:
#   METHODS="gptq awq" SCHEME=W4AFP8 sbatch pipeline/slurm/serve_eval.sbatch

set -uo pipefail
source /mnt/nfs/hoangduy/env.sh
source /mnt/nfs/hoangduy/venvs/quant/bin/activate
cd /mnt/nfs/hoangduy/projects/llm-compressor
source pipeline/slurm/_stage1_run.sh

echo "host=$(hostname) job=${SLURM_JOB_ID:-none}"
nvidia-smi --query-gpu=index,name,memory.total --format=csv || true

overall_rc=0

if [[ -n "${CKPTS:-}" ]]; then
  for CKPT in $CKPTS; do
    run_serve_eval "$CKPT" || overall_rc=1
  done
else
  SCHEME=${SCHEME:-W4AFP8}
  METHODS=${METHODS:-"gptq awq"}
  for METHOD in $METHODS; do
    CKPT=$(latest_ckpt "$METHOD" "$SCHEME")
    if [[ -z "$CKPT" ]]; then
      echo "[FAIL] no checkpoint found for method=$METHOD scheme=$SCHEME"
      overall_rc=1
      continue
    fi
    run_serve_eval "$CKPT" || overall_rc=1
  done
fi

echo ""
echo "SERVE+EVAL COMPLETE (overall_rc=$overall_rc)"
exit $overall_rc
