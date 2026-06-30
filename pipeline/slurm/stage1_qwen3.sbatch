#!/bin/bash
#SBATCH --job-name=quant-stage1
#SBATCH --partition=compute
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=200G
#SBATCH --time=12:00:00
#SBATCH --output=/mnt/nfs/hoangduy/logs/quant-stage1-%j.out
#
# Stage 1: Qwen3-30B-A3B end-to-end for BOTH GPTQ and AWQ, SEQUENTIALLY on one GPU.
# For PARALLEL runs (recommended, ~half the wall-clock) submit the two
# per-method scripts instead:
#   sbatch pipeline/slurm/stage1_qwen3_gptq.sbatch
#   sbatch pipeline/slurm/stage1_qwen3_awq.sbatch
#
# Submit (sequential):  sbatch pipeline/slurm/stage1_qwen3.sbatch
# Override grid:        SCHEME=W4A8 METHODS="gptq awq" sbatch pipeline/slurm/stage1_qwen3.sbatch

set -uo pipefail
source /mnt/nfs/hoangduy/env.sh
source /mnt/nfs/hoangduy/venvs/quant/bin/activate
cd /mnt/nfs/hoangduy/projects/llm-compressor
source pipeline/slurm/_stage1_run.sh

SCHEME=${SCHEME:-W4AFP8}
METHODS=${METHODS:-"gptq awq"}

echo "host=$(hostname) job=${SLURM_JOB_ID:-none} scheme=$SCHEME methods='$METHODS'"
nvidia-smi --query-gpu=index,name,memory.total --format=csv || true

overall_rc=0
for METHOD in $METHODS; do
  run_method "$METHOD" "$SCHEME" || overall_rc=1
done

echo ""
echo "ALL STAGE-1 RUNS COMPLETE (overall_rc=$overall_rc)"
echo "Artifacts under: artifacts/Qwen3-30B-A3B-*-${SCHEME}/"
exit $overall_rc
