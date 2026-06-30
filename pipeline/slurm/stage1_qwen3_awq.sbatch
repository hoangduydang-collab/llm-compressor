#!/bin/bash
#SBATCH --job-name=quant-s1-awq
#SBATCH --partition=compute
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=200G
#SBATCH --time=10:00:00
#SBATCH --output=/mnt/nfs/hoangduy/logs/quant-s1-awq-%j.out
#
# Stage 1 (AWQ only) - run in parallel with stage1_qwen3_gptq.sbatch.
#   sbatch pipeline/slurm/stage1_qwen3_awq.sbatch
#   SCHEME=W4A8 sbatch pipeline/slurm/stage1_qwen3_awq.sbatch

set -uo pipefail
source /mnt/nfs/hoangduy/env.sh
source /mnt/nfs/hoangduy/venvs/quant/bin/activate
cd /mnt/nfs/hoangduy/projects/llm-compressor
source pipeline/slurm/_stage1_run.sh

echo "host=$(hostname) job=${SLURM_JOB_ID:-none}"
nvidia-smi --query-gpu=index,name,memory.total --format=csv || true

run_method awq "${SCHEME:-W4AFP8}"
exit $?
