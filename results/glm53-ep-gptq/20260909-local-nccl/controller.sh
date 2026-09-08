#!/usr/bin/env bash
set -uo pipefail
cd /home/e/e1129930/llm-compressor
nccl_result=/home/e/e1129930/llm-compressor/results/glm53-ep-gptq/20260909-local-nccl
mkdir -p "$nccl_result"
git rev-parse HEAD > "$nccl_result/commit.txt"
git diff --exit-code HEAD -- src pipeline tests > "$nccl_result/protected_diff.txt" || exit 2
date -u +'%Y-%m-%dT%H:%M:%SZ' > "$nccl_result/controller_started.txt"
env -u SLURM_JOB_ID -u SLURM_JOBID -u SLURM_STEP_ID -u SLURM_STEPID srun --partition=gpu --nodelist=xgph12 --nodes=1 --ntasks=1 --cpus-per-task=4 --mem=16G --gres=gpu:a100-40:2 --time=00:15:00 --immediate=30 --job-name=glm53-ep-nccl bash /home/e/e1129930/llm-compressor/results/glm53-ep-gptq/20260909-local-nccl/worker.sh > "$nccl_result/controller.log" 2>&1
nccl_status=$?
printf '%s\n' "$nccl_status" > "$nccl_result/controller_exit_code.txt"
if test -f "$nccl_result/job_id.txt"; then
 nccl_job=$(cat "$nccl_result/job_id.txt")
 scontrol show job "$nccl_job" > "$nccl_result/scontrol.txt" 2>&1
 sacct -j "$nccl_job" --format=JobID,JobName,State,Elapsed,AllocTRES,ExitCode -P > "$nccl_result/sacct.txt" 2>&1
fi
exit "$nccl_status"
