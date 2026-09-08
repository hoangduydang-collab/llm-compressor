#!/usr/bin/env bash
set -uo pipefail
cd /home/e/e1129930/llm-compressor
nccl_result=$PWD/results/glm53-ep-gptq/20260909-h100-nccl
git rev-parse HEAD > "$nccl_result/controller_revision.txt"
date -u +'%Y-%m-%dT%H:%M:%SZ' > "$nccl_result/controller_started.txt"
# Queue at most eight hours, below this controller allocation's remaining lifetime.
# One attempt only; timeout forwards TERM so srun cancels its pending allocation.
timeout --signal=TERM --kill-after=30s 8h env -i HOME="$HOME" USER="$USER" PATH=/usr/local/bin:/usr/bin:/bin LANG=C.UTF-8 srun -vvv --partition=gpu --nodes=1 --ntasks=1 --cpus-per-task=4 --cpu-bind=cores --mem=16G --gres=gpu:h100-96:2 --time=00:15:00 --job-name=glm53-ep-h100 bash "$nccl_result/worker.sh" > "$nccl_result/controller.log" 2>&1
nccl_status=$?
printf '%s\n' "$nccl_status" > "$nccl_result/controller_exit_code.txt"
nccl_job=$(sed -n 's/.*JobId=\([0-9]*\) returned by the controller.*/\1/p' "$nccl_result/controller.log" | head -n 1)
if test -n "$nccl_job"; then
 printf '%s\n' "$nccl_job" > "$nccl_result/job_id.txt"
 scontrol show job "$nccl_job" > "$nccl_result/scontrol.final.txt" 2>&1
 sacct -j "$nccl_job" --format=JobID,JobName,State,Elapsed,AllocTRES,ExitCode -P > "$nccl_result/sacct.final.txt" 2>&1
fi
date -u +'%Y-%m-%dT%H:%M:%SZ' > "$nccl_result/controller_ended.txt"
exit "$nccl_status"
