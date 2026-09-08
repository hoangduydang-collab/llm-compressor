#!/usr/bin/env bash
set -uo pipefail
cd /home/e/e1129930/llm-compressor
nccl_result=$PWD/results/glm53-ep-gptq/20260909-local-nccl-retry4
git rev-parse HEAD > "$nccl_result/commit.txt"
git diff --exit-code HEAD -- src pipeline tests > "$nccl_result/protected_diff.txt" || exit 2
date -u +'%Y-%m-%dT%H:%M:%SZ' > "$nccl_result/controller_started.txt"
env -i HOME="$HOME" USER="$USER" PATH=/usr/local/bin:/usr/bin:/bin LANG=C.UTF-8 srun -vvv --partition=gpu --nodelist=xgpf11 --nodes=1 --ntasks=1 --cpus-per-task=4 --cpu-bind=cores --mem=16G --gres=gpu:nv:2 --time=00:15:00 --immediate=1 --job-name=glm53-ep-nccl-r4 bash "$nccl_result/worker.sh" > "$nccl_result/controller.log" 2>&1
nccl_status=$?
printf '%s\n' "$nccl_status" > "$nccl_result/controller_exit_code.txt"
nccl_job=$(sed -n 's/.*JobId=\([0-9]*\) returned by the controller.*/\1/p' "$nccl_result/controller.log" | head -n 1)
if test -n "$nccl_job"; then
 scontrol show job "$nccl_job" > "$nccl_result/scontrol.txt" 2>&1
 sacct -j "$nccl_job" --format=JobID,JobName,State,Elapsed,AllocTRES,ExitCode -P > "$nccl_result/sacct.txt" 2>&1
fi
exit "$nccl_status"
