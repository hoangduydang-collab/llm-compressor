#!/usr/bin/env bash
set -uo pipefail
cd /home/e/e1129930/llm-compressor
probe_root=$PWD/results/glm53-ep-gptq/20260909-launch-diagnosis
probe_args=(-vvv --partition=gpu --nodelist=xgph12 --nodes=1 --ntasks=1 --cpus-per-task=4 --mem=16G --gres=gpu:a100-40:2 --time=00:01:00 --immediate=30)
for probe_mode in inherited clean; do
  date -u +'%Y-%m-%dT%H:%M:%SZ' > "$probe_root/$probe_mode.started.txt"
  if test "$probe_mode" = inherited; then
    env -u SLURM_JOB_ID -u SLURM_JOBID -u SLURM_STEP_ID -u SLURM_STEPID srun "${probe_args[@]}" --job-name=ep-launch-inherited /bin/hostname > "$probe_root/$probe_mode.log" 2>&1
  else
    env -i HOME="$HOME" USER="$USER" PATH=/usr/local/bin:/usr/bin:/bin LANG=C.UTF-8 srun "${probe_args[@]}" --job-name=ep-launch-clean /bin/hostname > "$probe_root/$probe_mode.log" 2>&1
  fi
  printf '%s\n' "$?" > "$probe_root/$probe_mode.exit_code.txt"
done
date -u +'%Y-%m-%dT%H:%M:%SZ' > "$probe_root/ended.txt"
