#!/usr/bin/env bash
set -uo pipefail
cd /mnt/nfs/hoangduy/projects/llm-compressor
export RUN_ROOT=results/m3-quality/20260712T162609Z-m3-quality-smoke-tmux
export MATRIX=results/m3-quality/20260712-142048-m3-gptq-repaired/repaired_matrix.yaml
export REPAIRED_GPTQ=results/m3-quality/20260712-142048-m3-gptq-repaired/checkpoints/inhouse-gptq-portable
export LOG_ROOT=results/m3-quality/20260712T162609Z-m3-quality-smoke-tmux/logs
CONTROLLER_LOG=results/m3-quality/20260712T162609Z-m3-quality-smoke-tmux/logs/controller.log
RC_FILE=results/m3-quality/20260712T162609Z-m3-quality-smoke-tmux/controller.rc
exec >>"$CONTROLLER_LOG" 2>&1
echo "controller started=$(date -Is) host=$(hostname) pid=$$"
rc=0; bash /mnt/nfs/hoangduy/projects/llm-compressor/pipeline/slurm/run_m3_quality_smoke_srun.sh || rc=$?
printf "%s\n" "$rc" >"$RC_FILE.tmp"
mv "$RC_FILE.tmp" "$RC_FILE"
echo "controller finished=$(date -Is) rc=$rc"
exit "$rc"
