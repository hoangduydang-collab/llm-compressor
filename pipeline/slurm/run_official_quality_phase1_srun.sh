#!/usr/bin/env bash
# Controller for official-pipeline migration phase 1 (run inside tmux on the
# login node; srun-only cluster). Allocates ONE 8xH100 node and runs
# official_quality_phase1_node.sh on it. Writes controller.rc at the end.
set -uo pipefail

TS=$(date -u +%Y%m%dT%H%M%SZ)
ROOT=/mnt/nfs/hoangduy/results/m3-official-quality/${TS}-phase1-candidate-smoke
LOGDIR=/mnt/nfs/hoangduy/logs/m3-official-quality
mkdir -p "$ROOT" "$LOGDIR"
echo "$ROOT" > "$LOGDIR/phase1.latest_root"

echo "[controller] phase1 root: $ROOT"
srun --exclusive --nodes=1 --ntasks=1 --gres=gpu:8 --cpus-per-task=192 \
  --time=03:00:00 --kill-on-bad-exit=1 --job-name=m3-offq-p1 \
  bash /mnt/nfs/hoangduy/projects/llm-compressor/pipeline/slurm/official_quality_phase1_node.sh "$ROOT" \
  >"$LOGDIR/phase1-$TS.log" 2>&1
rc=$?
echo "$rc" > "$ROOT/controller.rc"
echo "[controller] phase1 rc=$rc (log: $LOGDIR/phase1-$TS.log)"
exit "$rc"
