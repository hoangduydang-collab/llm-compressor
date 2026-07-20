#!/usr/bin/env bash
# Controller for official-pipeline migration phase 2 (run inside tmux; srun-only).
# Launches TWO srun steps in parallel:
#   A. 2-node BF16 TP16/ray HTTP arm (official_quality_bf16_http_arm.sh, port 8001)
#   B. 1-node candidate W4A8 serve + quality.run_ab client (port 8000)
# The client signals $ROOT/client-done; the BF16 arm tears down on it.
set -uo pipefail

TS=$(date -u +%Y%m%dT%H%M%SZ)
ROOT=/mnt/nfs/hoangduy/results/m3-official-quality/${TS}-phase2-ab-smoke
LOGDIR=/mnt/nfs/hoangduy/logs/m3-official-quality
mkdir -p "$ROOT" "$LOGDIR"
echo "$ROOT" > "$LOGDIR/phase2.latest_root"
echo "[controller] phase2 root: $ROOT"

srun --exclusive --nodes=2 --ntasks=2 --ntasks-per-node=1 --gpus-per-node=8 \
  --cpus-per-task=192 --time=08:00:00 --kill-on-bad-exit=1 --job-name=m3-offq-bf16 \
  bash /mnt/nfs/hoangduy/projects/llm-compressor/pipeline/slurm/official_quality_bf16_http_arm.sh "$ROOT" \
  >"$LOGDIR/phase2-bf16-$TS.log" 2>&1 &
BF16_PID=$!

srun --exclusive --nodes=1 --ntasks=1 --gres=gpu:8 \
  --cpus-per-task=192 --time=08:00:00 --kill-on-bad-exit=1 --job-name=m3-offq-cli \
  bash /mnt/nfs/hoangduy/projects/llm-compressor/pipeline/slurm/official_quality_phase2_client.sh "$ROOT" \
  >"$LOGDIR/phase2-client-$TS.log" 2>&1 &
CLIENT_PID=$!

wait "$CLIENT_PID"; client_rc=$?
echo "[controller] client rc=$client_rc"
touch "$ROOT/client-done"
wait "$BF16_PID"; bf16_rc=$?
echo "[controller] bf16 arm rc=$bf16_rc"

rc=$client_rc
[ "$rc" = 0 ] && [ "$bf16_rc" != 0 ] && rc=$bf16_rc
echo "$rc" > "$ROOT/controller.rc"
echo "[controller] phase2 rc=$rc"
exit "$rc"
