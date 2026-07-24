#!/usr/bin/env bash
# Controller (tmux) for the shared-experts aux-stream fix matrix: one 8xH100
# node, 4 conditions x 3 trials via shared_stream_fix_matrix_node.sh.
# Results: /mnt/nfs/hoangduy/logs/m3-cudagraph-shared-stream/<session>-<cond>/
set -uo pipefail
REPO=/mnt/nfs/hoangduy/projects/llm-compressor
SESSION=${SESSION_OVERRIDE:-$(date +%Y%m%d-%H%M%S)}
ROOT=/mnt/nfs/hoangduy/logs/m3-cudagraph-shared-stream
mkdir -p "$ROOT"
echo "[controller] session=$SESSION root=$ROOT"

srun --exclusive --nodes=1 --ntasks=1 --gres=gpu:8 --cpus-per-task=192 \
     --time=05:00:00 --kill-on-bad-exit=1 --job-name=m3-stream-fix --export=ALL \
     bash "$REPO/pipeline/slurm/shared_stream_fix_matrix_node.sh" "$ROOT" "$SESSION" \
     > "$ROOT/$SESSION-srun.log" 2>&1
rc=$?
echo "[controller] node script rc=$rc"
echo "$rc" > "$ROOT/$SESSION-controller.rc"
echo "CONTROLLER_RC=$rc"
