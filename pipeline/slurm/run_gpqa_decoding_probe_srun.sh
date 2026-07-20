#!/usr/bin/env bash
# Controller for the GPQA decoding probe (run inside a persistent tmux).
set -uo pipefail
REPO=/mnt/nfs/hoangduy/projects/llm-compressor
TS=$(date -u +%Y%m%dT%H%M%SZ)
ROOT=/mnt/nfs/hoangduy/results/m3-official-quality/$TS-gpqa-decoding-probe
mkdir -p "$ROOT" /mnt/nfs/hoangduy/logs/m3-official-quality
echo "$ROOT" > /mnt/nfs/hoangduy/logs/m3-official-quality/gpqa-probe.latest_root
echo "[controller] root=$ROOT"
srun --exclusive --nodes=1 --ntasks=1 --gres=gpu:8 --cpus-per-task=192 \
     --time=04:00:00 --kill-on-bad-exit=1 --job-name=m3-gpqa-probe \
     bash "$REPO/pipeline/slurm/gpqa_decoding_probe_node.sh" "$ROOT"
rc=$?
echo "$rc" > "$ROOT/controller.rc"
echo "[controller] done rc=$rc"
