#!/usr/bin/env bash
# Controller for the FULL four-way official-pipeline quality run (tmux). 5 nodes:
#   bf16   2 nodes  TP16/ray serve (port 8001), held until all clients finish
#   gptq   1 node   in-house GPTQ W4AFP8 (port 8000) + owns the ONE baseline
#                   general-suite evaluation (standalone orchestrator vs bf16)
#   awq    1 node   in-house AWQ W4AFP8 r5 (port 8004)
#   mxfp8  1 node   official MiniMax MXFP8 (port 8002)
# Each candidate runs quality.run_ab FULL (7 general tasks + distribution +
# delta + per-arm report) with --reuse-baseline-general-wait-s, so BF16 is
# served once and evaluated once. run-id: full4.
set -uo pipefail
REPO=/mnt/nfs/hoangduy/projects/llm-compressor
BENCH_CFG=configs/minimax
TS=$(date -u +%Y%m%dT%H%M%SZ)
ROOT=/mnt/nfs/hoangduy/results/m3-official-quality/$TS-full4
mkdir -p "$ROOT" /mnt/nfs/hoangduy/logs/m3-official-quality
echo "$ROOT" > /mnt/nfs/hoangduy/logs/m3-official-quality/full4.latest_root
echo "[controller] root=$ROOT"

# BF16 serve arm: 30h hold budget (HOLD_MAX is in 10s ticks).
HOLD_MAX=10800 srun --exclusive --nodes=2 --ntasks=2 --ntasks-per-node=1 \
     --gpus-per-node=8 --cpus-per-task=192 --time=32:00:00 \
     --kill-on-bad-exit=1 --job-name=m3-full4-bf16 \
     bash "$REPO/pipeline/slurm/official_quality_bf16_http_arm.sh" "$ROOT" \
     > "$ROOT/bf16-srun.log" 2>&1 &
BF16_JOB=$!

launch_arm() {  # $1 arm  $2 ckpt  $3 port  $4 profile  $5 run_baseline_general
  ROOT="$ROOT" ARM="$1" CKPT="$2" PORT="$3" PROFILE="$4" RUN_BASELINE_GENERAL="$5" \
  srun --exclusive --nodes=1 --ntasks=1 --gres=gpu:8 --cpus-per-task=192 \
       --time=32:00:00 --kill-on-bad-exit=1 --job-name="m3-full4-$1" \
       --export=ALL \
       bash "$REPO/pipeline/slurm/official_quality_full4_candidate_client.sh" \
       > "$ROOT/$1-srun.log" 2>&1 &
  echo $!
}

GPTQ_PID=$(launch_arm gptq \
  "$REPO/artifacts/m3-awq-gptq-prepared/gptq-checkpoint-vllm-w123-abi-overlay" \
  8000 "$BENCH_CFG/minimax-m3.sh" 1)
AWQ_PID=$(launch_arm awq \
  "/mnt/nfs/hoangduy/results/m3-distributed-awq-full/20260720T060340Z-m3-ddp-awq-full-r5-deadchan/awq/MiniMax-M3-awq-W4AFP8/20260720-060402/checkpoint-vllm-w123" \
  8004 "$BENCH_CFG/minimax-m3-awq-inhouse.sh" 0)
MXFP8_PID=$(launch_arm mxfp8 \
  "/mnt/nfs/hoangduy/hf_assets/MiniMaxAI/MiniMax-M3-MXFP8" \
  8002 "$BENCH_CFG/minimax-m3-mxfp8.sh" 0)

rc_all=0
for spec in "gptq:$GPTQ_PID" "awq:$AWQ_PID" "mxfp8:$MXFP8_PID"; do
  arm=${spec%%:*}; pid=${spec##*:}
  wait "$pid"; rc=$?
  echo "$rc" > "$ROOT/client-$arm.rc"
  echo "[controller] arm $arm rc=$rc"
  [ "$rc" = 0 ] || rc_all=1
done

touch "$ROOT/client-done"
wait "$BF16_JOB"; bf16_rc=$?
echo "[controller] bf16 arm rc=$bf16_rc"
[ "$bf16_rc" = 0 ] || rc_all=1

echo "$rc_all" > "$ROOT/controller.rc"
echo "[controller] done rc=$rc_all"
