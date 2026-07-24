#!/usr/bin/env bash
# Controller (tmux) for the perf-eval RERUN under the new serve defaults
# (shared-experts stream ON + LLMC_M3_CAPTURE_SYNC=sync, commit 6e074b48).
# Same setup as run_perf_eval_srun.sh (perf_eval_arm.sh -> benchmarks aiperf
# suite: preflight gate + AA reasoning + agentic warm/cold; nonreasoning
# self-skips), 4 single-node local arms, one shared RUN_TS:
#   gptq   port 8000  original in-house GPTQ W4AFP8 (abi-overlay)
#   r8v2   port 8002  r8 dequant-qkv  (fp8rest, checkpoint-vllm-w123-v2)
#   r8v3   port 8003  r8 uniform-qkv  (checkpoint-vllm-w123-v3-uniformqkv)
#   r7     port 8004  AWQ r7 gate-alpha W4AFP8
# NOTE: absolute numbers are NOT comparable to the pre-2026-07-24 perf pass
# (that ran stream-off); cross-arm comparisons within this run are valid.
# Results: benchmarks/results/<profile>/vllm/perf/{reasoning,agentic}/$RUN_TS
set -uo pipefail
REPO=/mnt/nfs/hoangduy/projects/llm-compressor
TS=$(date -u +%Y%m%dT%H%M%SZ)
ROOT=${ROOT_OVERRIDE:-/mnt/nfs/hoangduy/results/m3-perf-eval/$TS}
mkdir -p "$ROOT" /mnt/nfs/hoangduy/logs/m3-perf-eval
echo "$ROOT" > /mnt/nfs/hoangduy/logs/m3-perf-eval/latest_root
echo "[controller] root=$ROOT"

export RUN_TS=$TS
export AGENTIC_ONLY=${AGENTIC_ONLY:-0}
echo "[controller] run_ts=$RUN_TS agentic_only=$AGENTIC_ONLY"

launch_local() {  # $1 arm  $2 profile  $3 ckpt  $4 port
  ROOT="$ROOT" ARM="$1" MODE=local PROFILE="$2" CKPT="$3" PORT="$4" \
  srun --exclusive --nodes=1 --ntasks=1 --gres=gpu:8 --cpus-per-task=192 \
       --time=12:00:00 --kill-on-bad-exit=1 --job-name="m3-perf-$1" --export=ALL \
       bash "$REPO/pipeline/slurm/perf_eval_arm.sh" \
       > "$ROOT/$1-srun.log" 2>&1 &
  LAST_PID=$!
}

launch_local gptq minimax-m3 \
  "$REPO/artifacts/m3-awq-gptq-prepared/gptq-checkpoint-vllm-w123-abi-overlay" \
  8000
GPTQ_PID=$LAST_PID
launch_local r8v2 minimax-m3-gptq-r8-fp8rest \
  "/mnt/nfs/hoangduy/results/m3-distributed-r8-full/20260723T160426Z-m3-ddp-gptq-full-r8-fp8rest/gptq/MiniMax-M3-gptq-W4AFP8/20260723-160454/checkpoint-vllm-w123-v2" \
  8002
R8V2_PID=$LAST_PID
launch_local r8v3 minimax-m3-gptq-r8-uniformqkv \
  "/mnt/nfs/hoangduy/results/m3-distributed-r8-full/20260723T160426Z-m3-ddp-gptq-full-r8-fp8rest/gptq/MiniMax-M3-gptq-W4AFP8/20260723-160454/checkpoint-vllm-w123-v3-uniformqkv" \
  8003
R8V3_PID=$LAST_PID
launch_local r7 minimax-m3-awq-inhouse-r7 \
  "/mnt/nfs/hoangduy/results/m3-distributed-awq-full/20260723T123927Z-m3-ddp-awq-full-r7-gatealpha/awq/MiniMax-M3-awq-W4AFP8/20260723-123953/checkpoint-vllm-w123" \
  8004
R7_PID=$LAST_PID

rc_all=0
for spec in "gptq:$GPTQ_PID" "r8v2:$R8V2_PID" "r8v3:$R8V3_PID" "r7:$R7_PID"; do
  arm=${spec%%:*}; pid=${spec##*:}
  wait "$pid"; rc=$?
  echo "$rc" > "$ROOT/perf-$arm.rc"
  echo "[controller] perf $arm rc=$rc"
  [ "$rc" = 0 ] || rc_all=1
done

echo "$rc_all" > "$ROOT/controller.rc"
echo "[controller] done rc=$rc_all"
echo "CONTROLLER_RC=$rc_all"
