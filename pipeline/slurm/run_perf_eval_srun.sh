#!/usr/bin/env bash
# Controller for the official PERFORMANCE eval run (tmux). First perf pass over
# the same five arms as the quality full4/tok64k runs, via the benchmarks-repo
# aiperf suite (preflight gate -> AA reasoning + agentic warm/cold; the
# nonreasoning workflow self-skips — M3 has no verified no-think mode). 6 GPU
# nodes + 1 CPU task:
#   bf16      2 nodes  TP16/ray serve (port 8001), held until client-done
#   gptq      1 node   in-house GPTQ W4AFP8      (serve+suite local, port 8000)
#   mxfp8     1 node   official MiniMax MXFP8    (serve+suite local, port 8002)
#   awq       1 node   in-house AWQ W4AFP8 r5    (serve+suite local, port 8004)
#   cyankiwi  1 node   community AWQ W4A16       (serve+suite local, port 8003)
#   bf16probe CPU-only aiperf client against the shared BF16 endpoint
# Agentic shape: BORROWED from the M2.5 tau2 telecom calibration (see the AG_*
# block in configs/minimax/minimax-m3*.sh) — identical across arms, so
# cross-arm comparisons are valid; absolute agentic numbers are indicative
# until an M3-specific calibration lands.
# Results: benchmarks/results/<profile>/vllm/perf/{reasoning,agentic}/$RUN_TS
set -uo pipefail
REPO=/mnt/nfs/hoangduy/projects/llm-compressor
TS=$(date -u +%Y%m%dT%H%M%SZ)
ROOT=${ROOT_OVERRIDE:-/mnt/nfs/hoangduy/results/m3-perf-eval/$TS}
mkdir -p "$ROOT" /mnt/nfs/hoangduy/logs/m3-perf-eval
echo "$ROOT" > /mnt/nfs/hoangduy/logs/m3-perf-eval/latest_root
echo "[controller] root=$ROOT"

# One shared timestamp so every arm's suite lands under the same RUN_TS
# (benchmarks/env.sh honors a preset RUN_TS; propagated via --export=ALL).
export RUN_TS=$TS
# AGENTIC_ONLY=1 -> every arm runs preflight + ONLY the agentic workload
# (shape-refresh mode; reasoning results from a prior full run stay valid).
export AGENTIC_ONLY=${AGENTIC_ONLY:-0}
echo "[controller] run_ts=$RUN_TS agentic_only=$AGENTIC_ONLY"

# BF16 2-node serve arm, held until client-done. Perf client is CPU-only and
# remote; 65536 ctx (arm default) is fine and matches the quality serves.
HOLD_MAX=10800 READY_MAX=540 \
  srun --exclusive --nodes=2 --ntasks=2 --ntasks-per-node=1 \
     --gpus-per-node=8 --cpus-per-task=192 --time=12:00:00 \
     --kill-on-bad-exit=1 --job-name=m3-perf-bf16 --export=ALL \
     bash "$REPO/pipeline/slurm/official_quality_bf16_http_arm.sh" "$ROOT" \
     > "$ROOT/bf16-srun.log" 2>&1 &
BF16_JOB=$!

# Launch in the MAIN shell (never $(fn) subshell: the srun would not be our
# child and `wait $pid` returns 127 instantly — observed 20260720T162402Z).
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
launch_local mxfp8 minimax-m3-mxfp8 \
  "/mnt/nfs/hoangduy/hf_assets/MiniMaxAI/MiniMax-M3-MXFP8" \
  8002
MXFP8_PID=$LAST_PID
launch_local awq minimax-m3-awq-inhouse \
  "/mnt/nfs/hoangduy/results/m3-distributed-awq-full/20260720T060340Z-m3-ddp-awq-full-r5-deadchan/awq/MiniMax-M3-awq-W4AFP8/20260720-060402/checkpoint-vllm-w123" \
  8004
AWQ_PID=$LAST_PID
launch_local cyankiwi minimax-m3-awq-cyankiwi \
  "/mnt/nfs/hoangduy/hf_assets/cyankiwi/MiniMax-M3-AWQ-INT4" \
  8003
CYAN_PID=$LAST_PID

# BF16 suite client: no GPU, just HTTP+CPU against the shared BF16 endpoint.
ROOT="$ROOT" ARM=bf16 MODE=remote PROFILE=minimax-m3-bf16 BF16_PORT=8001 \
  srun --nodes=1 --ntasks=1 --cpus-per-task=32 --time=12:00:00 \
       --job-name=m3-perf-bf16probe --export=ALL \
       bash "$REPO/pipeline/slurm/perf_eval_arm.sh" \
       > "$ROOT/bf16probe-srun.log" 2>&1 &
BF16P_PID=$!

rc_all=0
for spec in "gptq:$GPTQ_PID" "mxfp8:$MXFP8_PID" "awq:$AWQ_PID" \
            "cyankiwi:$CYAN_PID" "bf16:$BF16P_PID"; do
  arm=${spec%%:*}; pid=${spec##*:}
  wait "$pid"; rc=$?
  echo "$rc" > "$ROOT/perf-$arm.rc"
  echo "[controller] perf $arm rc=$rc"
  [ "$rc" = 0 ] || rc_all=1
done

touch "$ROOT/client-done"       # release the held BF16 serve
wait "$BF16_JOB"; echo "[controller] bf16 serve rc=$?"
echo "$rc_all" > "$ROOT/controller.rc"
echo "[controller] done rc=$rc_all"
