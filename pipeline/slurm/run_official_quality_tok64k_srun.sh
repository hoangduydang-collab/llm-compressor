#!/usr/bin/env bash
# Token-spend investigation run (tmux controller). Five arms, SIX nodes:
#   bf16      2 nodes  TP16/ray serve (port 8001), held until clients finish
#   gptq      1 node   in-house GPTQ W4AFP8 (8000) + owns the ONE baseline
#                      general-suite evaluation (standalone orchestrator vs bf16)
#   awq       1 node   in-house AWQ W4AFP8 r5 (8004)
#   mxfp8     1 node   official MiniMax MXFP8 (8002)
#   cyankiwi  1 node   community AWQ W4A16 (8003)
#
# Purpose: quantify the "quantized models spend more reasoning tokens"
# regression channel (QUANT_REGRESSION_METRICS_SURVEY.md; the AWQ-r5 GPQA
# collapse was 3x budget exhaustion at 32k). Differences vs the full4 run:
#   * tasks: ifeval + gpqa_diamond only (the two most damaged AWQ tasks)
#   * generation budget doubled: max_gen_toks 65536, served ctx 81920
#     (M3 max_position_embeddings is 1M — no positional ceiling)
#   * request timeout 3h (64k-token completions outlive 2h on slow items)
#   * token-usage capture is on by default in the benchmarks pipeline now:
#     per-request usage.jsonl (+ reasoning text), usage_delta / task_flips
#     sidecars per arm. run-id: tok64k.
# Profiles pick these up because profile_from_config sources configs/*.sh in
# a subshell inheriting this environment (every knob is ${VAR:-default}).
set -uo pipefail
REPO=/mnt/nfs/hoangduy/projects/llm-compressor
BENCH_CFG=configs/minimax
TS=$(date -u +%Y%m%dT%H%M%SZ)
ROOT=${ROOT_OVERRIDE:-/mnt/nfs/hoangduy/results/m3-official-quality/$TS-tok64k}
mkdir -p "$ROOT" /mnt/nfs/hoangduy/logs/m3-official-quality
echo "$ROOT" > /mnt/nfs/hoangduy/logs/m3-official-quality/tok64k.latest_root
echo "[controller] root=$ROOT"

export RUN_ID=tok64k
export GENERAL_TASKS="ifeval gpqa_diamond_cot_zeroshot"
export MAX_OUTPUT_REASONING=65536
export MAX_CONTEXT_LEN=81920
export MAX_MODEL_LEN=81920
export GENERAL_REQUEST_TIMEOUT_S=10800

# BF16 serve arm (HOLD_MAX in 10s ticks -> 30h).
HOLD_MAX=10800 srun --exclusive --nodes=2 --ntasks=2 --ntasks-per-node=1 \
     --gpus-per-node=8 --cpus-per-task=192 --time=32:00:00 \
     --kill-on-bad-exit=1 --job-name=m3-tok64k-bf16 --export=ALL \
     bash "$REPO/pipeline/slurm/official_quality_bf16_http_arm.sh" "$ROOT" \
     > "$ROOT/bf16-srun.log" 2>&1 &
BF16_JOB=$!

# NOTE: launch in the MAIN shell (no command substitution): $(fn) runs in a
# subshell, so the srun would not be our child — `wait $pid` then fails with
# 127 instantly and the run collapses (observed 20260720T162402Z).
launch_arm() {  # $1 arm  $2 ckpt  $3 port  $4 profile  $5 run_baseline_general
  ROOT="$ROOT" ARM="$1" CKPT="$2" PORT="$3" PROFILE="$4" RUN_BASELINE_GENERAL="$5" \
  srun --exclusive --nodes=1 --ntasks=1 --gres=gpu:8 --cpus-per-task=192 \
       --time=32:00:00 --kill-on-bad-exit=1 --job-name="m3-tok64k-$1" \
       --export=ALL \
       bash "$REPO/pipeline/slurm/official_quality_full4_candidate_client.sh" \
       > "$ROOT/$1-srun.log" 2>&1 &
  LAST_PID=$!
}

launch_arm gptq \
  "$REPO/artifacts/m3-awq-gptq-prepared/gptq-checkpoint-vllm-w123-abi-overlay" \
  8000 "$BENCH_CFG/minimax-m3.sh" 1
GPTQ_PID=$LAST_PID
launch_arm awq \
  "/mnt/nfs/hoangduy/results/m3-distributed-awq-full/20260720T060340Z-m3-ddp-awq-full-r5-deadchan/awq/MiniMax-M3-awq-W4AFP8/20260720-060402/checkpoint-vllm-w123" \
  8004 "$BENCH_CFG/minimax-m3-awq-inhouse.sh" 0
AWQ_PID=$LAST_PID
launch_arm mxfp8 \
  "/mnt/nfs/hoangduy/hf_assets/MiniMaxAI/MiniMax-M3-MXFP8" \
  8002 "$BENCH_CFG/minimax-m3-mxfp8.sh" 0
MXFP8_PID=$LAST_PID
launch_arm cyankiwi \
  "/mnt/nfs/hoangduy/hf_assets/cyankiwi/MiniMax-M3-AWQ-INT4" \
  8003 "$BENCH_CFG/minimax-m3-awq-cyankiwi.sh" 0
CYAN_PID=$LAST_PID

rc_all=0
for spec in "gptq:$GPTQ_PID" "awq:$AWQ_PID" "mxfp8:$MXFP8_PID" "cyankiwi:$CYAN_PID"; do
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
