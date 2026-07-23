#!/usr/bin/env bash
# tok64k official-quality rerun for the AWQ r6 arm ONLY (tmux controller).
# Same contract as run_official_quality_tok64k_srun.sh (2026-07-21 run):
# ifeval + gpqa_diamond, 65536 gen budget, 81920 ctx, 3h request timeout,
# token-usage capture on. Two arms, THREE nodes:
#   bf16    2 nodes (debug partition pair) TP16/ray serve (8001), held
#   awq-r6  1 node  in-house AWQ W4AFP8 r6 (8004), profile
#           minimax-m3-awq-inhouse-r6.sh
#
# Baseline reuse: the 2026-07-21 tok64k baseline GENERAL results are copied
# into this root, so quality.run_ab scores r6 against the IDENTICAL bf16
# sample set the other arms were scored against (RUN_BASELINE_GENERAL=0 — no
# bf16 general re-eval; the live bf16 endpoint is still required for
# capability probes and the distribution suite). If run_ab refuses the reused
# baseline, rerun with RUN_BASELINE_GENERAL=1.
set -uo pipefail
REPO=/mnt/nfs/hoangduy/projects/llm-compressor
BENCH_CFG=configs/minimax
TS=$(date -u +%Y%m%dT%H%M%SZ)
ROOT=${ROOT_OVERRIDE:-/mnt/nfs/hoangduy/results/m3-official-quality/$TS-tok64k-awqr6}
OLD_ROOT=/mnt/nfs/hoangduy/results/m3-official-quality/20260721T154830Z-tok64k
mkdir -p "$ROOT" /mnt/nfs/hoangduy/logs/m3-official-quality
echo "$ROOT" > /mnt/nfs/hoangduy/logs/m3-official-quality/tok64k-awqr6.latest_root
echo "[controller] root=$ROOT"

# Reuse the 07-21 baseline general results (identical sample set across arms).
mkdir -p "$ROOT/results"
cp -a "$OLD_ROOT/results/minimax-m3-bf16" "$ROOT/results/" \
  || { echo "[controller] baseline copy FAILED"; exit 1; }

export RUN_ID=tok64k
export GENERAL_TASKS="ifeval gpqa_diamond_cot_zeroshot"
export MAX_OUTPUT_REASONING=65536
export MAX_CONTEXT_LEN=81920
export MAX_MODEL_LEN=81920
export GENERAL_REQUEST_TIMEOUT_S=10800

# BF16 serve arm on the debug pair (HOLD_MAX in 10s ticks -> 30h).
HOLD_MAX=10800 srun --exclusive --partition=debug --nodes=2 --ntasks=2 \
     --ntasks-per-node=1 --gpus-per-node=8 --cpus-per-task=192 \
     --time=32:00:00 --kill-on-bad-exit=1 --job-name=m3-tok64k-r6-bf16 \
     --export=ALL \
     bash "$REPO/pipeline/slurm/official_quality_bf16_http_arm.sh" "$ROOT" \
     > "$ROOT/bf16-srun.log" 2>&1 &
BF16_JOB=$!

# Candidate leg in the MAIN shell (no command substitution — see the base
# tok64k controller's note on wait/subshell collapse).
ROOT="$ROOT" ARM=awq-r6 \
CKPT="/mnt/nfs/hoangduy/results/m3-distributed-awq-full/20260723T092202Z-m3-ddp-awq-full-r6-noupdown/awq/MiniMax-M3-awq-W4AFP8/20260723-092256/checkpoint-vllm-w123" \
PORT=8004 PROFILE="$BENCH_CFG/minimax-m3-awq-inhouse-r6.sh" \
RUN_BASELINE_GENERAL=0 \
srun --exclusive --nodes=1 --ntasks=1 --gres=gpu:8 --cpus-per-task=192 \
     --time=32:00:00 --kill-on-bad-exit=1 --job-name=m3-tok64k-awq-r6 \
     --export=ALL \
     bash "$REPO/pipeline/slurm/official_quality_full4_candidate_client.sh" \
     > "$ROOT/awq-r6-srun.log" 2>&1 &
R6_PID=$!

rc_all=0
wait "$R6_PID"; rc=$?
echo "$rc" > "$ROOT/client-awq-r6.rc"
echo "[controller] arm awq-r6 rc=$rc"
[ "$rc" = 0 ] || rc_all=1

touch "$ROOT/client-done"
wait "$BF16_JOB"; bf16_rc=$?
echo "[controller] bf16 arm rc=$bf16_rc"
[ "$bf16_rc" = 0 ] || rc_all=1

echo "$rc_all" > "$ROOT/controller.rc"
echo "CONTROLLER_RC=$rc_all"
