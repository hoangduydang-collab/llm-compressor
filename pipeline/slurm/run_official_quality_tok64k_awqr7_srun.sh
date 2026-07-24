#!/usr/bin/env bash
# tok64k official-quality rerun for the AWQ r7 arm ONLY (tmux controller).
# Same contract as run_official_quality_tok64k_awqr6_srun.sh (and the base
# 2026-07-21 tok64k run): ifeval + gpqa_diamond, 65536 gen budget, 81920 ctx,
# 3h request timeout, token-usage capture on. Two arms, THREE nodes:
#   bf16    2 nodes (default: debug partition pair) TP16/ray serve (8001), held
#   awq-r7  1 node  in-house AWQ W4AFP8 r7 gate-alpha fold (8007), profile
#           minimax-m3-awq-inhouse-r7.sh
#
# r7 serving REQUIRES the gate-alpha overlay from
# pipeline/slurm/patch_vllm_m3_serve.py in the serving vLLM (quant venv is
# patched; verified 2026-07-24 via --check + capture-safe ABI smoke v4).
#
# Baseline reuse: the 2026-07-21 tok64k baseline GENERAL results are copied
# into this root, so quality.run_ab scores r7 against the IDENTICAL bf16
# sample set the other arms were scored against (RUN_BASELINE_GENERAL=0 — no
# bf16 general re-eval; the live bf16 endpoint is still required for
# capability probes and the distribution suite). If run_ab refuses the reused
# baseline, rerun with RUN_BASELINE_GENERAL=1.
#
# Partitions: BF16_PARTITION defaults to debug (compute is saturated by the
# r8 lanes as of 2026-07-24); override BF16_PARTITION / ARM_PARTITION.
set -uo pipefail
REPO=/mnt/nfs/hoangduy/projects/llm-compressor
BENCH_CFG=configs/minimax
TS=$(date -u +%Y%m%dT%H%M%SZ)
ROOT=${ROOT_OVERRIDE:-/mnt/nfs/hoangduy/results/m3-official-quality/$TS-tok64k-awqr7}
OLD_ROOT=/mnt/nfs/hoangduy/results/m3-official-quality/20260721T154830Z-tok64k
mkdir -p "$ROOT" /mnt/nfs/hoangduy/logs/m3-official-quality
echo "$ROOT" > /mnt/nfs/hoangduy/logs/m3-official-quality/tok64k-awqr7.latest_root
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

# BF16 serve arm, 2 nodes (HOLD_MAX in 10s ticks -> 30h).
BF16_PARTITION=${BF16_PARTITION:-debug}
BF16_PART_OPT=()
[ -n "${BF16_PARTITION:-}" ] && BF16_PART_OPT=(--partition="$BF16_PARTITION")
HOLD_MAX=10800 srun --exclusive "${BF16_PART_OPT[@]}" --nodes=2 --ntasks=2 \
     --ntasks-per-node=1 --gpus-per-node=8 --cpus-per-task=192 \
     --time=32:00:00 --kill-on-bad-exit=1 --job-name=m3-tok64k-r7-bf16 \
     --export=ALL \
     bash "$REPO/pipeline/slurm/official_quality_bf16_http_arm.sh" "$ROOT" \
     > "$ROOT/bf16-srun.log" 2>&1 &
BF16_JOB=$!

# Candidate leg in the MAIN shell (no command substitution — see the base
# tok64k controller's note on wait/subshell collapse).
ARM_PART_OPT=()
[ -n "${ARM_PARTITION:-}" ] && ARM_PART_OPT=(--partition="$ARM_PARTITION")
ROOT="$ROOT" ARM=awq-r7 \
CKPT="/mnt/nfs/hoangduy/results/m3-distributed-awq-full/20260723T123927Z-m3-ddp-awq-full-r7-gatealpha/awq/MiniMax-M3-awq-W4AFP8/20260723-123953/checkpoint-vllm-w123" \
PORT=8007 PROFILE="$BENCH_CFG/minimax-m3-awq-inhouse-r7.sh" \
RUN_BASELINE_GENERAL=0 \
srun --exclusive "${ARM_PART_OPT[@]}" --nodes=1 --ntasks=1 --gres=gpu:8 --cpus-per-task=192 \
     --time=32:00:00 --kill-on-bad-exit=1 --job-name=m3-tok64k-awq-r7 \
     --export=ALL \
     bash "$REPO/pipeline/slurm/official_quality_full4_candidate_client.sh" \
     > "$ROOT/awq-r7-srun.log" 2>&1 &
R7_PID=$!

rc_all=0
wait "$R7_PID"; rc=$?
echo "$rc" > "$ROOT/client-awq-r7.rc"
echo "[controller] arm awq-r7 rc=$rc"
[ "$rc" = 0 ] || rc_all=1

touch "$ROOT/client-done"
wait "$BF16_JOB"; bf16_rc=$?
echo "[controller] bf16 arm rc=$bf16_rc"
[ "$bf16_rc" = 0 ] || rc_all=1

echo "$rc_all" > "$ROOT/controller.rc"
echo "CONTROLLER_RC=$rc_all"
