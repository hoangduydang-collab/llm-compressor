#!/usr/bin/env bash
# Sampling-sensitivity probe controller (tmux). Tests whether AWQ's GPQA
# non-termination (tok64k: in-house 38.9%, cyankiwi 55.6% budget-exhausted vs
# ~12% for healthy arms, all under GREEDY temp=0) is a greedy x quant artifact.
# Re-runs the tok64k GPQA exhausted docs (+ terminated control) under greedy
# AND sampled (temp 1.0 / top_p 0.95) at a fixed 32k budget. Arms:
#   bf16      2 nodes  TP16/ray serve (port 8001), held until client-done
#   gptq      1 node   healthy quant control (serve+probe local)
#   awq       1 node   in-house AWQ r5      (serve+probe local)
#   cyankiwi  1 node   community AWQ W4A16  (serve+probe local)
#   bf16probe 1 node   no-GPU: probe driver against the shared BF16 endpoint
# Per-generation output: $ROOT/sampling/<arm>.jsonl (finish_reason,
# completion_tokens, reasoning_rep_ratio, best-effort answer/correct).
set -uo pipefail
REPO=/mnt/nfs/hoangduy/projects/llm-compressor
TS=$(date -u +%Y%m%dT%H%M%SZ)
ROOT=${ROOT_OVERRIDE:-/mnt/nfs/hoangduy/results/m3-sampling-probe/$TS}
mkdir -p "$ROOT" /mnt/nfs/hoangduy/logs/m3-sampling-probe
echo "$ROOT" > /mnt/nfs/hoangduy/logs/m3-sampling-probe/latest_root
echo "[controller] root=$ROOT"

TOK=/mnt/nfs/hoangduy/results/m3-official-quality/20260721T154830Z-tok64k/results
G() { echo "$TOK/$1/vllm/quality/_lm_eval/gpqa_diamond_cot_zeroshot/**/samples_*.jsonl"; }

export N_SAMPLES=${N_SAMPLES:-3} MAX_TOKENS=${MAX_TOKENS:-32768}
export N_CONTROL=${N_CONTROL:-25} CONCURRENCY=${CONCURRENCY:-24}
export REQUEST_TIMEOUT_S=${REQUEST_TIMEOUT_S:-3600}
# AWQ arms have 77/101 exhausted docs; cap to 50 (even-spaced) to bound cost.
# gptq/bf16 have only ~23 exhausted -> uncapped (N_EXHAUSTED=0 = all).
AWQ_EXHAUSTED_CAP=${AWQ_EXHAUSTED_CAP:-50}

# BF16 2-node serve arm, held until client-done. Longer ready budget for the
# ~920 GB TP16 load; 40960 ctx is plenty for 32k gen on short GPQA prompts.
HOLD_MAX=10800 READY_MAX=540 MAX_MODEL_LEN=40960 \
  srun --exclusive --nodes=2 --ntasks=2 --ntasks-per-node=1 \
     --gpus-per-node=8 --cpus-per-task=192 --time=12:00:00 \
     --kill-on-bad-exit=1 --job-name=m3-sprobe-bf16 --export=ALL \
     bash "$REPO/pipeline/slurm/official_quality_bf16_http_arm.sh" "$ROOT" \
     > "$ROOT/bf16-srun.log" 2>&1 &
BF16_JOB=$!

# Launch in the MAIN shell (never $(fn) subshell: the srun would not be our
# child and `wait $pid` returns 127 instantly — observed 20260720T162402Z).
launch_local() {  # $1 arm  $2 ckpt  $3 port  $4 samples_glob  $5 n_exhausted(0=all)
  ROOT="$ROOT" ARM="$1" MODE=local CKPT="$2" PORT="$3" SAMPLES_GLOB="$4" \
  N_EXHAUSTED="${5:-0}" \
  srun --exclusive --nodes=1 --ntasks=1 --gres=gpu:8 --cpus-per-task=192 \
       --time=12:00:00 --kill-on-bad-exit=1 --job-name="m3-sprobe-$1" --export=ALL \
       bash "$REPO/pipeline/slurm/sampling_probe_arm.sh" \
       > "$ROOT/$1-srun.log" 2>&1 &
  LAST_PID=$!
}

launch_local gptq \
  "$REPO/artifacts/m3-awq-gptq-prepared/gptq-checkpoint-vllm-w123-abi-overlay" \
  8000 "$(G minimax-m3)" 0
GPTQ_PID=$LAST_PID
launch_local awq \
  "/mnt/nfs/hoangduy/results/m3-distributed-awq-full/20260720T060340Z-m3-ddp-awq-full-r5-deadchan/awq/MiniMax-M3-awq-W4AFP8/20260720-060402/checkpoint-vllm-w123" \
  8004 "$(G minimax-m3-awq-inhouse)" "$AWQ_EXHAUSTED_CAP"
AWQ_PID=$LAST_PID
launch_local cyankiwi \
  "/mnt/nfs/hoangduy/hf_assets/cyankiwi/MiniMax-M3-AWQ-INT4" \
  8003 "$(G minimax-m3-awq-cyankiwi)" "$AWQ_EXHAUSTED_CAP"
CYAN_PID=$LAST_PID

# BF16 probe: no GPU, just HTTP+CPU against the shared BF16 endpoint.
ROOT="$ROOT" ARM=bf16 MODE=remote BF16_PORT=8001 SAMPLES_GLOB="$(G minimax-m3-bf16)" \
  srun --nodes=1 --ntasks=1 --cpus-per-task=16 --time=12:00:00 \
       --job-name=m3-sprobe-bf16probe --export=ALL \
       bash "$REPO/pipeline/slurm/sampling_probe_arm.sh" \
       > "$ROOT/bf16probe-srun.log" 2>&1 &
BF16P_PID=$!

rc_all=0
for spec in "gptq:$GPTQ_PID" "awq:$AWQ_PID" "cyankiwi:$CYAN_PID" "bf16:$BF16P_PID"; do
  arm=${spec%%:*}; pid=${spec##*:}
  wait "$pid"; rc=$?
  echo "$rc" > "$ROOT/probe-$arm.rc"
  echo "[controller] probe $arm rc=$rc"
  [ "$rc" = 0 ] || rc_all=1
done

touch "$ROOT/client-done"       # release the held BF16 serve
wait "$BF16_JOB"; echo "[controller] bf16 serve rc=$?"
echo "$rc_all" > "$ROOT/controller.rc"
echo "[controller] done rc=$rc_all"
