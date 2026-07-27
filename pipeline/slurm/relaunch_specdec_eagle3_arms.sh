#!/usr/bin/env bash
# Relaunch selected arms of an existing spec-dec window (M3_SPECDEC_EAGLE3_PLAN.md)
# into the SAME ROOT/RUN_TS, retrying on a different node when the serve preflight
# refuses to start.
#
# Why this exists: slurm reports nodes as `idle` that carry another user's
# out-of-band GPU processes (2026-07-27: gpu-h98 and gpu-h101 were each fully
# occupied by a foreign DeepSeek-V4 run, ~32 GiB free per GPU). The serve
# preflight fails closed there -- correctly -- so the arm must move nodes rather
# than share. Kept separate from run_specdec_eagle3_srun.sh because bash reads a
# script incrementally: editing a launcher while its controller is running can
# corrupt the running shell.
#
#   ROOT=/mnt/nfs/.../20260727T061506Z RUN_TS=20260727T061506Z \
#     EXCLUDE=gpu-h98,gpu-h101 bash relaunch_specdec_eagle3_arms.sh k0-control:0:8020 k5:5:8023
set -uo pipefail
ROOT=${ROOT:?}; RUN_TS=${RUN_TS:?}
EXCLUDE=${EXCLUDE:-}
MAX_TRIES=${MAX_TRIES:-3}
REPO=/mnt/nfs/hoangduy/projects/llm-compressor
GPTQ_CKPT="$REPO/artifacts/m3-awq-gptq-prepared/gptq-checkpoint-vllm-w123-abi-overlay"
DRAFTER=/mnt/nfs/hoangduy/hf_assets/Inferact/MiniMax-M3-EAGLE3
SITE_0110=/mnt/nfs/hoangduy/venvs/humming-0.1.10-site

export RUN_TS
export LLMC_M3_CAPTURE_SYNC=sync

run_arm() {  # $1 = arm:spec_k:port
  local spec=$1
  local arm=${spec%%:*} rest=${spec#*:}
  local k=${rest%%:*} port=${rest##*:}
  local try=1 rc=1
  while [ "$try" -le "$MAX_TRIES" ]; do
    local -a exclude_arg=()
    [ -n "$EXCLUDE" ] && exclude_arg=(--exclude="$EXCLUDE")
    echo "[relaunch] $arm k=$k port=$port try=$try exclude='${EXCLUDE:-none}'"
    local -a arm_env=(
      "ROOT=$ROOT" "ARM=$arm" "RUN_TS=$RUN_TS" "SPEC_K=$k"
      "CKPT=$GPTQ_CKPT" "PORT=$port" "DRAFTER=$DRAFTER"
      "HUMMING_M3_W4A8_CACHE_ROOT=/mnt/nfs/hoangduy/.humming"
      "M3_W4A8_BACKEND=humming" "VLLM_HUMMING_MOE_GEMM_TYPE=indexed"
      "VLLM_HUMMING_USE_F16_ACCUM=0" "PYTHONPATH=$SITE_0110:$REPO"
      "AA_INPUTS=1k,10k" "AA_CONC=1,10"
    )
    env "${arm_env[@]}" \
    srun --exclusive --nodes=1 --ntasks=1 --gres=gpu:8 --cpus-per-task=192 \
         --time=04:00:00 --kill-on-bad-exit=1 --partition=compute \
         "${exclude_arg[@]}" --job-name="m3-spec-$arm" --export=ALL \
         bash "$REPO/pipeline/slurm/specdec_eagle3_arm.sh" \
         > "$ROOT/$arm-srun.try$try.log" 2>&1
    rc=$?
    printf 'arm=%s spec_k=%s port=%s try=%s rc=%s node=%s at=%s\n' \
      "$arm" "$k" "$port" "$try" "$rc" \
      "$(grep -o 'srun: error: [a-z0-9-]*' "$ROOT/$arm-srun.try$try.log" | head -1 | awk '{print $3}')" \
      "$(date -u +%Y-%m-%dT%H:%M:%SZ)" >>"$ROOT/arm-provenance.txt"
    [ "$rc" = 0 ] && { echo "[relaunch] $arm rc=0"; return 0; }
    # Preflight refusal => the node is dirty; exclude it and try elsewhere.
    if grep -q "GPUs are not free" "$ROOT/arm-$arm/client.log" 2>/dev/null; then
      local bad
      bad=$(grep -o 'srun: error: [a-z0-9-]*' "$ROOT/$arm-srun.try$try.log" | head -1 | awk '{print $3}')
      if [ -n "$bad" ]; then
        EXCLUDE="${EXCLUDE:+$EXCLUDE,}$bad"
        echo "[relaunch] $arm: node $bad busy with a foreign job; excluding and retrying"
      fi
      # Stale refusal text from the previous try must not decide the next one.
      mv "$ROOT/arm-$arm/client.log" "$ROOT/arm-$arm/client.log.try$try" 2>/dev/null || true
    else
      echo "[relaunch] $arm rc=$rc is not a preflight refusal -- stopping"
      return "$rc"
    fi
    try=$((try + 1))
  done
  echo "[relaunch] $arm exhausted $MAX_TRIES tries"
  return 1
}

rc_all=0
for spec in "$@"; do
  run_arm "$spec" || rc_all=1
done
echo "[relaunch] done rc_all=$rc_all"
exit "$rc_all"
