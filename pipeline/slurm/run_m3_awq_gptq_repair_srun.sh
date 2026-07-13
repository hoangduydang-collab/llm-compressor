#!/usr/bin/env bash
# Launch the MiniMax-M3 AWQ/GPTQ repair matrix concurrently with srun.

set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
cd "$REPO_ROOT"

DRY_RUN="${DRY_RUN:-0}"
MATRIX_ID="${MATRIX_ID:-$(date +%Y%m%d-%H%M%S)-awq-gptq-repair}"
RESULTS_ROOT="${RESULTS_ROOT:-/mnt/nfs/hoangduy/logs/m3-awq-gptq-repair}"
EVIDENCE_ROOT="${EVIDENCE_ROOT:-$REPO_ROOT/results/m3-awq-gptq-repair}"
LOG_ROOT="${LOG_ROOT:-/mnt/nfs/hoangduy/logs/m3-awq-gptq-repair-srun}"
TIME_LIMIT="${TIME_LIMIT:-04:00:00}"
DIAGNOSTIC_LAYERS="${M3_DIAGNOSTIC_LAYERS:-$(seq -s, 3 59)}"
ARMS=(
  reference_w4a16 awq_control_w4a8
  gptq_w4a8 gptq_w4a16 gptq_http
  awq_offsetfix_w4a8 awq_offsetfix_w4a16 awq_offsetfix_http
  awq_nosmooth_w4a8 awq_nosmooth_w4a16 awq_nosmooth_http
)
# shellcheck disable=SC2206
EXTRA_SRUN_ARGS=(${SRUN_ARGS:-})

mkdir -p "$EVIDENCE_ROOT/$MATRIX_ID"
pids=()
for arm in "${ARMS[@]}"; do
  log="$LOG_ROOT/$MATRIX_ID-$arm.log"
  command=(
    srun --exclusive --nodes=1 --ntasks=1 --gres=gpu:8
    --time="$TIME_LIMIT" --kill-on-bad-exit=0 "${EXTRA_SRUN_ARGS[@]}"
    env "MATRIX_ID=$MATRIX_ID" "ARM=$arm" "RESULTS_ROOT=$RESULTS_ROOT"
    "EVIDENCE_ROOT=$EVIDENCE_ROOT" "M3_REPAIR_SRUN_LOG=$log"
    "M3_DIAGNOSTIC_LAYERS=$DIAGNOSTIC_LAYERS"
    bash "$SCRIPT_DIR/test_m3_layer_boundary_arm.sh"
  )
  if [[ "$DRY_RUN" == 1 || "$DRY_RUN" == true ]]; then
    printf '%q ' "${command[@]}"; printf '\n'
  else
    mkdir -p "$LOG_ROOT"
    "${command[@]}" >"$log" 2>&1 &
    pids+=("$!")
    echo "$arm pid=$! log=$log"
  fi
done

audit_log="$LOG_ROOT/$MATRIX_ID-checkpoint-scale-audit.log"
audit_command=(
  srun --exclusive --nodes=1 --ntasks=1 --gres=gpu:1
  --time="$TIME_LIMIT" --kill-on-bad-exit=0 "${EXTRA_SRUN_ARGS[@]}"
  env "OUTPUT=$EVIDENCE_ROOT/$MATRIX_ID/checkpoint_scale_audit.json"
  "M3_DIAGNOSTIC_LAYERS=$DIAGNOSTIC_LAYERS"
  bash "$SCRIPT_DIR/test_m3_checkpoint_scale_audit.sh"
)
if [[ "$DRY_RUN" == 1 || "$DRY_RUN" == true ]]; then
  printf '%q ' "${audit_command[@]}"; printf '\n'
  echo "matrix_id=$MATRIX_ID"
  exit 0
fi
"${audit_command[@]}" >"$audit_log" 2>&1 &
audit_pid="$!"

overall=0
for index in "${!ARMS[@]}"; do
  rc=0
  wait "${pids[$index]}" || rc=$?
  echo "${ARMS[$index]} srun_rc=$rc"
  [[ "$rc" -eq 0 ]] || overall=1
done
audit_rc=0
wait "$audit_pid" || audit_rc=$?
echo "checkpoint_scale_audit srun_rc=$audit_rc"
cp "$audit_log" "$EVIDENCE_ROOT/$MATRIX_ID/checkpoint_scale_audit.log" || true
echo "$audit_rc" >"$EVIDENCE_ROOT/$MATRIX_ID/checkpoint_scale_audit.return_code.txt"
[[ "$audit_rc" -eq 0 ]] || overall=1

python -m pipeline.m3_awq_gptq_repair aggregate \
  --evidence-root "$EVIDENCE_ROOT/$MATRIX_ID"
echo "matrix_id=$MATRIX_ID"
echo "comparison=$EVIDENCE_ROOT/$MATRIX_ID/comparison.json"
exit "$overall"
