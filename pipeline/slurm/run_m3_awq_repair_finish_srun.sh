#!/usr/bin/env bash
# Add the six fresh-AWQ arms to an existing early GPTQ matrix and aggregate.

set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
cd "$REPO_ROOT"

DRY_RUN="${DRY_RUN:-0}"
MATRIX_ID="${MATRIX_ID:?reuse MATRIX_ID printed by run_m3_gptq_early_srun.sh}"
RESULTS_ROOT="${RESULTS_ROOT:-/mnt/nfs/hoangduy/logs/m3-awq-gptq-repair}"
EVIDENCE_ROOT="${EVIDENCE_ROOT:-$REPO_ROOT/results/m3-awq-gptq-repair}"
LOG_ROOT="${LOG_ROOT:-/mnt/nfs/hoangduy/logs/m3-awq-gptq-repair-srun}"
TIME_LIMIT="${TIME_LIMIT:-04:00:00}"
DIAGNOSTIC_LAYERS="${M3_DIAGNOSTIC_LAYERS:-$(seq -s, 3 59)}"
ARMS=(
  awq_offsetfix_w4a8 awq_offsetfix_w4a16 awq_offsetfix_http
  awq_nosmooth_w4a8 awq_nosmooth_w4a16 awq_nosmooth_http
)
# shellcheck disable=SC2206
EXTRA_SRUN_ARGS=(${SRUN_ARGS:-})

if [[ "$DRY_RUN" != 1 && "$DRY_RUN" != true \
      && ! -f "$EVIDENCE_ROOT/$MATRIX_ID/comparison_early.json" ]]; then
  echo "ERROR: early evidence missing for MATRIX_ID=$MATRIX_ID" >&2
  exit 2
fi

pids=()
for arm in "${ARMS[@]}"; do
  log="$LOG_ROOT/$MATRIX_ID-$arm.log"
  command=(srun --exclusive --nodes=1 --ntasks=1 --gres=gpu:8
    --time="$TIME_LIMIT" --kill-on-bad-exit=0 "${EXTRA_SRUN_ARGS[@]}"
    env "MATRIX_ID=$MATRIX_ID" "ARM=$arm" "RESULTS_ROOT=$RESULTS_ROOT"
    "EVIDENCE_ROOT=$EVIDENCE_ROOT" "M3_REPAIR_SRUN_LOG=$log"
    "M3_DIAGNOSTIC_LAYERS=$DIAGNOSTIC_LAYERS"
    bash "$SCRIPT_DIR/test_m3_layer_boundary_arm.sh")
  if [[ "$DRY_RUN" == 1 || "$DRY_RUN" == true ]]; then
    printf '%q ' "${command[@]}"; printf '\n'
  else
    mkdir -p "$LOG_ROOT"
    "${command[@]}" >"$log" 2>&1 &
    pids+=("$!")
    echo "$arm pid=$! log=$log"
  fi
done
if [[ "$DRY_RUN" == 1 || "$DRY_RUN" == true ]]; then
  echo "matrix_id=$MATRIX_ID"
  exit 0
fi

overall=0
for index in "${!ARMS[@]}"; do
  rc=0; wait "${pids[$index]}" || rc=$?
  echo "${ARMS[$index]} srun_rc=$rc"
  [[ "$rc" -eq 0 ]] || overall=1
done
python -m pipeline.m3_awq_gptq_repair aggregate \
  --phase final --evidence-root "$EVIDENCE_ROOT/$MATRIX_ID"
echo "matrix_id=$MATRIX_ID"
echo "comparison=$EVIDENCE_ROOT/$MATRIX_ID/comparison.json"
exit "$overall"
