#!/usr/bin/env bash
# Launch eleven MiniMax-M3 layer-boundary arms concurrently with srun.

set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
cd "$REPO_ROOT"

DRY_RUN="${DRY_RUN:-0}"
MATRIX_ID="${MATRIX_ID:-$(date +%Y%m%d-%H%M%S)-layer-boundary}"
RESULTS_ROOT="${RESULTS_ROOT:-/mnt/nfs/hoangduy/logs/m3-layer-boundary}"
EVIDENCE_ROOT="${EVIDENCE_ROOT:-$REPO_ROOT/results/m3-layer-boundary}"
LOG_ROOT="${LOG_ROOT:-/mnt/nfs/hoangduy/logs/m3-layer-boundary-srun}"
TIME_LIMIT="${TIME_LIMIT:-02:00:00}"
ENV_FILE="${ENV_FILE:-/mnt/nfs/hoangduy/env.sh}"
VENV_ACTIVATE="${VENV_ACTIVATE:-/mnt/nfs/hoangduy/venvs/quant/bin/activate}"
ARMS=(
  reference_w4a16_ep_fp8kv candidate_w4a8_ep_fp8kv candidate_w4a16_ep_fp8kv
  candidate_w4a8_router_alias candidate_w4a16_router_alias
  reference_w4a16_tp_fp8kv candidate_w4a8_tp_fp8kv candidate_w4a16_tp_fp8kv
  candidate_w4a8_ep_autokv candidate_w4a16_ep_autokv candidate_w4a8_router_http
)
# shellcheck disable=SC2206
EXTRA_SRUN_ARGS=(${SRUN_ARGS:-})

if [[ "$DRY_RUN" != 1 && "$DRY_RUN" != true ]]; then
  mkdir -p "$LOG_ROOT"
  if [[ ! -f "$ENV_FILE" || ! -f "$VENV_ACTIVATE" ]]; then
    echo "ERROR: missing environment files: $ENV_FILE $VENV_ACTIVATE" >&2; exit 2
  fi
  # shellcheck disable=SC1090
  source "$ENV_FILE"
  # shellcheck disable=SC1090
  source "$VENV_ACTIVATE"
  export PYTHONPATH="$REPO_ROOT"
  python "$SCRIPT_DIR/patch_vllm_m3_serve.py"
  python "$SCRIPT_DIR/patch_vllm_m3_serve.py" --check
fi

pids=()
for arm in "${ARMS[@]}"; do
  log="$LOG_ROOT/$MATRIX_ID-$arm.log"
  command=(
    srun --exclusive --nodes=1 --ntasks=1 --gres=gpu:8
    --time="$TIME_LIMIT" --kill-on-bad-exit=0 "${EXTRA_SRUN_ARGS[@]}"
    env "MATRIX_ID=$MATRIX_ID" "ARM=$arm" "RESULTS_ROOT=$RESULTS_ROOT"
    "EVIDENCE_ROOT=$EVIDENCE_ROOT" "M3_BOUNDARY_SRUN_LOG=$log"
    bash "$SCRIPT_DIR/test_m3_layer_boundary_arm.sh"
  )
  if [[ "$DRY_RUN" == 1 || "$DRY_RUN" == true ]]; then
    printf '%q ' "${command[@]}"; printf '\n'
  else
    "${command[@]}" >"$log" 2>&1 &
    pids+=("$!")
    echo "$arm pid=$! log=$log"
  fi
done
if [[ "$DRY_RUN" == 1 || "$DRY_RUN" == true ]]; then
  echo "matrix_id=$MATRIX_ID"; exit 0
fi

overall=0
for index in "${!ARMS[@]}"; do
  arm="${ARMS[$index]}"; pid="${pids[$index]}"; rc=0
  wait "$pid" || rc=$?
  echo "$rc" >"$LOG_ROOT/$MATRIX_ID-$arm.srun_return_code.txt"
  echo "$arm srun_rc=$rc"
  [[ "$rc" -eq 0 ]] || overall=1
done
python -m pipeline.m3_layer_boundary_diagnostics aggregate \
  --evidence-root "$EVIDENCE_ROOT/$MATRIX_ID"
echo "matrix_id=$MATRIX_ID"
echo "comparison=$EVIDENCE_ROOT/$MATRIX_ID/comparison.json"
exit "$overall"
