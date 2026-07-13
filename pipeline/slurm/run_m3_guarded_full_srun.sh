#!/usr/bin/env bash
# Smoke the real trace, then run three guarded full quantization hypotheses.

set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
cd "$REPO_ROOT"

DRY_RUN="${DRY_RUN:-0}"
RUN_ID="${RUN_ID:-$(date -u +%Y%m%dT%H%M%SZ)-m3-guarded-full}"
LOG_ROOT="${LOG_ROOT:-/mnt/nfs/hoangduy/logs/m3-guarded-full/$RUN_ID}"
RESULT_ROOT="${RESULT_ROOT:-/mnt/nfs/hoangduy/results/m3-guarded-full/$RUN_ID}"
CONFIG="${CONFIG:-pipeline/configs/minimax_m3_full_calib.yaml}"
MODEL_ID="${MODEL_ID:-}"
ENV_FILE="${ENV_FILE:-/mnt/nfs/hoangduy/env.sh}"
VENV_ACTIVATE="${VENV_ACTIVATE:-/mnt/nfs/hoangduy/venvs/quant/bin/activate}"
TRACE_TIME_LIMIT="${TRACE_TIME_LIMIT:-02:00:00}"
ARM_TIME_LIMIT="${ARM_TIME_LIMIT:-24:00:00}"
SRUN_ARGS="${SRUN_ARGS:-}"
read -r -a EXTRA_SRUN_ARGS <<< "$SRUN_ARGS"

if [[ -n "$MODEL_ID" ]]; then
  MODEL_ID_ARGS=(--model-id "$MODEL_ID")
else
  MODEL_ID_ARGS=()
fi

trace_command=(
  srun --exclusive --nodes=1 --ntasks=1 --gres=gpu:1
  --time="$TRACE_TIME_LIMIT" --kill-on-bad-exit=1 "${EXTRA_SRUN_ARGS[@]}"
  python -m pipeline.m3_trace_diagnostic
  --config "$CONFIG" --output-dir "$RESULT_ROOT/trace-smoke"
  "${MODEL_ID_ARGS[@]}"
)

variants=(offsetfix nosmooth quant_only)

print_arm() {
  local variant="$1"
  local command=(
    srun --exclusive --nodes=1 --ntasks=1 --gres=gpu:1
    --time="$ARM_TIME_LIMIT" --kill-on-bad-exit=1 "${EXTRA_SRUN_ARGS[@]}"
    python -m pipeline.m3_guarded_full arm
    --variant "$variant" --config "$CONFIG"
    --output-dir "$RESULT_ROOT/$variant" "${MODEL_ID_ARGS[@]}"
  )
  printf '%q ' "${command[@]}"
  printf '> %q 2>&1\n' "$LOG_ROOT/$variant.log"
}

if [[ "$DRY_RUN" == 1 || "$DRY_RUN" == true ]]; then
  echo "RUN_ID=$RUN_ID"
  echo "LOG_ROOT=$LOG_ROOT"
  echo "RESULT_ROOT=$RESULT_ROOT"
  printf '%q ' python -m pipeline.prequant_compatibility --config "$CONFIG" \
    --output "$RESULT_ROOT/prequant_compatibility.json" "${MODEL_ID_ARGS[@]}"
  printf '> %q 2>&1\n' "$LOG_ROOT/prequant-compatibility.log"
  printf '%q ' "${trace_command[@]}"
  printf '> %q 2>&1\n' "$LOG_ROOT/trace-smoke.log"
  for variant in "${variants[@]}"; do print_arm "$variant"; done
  printf '%q ' python -m pipeline.m3_guarded_full aggregate --result-root "$RESULT_ROOT"
  printf '\n'
  exit 0
fi

if [[ -n "${SLURM_JOB_ID:-}" ]]; then
  echo "refusing nested srun under SLURM_JOB_ID=$SLURM_JOB_ID; launch from a login/control shell" >&2
  exit 2
fi
# shellcheck disable=SC1090
source "$ENV_FILE" || exit 2
# shellcheck disable=SC1090
source "$VENV_ACTIVATE" || exit 2
export PYTHONPATH="$REPO_ROOT/src:$REPO_ROOT${PYTHONPATH:+:$PYTHONPATH}"
mkdir -p "$LOG_ROOT" "$RESULT_ROOT/trace-smoke"

prequant_args=(
  python -m pipeline.prequant_compatibility
  --config "$CONFIG" --output "$RESULT_ROOT/prequant_compatibility.json"
  "${MODEL_ID_ARGS[@]}"
)
prequant_rc=0
"${prequant_args[@]}" >"$LOG_ROOT/prequant-compatibility.log" 2>&1 || prequant_rc=$?
printf '%s\n' "$prequant_rc" >"$RESULT_ROOT/prequant_compatibility.rc"
if [[ "$prequant_rc" -ne 0 ]]; then
  echo "prequant compatibility gate failed rc=$prequant_rc; no srun was started" >&2
  exit "$prequant_rc"
fi

trace_rc=0
"${trace_command[@]}" >"$LOG_ROOT/trace-smoke.log" 2>&1 || trace_rc=$?
printf '%s\n' "$trace_rc" >"$RESULT_ROOT/trace-smoke/rc.tmp"
mv "$RESULT_ROOT/trace-smoke/rc.tmp" "$RESULT_ROOT/trace-smoke/rc"
if [[ "$trace_rc" -ne 0 ]]; then
  echo "trace smoke failed rc=$trace_rc; full arms were not started" >&2
  exit "$trace_rc"
fi

pids=()
for variant in "${variants[@]}"; do
  mkdir -p "$RESULT_ROOT/$variant"
  (
    rc=0
    command=(
      srun --exclusive --nodes=1 --ntasks=1 --gres=gpu:1
      --time="$ARM_TIME_LIMIT" --kill-on-bad-exit=1 "${EXTRA_SRUN_ARGS[@]}"
      python -m pipeline.m3_guarded_full arm
      --variant "$variant" --config "$CONFIG"
      --output-dir "$RESULT_ROOT/$variant" "${MODEL_ID_ARGS[@]}"
    )
    "${command[@]}" >"$LOG_ROOT/$variant.log" 2>&1 || rc=$?
    printf '%s\n' "$rc" >"$RESULT_ROOT/$variant/rc.tmp"
    mv "$RESULT_ROOT/$variant/rc.tmp" "$RESULT_ROOT/$variant/rc"
    exit "$rc"
  ) &
  pids+=("$!")
  echo "$variant controller_pid=$! log=$LOG_ROOT/$variant.log"
done

overall=0
for index in "${!pids[@]}"; do
  rc=0
  wait "${pids[$index]}" || rc=$?
  echo "${variants[$index]} rc=$rc"
  [[ "$rc" -eq 0 ]] || overall=1
done
python -m pipeline.m3_guarded_full aggregate --result-root "$RESULT_ROOT"
exit "$overall"
