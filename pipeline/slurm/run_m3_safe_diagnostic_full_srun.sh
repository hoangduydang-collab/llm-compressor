#!/usr/bin/env bash
# Preflight MiniMax-M3, then run three safe and two diagnostic lanes concurrently.

set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
cd "$REPO_ROOT"

DRY_RUN="${DRY_RUN:-0}"
RUN_ID="${RUN_ID:-$(date -u +%Y%m%dT%H%M%SZ)-m3-safe-diagnostic-full}"
LOG_ROOT="${LOG_ROOT:-/mnt/nfs/hoangduy/logs/m3-safe-diagnostic-full/$RUN_ID}"
RESULT_ROOT="${RESULT_ROOT:-/mnt/nfs/hoangduy/results/m3-safe-diagnostic-full/$RUN_ID}"
CONFIG="${CONFIG:-pipeline/configs/minimax_m3_full_calib.yaml}"
MODEL_ID="${MODEL_ID:-}"
ENV_FILE="${ENV_FILE:-/mnt/nfs/hoangduy/env.sh}"
VENV_ACTIVATE="${VENV_ACTIVATE:-/mnt/nfs/hoangduy/venvs/quant/bin/activate}"
TRACE_TIME_LIMIT="${TRACE_TIME_LIMIT:-02:00:00}"
LANE_TIME_LIMIT="${LANE_TIME_LIMIT:-24:00:00}"
SRUN_ARGS="${SRUN_ARGS:-}"
LANE_FILTER="${LANE_FILTER:-}"
SAFE_NUM_SAMPLES="${SAFE_NUM_SAMPLES:-}"
SAFE_MAX_SEQ_LENGTH="${SAFE_MAX_SEQ_LENGTH:-}"
read -r -a EXTRA_SRUN_ARGS <<< "$SRUN_ARGS"

all_lanes=(
  safe-offsetfix safe-nosmooth safe-quant_only
  diag-heavy-offsetfix diag-light-offsetfix
)
lanes=()
for lane in "${all_lanes[@]}"; do
  if [[ -z "$LANE_FILTER" || "$lane" == "$LANE_FILTER" ]]; then
    lanes+=("$lane")
  fi
done
if [[ "${#lanes[@]}" -eq 0 ]]; then
  echo "LANE_FILTER did not match a known lane: $LANE_FILTER" >&2
  exit 2
fi

model_args=()
[[ -z "$MODEL_ID" ]] || model_args=(--model-id "$MODEL_ID")
trace_command=(
  srun --exclusive --nodes=1 --ntasks=1 --gres=gpu:1
  --time="$TRACE_TIME_LIMIT" --kill-on-bad-exit=1 "${EXTRA_SRUN_ARGS[@]}"
  python -m pipeline.m3_trace_diagnostic
  --config "$CONFIG" --output-dir "$RESULT_ROOT/trace-smoke"
  "${model_args[@]}"
)

safe_variant() {
  case "$1" in
    safe-offsetfix) echo offsetfix ;;
    safe-nosmooth) echo nosmooth ;;
    safe-quant_only) echo quant_only ;;
    *) return 2 ;;
  esac
}

print_lane() {
  local lane="$1"
  local command
  if [[ "$lane" == safe-* ]]; then
    local variant
    variant="$(safe_variant "$lane")"
    echo "lane=$lane production=python -m pipeline.run --stage quantize variant=$variant"
    command=(
      srun --exclusive --nodes=1 --ntasks=1 --gres=gpu:1
      --time="$LANE_TIME_LIMIT" --kill-on-bad-exit=1 "${EXTRA_SRUN_ARGS[@]}"
      bash "$SCRIPT_DIR/run_m3_safe_full_lane.sh"
      --variant "$variant" --lane-root "$RESULT_ROOT/lanes/$lane"
      --config "$CONFIG"
    )
    [[ -z "$MODEL_ID" ]] || command+=(--model-id "$MODEL_ID")
    [[ -z "$SAFE_NUM_SAMPLES" ]] || command+=(--num-samples "$SAFE_NUM_SAMPLES")
    [[ -z "$SAFE_MAX_SEQ_LENGTH" ]] || command+=(--max-seq-length "$SAFE_MAX_SEQ_LENGTH")
  else
    local mode="${lane#diag-}"
    mode="${mode%-offsetfix}"
    command=(
      srun --exclusive --nodes=1 --ntasks=1 --gres=gpu:1
      --time="$LANE_TIME_LIMIT" --kill-on-bad-exit=1 "${EXTRA_SRUN_ARGS[@]}"
      python -m pipeline.m3_guarded_full arm
      --variant offsetfix --diagnostic-mode "$mode" --config "$CONFIG"
      --output-dir "$RESULT_ROOT/lanes/$lane" "${model_args[@]}"
    )
  fi
  printf '%q ' "${command[@]}"
  printf '> %q 2>&1\n' "$LOG_ROOT/$lane.log"
}

if [[ "$DRY_RUN" == 1 || "$DRY_RUN" == true ]]; then
  echo "RUN_ID=$RUN_ID"
  echo "LOG_ROOT=$LOG_ROOT"
  echo "RESULT_ROOT=$RESULT_ROOT"
  printf '%q ' python -m pipeline.prequant_compatibility --config "$CONFIG" \
    --output "$RESULT_ROOT/prequant_compatibility.json" "${model_args[@]}"
  printf '> %q 2>&1\n' "$LOG_ROOT/prequant-compatibility.log"
  printf '%q ' "${trace_command[@]}"
  printf '> %q 2>&1\n' "$LOG_ROOT/trace-smoke.log"
  for lane in "${lanes[@]}"; do
    print_lane "$lane"
  done
  exit 0
fi

if [[ -n "${SLURM_JOB_ID:-}" ]]; then
  echo "refusing nested srun under SLURM_JOB_ID=$SLURM_JOB_ID; launch from login/control shell" >&2
  exit 2
fi
# shellcheck disable=SC1090
source "$ENV_FILE" || exit 2
# shellcheck disable=SC1090
source "$VENV_ACTIVATE" || exit 2
export PYTHONPATH="$REPO_ROOT/src:$REPO_ROOT${PYTHONPATH:+:$PYTHONPATH}"

for lane in "${lanes[@]}"; do
  if [[ -e "$RESULT_ROOT/lanes/$lane" ]]; then
    echo "refusing non-fresh lane root: $RESULT_ROOT/lanes/$lane" >&2
    exit 2
  fi
done
mkdir -p "$LOG_ROOT" "$RESULT_ROOT/status" "$RESULT_ROOT/trace-smoke"

prequant_rc=0
python -m pipeline.prequant_compatibility --config "$CONFIG" \
  --output "$RESULT_ROOT/prequant_compatibility.json" "${model_args[@]}" \
  >"$LOG_ROOT/prequant-compatibility.log" 2>&1 || prequant_rc=$?
printf '%s\n' "$prequant_rc" >"$RESULT_ROOT/prequant_compatibility.rc.tmp"
mv "$RESULT_ROOT/prequant_compatibility.rc.tmp" "$RESULT_ROOT/prequant_compatibility.rc"
if [[ "$prequant_rc" -ne 0 ]]; then
  echo "prequant compatibility gate failed rc=$prequant_rc; no srun was started" >&2
  exit "$prequant_rc"
fi

trace_rc=0
"${trace_command[@]}" >"$LOG_ROOT/trace-smoke.log" 2>&1 || trace_rc=$?
printf '%s\n' "$trace_rc" >"$RESULT_ROOT/trace-smoke/rc.tmp"
mv "$RESULT_ROOT/trace-smoke/rc.tmp" "$RESULT_ROOT/trace-smoke/rc"
if [[ "$trace_rc" -ne 0 ]]; then
  echo "trace smoke failed rc=$trace_rc; full lanes were not started" >&2
  exit "$trace_rc"
fi

pids=()
for lane in "${lanes[@]}"; do
  (
    rc=0
    if [[ "$lane" == safe-* ]]; then
      variant="$(safe_variant "$lane")"
      command=(
        srun --exclusive --nodes=1 --ntasks=1 --gres=gpu:1
        --time="$LANE_TIME_LIMIT" --kill-on-bad-exit=1 "${EXTRA_SRUN_ARGS[@]}"
        bash "$SCRIPT_DIR/run_m3_safe_full_lane.sh"
        --variant "$variant" --lane-root "$RESULT_ROOT/lanes/$lane"
        --config "$CONFIG"
      )
      [[ -z "$MODEL_ID" ]] || command+=(--model-id "$MODEL_ID")
      [[ -z "$SAFE_NUM_SAMPLES" ]] || command+=(--num-samples "$SAFE_NUM_SAMPLES")
      [[ -z "$SAFE_MAX_SEQ_LENGTH" ]] || command+=(--max-seq-length "$SAFE_MAX_SEQ_LENGTH")
    else
      mode="${lane#diag-}"
      mode="${mode%-offsetfix}"
      command=(
        srun --exclusive --nodes=1 --ntasks=1 --gres=gpu:1
        --time="$LANE_TIME_LIMIT" --kill-on-bad-exit=1 "${EXTRA_SRUN_ARGS[@]}"
        python -m pipeline.m3_guarded_full arm
        --variant offsetfix --diagnostic-mode "$mode" --config "$CONFIG"
        --output-dir "$RESULT_ROOT/lanes/$lane" "${model_args[@]}"
      )
    fi
    "${command[@]}" >"$LOG_ROOT/$lane.log" 2>&1 || rc=$?
    printf '%s\n' "$rc" >"$RESULT_ROOT/status/$lane.rc.tmp"
    mv "$RESULT_ROOT/status/$lane.rc.tmp" "$RESULT_ROOT/status/$lane.rc"
    exit "$rc"
  ) &
  pids+=("$!")
  echo "$lane controller_pid=$! log=$LOG_ROOT/$lane.log"
done

overall=0
: >"$RESULT_ROOT/lane-status.tsv.tmp"
for index in "${!pids[@]}"; do
  rc=0
  wait "${pids[$index]}" || rc=$?
  lane="${lanes[$index]}"
  printf '%s\t%s\n' "$lane" "$rc" >>"$RESULT_ROOT/lane-status.tsv.tmp"
  echo "$lane rc=$rc"
  [[ "$rc" -eq 0 ]] || overall=1
done
mv "$RESULT_ROOT/lane-status.tsv.tmp" "$RESULT_ROOT/lane-status.tsv"
exit "$overall"
