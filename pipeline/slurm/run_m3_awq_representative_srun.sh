#!/usr/bin/env bash
# Run the six independent MiniMax-M3 representative-layer AWQ arms.

set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
cd "$REPO_ROOT"

DRY_RUN="${DRY_RUN:-0}"
TIME_LIMIT="${TIME_LIMIT:-12:00:00}"
RUN_ID="${RUN_ID:-$(date -u +%Y%m%dT%H%M%SZ)-$$}"
LOG_ROOT="${LOG_ROOT:-/mnt/nfs/hoangduy/logs/m3-awq-representative/$RUN_ID}"
RESULT_ROOT="${RESULT_ROOT:-/mnt/nfs/hoangduy/results/m3-awq-representative/$RUN_ID}"
ENV_FILE="${ENV_FILE:-/mnt/nfs/hoangduy/env.sh}"
VENV_ACTIVATE="${VENV_ACTIVATE:-/mnt/nfs/hoangduy/venvs/quant/bin/activate}"
SRUN_ARGS="${SRUN_ARGS:-}"
ARM_FILTER="${ARM_FILTER:-}"
read -r -a EXTRA_SRUN_ARGS <<< "$SRUN_ARGS"

names=(
  offsetfix-layer8 offsetfix-layer31 offsetfix-layer59
  nosmooth-layer8 nosmooth-layer31 nosmooth-layer59
)
variants=(offsetfix offsetfix offsetfix nosmooth nosmooth nosmooth)
layers=(8 31 59 8 31 59)
pids=()
launched_names=()
if [[ -n "$ARM_FILTER" ]]; then
  valid=0
  for name in "${names[@]}"; do [[ "$name" == "$ARM_FILTER" ]] && valid=1; done
  ((valid == 1)) || { echo "unknown ARM_FILTER=$ARM_FILTER" >&2; exit 2; }
fi

if [[ "$DRY_RUN" != 1 && "$DRY_RUN" != true ]]; then
  if [[ -n "${SLURM_JOB_ID:-}" ]]; then
    echo "refusing nested srun under SLURM_JOB_ID=$SLURM_JOB_ID; launch from outside any Slurm allocation" >&2
    exit 2
  fi
  # shellcheck disable=SC1090
  if ! source "$ENV_FILE"; then
    echo "failed to source ENV_FILE=$ENV_FILE" >&2
    exit 2
  fi
  # shellcheck disable=SC1090
  if ! source "$VENV_ACTIVATE"; then
    echo "failed to source VENV_ACTIVATE=$VENV_ACTIVATE" >&2
    exit 2
  fi
  export PYTHONPATH="$REPO_ROOT/src:$REPO_ROOT${PYTHONPATH:+:$PYTHONPATH}"
  if ! mkdir -p "$LOG_ROOT" "$RESULT_ROOT"; then
    echo "failed to create evidence roots: $LOG_ROOT $RESULT_ROOT" >&2
    exit 2
  fi
fi

for index in "${!names[@]}"; do
  name="${names[$index]}"
  [[ -z "$ARM_FILTER" || "$name" == "$ARM_FILTER" ]] || continue
  launched_names+=("$name")
  output_dir="$RESULT_ROOT/$name"
  command=(
    srun --exclusive --nodes=1 --ntasks=1 --gres=gpu:1 --time="$TIME_LIMIT"
    --kill-on-bad-exit=0 "${EXTRA_SRUN_ARGS[@]}"
    python -m pipeline.m3_awq_representative arm
    --layer "${layers[$index]}"
    --variant "${variants[$index]}"
    --output-dir "$output_dir"
  )

  if [[ "$DRY_RUN" == 1 || "$DRY_RUN" == true ]]; then
    printf '%q ' "${command[@]}"
    printf '> %q 2>&1\n' "$LOG_ROOT/$name.log"
    continue
  fi

  if ! mkdir -p "$output_dir"; then
    echo "$name: failed to create arm output directory $output_dir" >&2
    exit 2
  fi

  (
    rc_file="$output_dir/rc"
    record_rc() {
      local rc="$1"
      trap - EXIT TERM HUP INT
      if ! printf '%s\n' "$rc" >"$rc_file.tmp" || ! mv "$rc_file.tmp" "$rc_file"; then
        echo "$name: failed to atomically write return code to $rc_file" >&2
        exit 125
      fi
      exit "$rc"
    }
    trap 'record_rc $?' EXIT
    trap 'exit 143' TERM
    trap 'exit 129' HUP
    trap 'exit 130' INT
    # SIGKILL cannot be trapped; scheduler accounting is required if its rc file is absent.
    "${command[@]}" >"$LOG_ROOT/$name.log" 2>&1
  ) &
  pids+=("$!")
  echo "$name pid=$! log=$LOG_ROOT/$name.log"
done

if [[ "$DRY_RUN" == 1 || "$DRY_RUN" == true ]]; then
  exit 0
fi

overall=0
for index in "${!pids[@]}"; do
  rc=0
  wait "${pids[$index]}" || rc=$?
  echo "${launched_names[$index]} rc=$rc log=$LOG_ROOT/${launched_names[$index]}.log"
  [[ "$rc" -eq 0 ]] || overall=1
done

if [[ -z "$ARM_FILTER" ]]; then
  aggregate_rc=0
  python -m pipeline.m3_awq_representative aggregate \
    --result-root "$RESULT_ROOT" \
    --matrix-json "$RESULT_ROOT/matrix.json" \
    --report-md "$RESULT_ROOT/report.md" || aggregate_rc=$?
  [[ "$aggregate_rc" -eq 0 ]] || overall=1
  echo "aggregate rc=$aggregate_rc matrix=$RESULT_ROOT/matrix.json report=$RESULT_ROOT/report.md"
else
  echo "one-arm smoke complete arm=$ARM_FILTER rc=$overall result=$RESULT_ROOT/$ARM_FILTER"
fi
exit "$overall"
