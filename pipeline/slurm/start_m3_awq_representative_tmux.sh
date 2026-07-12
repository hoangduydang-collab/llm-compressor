#!/usr/bin/env bash
# Start the representative-layer srun controller inside a detached tmux server.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
cd "$REPO_ROOT"

DRY_RUN="${DRY_RUN:-0}"
RUN_ID="${RUN_ID:-$(date -u +%Y%m%dT%H%M%SZ)-m3-awq-representative}"
SESSION_NAME="${SESSION_NAME:-m3-awq-${RUN_ID}}"
LOG_ROOT="${LOG_ROOT:-/mnt/nfs/hoangduy/logs/m3-awq-representative/$RUN_ID}"
RESULT_ROOT="${RESULT_ROOT:-/mnt/nfs/hoangduy/results/m3-awq-representative/$RUN_ID}"
CONTROLLER_LOG="${CONTROLLER_LOG:-$LOG_ROOT/controller.log}"
TIME_LIMIT="${TIME_LIMIT:-12:00:00}"
SRUN_ARGS="${SRUN_ARGS:-}"
CONTROLLER_SCRIPT="$RESULT_ROOT/controller.sh"

if [[ ! "$SESSION_NAME" =~ ^[A-Za-z0-9_.-]+$ ]]; then
  echo "invalid tmux SESSION_NAME=$SESSION_NAME" >&2
  exit 2
fi

print_commands() {
  printf '%q ' tmux new-session -d -s "$SESSION_NAME" -c "$REPO_ROOT" \
    "bash $(printf '%q' "$CONTROLLER_SCRIPT")"
  printf '\n'
  printf '%q ' tmux has-session -t "=$SESSION_NAME"; printf '\n'
  printf '%q ' tmux capture-pane -pt "=$SESSION_NAME" -S -80; printf '\n'
  printf '%q ' tmux attach-session -t "=$SESSION_NAME"; printf '\n'
  printf 'tail -f %q\n' "$CONTROLLER_LOG"
  printf 'squeue -u "$USER" -o %q\n' '%.18i %.28j %.8T %.10M %.10l %.6D %R'
  printf 'cat %q\n' "$RESULT_ROOT/controller.rc"
}

if [[ "$DRY_RUN" == 1 || "$DRY_RUN" == true ]]; then
  echo "RUN_ID=$RUN_ID"
  echo "SESSION_NAME=$SESSION_NAME"
  echo "LOG_ROOT=$LOG_ROOT"
  echo "RESULT_ROOT=$RESULT_ROOT"
  echo "CONTROLLER_LOG=$CONTROLLER_LOG"
  echo "controller executes: $SCRIPT_DIR/run_m3_awq_representative_srun.sh"
  print_commands
  echo "six-arm srun dry run:"
  DRY_RUN=1 RUN_ID="$RUN_ID" LOG_ROOT="$LOG_ROOT" RESULT_ROOT="$RESULT_ROOT" \
    TIME_LIMIT="$TIME_LIMIT" SRUN_ARGS="$SRUN_ARGS" \
    bash "$SCRIPT_DIR/run_m3_awq_representative_srun.sh"
  exit 0
fi

if [[ -n "${SLURM_JOB_ID:-}" ]]; then
  echo "refusing nested srun under SLURM_JOB_ID=$SLURM_JOB_ID; start this tmux wrapper from a login/control shell outside any Slurm allocation so each top-level srun receives an exclusive node" >&2
  exit 2
fi
if ! command -v tmux >/dev/null 2>&1; then
  echo "tmux is required; install/load it before launching srun" >&2
  exit 2
fi
if tmux has-session -t "=$SESSION_NAME" 2>/dev/null; then
  echo "tmux session already exists: $SESSION_NAME" >&2
  echo "inspect with: tmux capture-pane -pt =$SESSION_NAME -S -80" >&2
  exit 2
fi
for stale in \
  "$RESULT_ROOT/controller.sh" \
  "$RESULT_ROOT/controller.rc" \
  "$RESULT_ROOT/matrix.json" \
  "$RESULT_ROOT/report.md" \
  "$CONTROLLER_LOG"; do
  if [[ -e "$stale" ]]; then
    echo "existing run evidence at $stale; choose a fresh RUN_ID" >&2
    exit 2
  fi
done

mkdir -p "$LOG_ROOT" "$RESULT_ROOT"
{
  echo '#!/usr/bin/env bash'
  echo 'set -uo pipefail'
  printf 'cd %q\n' "$REPO_ROOT"
  printf 'export RUN_ID=%q\n' "$RUN_ID"
  printf 'export LOG_ROOT=%q\n' "$LOG_ROOT"
  printf 'export RESULT_ROOT=%q\n' "$RESULT_ROOT"
  printf 'export TIME_LIMIT=%q\n' "$TIME_LIMIT"
  printf 'export SRUN_ARGS=%q\n' "$SRUN_ARGS"
  printf 'CONTROLLER_LOG=%q\n' "$CONTROLLER_LOG"
  printf 'RC_FILE=%q\n' "$RESULT_ROOT/controller.rc"
  echo 'exec >>"$CONTROLLER_LOG" 2>&1'
  echo 'echo "controller started=$(date -Is) host=$(hostname) pid=$$ run_id=$RUN_ID"'
  printf 'rc=0; bash %q || rc=$?\n' "$SCRIPT_DIR/run_m3_awq_representative_srun.sh"
  echo 'printf "%s\n" "$rc" >"$RC_FILE.tmp"'
  echo 'mv "$RC_FILE.tmp" "$RC_FILE"'
  echo 'echo "controller finished=$(date -Is) rc=$rc"'
  echo 'exit "$rc"'
} >"$CONTROLLER_SCRIPT"
chmod 700 "$CONTROLLER_SCRIPT"

tmux new-session -d -s "$SESSION_NAME" -c "$REPO_ROOT" \
  "bash $(printf '%q' "$CONTROLLER_SCRIPT")"

if ! tmux has-session -t "=$SESSION_NAME" 2>/dev/null; then
  echo "tmux session exited during startup: $SESSION_NAME" >&2
  echo "inspect controller log: $CONTROLLER_LOG" >&2
  exit 1
fi

echo "verified detached tmux session: $SESSION_NAME"
echo "RUN_ID=$RUN_ID"
echo "SESSION_NAME=$SESSION_NAME"
echo "LOG_ROOT=$LOG_ROOT"
echo "RESULT_ROOT=$RESULT_ROOT"
echo "CONTROLLER_LOG=$CONTROLLER_LOG"
echo "The Cursor tool may exit now; do not kill the tmux server."
print_commands
