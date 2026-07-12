#!/usr/bin/env bash
# Start the MiniMax-M3 quality smoke srun controller inside detached tmux.
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
cd "$REPO_ROOT"

DRY_RUN="${DRY_RUN:-0}"
RUN_ID="${RUN_ID:-$(date -u +%Y%m%dT%H%M%SZ)-m3-quality-smoke}"
SESSION_NAME="${SESSION_NAME:-m3-quality-${RUN_ID}}"
RUN_ROOT="${RUN_ROOT:?set RUN_ROOT from completed preflight}"
MATRIX="${MATRIX:?set MATRIX to repaired matrix}"
REPAIRED_GPTQ="${REPAIRED_GPTQ:?set REPAIRED_GPTQ}"
LOG_ROOT="${LOG_ROOT:-$RUN_ROOT/logs}"
QUALITY_ARM_FILTER="${QUALITY_ARM_FILTER:-}"
CONTROLLER_LOG="${CONTROLLER_LOG:-$LOG_ROOT/controller.log}"
CONTROLLER_SCRIPT="$RUN_ROOT/controller.sh"

[[ "$SESSION_NAME" =~ ^[A-Za-z0-9_.-]+$ ]] || { echo "invalid tmux SESSION_NAME=$SESSION_NAME" >&2; exit 2; }
print_commands() {
  printf '%q ' tmux new-session -d -s "$SESSION_NAME" -c "$REPO_ROOT" "bash $(printf '%q' "$CONTROLLER_SCRIPT")"; printf '\n'
  printf '%q ' tmux has-session -t "=$SESSION_NAME"; printf '\n'
  printf '%q ' tmux capture-pane -pt "=$SESSION_NAME" -S -80; printf '\n'
  printf '%q ' tmux attach-session -t "=$SESSION_NAME"; printf '\n'
  printf 'tail -f %q\n' "$CONTROLLER_LOG"
  printf 'squeue -u "$USER" -o %q\n' '%.18i %.28j %.8T %.10M %.10l %.6D %R'
  printf 'cat %q\n' "$RUN_ROOT/controller.rc"
}
if [[ "$DRY_RUN" == 1 || "$DRY_RUN" == true ]]; then
  echo "RUN_ID=$RUN_ID SESSION_NAME=$SESSION_NAME RUN_ROOT=$RUN_ROOT"
  echo "controller executes: $SCRIPT_DIR/run_m3_quality_smoke_srun.sh"
  print_commands
  DRY_RUN=1 RUN_ROOT="$RUN_ROOT" MATRIX="$MATRIX" REPAIRED_GPTQ="$REPAIRED_GPTQ" LOG_ROOT="$LOG_ROOT" QUALITY_ARM_FILTER="$QUALITY_ARM_FILTER" bash "$SCRIPT_DIR/run_m3_quality_smoke_srun.sh"
  exit 0
fi
if [[ -n "${SLURM_JOB_ID:-}" ]]; then
  echo "refusing nested srun under SLURM_JOB_ID=$SLURM_JOB_ID; start this tmux wrapper from a login/control shell outside any Slurm allocation so each top-level srun receives an exclusive node" >&2
  exit 2
fi
command -v tmux >/dev/null 2>&1 || { echo "tmux is required; install/load it before launching srun" >&2; exit 2; }
if tmux has-session -t "=$SESSION_NAME" 2>/dev/null; then echo "tmux session already exists: $SESSION_NAME" >&2; exit 2; fi
for stale in "$RUN_ROOT/controller.sh" "$RUN_ROOT/controller.rc" "$CONTROLLER_LOG"; do
  [[ ! -e "$stale" ]] || { echo "existing run evidence at $stale; choose a fresh RUN_ID/RUN_ROOT" >&2; exit 2; }
done
mkdir -p "$LOG_ROOT" "$RUN_ROOT"
{
  echo '#!/usr/bin/env bash'; echo 'set -uo pipefail'; printf 'cd %q\n' "$REPO_ROOT"
  printf 'export RUN_ROOT=%q\n' "$RUN_ROOT"; printf 'export MATRIX=%q\n' "$MATRIX"
  printf 'export REPAIRED_GPTQ=%q\n' "$REPAIRED_GPTQ"; printf 'export LOG_ROOT=%q\n' "$LOG_ROOT"
  printf 'CONTROLLER_LOG=%q\n' "$CONTROLLER_LOG"; printf 'RC_FILE=%q\n' "$RUN_ROOT/controller.rc"
  echo 'exec >>"$CONTROLLER_LOG" 2>&1'; echo 'echo "controller started=$(date -Is) host=$(hostname) pid=$$"'
  printf 'rc=0; bash %q || rc=$?\n' "$SCRIPT_DIR/run_m3_quality_smoke_srun.sh"
  echo 'printf "%s\n" "$rc" >"$RC_FILE.tmp"'; echo 'mv "$RC_FILE.tmp" "$RC_FILE"'
  echo 'echo "controller finished=$(date -Is) rc=$rc"'; echo 'exit "$rc"'
} >"$CONTROLLER_SCRIPT"
chmod 700 "$CONTROLLER_SCRIPT"
tmux new-session -d -s "$SESSION_NAME" -c "$REPO_ROOT" "bash $(printf '%q' "$CONTROLLER_SCRIPT")"
tmux has-session -t "=$SESSION_NAME" 2>/dev/null || { echo "tmux session exited during startup: $SESSION_NAME; inspect $CONTROLLER_LOG" >&2; exit 1; }
echo "verified detached tmux session: $SESSION_NAME"
echo "The Cursor tool may exit now; do not kill the tmux server."
print_commands
