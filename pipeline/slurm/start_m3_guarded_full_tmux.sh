#!/usr/bin/env bash
# Own the guarded-full srun controller in detached tmux.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
cd "$REPO_ROOT"

DRY_RUN="${DRY_RUN:-0}"
RUN_ID="${RUN_ID:-$(date -u +%Y%m%dT%H%M%SZ)-m3-guarded-full}"
SESSION_NAME="${SESSION_NAME:-m3-guarded-${RUN_ID}}"
LOG_ROOT="${LOG_ROOT:-/mnt/nfs/hoangduy/logs/m3-guarded-full/$RUN_ID}"
RESULT_ROOT="${RESULT_ROOT:-/mnt/nfs/hoangduy/results/m3-guarded-full/$RUN_ID}"
CONTROLLER_LOG="${CONTROLLER_LOG:-$LOG_ROOT/controller.log}"
CONTROLLER_SCRIPT="$RESULT_ROOT/controller.sh"

if [[ "$DRY_RUN" == 1 || "$DRY_RUN" == true ]]; then
  echo "RUN_ID=$RUN_ID"
  echo "SESSION_NAME=$SESSION_NAME"
  echo "controller executes: $SCRIPT_DIR/run_m3_guarded_full_srun.sh"
  DRY_RUN=1 RUN_ID="$RUN_ID" LOG_ROOT="$LOG_ROOT" RESULT_ROOT="$RESULT_ROOT" \
    bash "$SCRIPT_DIR/run_m3_guarded_full_srun.sh"
  exit 0
fi
if [[ -n "${SLURM_JOB_ID:-}" ]]; then
  echo "refusing nested srun under SLURM_JOB_ID=$SLURM_JOB_ID" >&2
  exit 2
fi
command -v tmux >/dev/null 2>&1 || { echo "tmux is required" >&2; exit 2; }
tmux has-session -t "=$SESSION_NAME" 2>/dev/null && {
  echo "tmux session already exists: $SESSION_NAME" >&2; exit 2;
}
mkdir -p "$LOG_ROOT" "$RESULT_ROOT"
{
  echo '#!/usr/bin/env bash'
  echo 'set -uo pipefail'
  printf 'cd %q\n' "$REPO_ROOT"
  printf 'export RUN_ID=%q LOG_ROOT=%q RESULT_ROOT=%q\n' "$RUN_ID" "$LOG_ROOT" "$RESULT_ROOT"
  printf 'exec >>%q 2>&1\n' "$CONTROLLER_LOG"
  echo 'echo "controller started=$(date -Is) host=$(hostname) pid=$$"'
  printf 'rc=0; bash %q || rc=$?\n' "$SCRIPT_DIR/run_m3_guarded_full_srun.sh"
  printf 'printf "%%s\\n" "$rc" >%q\n' "$RESULT_ROOT/controller.rc.tmp"
  printf 'mv %q %q\n' "$RESULT_ROOT/controller.rc.tmp" "$RESULT_ROOT/controller.rc"
  echo 'echo "controller finished=$(date -Is) rc=$rc"'
  echo 'exit "$rc"'
} >"$CONTROLLER_SCRIPT"
chmod 700 "$CONTROLLER_SCRIPT"
tmux new-session -d -s "$SESSION_NAME" -c "$REPO_ROOT" \
  "bash $(printf '%q' "$CONTROLLER_SCRIPT")"
tmux has-session -t "=$SESSION_NAME"
echo "verified detached tmux session: $SESSION_NAME"
echo "controller log: $CONTROLLER_LOG"
echo "results: $RESULT_ROOT"
echo "The Cursor tool may exit now; do not kill the tmux server."
