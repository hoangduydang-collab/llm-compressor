#!/usr/bin/env bash
# Rerun only the checkpoint-scale audit for an existing staged matrix.

set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
cd "$REPO_ROOT"

DRY_RUN="${DRY_RUN:-0}"
MATRIX_ID="${MATRIX_ID:?set the staged MATRIX_ID}"
EVIDENCE_ROOT="${EVIDENCE_ROOT:-$REPO_ROOT/results/m3-awq-gptq-repair}"
LOG_ROOT="${LOG_ROOT:-/mnt/nfs/hoangduy/logs/m3-awq-gptq-repair-srun}"
TIME_LIMIT="${TIME_LIMIT:-04:00:00}"
DIAGNOSTIC_LAYERS="${M3_DIAGNOSTIC_LAYERS:-$(seq -s, 3 59)}"
# shellcheck disable=SC2206
EXTRA_SRUN_ARGS=(${SRUN_ARGS:-})

matrix_dir="$EVIDENCE_ROOT/$MATRIX_ID"
log="$LOG_ROOT/$MATRIX_ID-checkpoint-scale-audit-rerun.log"
command=(srun --exclusive --nodes=1 --ntasks=1 --gres=gpu:1
  --time="$TIME_LIMIT" --kill-on-bad-exit=0 "${EXTRA_SRUN_ARGS[@]}"
  env "OUTPUT=$matrix_dir/checkpoint_scale_audit.json"
  "M3_DIAGNOSTIC_LAYERS=$DIAGNOSTIC_LAYERS"
  bash "$SCRIPT_DIR/test_m3_checkpoint_scale_audit.sh")

if [[ "$DRY_RUN" == 1 || "$DRY_RUN" == true ]]; then
  printf '%q ' "${command[@]}"; printf '\n'
  exit 0
fi
[[ -f "$matrix_dir/comparison_early.json" ]] || {
  echo "ERROR: early matrix evidence missing: $matrix_dir" >&2
  exit 2
}
mkdir -p "$LOG_ROOT"
rc=0
"${command[@]}" >"$log" 2>&1 || rc=$?
cp "$log" "$matrix_dir/checkpoint_scale_audit.log"
echo "$rc" >"$matrix_dir/checkpoint_scale_audit.return_code.txt"
python -m pipeline.m3_awq_gptq_repair aggregate \
  --phase early --evidence-root "$matrix_dir"
echo "checkpoint_scale_audit srun_rc=$rc log=$log"
exit "$rc"
