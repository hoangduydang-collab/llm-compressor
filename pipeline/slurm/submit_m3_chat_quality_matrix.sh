#!/usr/bin/env bash
# Submit four independent canonical MiniMax-M3 chat quality arms.

set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
cd "$REPO_ROOT"

DRY_RUN="${DRY_RUN:-0}"
MATRIX_ID="${MATRIX_ID:-$(date +%Y%m%d-%H%M%S)-canonical-chat}"
LOG_ROOT="${LOG_ROOT:-/mnt/nfs/hoangduy/logs/m3-chat-quality-submit}"
TIME_LIMIT="${TIME_LIMIT:-02:00:00}"
ARMS=(
  reference_offline_chat
  candidate_offline_chat
  reference_http_chat
  candidate_http_chat
)
# Runtime-only scheduler adaptations may be supplied, for example:
# SBATCH_ARGS="--partition=h100 --nodelist=gpu-h101,gpu-h102"
# shellcheck disable=SC2206
EXTRA_SBATCH_ARGS=(${SBATCH_ARGS:-})

if [[ "$DRY_RUN" != 1 && "$DRY_RUN" != true ]]; then
  mkdir -p "$LOG_ROOT"
fi

for arm in "${ARMS[@]}"; do
  out="$LOG_ROOT/$MATRIX_ID-$arm.out"
  err="$LOG_ROOT/$MATRIX_ID-$arm.err"
  command=(
    sbatch
    --parsable
    --job-name="m3q-${arm:0:18}"
    --nodes=1
    --ntasks=1
    --gres=gpu:8
    --exclusive
    --time="$TIME_LIMIT"
    --output="$out"
    --error="$err"
    --export="ALL,MATRIX_ID=$MATRIX_ID,ARM=$arm,M3_MATRIX_STDOUT=$out,M3_MATRIX_STDERR=$err"
    "${EXTRA_SBATCH_ARGS[@]}"
    --wrap="bash $REPO_ROOT/pipeline/slurm/test_m3_chat_quality_arm.sh"
  )
  if [[ "$DRY_RUN" == 1 || "$DRY_RUN" == true ]]; then
    printf '%q ' "${command[@]}"
    printf '\n'
  else
    job_id="$("${command[@]}")"
    echo "$arm job_id=$job_id out=$out err=$err"
  fi
done

echo "matrix_id=$MATRIX_ID"
echo "aggregate after all arms: python -m pipeline.m3_chat_quality aggregate --evidence-root results/m3-chat-quality/$MATRIX_ID"
