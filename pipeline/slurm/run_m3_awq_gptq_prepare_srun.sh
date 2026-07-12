#!/usr/bin/env bash
# Prepare GPTQ portable and two fresh AWQ variants concurrently.

set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
cd "$REPO_ROOT"

DRY_RUN="${DRY_RUN:-0}"
LOG_ROOT="${LOG_ROOT:-/mnt/nfs/hoangduy/logs/m3-awq-gptq-prepare}"
TIME_LIMIT="${TIME_LIMIT:-96:00:00}"
ENV_FILE="${ENV_FILE:-/mnt/nfs/hoangduy/env.sh}"
VENV_ACTIVATE="${VENV_ACTIVATE:-/mnt/nfs/hoangduy/venvs/quant/bin/activate}"
# shellcheck disable=SC2206
EXTRA_SRUN_ARGS=(${SRUN_ARGS:-})

commands=(
  "env VARIANT=offsetfix bash $SCRIPT_DIR/test_m3_awq_variant_quantize.sh"
  "env VARIANT=nosmooth bash $SCRIPT_DIR/test_m3_awq_variant_quantize.sh"
  "bash $SCRIPT_DIR/test_m3_gptq_prepare.sh"
)
names=(awq-offsetfix awq-nosmooth gptq-portable)
pids=()
if [[ "$DRY_RUN" != 1 && "$DRY_RUN" != true ]]; then
  source "$ENV_FILE"
  source "$VENV_ACTIVATE"
  export PYTHONPATH="$REPO_ROOT/src:$REPO_ROOT"
  python -m pytest -q \
    pipeline/tests/test_minimax_m3_config.py \
    tests/llmcompressor/modeling/test_offset_norm_minimax_m3.py \
    pipeline/tests/test_m3_checkpoint_scale_audit.py \
    pipeline/tests/test_m3_awq_gptq_repair.py \
    pipeline/tests/test_m3_awq_gptq_repair_runner.py
  mkdir -p "$LOG_ROOT"
fi
for index in "${!commands[@]}"; do
  command=(srun --exclusive --nodes=1 --ntasks=1 --gres=gpu:1
    --time="$TIME_LIMIT" --kill-on-bad-exit=0 "${EXTRA_SRUN_ARGS[@]}"
    bash -lc "${commands[$index]}")
  if [[ "$DRY_RUN" == 1 || "$DRY_RUN" == true ]]; then
    printf '%q ' "${command[@]}"; printf '\n'
  else
    "${command[@]}" >"$LOG_ROOT/${names[$index]}.log" 2>&1 &
    pids+=("$!")
    echo "${names[$index]} pid=$!"
  fi
done
if [[ "$DRY_RUN" == 1 || "$DRY_RUN" == true ]]; then exit 0; fi

overall=0
for index in "${!pids[@]}"; do
  rc=0; wait "${pids[$index]}" || rc=$?
  echo "${names[$index]} rc=$rc log=$LOG_ROOT/${names[$index]}.log"
  [[ "$rc" -eq 0 ]] || overall=1
done
exit "$overall"
