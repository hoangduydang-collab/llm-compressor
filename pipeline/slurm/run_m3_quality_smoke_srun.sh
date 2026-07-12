#!/usr/bin/env bash
# Run the repaired-GPTQ MiniMax-M3 quality smoke matrix under a durable controller.
set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
cd "$REPO_ROOT"

DRY_RUN="${DRY_RUN:-0}"
RUN_ROOT="${RUN_ROOT:?set RUN_ROOT from quality preflight}"
MATRIX="${MATRIX:?set MATRIX to repaired matrix}"
REPAIRED_GPTQ="${REPAIRED_GPTQ:?set REPAIRED_GPTQ}"
LOG_ROOT="${LOG_ROOT:-$RUN_ROOT/logs}"
ENV_FILE="${ENV_FILE:-/mnt/nfs/hoangduy/env.sh}"
VENV_ACTIVATE="${VENV_ACTIVATE:-/mnt/nfs/hoangduy/venvs/quant/bin/activate}"
AWQ="${AWQ:-/mnt/nfs/hoangduy/hf_assets/cyankiwi/MiniMax-M3-AWQ-INT4}"
BF16="${BF16:-/mnt/nfs/hoangduy/hf_assets/MiniMaxAI/MiniMax-M3}"
QUALITY_ARM_FILTER="${QUALITY_ARM_FILTER:-}"
case "$QUALITY_ARM_FILTER" in
  ""|gptq|awq|bf16) ;;
  *) echo "unknown QUALITY_ARM_FILTER=$QUALITY_ARM_FILTER" >&2; exit 2 ;;
esac

if [[ "$DRY_RUN" == 1 || "$DRY_RUN" == true ]]; then
  TASKS=resolved-smoke-tasks
else
  if [[ -n "${SLURM_JOB_ID:-}" ]]; then
    echo "refusing nested srun under SLURM_JOB_ID=$SLURM_JOB_ID; launch from outside any Slurm allocation" >&2
    exit 2
  fi
  if ! source "$ENV_FILE"; then
    echo "failed to source ENV_FILE=$ENV_FILE" >&2
    exit 2
  fi
  if ! source "$VENV_ACTIVATE"; then
    echo "failed to source VENV_ACTIVATE=$VENV_ACTIVATE" >&2
    exit 2
  fi
  export PYTHONPATH="$REPO_ROOT${PYTHONPATH:+:$PYTHONPATH}"
  mkdir -p "$LOG_ROOT"
  TASKS=$(python - "$RUN_ROOT/preflight/resolved_tasks.json" <<'PYTASKS'
import json, sys
a = json.load(open(sys.argv[1]))["aliases"]
print(",".join(a[x] for x in ("gpqa_diamond", "ifeval", "aime_2025", "mmlu_pro", "gsm8k")))
PYTASKS
  )
fi

gptq=(srun --exclusive --nodes=1 --ntasks=1 --gpus-per-node=8 --kill-on-bad-exit=1
  pipeline/slurm/test_m3_quality_eval_arm.sh --profile smoke --run-root "$RUN_ROOT"
  --matrix "$MATRIX" --model-label inhouse_gptq --model "$REPAIRED_GPTQ"
  --shard smoke --tasks "$TASKS" --tensor-parallel-size 8
  --distributed-executor-backend mp --run-probe 1 --probe-tokens 2048)
awq=(srun --exclusive --nodes=1 --ntasks=1 --gpus-per-node=8 --kill-on-bad-exit=1
  pipeline/slurm/test_m3_quality_eval_arm.sh --profile smoke --run-root "$RUN_ROOT"
  --matrix "$MATRIX" --model-label cyankiwi_awq --model "$AWQ"
  --shard smoke --tasks "$TASKS" --tensor-parallel-size 8
  --distributed-executor-backend mp --run-probe 1 --probe-tokens 2048)
bf16=(timeout --signal=TERM --kill-after=60s 45m
  srun --exclusive --nodes=1 --ntasks=1 --gpus-per-node=8 --kill-on-bad-exit=1
  pipeline/slurm/test_m3_quality_eval_arm.sh --profile smoke --run-root "$RUN_ROOT"
  --matrix "$MATRIX" --model-label bf16 --model "$BF16" --shard smoke
  --tasks "$TASKS" --tensor-parallel-size 8 --distributed-executor-backend mp
  --run-probe 1 --probe-tokens 2048)

selected=()
for name in gptq awq bf16; do
  [[ -z "$QUALITY_ARM_FILTER" || "$name" == "$QUALITY_ARM_FILTER" ]] || continue
  selected+=("$name")
done

if [[ "$DRY_RUN" == 1 || "$DRY_RUN" == true ]]; then
  for name in "${selected[@]}"; do
    declare -n command="$name"
    printf '%q ' "${command[@]}"
    printf '> %q 2> %q\n' "$LOG_ROOT/$name-smoke.out" "$LOG_ROOT/$name-smoke.err"
  done
  exit 0
fi

pids=()
for name in "${selected[@]}"; do
  declare -n command="$name"
  "${command[@]}" >"$LOG_ROOT/$name-smoke.out" 2>"$LOG_ROOT/$name-smoke.err" &
  pids+=("$!")
  echo "$name pid=$!"
done

overall=0
: >"$RUN_ROOT/executor_return_codes.txt"
for index in "${!pids[@]}"; do
  rc=0
  wait "${pids[$index]}" || rc=$?
  name="${selected[$index]}"
  printf '%s=%s\n' "$name" "$rc" | tee -a "$RUN_ROOT/executor_return_codes.txt"
  [[ "$rc" -eq 0 ]] || overall=1
done
exit "$overall"
