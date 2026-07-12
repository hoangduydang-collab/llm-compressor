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
ray=(srun --exclusive --nodes=2 --ntasks=2 --gpus-per-node=8 --kill-on-bad-exit=1
  pipeline/slurm/test_m3_ray_placement_group.sh --out "$RUN_ROOT/ray_placement"
  --expected-bundles 16 --timeout-seconds 120)
bf16=(timeout --signal=TERM --kill-after=60s 10m
  srun --exclusive --nodes=2 --ntasks=2 --gpus-per-node=8 --kill-on-bad-exit=1
  pipeline/slurm/test_m3_quality_eval_arm.sh --profile smoke --run-root "$RUN_ROOT"
  --matrix "$MATRIX" --model-label bf16 --model "$BF16" --shard smoke
  --tasks "$TASKS" --tensor-parallel-size 16 --distributed-executor-backend ray
  --run-probe 1 --probe-tokens 2048)

if [[ "$DRY_RUN" == 1 || "$DRY_RUN" == true ]]; then
  for name in gptq awq ray bf16; do
    declare -n command="$name"
    printf '%q ' "${command[@]}"
    printf '> %q 2> %q\n' "$LOG_ROOT/$name-smoke.out" "$LOG_ROOT/$name-smoke.err"
  done
  exit 0
fi

"${gptq[@]}" >"$LOG_ROOT/gptq-smoke.out" 2>"$LOG_ROOT/gptq-smoke.err" & GPTQ_PID=$!
"${awq[@]}" >"$LOG_ROOT/awq-smoke.out" 2>"$LOG_ROOT/awq-smoke.err" & AWQ_PID=$!
"${ray[@]}" >"$LOG_ROOT/ray-placement.out" 2>"$LOG_ROOT/ray-placement.err" & RAY_PID=$!

RAY_RC=0; wait "$RAY_PID" || RAY_RC=$?
BF16_RC=0; "${bf16[@]}" >"$LOG_ROOT/bf16-smoke.out" 2>"$LOG_ROOT/bf16-smoke.err" || BF16_RC=$?
GPTQ_RC=0; wait "$GPTQ_PID" || GPTQ_RC=$?
AWQ_RC=0; wait "$AWQ_PID" || AWQ_RC=$?
printf 'ray=%s\nbf16=%s\ngptq=%s\nawq=%s\n' "$RAY_RC" "$BF16_RC" "$GPTQ_RC" "$AWQ_RC" | tee "$RUN_ROOT/executor_return_codes.txt"
[[ "$RAY_RC" -eq 0 && "$BF16_RC" -eq 0 && "$GPTQ_RC" -eq 0 && "$AWQ_RC" -eq 0 ]]
