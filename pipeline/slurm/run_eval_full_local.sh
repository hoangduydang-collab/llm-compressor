#!/usr/bin/env bash
# Run full static eval in the CURRENT shell (no sbatch). Use when Slurm submission fails.
#
#   METHODS=gptq bash pipeline/slurm/run_eval_full_local.sh
#
# Needs an idle GPU on the node you're on (salloc / interactive session).

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
cd "$REPO_ROOT"

source /mnt/nfs/hoangduy/env.sh
source /mnt/nfs/hoangduy/venvs/quant/bin/activate
export HOME=${WORK_ROOT:-/mnt/nfs/hoangduy}
export FLASHINFER_WORKSPACE_DIR=${FLASHINFER_WORKSPACE_DIR:-$HOME/cache/flashinfer}
mkdir -p "$FLASHINFER_WORKSPACE_DIR" 2>/dev/null || true

export CONFIG="${CONFIG:-pipeline/configs/qwen3_30b_a3b.yaml}"
export METHODS="${METHODS:-gptq awq}"
export SCHEME="${SCHEME:-W4AFP8}"
export SERVE_TP="${SERVE_TP:-1}"
export AGENTIC="${AGENTIC:-0}"

source pipeline/slurm/_stage1_run.sh

echo "host=$(hostname) local full static eval (no sbatch)"
nvidia-smi --query-gpu=index,name,memory.total --format=csv || true

run_eval() {
  local CKPT=$1
  if [[ -z "${CKPT:-}" || ! -d "$CKPT" ]]; then
    echo "[FAIL] checkpoint dir not found: '${CKPT:-}'"
    return 1
  fi
  echo "================================================================"
  echo "  FULL STATIC EVAL: $CKPT"
  echo "================================================================"
  local -a EVAL_EXTRA=()
  [[ "${AGENTIC:-0}" == "1" ]] && EVAL_EXTRA+=(--agentic)
  [[ -n "${AGENT_BASE:-}" ]] && EVAL_EXTRA+=(--agent-base "$AGENT_BASE")
  [[ -n "${AGENT_MODEL:-}" ]] && EVAL_EXTRA+=(--agent-model "$AGENT_MODEL")
  python -m pipeline.run --config "$CONFIG" --stage eval --checkpoint "$CKPT" \
    --set serve.tensor_parallel_size="$SERVE_TP" \
    --set eval.log_samples=true \
    "${EVAL_EXTRA[@]}"
}

overall_rc=0
if [[ -n "${CKPTS:-}" ]]; then
  for c in $CKPTS; do run_eval "$c" || overall_rc=1; done
else
  for m in $METHODS; do
    CKPT=$(latest_ckpt "$m" "$SCHEME")
    if [[ -z "${CKPT:-}" ]]; then
      echo "[FAIL] no checkpoint for method=$m scheme=$SCHEME"
      overall_rc=1
      continue
    fi
    run_eval "$CKPT" || overall_rc=1
  done
fi

echo ""
echo "FULL STATIC EVAL COMPLETE (overall_rc=$overall_rc)"
exit $overall_rc
