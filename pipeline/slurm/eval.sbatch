#!/bin/bash
#SBATCH --job-name=eval-gate
#SBATCH --partition=compute
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=200G
#SBATCH --time=24:00:00
#SBATCH --output=/mnt/nfs/hoangduy/logs/eval-%j.out
#
# Accuracy gate on EXISTING checkpoint(s) - eval only (serve already verified).
# For the full 8-task static suite, use pipeline/slurm/submit_eval_full.sh instead.
#
# Option A: explicit checkpoint dirs (space-separated):
#   CKPTS="artifacts/Qwen3-30B-A3B-gptq-W4AFP8/20260630-080115/checkpoint \
#          artifacts/Qwen3-30B-A3B-awq-W4AFP8/20260630-080112/checkpoint" \
#     sbatch pipeline/slurm/eval.sbatch
#
# Option B: auto-resolve newest per method:
#   METHODS="gptq awq" SCHEME=W4AFP8 sbatch pipeline/slurm/eval.sbatch
#
# Agentic tau2 eval (model must be served at AGENT_BASE first):
#   AGENTIC=1 AGENT_BASE=http://localhost:8000/v1 sbatch pipeline/slurm/eval.sbatch

set -uo pipefail
source /mnt/nfs/hoangduy/env.sh
source /mnt/nfs/hoangduy/venvs/quant/bin/activate
cd /mnt/nfs/hoangduy/projects/llm-compressor
source pipeline/slurm/_stage1_run.sh   # HOME/FlashInfer env + CONFIG + latest_ckpt()

SERVE_TP=${SERVE_TP:-1}
echo "host=$(hostname) job=${SLURM_JOB_ID:-none}"
nvidia-smi --query-gpu=index,name,memory.total --format=csv || true

run_eval() {
  local CKPT=$1
  if [[ -z "${CKPT:-}" || ! -d "$CKPT" ]]; then
    echo "[FAIL] checkpoint dir not found: '${CKPT:-}'"
    return 1
  fi
  echo "================================================================"
  echo "  EVAL: $CKPT (tp=$SERVE_TP agentic=${AGENTIC:-0})"
  echo "================================================================"
  local -a EVAL_EXTRA=()
  [[ "${AGENTIC:-0}" == "1" ]] && EVAL_EXTRA+=(--agentic)
  [[ -n "${AGENT_BASE:-}" ]] && EVAL_EXTRA+=(--agent-base "$AGENT_BASE")
  [[ -n "${AGENT_MODEL:-}" ]] && EVAL_EXTRA+=(--agent-model "$AGENT_MODEL")
  python -m pipeline.run --config "$CONFIG" --stage eval --checkpoint "$CKPT" \
    --set serve.tensor_parallel_size="$SERVE_TP" \
    "${EVAL_EXTRA[@]}"
}

overall_rc=0
if [[ -n "${CKPTS:-}" ]]; then
  for c in $CKPTS; do run_eval "$c" || overall_rc=1; done
else
  SCHEME=${SCHEME:-W4AFP8}
  METHODS=${METHODS:-"gptq awq"}
  for m in $METHODS; do run_eval "$(latest_ckpt "$m" "$SCHEME")" || overall_rc=1; done
fi

echo ""
echo "EVAL COMPLETE (overall_rc=$overall_rc)"
exit $overall_rc
