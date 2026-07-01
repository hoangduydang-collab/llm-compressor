#!/bin/bash
#SBATCH --job-name=eval-full
#SBATCH --partition=compute
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=200G
#SBATCH --time=24:00:00
#SBATCH --output=/mnt/nfs/hoangduy/logs/eval-full-%j.out
#
# Full static lm-eval suite on EXISTING quantized checkpoint(s).
# Agentic tau2 eval is OFF unless AGENTIC=1 (model must be served at AGENT_BASE).
#
# Tasks (default, full splits): wikitext, mmlu, arc_challenge, hellaswag,
# winogrande, gsm8k, truthfulqa_mc2, bbh — see full_static_tasks() in config.py.
#
# Submit via wrapper (recommended):
#   bash pipeline/slurm/submit_eval_full.sh
#
# Or sbatch directly:
#   METHODS="gptq awq" SCHEME=W4AFP8 sbatch pipeline/slurm/eval_full.sbatch
#   CKPTS="artifacts/Qwen3-30B-A3B-gptq-W4AFP8/<ts>/checkpoint" sbatch pipeline/slurm/eval_full.sbatch
#   AGENTIC=1 AGENT_BASE=http://localhost:8000/v1 sbatch pipeline/slurm/eval_full.sbatch

set -uo pipefail
source /mnt/nfs/hoangduy/env.sh
source /mnt/nfs/hoangduy/venvs/quant/bin/activate
cd /mnt/nfs/hoangduy/projects/llm-compressor
source pipeline/slurm/_stage1_run.sh   # HOME/FlashInfer env + CONFIG + latest_ckpt()

CONFIG=${CONFIG:-pipeline/configs/qwen3_30b_a3b.yaml}
SERVE_TP=${SERVE_TP:-1}
AGENTIC=${AGENTIC:-0}

echo "host=$(hostname) job=${SLURM_JOB_ID:-none}"
echo "config=$CONFIG scheme=${SCHEME:-W4AFP8} methods=${METHODS:-gptq awq} serve_tp=$SERVE_TP agentic=$AGENTIC"
echo "static tasks: wikitext mmlu arc_challenge hellaswag winogrande gsm8k truthfulqa_mc2 bbh (full splits)"
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
  [[ "$AGENTIC" == "1" ]] && EVAL_EXTRA+=(--agentic)
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
  SCHEME=${SCHEME:-W4AFP8}
  METHODS=${METHODS:-"gptq awq"}
  for m in $METHODS; do
    CKPT=$(latest_ckpt "$m" "$SCHEME")
    if [[ -z "${CKPT:-}" ]]; then
      echo "[FAIL] no checkpoint found for method=$m scheme=$SCHEME"
      overall_rc=1
      continue
    fi
    run_eval "$CKPT" || overall_rc=1
  done
fi

echo ""
echo "FULL STATIC EVAL COMPLETE (overall_rc=$overall_rc)"
echo "Results: artifacts/Qwen3-30B-A3B-*/<ts>/evalsuite/ + eval_report.json"
exit $overall_rc
