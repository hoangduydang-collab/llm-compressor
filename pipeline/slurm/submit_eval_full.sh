#!/usr/bin/env bash
# Submit full static eval (8 lm-eval tasks, agentic OFF) on quantized Qwen3 checkpoints.
#
# Usage (from repo root on the cluster):
#   bash pipeline/slurm/submit_eval_full.sh
#
# Options (env vars):
#   METHODS="gptq awq"     newest checkpoint per method (default)
#   SCHEME=W4AFP8          quantization scheme slug (default W4AFP8)
#   METHODS=gptq           single method only
#   CKPTS="artifacts/.../checkpoint ..."   explicit checkpoint dirs (overrides METHODS)
#   CONFIG=pipeline/configs/qwen3_30b_a3b.yaml
#   SERVE_TP=1
#   AGENTIC=1 AGENT_BASE=http://...   optional tau2 agentic (off by default)
#
# Examples:
#   bash pipeline/slurm/submit_eval_full.sh
#   METHODS=gptq bash pipeline/slurm/submit_eval_full.sh
#   SCHEME=W4A8 METHODS="gptq awq" bash pipeline/slurm/submit_eval_full.sh

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
cd "$REPO_ROOT"

export CONFIG="${CONFIG:-pipeline/configs/qwen3_30b_a3b.yaml}"
export METHODS="${METHODS:-gptq awq}"
export SCHEME="${SCHEME:-W4AFP8}"
export SERVE_TP="${SERVE_TP:-1}"
export AGENTIC="${AGENTIC:-0}"

# Export only job knobs — NOT --export=ALL (bloated SSH env can break sbatch on NFS).
EXPORT_VARS=(NONE CONFIG METHODS SCHEME SERVE_TP AGENTIC)
[[ -n "${CKPTS:-}" ]] && EXPORT_VARS+=(CKPTS)
[[ -n "${AGENT_BASE:-}" ]] && EXPORT_VARS+=(AGENT_BASE)
[[ -n "${AGENT_MODEL:-}" ]] && EXPORT_VARS+=(AGENT_MODEL)
EXPORT_LIST=$(IFS=,; echo "${EXPORT_VARS[*]}")

echo "Submitting full static eval (agentic=${AGENTIC})"
echo "  config:  $CONFIG"
echo "  scheme:  $SCHEME"
if [[ -n "${CKPTS:-}" ]]; then
  echo "  ckpts:   $CKPTS"
else
  echo "  methods: $METHODS (newest per slug)"
fi
echo "  serve_tp: $SERVE_TP"
echo "  tasks:   wikitext mmlu arc_challenge hellaswag winogrande gsm8k truthfulqa_mc2 bbh"

JOB_LINE=$(sbatch --export="$EXPORT_LIST" pipeline/slurm/eval_full.sbatch)
JOB_ID="${JOB_LINE##* }"
LOG_PATH="/mnt/nfs/hoangduy/logs/eval-full-${JOB_ID}.out"

echo ""
echo "$JOB_LINE"
echo "  tail -f $LOG_PATH"
