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

CONFIG="${CONFIG:-pipeline/configs/qwen3_30b_a3b.yaml}"
METHODS="${METHODS:-gptq awq}"
SCHEME="${SCHEME:-W4AFP8}"
SERVE_TP="${SERVE_TP:-1}"
AGENTIC="${AGENTIC:-0}"
CKPTS="${CKPTS:-}"
AGENT_BASE="${AGENT_BASE:-}"
AGENT_MODEL="${AGENT_MODEL:-}"

echo "Submitting full static eval (agentic=${AGENTIC})"
echo "  config:  $CONFIG"
echo "  scheme:  $SCHEME"
if [[ -n "$CKPTS" ]]; then
  echo "  ckpts:   $CKPTS"
else
  echo "  methods: $METHODS (newest per slug)"
fi
echo "  serve_tp: $SERVE_TP"
echo "  tasks:   wikitext mmlu arc_challenge hellaswag winogrande gsm8k truthfulqa_mc2 bbh"

# Stage on local disk: some Slurm controllers fail to spool batch scripts from NFS
# ("I/O error writing script/environment to file").
TMP_JOB="$(mktemp /tmp/eval_full_${USER:-user}_XXXXXX.sbatch)"
cp pipeline/slurm/eval_full.sbatch "$TMP_JOB"

# Inject job knobs directly — no sbatch --export needed.
inject=$'export CONFIG='"$(printf '%q' "$CONFIG")"$'\n'
inject+=$'export METHODS='"$(printf '%q' "$METHODS")"$'\n'
inject+=$'export SCHEME='"$(printf '%q' "$SCHEME")"$'\n'
inject+=$'export SERVE_TP='"$(printf '%q' "$SERVE_TP")"$'\n'
inject+=$'export AGENTIC='"$(printf '%q' "$AGENTIC")"$'\n'
[[ -n "$CKPTS" ]] && inject+=$'export CKPTS='"$(printf '%q' "$CKPTS")"$'\n'
[[ -n "$AGENT_BASE" ]] && inject+=$'export AGENT_BASE='"$(printf '%q' "$AGENT_BASE")"$'\n'
[[ -n "$AGENT_MODEL" ]] && inject+=$'export AGENT_MODEL='"$(printf '%q' "$AGENT_MODEL")"$'\n'

awk -v inject="$inject" '
  /^set -uo pipefail/ { print; print inject; next }
  { print }
' "$TMP_JOB" > "${TMP_JOB}.new"
mv "${TMP_JOB}.new" "$TMP_JOB"
chmod +x "$TMP_JOB"

submit_job() {
  # Clean submission environment; job script carries all knobs.
  env -i \
    HOME="${HOME:-/mnt/nfs/hoangduy}" \
    USER="${USER:-hoangduy}" \
    PATH="${PATH:-/usr/bin:/bin}" \
    LANG="${LANG:-C.UTF-8}" \
    sbatch --export=NONE "$TMP_JOB"
}

if ! JOB_LINE="$(submit_job 2>&1)"; then
  echo ""
  echo "sbatch failed:"
  echo "$JOB_LINE"
  echo ""
  echo "Cluster-side checks (ask admin if these look bad):"
  echo "  df -h /tmp /var/spool/slurm 2>/dev/null || true"
  echo "  sacctmgr show user $USER 2>/dev/null || true"
  echo ""
  echo "Interactive fallback (needs a free GPU on this node):"
  echo "  bash pipeline/slurm/run_eval_full_local.sh"
  exit 1
fi

JOB_ID="${JOB_LINE##* }"
LOG_PATH="/mnt/nfs/hoangduy/logs/eval-full-${JOB_ID}.out"

echo ""
echo "$JOB_LINE"
echo "  staged job script: $TMP_JOB (kept in /tmp for debugging)"
echo "  tail -f $LOG_PATH"
rm -f "$TMP_JOB" 2>/dev/null || true
