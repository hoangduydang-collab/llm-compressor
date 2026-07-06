#!/usr/bin/env bash
# Run GLM-5.2 SGLang eval in the current shell (no sbatch).
# Use when Slurm submission fails but you have an idle 8-GPU node (salloc / interactive).
#
#   bash pipeline/slurm/run_eval_glm52_sglang_local.sh
#
# Options: CONFIG, OUT_DIR, MODEL_PATH (same as submit_eval_glm52_sglang.sh)

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
cd "$REPO_ROOT"

export CONFIG="${CONFIG:-pipeline/configs/eval_glm52_w4afp8_sglang_h100.yaml}"
export OUT_DIR="${OUT_DIR:-evals/glm52-w4afp8-phala}"
export MODEL_PATH="${MODEL_PATH:-/mnt/nfs/hoangduy/hf_assets/PhalaCloud/GLM-5.2-W4AFP8}"

echo "host=$(hostname) local GLM-5.2 SGLang eval (no sbatch)"
nvidia-smi --query-gpu=index,name,memory.total --format=csv || true

bash pipeline/slurm/eval_glm52_sglang.sbatch
