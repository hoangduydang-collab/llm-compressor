#!/usr/bin/env bash
# Install the team vLLM build for MiniMax-M3 compressed-tensors serve.
#
# Our W4AFP8 checkpoint keeps the MSA indexer in bf16 while quantizing q/k/v, and
# saves per-expert linearized MoE weights (block_sparse_moe.experts.N.*). The
# toncao branch handles that compressed-tensors layout:
#   https://github.com/toncao/vllm/tree/minimax-m3-compressed-tensors
#
# MoE w13 uninterleaved layout + SWIGLUOAI_UNINTERLEAVE is a separate gap — fixed
# by patch_vllm_m3_serve.py patches 1–2 (not by switching branches).
#
# Uses venvs/quant (NOT sglang-eval). Re-run serve after install.
#
#   bash pipeline/slurm/install_vllm_m3_serve.sh
#
# Env:
#   VLLM_M3_REF  default: minimax-m3-compressed-tensors

set -euo pipefail

VLLM_M3_REF="${VLLM_M3_REF:-minimax-m3-compressed-tensors}"

source /mnt/nfs/hoangduy/env.sh
source /mnt/nfs/hoangduy/venvs/quant/bin/activate

echo "Installing vLLM from toncao/vllm@${VLLM_M3_REF} into quant venv"
echo "  python: $(which python)"
echo "  before: $(python -c 'import vllm; print(vllm.__version__)' 2>/dev/null || echo 'not installed')"

# Keep existing torch in quant venv when possible; vLLM is pure Python + extensions
# that compile against the installed torch.
"$UV" pip install --force-reinstall \
  "vllm @ git+https://github.com/toncao/vllm.git@${VLLM_M3_REF}"

echo "  after:  $(python -c 'import vllm; print(vllm.__version__)')"

# Re-apply the persistent W4A8 MoE + SwiGLU-OAI (uninterleave) source patch, which a
# fresh install overwrites. Idempotent; see BUGS_AND_FIXES.md for the root cause.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
python "${SCRIPT_DIR}/patch_vllm_m3_serve.py" || {
  echo "WARNING: W4A8 SwiGLU-OAI patch did not apply; M3 W4A8 serve will fail until fixed." >&2
}

echo "Done. Re-run: bash pipeline/slurm/run_serve_minimax_m3_detached.sh"
