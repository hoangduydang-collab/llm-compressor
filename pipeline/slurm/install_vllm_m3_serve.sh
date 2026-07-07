#!/usr/bin/env bash
# Install the patched vLLM build for MiniMax-M3 compressed-tensors checkpoints.
#
# Our W4AFP8 checkpoint keeps the MSA indexer in bf16 while quantizing q/k/v, and
# saves per-expert linearized MoE weights (block_sparse_moe.experts.N.*). Stock
# vLLM may fail worker init without Ton Cao's branch:
#   https://github.com/toncao/vllm/tree/minimax-m3-compressed-tensors
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
echo "Done. Re-run: bash pipeline/slurm/run_serve_minimax_m3_detached.sh"
