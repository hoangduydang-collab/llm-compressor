#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/../.." && pwd)"
VENV_ACTIVATE="/mnt/nfs/hoangduy/venvs/quant/bin/activate"

if [[ ! -f "${VENV_ACTIVATE}" ]]; then
  echo "missing quant environment: ${VENV_ACTIVATE}" >&2
  exit 2
fi
source "${VENV_ACTIVATE}"

# Use a specialization-owned namespace. Humming keys JIT artifacts by source
# metadata, so a fresh namespace is sufficient and no recursive deletion is
# needed or allowed here.
export HUMMING_CACHE_DIR="${HUMMING_NVFP4_W4A8_CACHE_DIR:-/mnt/nfs/hoangduy/.humming/cache-nvfp4-w4a8-v1}"

HUMMING_VERSION="$(python -c 'import importlib.metadata; print(importlib.metadata.version("humming-kernels"))')"
if [[ "${HUMMING_VERSION}" != "0.1.10" ]]; then
  echo "humming-kernels==0.1.10 is required; found ${HUMMING_VERSION}" >&2
  exit 2
fi

python "${REPO_ROOT}/pipeline/slurm/patch_humming_nvfp4_w4a8.py"

HUMMING_JIT_ROOT="$(python -c 'from pathlib import Path; from humming.utils.jit import get_humming_cache_dir; print(Path(get_humming_cache_dir()).resolve())')"
EXPECTED_JIT_ROOT="$(python -c 'import os; from pathlib import Path; print(Path(os.environ["HUMMING_CACHE_DIR"]).resolve())')"
if [[ "${HUMMING_JIT_ROOT}" != "${EXPECTED_JIT_ROOT}" || "$(basename -- "${HUMMING_JIT_ROOT}")" != "cache-nvfp4-w4a8-v1" ]]; then
  echo "Humming did not honor the dedicated cache root: ${HUMMING_JIT_ROOT}" >&2
  exit 2
fi

python "${REPO_ROOT}/pipeline/slurm/patch_humming_nvfp4_w4a8.py" --check

echo "Humming NVFP4 W4A8 g16 overlay installed; qualification is still required."
echo "Separate Marlin fallback (new process): vllm serve <MODEL_PATH> --quantization compressed-tensors"
