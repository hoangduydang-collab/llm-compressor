set -uo pipefail
source /mnt/nfs/hoangduy/venvs/quant/bin/activate
export LD_LIBRARY_PATH="/mnt/nfs/hoangduy/venvs/quant/lib/python3.12/site-packages/nvidia/cu13/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export PYTHONPATH="/mnt/nfs/hoangduy/venvs/humming-0.1.10-site:/mnt/nfs/hoangduy/projects/llm-compressor"
export HOME=/mnt/nfs/hoangduy
python -m pipeline.m3_humming_gemm_type_probe --out "$1"
