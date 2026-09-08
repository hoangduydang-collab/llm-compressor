#!/usr/bin/env bash
set -euo pipefail
cd /home/e/e1129930/llm-compressor
nccl_result=$PWD/results/glm53-ep-gptq/20260909-local-nccl-retry2
trap 'nccl_status=$?; printf "%s\n" "$nccl_status" > "$nccl_result/worker_exit_code.txt"; date -u +"%Y-%m-%dT%H:%M:%SZ" > "$nccl_result/ended.txt"' EXIT
printf '%s\n' "$SLURM_JOB_ID" > "$nccl_result/job_id.txt"
date -u +'%Y-%m-%dT%H:%M:%SZ' > "$nccl_result/started.txt"
hostname > "$nccl_result/node.txt"
nvidia-smi -L > "$nccl_result/gpu-list.txt"
nvidia-smi > "$nccl_result/nvidia-smi.txt"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NCCL_DEBUG=INFO PYTHONUNBUFFERED=1
.venv-prequant/bin/python - <<'ENV' > "$nccl_result/environment.json"
import json,torch,transformers,compressed_tensors,sys
print(json.dumps({'python':sys.version,'torch':torch.__version__, 'transformers':transformers.__version__, 'compressed_tensors':compressed_tensors.__version__, 'gpus':[torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())], 'cuda_available':torch.cuda.is_available(), 'cuda_device_count':torch.cuda.device_count(), 'compiled_arches':torch.cuda.get_arch_list()},indent=2), flush=True)
assert torch.cuda.is_available() and torch.cuda.device_count()==2
assert all('MIG' not in torch.cuda.get_device_name(i) for i in range(2))
ENV
set +e
env -u SLURM_STEP_ID .venv-prequant/bin/python -m pytest -vv -ra -o tmp_path_retention_policy=all --basetemp="/tmp/pytest-of-e1129930/glm53-ep-nccl-${SLURM_JOB_ID}" tests/llmcompressor/modifiers/gptq/test_expert_parallel_nccl.py > "$nccl_result/pytest.log" 2>&1
nccl_status=$?
printf '%s\n' "$nccl_status" > "$nccl_result/pytest_exit_code.txt"
.venv-prequant/bin/python - <<'ARTIFACTS'
import os, shutil
from pathlib import Path
src=Path('/tmp/pytest-of-e1129930')/f"glm53-ep-nccl-{os.environ['SLURM_JOB_ID']}"
dst=Path('results/glm53-ep-gptq/20260909-local-nccl-retry2/artifacts')
for pattern in ('rank-*-phases.txt','rank-*-packed-differences.json'):
 for path in src.rglob(pattern):
  target=dst/path.relative_to(src);target.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(path,target)
ARTIFACTS
exit "$nccl_status"
