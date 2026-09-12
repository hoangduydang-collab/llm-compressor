#!/usr/bin/env bash
set -euo pipefail
cd /home/e/e1129930/llm-compressor
nccl_repo=$PWD
nccl_result=$nccl_repo/results/glm53-ep-gptq/20260912-fp8-before-gptq-h100
nccl_python=$nccl_repo/.venv-prequant/bin/python
nccl_revision=d90ea79c49b4e62d314cc8d9eba190fdfba0ddf4
trap 'nccl_status=$?; printf "%s\n" "$nccl_status" > "$nccl_result/worker_exit_code.txt"; date -u +"%Y-%m-%dT%H:%M:%SZ" > "$nccl_result/ended.txt"' EXIT
printf '%s\n' "$SLURM_JOB_ID" > "$nccl_result/job_id.txt"
date -u +'%Y-%m-%dT%H:%M:%SZ' > "$nccl_result/started.txt"
hostname > "$nccl_result/node.txt"
printf '%s\n' "$nccl_revision" > "$nccl_result/code_revision.txt"
nccl_scratch=/tmp/pytest-of-e1129930/glm53-fp8-h100-${SLURM_JOB_ID}
mkdir -p "$nccl_scratch/code"
git -C "$nccl_repo" archive "$nccl_revision" src pipeline tests pyproject.toml | tar -x -C "$nccl_scratch/code"
export PYTHONPATH="$nccl_scratch/code/src:$nccl_scratch/code"
cd "$nccl_scratch/code"
nvidia-smi -L > "$nccl_result/gpu-list.txt"
nvidia-smi > "$nccl_result/nvidia-smi.txt"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NCCL_DEBUG=INFO PYTHONUNBUFFERED=1
"$nccl_python" - <<'ENV' > "$nccl_result/environment.json"
import json,torch,transformers,compressed_tensors,sys,os
from pathlib import Path
import llmcompressor,pipeline,tests
print(json.dumps({'python':sys.version,'torch':torch.__version__, 'transformers':transformers.__version__, 'compressed_tensors':compressed_tensors.__version__, 'gpus':[torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())], 'cuda_available':torch.cuda.is_available(), 'cuda_device_count':torch.cuda.device_count(), 'compiled_arches':torch.cuda.get_arch_list(), 'llmcompressor_path':llmcompressor.__file__, 'pipeline_path':pipeline.__file__, 'tests_path':tests.__file__, 'capabilities':[torch.cuda.get_device_capability(i) for i in range(torch.cuda.device_count())]},indent=2), flush=True)
assert torch.cuda.is_available() and torch.cuda.device_count()==2
assert all('H100' in torch.cuda.get_device_name(i) and 'MIG' not in torch.cuda.get_device_name(i) for i in range(2))
assert all(torch.cuda.get_device_capability(i)==(9,0) for i in range(2))
assert torch.__version__=='2.11.0+cu128' and transformers.__version__=='5.12.1'
assert compressed_tensors.__version__=='0.17.2.a20260707'
assert all(Path(module.__file__).resolve().is_relative_to(Path.cwd()) for module in (llmcompressor,pipeline,tests))
ENV
mkdir -p /tmp/pytest-of-e1129930
set +e
env -u SLURM_STEP_ID "$nccl_python" -m pytest -vv -rA -o tmp_path_retention_policy=all --basetemp="/tmp/pytest-of-e1129930/glm53-fp8-nccl-${SLURM_JOB_ID}" --junitxml="$nccl_result/pytest.xml" tests/llmcompressor/modifiers/gptq/test_expert_parallel_nccl.py > "$nccl_result/pytest.log" 2>&1
nccl_status=$?
printf '%s\n' "$nccl_status" > "$nccl_result/pytest_exit_code.txt"
"$nccl_python" - <<'ARTIFACTS'
import os, shutil
from pathlib import Path
src=Path('/tmp/pytest-of-e1129930')/f"glm53-fp8-nccl-{os.environ['SLURM_JOB_ID']}"
dst=Path('/home/e/e1129930/llm-compressor/results/glm53-ep-gptq/20260912-fp8-before-gptq-h100/artifacts')
for pattern in ('rank-*-phases.txt','rank-*-packed-differences.json','rank-*-fp8-parity.json','rank-*-activation-logits.json'):
 for path in src.rglob(pattern):
  target=dst/path.relative_to(src);target.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(path,target)
ARTIFACTS
if test "$nccl_status" -eq 0; then
 "$nccl_python" - "$nccl_result/pytest.xml" <<'GATE'
import sys, xml.etree.ElementTree as ET
cases = ET.parse(sys.argv[1]).getroot().findall('.//testcase')
assert len(cases) == 5, f"expected five GPU cases, got {len(cases)}"
assert all(not any(c.find(t) is not None for t in ('skipped', 'failure', 'error')) for c in cases)
GATE
fi
exit "$nccl_status"
