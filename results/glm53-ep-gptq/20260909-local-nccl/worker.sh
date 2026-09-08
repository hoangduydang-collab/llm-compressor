#!/usr/bin/env bash
set -euo pipefail
cd /home/e/e1129930/llm-compressor
nccl_result=/home/e/e1129930/llm-compressor/results/glm53-ep-gptq/20260909-local-nccl
printf '%s\n' "$SLURM_JOB_ID" > "$nccl_result/job_id.txt"
date -u +'%Y-%m-%dT%H:%M:%SZ' > "$nccl_result/started.txt"
hostname > "$nccl_result/node.txt"
nvidia-smi -L > "$nccl_result/gpus.txt"
.venv-prequant/bin/python - <<'PY' > "$nccl_result/environment.json"
import json,torch,transformers,compressed_tensors,sys
assert torch.cuda.is_available() and torch.cuda.device_count()==2
print(json.dumps({'python':sys.version,'torch':torch.__version__, 'transformers':transformers.__version__, 'compressed_tensors':compressed_tensors.__version__, 'gpus':[torch.cuda.get_device_name(i) for i in range(2)]},indent=2))
PY
set +e
env -u SLURM_STEP_ID OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 .venv-prequant/bin/python -m pytest -q -rs -o tmp_path_retention_policy=all --basetemp="/tmp/pytest-of-e1129930/glm53-ep-nccl-${SLURM_JOB_ID}" tests/llmcompressor/modifiers/gptq/test_expert_parallel_nccl.py > "$nccl_result/pytest.log" 2>&1
nccl_status=$?
printf '%s\n' "$nccl_status" > "$nccl_result/pytest_exit_code.txt"
date -u +'%Y-%m-%dT%H:%M:%SZ' > "$nccl_result/ended.txt"
exit "$nccl_status"
