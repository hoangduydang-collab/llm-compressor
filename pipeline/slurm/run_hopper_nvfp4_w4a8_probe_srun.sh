#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/../.." && pwd)"
VENV_ACTIVATE="/mnt/nfs/hoangduy/venvs/quant/bin/activate"
RESULT_BASE="${HOPPER_NVFP4_W4A8_RESULT_BASE:-/mnt/nfs/hoangduy/hopper_nvfp4_w4a8_probe}"
RUN_ID="$(date -u +%Y%m%dT%H%M%SZ)-${RANDOM}-$$"
RESULT_ROOT="${RESULT_BASE}/${RUN_ID}"

mkdir "${RESULT_ROOT}"
git -C "${REPO_ROOT}" rev-parse HEAD > "${RESULT_ROOT}/git-revision.txt"
git -C "${REPO_ROOT}" status --short > "${RESULT_ROOT}/git-status.txt"

source "${VENV_ACTIVATE}"
export HUMMING_CACHE_DIR="${HUMMING_NVFP4_W4A8_CACHE_DIR:-/mnt/nfs/hoangduy/.humming/cache-nvfp4-w4a8-v1}"
python "${REPO_ROOT}/pipeline/slurm/patch_humming_nvfp4_w4a8.py" \
  --check --json "${RESULT_ROOT}/patch-report.json"
python -c 'import importlib.metadata, torch; print("torch=" + torch.__version__); print("humming-kernels=" + importlib.metadata.version("humming-kernels"))' \
  > "${RESULT_ROOT}/python-environment.txt"
nvidia-smi -q > "${RESULT_ROOT}/nvidia-smi.txt" 2>&1 || true

cat > "${RESULT_ROOT}/worker.sh" <<EOF
#!/usr/bin/env bash
set -euo pipefail
source "${VENV_ACTIVATE}"
export HUMMING_CACHE_DIR="${HUMMING_NVFP4_W4A8_CACHE_DIR:-/mnt/nfs/hoangduy/.humming/cache-nvfp4-w4a8-v1}"
cd "${REPO_ROOT}"
python "${REPO_ROOT}/pipeline/slurm/patch_humming_nvfp4_w4a8.py" --check
python - <<'PY'
import torch
torch.manual_seed(20260719)
torch.cuda.manual_seed_all(20260719)
capability = torch.cuda.get_device_capability()
if capability != (9, 0):
    raise SystemExit(f"SM90 required, found {capability}")
print({"device_capability": capability, "seed": 20260719})
PY
python "${REPO_ROOT}/pipeline/hopper_nvfp4_w4a8/gpu_probe.py" \
  --output "${RESULT_ROOT}/probe.json"
EOF

set +e
srun --nodes=1 --ntasks=1 --gres=gpu:1 --time=00:30:00 \
  bash "${RESULT_ROOT}/worker.sh" \
  > "${RESULT_ROOT}/srun.stdout" 2> "${RESULT_ROOT}/srun.stderr"
SRUN_RC=$?
set -e
printf '%s\n' "${SRUN_RC}" > "${RESULT_ROOT}/srun.returncode"

set +e
python - "${RESULT_ROOT}/probe.json" <<'PY'
import json
import sys
from pathlib import Path

path = Path(sys.argv[1])
payload = json.loads(path.read_text(encoding="utf-8"))
assert payload["passed"] is True
assert payload["device_capability"] == [9, 0]
assert payload["sass"]["fp8_wgmma_found"] is True
assert payload["memory"]["persistent_ratio"] <= 1.10
assert payload["layer_transform"]["passed"] is True
assert all(item["passed"] for item in payload["shapes"])
assert payload["k16_isolation"]["passed"] is True
assert payload["fragment_scale_isolation"]["passed"] is True
PY
VALIDATE_RC=$?
set -e
printf '%s\n' "${VALIDATE_RC}" > "${RESULT_ROOT}/validation.returncode"

(
  cd "${RESULT_ROOT}"
  find . -type f ! -name SHA256SUMS -print0 \
    | sort -z \
    | xargs -0 sha256sum > SHA256SUMS
)

echo "RESULT_ROOT=${RESULT_ROOT}"
if [[ "${SRUN_RC}" -ne 0 || "${VALIDATE_RC}" -ne 0 ]]; then
  exit 1
fi
