#!/usr/bin/env bash
# Paired MiniMax-M3 quality comparison: working cyankiwi control vs portable W4A8.
# Live execution requires one idle 8-GPU node. CPU validation uses DRY_RUN=1.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
cd "$REPO_ROOT"

DRY_RUN="${DRY_RUN:-0}"
RUN_MODE="${RUN_MODE:-paired}"
if [[ "$RUN_MODE" != "paired" && "$RUN_MODE" != "reference_only" ]]; then
  echo "ERROR: RUN_MODE must be paired or reference_only: $RUN_MODE"
  exit 2
fi
M3_LOAD_AUDIT="${M3_LOAD_AUDIT:-1}"
M3_MOE_PROBE="${M3_MOE_PROBE:-1}"
M3_PARAM_FINGERPRINT="${M3_PARAM_FINGERPRINT:-1}"
RUN_ID="${RUN_ID:-$(date +%Y%m%d-%H%M%S)}"
RESULTS_ROOT="${RESULTS_ROOT:-/mnt/nfs/hoangduy/logs/m3-paired-quality}"
EVIDENCE_ROOT="${EVIDENCE_ROOT:-$REPO_ROOT/results/m3-paired-quality}"
RUN_DIR="$RESULTS_ROOT/$RUN_ID"
EVIDENCE_DIR="$EVIDENCE_ROOT/$RUN_ID"

REFERENCE_CKPT="${REFERENCE_CKPT:-/mnt/nfs/hoangduy/hf_assets/cyankiwi/MiniMax-M3-AWQ-INT4}"
CANDIDATE_CKPT="${CANDIDATE_CKPT:-artifacts/MiniMax-M3-awq-W4AFP8/20260709-064104/checkpoint-vllm-w123}"
MODEL_ID="${MODEL_ID:-/mnt/nfs/hoangduy/hf_assets/MiniMaxAI/MiniMax-M3}"
CONFIG="${CONFIG:-pipeline/configs/minimax_m3_full_calib.yaml}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-2048}"
GPU_UTIL="${GPU_UTIL:-0.85}"
ENV_FILE="${ENV_FILE:-/mnt/nfs/hoangduy/env.sh}"
VENV_ACTIVATE="${VENV_ACTIVATE:-/mnt/nfs/hoangduy/venvs/quant/bin/activate}"

checkpoints=("$REFERENCE_CKPT")
if [[ "$RUN_MODE" == "paired" ]]; then
  checkpoints+=("$CANDIDATE_CKPT")
fi
for checkpoint in "${checkpoints[@]}"; do
  if [[ ! -f "$checkpoint/config.json" ]]; then
    echo "ERROR: checkpoint config missing: $checkpoint/config.json"
    exit 2
  fi
done
REFERENCE_CKPT="$(realpath "$REFERENCE_CKPT")"
if [[ "$RUN_MODE" == "paired" ]]; then
  CANDIDATE_CKPT="$(realpath "$CANDIDATE_CKPT")"
fi
if [[ ! -f "$CONFIG" ]]; then
  echo "ERROR: pipeline config missing: $CONFIG"
  exit 2
fi

mkdir -p "$RUN_DIR" "$EVIDENCE_DIR"
manifest_args=(
  python -m pipeline.m3_quality_evidence manifest
  --run-dir "$RUN_DIR"
  --evidence-dir "$EVIDENCE_DIR"
  --reference "$REFERENCE_CKPT"
  --candidate "$CANDIDATE_CKPT"
  --config "$CONFIG"
  --model-id "$MODEL_ID"
  --max-model-len "$MAX_MODEL_LEN"
  --gpu-util "$GPU_UTIL"
  --run-mode "$RUN_MODE"
  --load-audit "$M3_LOAD_AUDIT"
  --moe-probe "$M3_MOE_PROBE"
  --param-fingerprint "$M3_PARAM_FINGERPRINT"
)
if [[ "$DRY_RUN" == "1" || "$DRY_RUN" == "true" ]]; then
  manifest_args+=(--dry-run)
fi
"${manifest_args[@]}"

echo "[m3-quality] run_dir=$RUN_DIR"
echo "[m3-quality] evidence_dir=$EVIDENCE_DIR"

if [[ "$DRY_RUN" == "1" || "$DRY_RUN" == "true" ]]; then
  python -m pipeline.m3_quality_evidence bundle \
    --run-dir "$RUN_DIR" \
    --evidence-dir "$EVIDENCE_DIR"
  echo "[m3-quality] DRY_RUN complete; no GPU or vLLM command executed"
  exit 0
fi

if [[ ! -f "$ENV_FILE" || ! -f "$VENV_ACTIVATE" ]]; then
  echo "ERROR: live environment files missing:"
  echo "  ENV_FILE=$ENV_FILE"
  echo "  VENV_ACTIVATE=$VENV_ACTIVATE"
  exit 2
fi

# shellcheck disable=SC1090
source "$ENV_FILE"
# shellcheck disable=SC1090
source "$VENV_ACTIVATE"
export PYTHONPATH="$REPO_ROOT"
export PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM=false
export FLASHINFER_USE_CUDA_NORM=1
export VLLM_DISABLE_SHARED_EXPERTS_STREAM=1
export M3_LOAD_AUDIT
export M3_MOE_PROBE
export M3_PARAM_FINGERPRINT
export M3_PARAM_FINGERPRINT_LAYERS="${M3_PARAM_FINGERPRINT_LAYERS:-3,59}"

{
  echo "recorded_at=$(date -Is)"
  echo "python=$(command -v python)"
  python --version
  python -c 'import vllm; print("vllm", vllm.__version__)'
  python -c 'import torch; print("torch", torch.__version__, "cuda", torch.version.cuda)'
  for package in compressed-tensors flashinfer-python safetensors transformers; do
    python -c 'import importlib.metadata as m,sys; print(sys.argv[1], m.version(sys.argv[1]))' "$package" || echo "$package NOT_INSTALLED"
  done
} >"$RUN_DIR/software_versions.txt" 2>&1
nvidia-smi --query-gpu=index,name,uuid,driver_version,memory.total \
  --format=csv >"$RUN_DIR/nvidia_smi.csv" 2>&1
nvidia-smi topo -m >"$RUN_DIR/nvidia_topology.txt" 2>&1

# Establish identical persistent source state before running either case.
python pipeline/slurm/patch_vllm_m3_serve.py >"$RUN_DIR/patch_status.txt" 2>&1

_bundle() {
  python -m pipeline.m3_quality_evidence bundle \
    --run-dir "$RUN_DIR" \
    --evidence-dir "$EVIDENCE_DIR"
}

_run_case() {
  local case_name="$1"
  local checkpoint="$2"
  local case_dir="$RUN_DIR/$case_name"
  local log="$case_dir/serve.log"
  mkdir -p "$case_dir"
  ln -s "$checkpoint" "$case_dir/checkpoint"
  MIN_FREE_GIB="${MIN_FREE_GIB:-70}" bash "$SCRIPT_DIR/free_gpus.sh"
  export M3_QUALITY_CASE="$case_name"
  date -Is >"$case_dir/started_at.txt"
  echo "[m3-quality] starting $case_name at $(date -Is)"
  set +e
  python -m pipeline.run --config "$CONFIG" --stage serve \
    --checkpoint "$case_dir/checkpoint" \
    --set model.id="$MODEL_ID" \
    --set serve.tensor_parallel_size=8 \
    --set serve.enable_expert_parallel=true \
    --set serve.block_size=128 \
    --set serve.kv_cache_dtype=fp8 \
    --set serve.max_model_len="$MAX_MODEL_LEN" \
    --set serve.gpu_memory_utilization="$GPU_UTIL" \
    --set serve.enforce_eager=true \
    --set serve.disable_custom_all_reduce=true \
    --set eval.enabled=false \
    2>&1 | tee "$log"
  local rc="${PIPESTATUS[0]}"
  set -e
  echo "$rc" >"$case_dir/return_code.txt"
  date -Is >"$case_dir/finished_at.txt"
  echo "[m3-quality] finished $case_name rc=$rc at $(date -Is)"
  return "$rc"
}

reference_rc=0
_run_case "cyankiwi_reference" "$REFERENCE_CKPT" || reference_rc=$?
_bundle
if [[ "$RUN_MODE" == "reference_only" ]]; then
  echo "[m3-quality] reference-only evidence complete: $EVIDENCE_DIR (rc=$reference_rc)"
  exit "$reference_rc"
fi
if [[ "$reference_rc" -ne 0 ]]; then
  echo "ERROR: reference infrastructure failed (rc=$reference_rc); candidate skipped"
  exit "$reference_rc"
fi
if ! python -c 'import json,sys; report=json.load(open(sys.argv[1])); raise SystemExit(0 if report.get("quality_ok") is True else 1)' \
  "$RUN_DIR/cyankiwi_reference/serve_report.json"; then
  echo "ERROR: reference failed smoke quality; candidate skipped"
  exit 3
fi

candidate_rc=0
_run_case "portable_awq_w4a8" "$CANDIDATE_CKPT" || candidate_rc=$?
_bundle
if [[ "$candidate_rc" -ne 0 ]]; then
  echo "ERROR: candidate infrastructure failed (rc=$candidate_rc); evidence preserved"
  exit "$candidate_rc"
fi

echo "[m3-quality] paired evidence complete: $EVIDENCE_DIR"
