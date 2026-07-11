#!/usr/bin/env bash
# Run one MiniMax-M3 routed-expert diagnostic arm on an allocated 8-GPU node.

set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
cd "$REPO_ROOT"

DRY_RUN="${DRY_RUN:-0}"
MATRIX_ID="${MATRIX_ID:-$(date +%Y%m%d-%H%M%S)-routed-diagnostics}"
ARM="${ARM:-}"
case "$ARM" in
  reference_w4a16) CHECKPOINT_ROLE=reference; DISABLE_ACTIVATIONS=0 ;;
  candidate_w4a8) CHECKPOINT_ROLE=candidate; DISABLE_ACTIVATIONS=0 ;;
  candidate_w4a16) CHECKPOINT_ROLE=candidate; DISABLE_ACTIVATIONS=1 ;;
  *) echo "ERROR: unknown ARM=$ARM" >&2; exit 2 ;;
esac

REFERENCE_CKPT="${REFERENCE_CKPT:-/mnt/nfs/hoangduy/hf_assets/cyankiwi/MiniMax-M3-AWQ-INT4}"
CANDIDATE_CKPT="${CANDIDATE_CKPT:-/mnt/nfs/hoangduy/projects/llm-compressor/artifacts/MiniMax-M3-awq-W4AFP8/20260709-064104/checkpoint-vllm-w123}"
MODEL_ID="${MODEL_ID:-/mnt/nfs/hoangduy/hf_assets/MiniMaxAI/MiniMax-M3}"
CONFIG="${CONFIG:-pipeline/configs/minimax_m3_full_calib.yaml}"
RESULTS_ROOT="${RESULTS_ROOT:-/mnt/nfs/hoangduy/logs/m3-routed-diagnostics}"
EVIDENCE_ROOT="${EVIDENCE_ROOT:-$REPO_ROOT/results/m3-routed-diagnostics}"
ENV_FILE="${ENV_FILE:-/mnt/nfs/hoangduy/env.sh}"
VENV_ACTIVATE="${VENV_ACTIVATE:-/mnt/nfs/hoangduy/venvs/quant/bin/activate}"
RUN_DIR="$RESULTS_ROOT/$MATRIX_ID/$ARM"
EVIDENCE_DIR="$EVIDENCE_ROOT/$MATRIX_ID/$ARM"
CHECKPOINT="$REFERENCE_CKPT"
[[ "$CHECKPOINT_ROLE" == candidate ]] && CHECKPOINT="$CANDIDATE_CKPT"

if [[ ! -f "$CHECKPOINT/config.json" || ! -f "$CHECKPOINT/model.safetensors.index.json" ]]; then
  echo "ERROR: selected checkpoint metadata missing: $CHECKPOINT" >&2
  exit 2
fi
CHECKPOINT="$(realpath "$CHECKPOINT")"
mkdir -p "$RUN_DIR" "$EVIDENCE_DIR"
manifest_args=(
  python -m pipeline.m3_routed_diagnostics manifest
  --arm "$ARM"
  --run-dir "$RUN_DIR"
  --evidence-dir "$EVIDENCE_DIR"
  --checkpoint "$CHECKPOINT"
  --model-id "$MODEL_ID"
)
if [[ "$DRY_RUN" == 1 || "$DRY_RUN" == true ]]; then
  manifest_args+=(--dry-run)
fi
"${manifest_args[@]}"

if [[ "$DRY_RUN" == 1 || "$DRY_RUN" == true ]]; then
  python -m pipeline.m3_routed_diagnostics bundle-arm \
    --run-dir "$RUN_DIR" --evidence-dir "$EVIDENCE_DIR"
  echo "[m3-routed-diag] DRY_RUN arm=$ARM run_dir=$RUN_DIR"
  exit 0
fi

_cleanup() {
  local rc="$?"
  trap - EXIT
  echo "$rc" >"$RUN_DIR/return_code.txt"
  python -m pipeline.m3_routed_diagnostics bundle-arm \
    --run-dir "$RUN_DIR" --evidence-dir "$EVIDENCE_DIR" || true
  exit "$rc"
}
trap _cleanup EXIT

if [[ ! -f "$ENV_FILE" || ! -f "$VENV_ACTIVATE" ]]; then
  echo "ERROR: live environment files missing: $ENV_FILE $VENV_ACTIVATE" >&2
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
export M3_LOAD_AUDIT=1
export M3_MOE_PROBE=1
export M3_MOE_PROBE_RECOMPUTE=1
export M3_MOE_PROBE_MAX_TOKENS=256
export M3_PARAM_FINGERPRINT=1
export M3_PARAM_FINGERPRINT_LAYERS=3,59
export M3_QUALITY_CASE="$ARM"

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
python "$SCRIPT_DIR/patch_vllm_m3_serve.py" --check \
  >"$RUN_DIR/patch_status.txt" 2>&1

FORCE="${FORCE:-0}" MIN_FREE_GIB="${MIN_FREE_GIB:-70}" \
  bash "$SCRIPT_DIR/free_gpus.sh"

CHECKPOINT_VIEW="$RUN_DIR/checkpoint"
overlay_args=(
  python -m pipeline.m3_routed_diagnostics prepare-overlay
  --source "$CHECKPOINT"
  --destination "$CHECKPOINT_VIEW"
)
if [[ "$DISABLE_ACTIVATIONS" == 1 ]]; then
  overlay_args+=(--disable-activations)
fi
"${overlay_args[@]}"

python -m pipeline.run --config "$CONFIG" --stage serve \
  --checkpoint "$CHECKPOINT_VIEW" \
  --set model.id="$MODEL_ID" \
  --set serve.tensor_parallel_size=8 \
  --set serve.enable_expert_parallel=true \
  --set serve.block_size=128 \
  --set serve.kv_cache_dtype=fp8 \
  --set serve.max_model_len=2048 \
  --set serve.gpu_memory_utilization=0.85 \
  --set serve.enforce_eager=true \
  --set serve.disable_custom_all_reduce=true \
  --set eval.enabled=false \
  2>&1 | tee "$RUN_DIR/serve.log"
