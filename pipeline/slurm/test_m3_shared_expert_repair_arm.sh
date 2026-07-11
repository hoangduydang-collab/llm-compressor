#!/usr/bin/env bash
# Run one MiniMax-M3 shared-expert repair arm on an allocated 8-GPU node.

set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
cd "$REPO_ROOT"

DRY_RUN="${DRY_RUN:-0}"
MATRIX_ID="${MATRIX_ID:-$(date +%Y%m%d-%H%M%S)-shared-repair}"
ARM="${ARM:-}"
case "$ARM" in
  repaired_w4a8_offline) INTERFACE=offline; DISABLE_ACTIVATIONS=0 ;;
  repaired_w4a16_offline) INTERFACE=offline; DISABLE_ACTIVATIONS=1 ;;
  repaired_w4a8_http) INTERFACE=http; DISABLE_ACTIVATIONS=0 ;;
  *) echo "ERROR: unknown ARM=$ARM" >&2; exit 2 ;;
esac

CANDIDATE_CKPT="${CANDIDATE_CKPT:-/mnt/nfs/hoangduy/projects/llm-compressor/artifacts/MiniMax-M3-awq-W4AFP8/20260709-064104/checkpoint-vllm-w123}"
MODEL_ID="${MODEL_ID:-/mnt/nfs/hoangduy/hf_assets/MiniMaxAI/MiniMax-M3}"
CONFIG="${CONFIG:-pipeline/configs/minimax_m3_full_calib.yaml}"
RESULTS_ROOT="${RESULTS_ROOT:-/mnt/nfs/hoangduy/logs/m3-shared-expert-repair}"
EVIDENCE_ROOT="${EVIDENCE_ROOT:-$REPO_ROOT/results/m3-shared-expert-repair}"
ENV_FILE="${ENV_FILE:-/mnt/nfs/hoangduy/env.sh}"
VENV_ACTIVATE="${VENV_ACTIVATE:-/mnt/nfs/hoangduy/venvs/quant/bin/activate}"
RUN_DIR="$RESULTS_ROOT/$MATRIX_ID/$ARM"
EVIDENCE_DIR="$EVIDENCE_ROOT/$MATRIX_ID/$ARM"

if [[ ! -f "$CANDIDATE_CKPT/config.json" || ! -f "$CANDIDATE_CKPT/model.safetensors.index.json" ]]; then
  echo "ERROR: candidate checkpoint metadata missing: $CANDIDATE_CKPT" >&2
  exit 2
fi
CANDIDATE_CKPT="$(realpath "$CANDIDATE_CKPT")"
mkdir -p "$RUN_DIR" "$EVIDENCE_DIR"
CHECKPOINT_VIEW="$RUN_DIR/checkpoint"
overlay_args=(
  python -m pipeline.m3_routed_diagnostics prepare-overlay
  --source "$CANDIDATE_CKPT"
  --destination "$CHECKPOINT_VIEW"
  --add-vllm-shared-expert-ignore
)
if [[ "$DISABLE_ACTIVATIONS" == 1 ]]; then
  overlay_args+=(--disable-activations)
fi
"${overlay_args[@]}"

manifest_args=(
  python -m pipeline.m3_shared_expert_repair manifest
  --arm "$ARM"
  --run-dir "$RUN_DIR"
  --evidence-dir "$EVIDENCE_DIR"
  --source-checkpoint "$CANDIDATE_CKPT"
  --overlay-checkpoint "$CHECKPOINT_VIEW"
  --model-id "$MODEL_ID"
)
if [[ "$DRY_RUN" == 1 || "$DRY_RUN" == true ]]; then
  manifest_args+=(--dry-run)
fi
"${manifest_args[@]}"

if [[ "$DRY_RUN" == 1 || "$DRY_RUN" == true ]]; then
  python -m pipeline.m3_shared_expert_repair bundle-arm \
    --run-dir "$RUN_DIR" --evidence-dir "$EVIDENCE_DIR"
  echo "[m3-shared-repair] DRY_RUN arm=$ARM run_dir=$RUN_DIR"
  exit 0
fi

SERVER_PID_FILE="$RUN_DIR/server.pid"
_cleanup() {
  local rc="$?"
  trap - EXIT
  if [[ -f "$SERVER_PID_FILE" ]]; then
    server_pid="$(cat "$SERVER_PID_FILE" 2>/dev/null || true)"
    if [[ -n "$server_pid" ]] && kill -0 "$server_pid" 2>/dev/null; then
      kill -TERM -- "-$server_pid" 2>/dev/null || kill "$server_pid" 2>/dev/null || true
    fi
  fi
  echo "$rc" >"$RUN_DIR/return_code.txt"
  python -m pipeline.m3_shared_expert_repair bundle-arm \
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
if [[ "$INTERFACE" == offline ]]; then
  export M3_LOAD_AUDIT=1 M3_MOE_PROBE=1 M3_MOE_PROBE_RECOMPUTE=1
  export M3_MOE_PROBE_MAX_TOKENS=256 M3_PARAM_FINGERPRINT=1
  export M3_PARAM_FINGERPRINT_LAYERS=3,59 M3_QUALITY_CASE="$ARM"
else
  export M3_LOAD_AUDIT=0 M3_MOE_PROBE=0 M3_MOE_PROBE_RECOMPUTE=0
  export M3_PARAM_FINGERPRINT=0
fi

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
python "$SCRIPT_DIR/patch_vllm_m3_serve.py" --check >"$RUN_DIR/patch_status.txt" 2>&1
FORCE="${FORCE:-0}" MIN_FREE_GIB="${MIN_FREE_GIB:-70}" bash "$SCRIPT_DIR/free_gpus.sh"

if [[ "$INTERFACE" == offline ]]; then
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
    --set eval.enabled=false 2>&1 | tee "$RUN_DIR/serve.log"
else
  SERVED_NAME="m3-shared-repair"
  PORT="${PORT:-8000}"
  CKPT="$CHECKPOINT_VIEW" SERVED_NAME="$SERVED_NAME" MODEL_ID="$MODEL_ID" \
    HOST=0.0.0.0 PORT="$PORT" MAX_MODEL_LEN=2048 GPU_UTIL=0.85 \
    ENFORCE_EAGER=1 ENABLE_EP=1 DISABLE_CUSTOM_AR=1 LANGUAGE_MODEL_ONLY=1 \
    DEBUG_CUDAGRAPH=0 SKIP_GPU_PREFLIGHT=1 LOG="$RUN_DIR/serve.log" \
    PID_FILE="$SERVER_PID_FILE" bash "$SCRIPT_DIR/run_vllm_http_serve_smoke.sh" \
    >"$RUN_DIR/server_start.txt" 2>&1
  ready=0
  for _ in $(seq 1 120); do
    if curl -sf "http://127.0.0.1:$PORT/health" >/dev/null; then ready=1; break; fi
    if [[ -f "$SERVER_PID_FILE" ]]; then
      server_pid="$(cat "$SERVER_PID_FILE" 2>/dev/null || true)"
      if [[ -n "$server_pid" ]] && ! kill -0 "$server_pid" 2>/dev/null; then break; fi
    fi
    sleep 10
  done
  if [[ "$ready" -ne 1 ]]; then echo "ERROR: HTTP server did not become healthy" >&2; exit 1; fi
  python - "$PORT" "$SERVED_NAME" "$RUN_DIR" <<'PYHTTP'
import json
import sys
import urllib.request
from pathlib import Path

from pipeline.m3_quality_evidence import M3_QUALITY_CASES

port, model, raw_dir = sys.argv[1], sys.argv[2], Path(sys.argv[3])
url = f"http://127.0.0.1:{port}/v1/chat/completions"
for index, case in enumerate(M3_QUALITY_CASES):
    body = {
        "model": model,
        "messages": [{"role": "user", "content": case.prompt}],
        "max_tokens": 64,
        "temperature": 0.0,
        "chat_template_kwargs": {"enable_thinking": False, "thinking_mode": "disabled"},
    }
    request = urllib.request.Request(
        url, data=json.dumps(body).encode(), headers={"Content-Type": "application/json"}
    )
    try:
        with urllib.request.urlopen(request, timeout=300) as response:
            payload = json.loads(response.read().decode())
    except Exception as exc:
        payload = {"error": {"type": type(exc).__name__, "message": str(exc)}}
    (raw_dir / f"http_request_{index}.json").write_text(json.dumps(body, indent=2, sort_keys=True) + "\n")
    (raw_dir / f"http_response_{index}.json").write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
PYHTTP
fi
