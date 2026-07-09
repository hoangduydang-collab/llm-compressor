#!/usr/bin/env bash
# Nemotron-style HTTP smoke serve: ``vllm serve`` + OpenAI chat/completions.
#
# Unlike ``run_serve_minimax_m3_detached.sh`` (offline ``LLM.generate`` via
# ``pipeline.run --stage serve``), this matches the usable Nemotron Ultra flow:
# long-lived HTTP server, then curl ``/v1/chat/completions``.
#
# Default target: cyankiwi MiniMax-M3 AWQ-INT4 on 8 GPUs.
#
#   bash pipeline/slurm/run_vllm_http_serve_smoke.sh
#   CKPT=/path/to/ckpt SERVED_NAME=... bash pipeline/slurm/run_vllm_http_serve_smoke.sh
#
# IMPORTANT: do NOT reuse a shell that still has ``export MODEL_CKPT=...Nemotron...``.
# This script intentionally ignores inherited ``MODEL_CKPT`` (use ``CKPT=`` instead)
# so a prior Nemotron smoke cannot silently load the wrong weights under MiniMax
# parsers (that fails with MiniMaxM3ReasoningParser missing think tokens).
#
# Then smoke:
#   bash pipeline/slurm/smoke_chat_completions.sh
#
# Monitor / stop:
#   tail -f "$LOG"
#   kill "$(cat "$PID_FILE")"
#   kill -9 -"$(cat "$PID_FILE")"

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
cd "$REPO_ROOT"

source /mnt/nfs/hoangduy/env.sh
source /mnt/nfs/hoangduy/venvs/quant/bin/activate

export HOME="${HOME:-/mnt/nfs/hoangduy}"
export WORK_ROOT="${WORK_ROOT:-/mnt/nfs/hoangduy}"
export PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM=false
export FLASHINFER_USE_CUDA_NORM=1
export FLASHINFER_WORKSPACE_DIR="${FLASHINFER_WORKSPACE_DIR:-${HOME}/cache/flashinfer}"
export VLLM_WORKER_MULTIPROC_METHOD=spawn
export VLLM_MEMORY_PROFILER_ESTIMATE_CUDAGRAPHS="${VLLM_MEMORY_PROFILER_ESTIMATE_CUDAGRAPHS:-1}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
# Nemotron smoke sets this; it is meaningless for MiniMax and only confuses logs.
unset VLLM_USE_FLASHINFER_MOE_FP4 2>/dev/null || true
mkdir -p "$FLASHINFER_WORKSPACE_DIR" /mnt/nfs/hoangduy/logs

DEFAULT_CKPT="/mnt/nfs/hoangduy/hf_assets/cyankiwi/MiniMax-M3-AWQ-INT4"
DEFAULT_NAME="cyankiwi/MiniMax-M3-AWQ-INT4"

# Prefer CKPT= (explicit). Ignore inherited MODEL_CKPT from Nemotron sessions.
if [[ -n "${CKPT:-}" ]]; then
  MODEL_CKPT="$CKPT"
elif [[ -n "${MODEL_CKPT:-}" && "$MODEL_CKPT" != "$DEFAULT_CKPT" ]]; then
  echo "WARNING: ignoring inherited MODEL_CKPT=$MODEL_CKPT"
  echo "         (Nemotron leftover). Using default cyankiwi path."
  echo "         Override with: CKPT=/path/to/minimax bash $0"
  MODEL_CKPT="$DEFAULT_CKPT"
else
  MODEL_CKPT="${MODEL_CKPT:-$DEFAULT_CKPT}"
fi

SERVED_NAME="${SERVED_NAME:-$DEFAULT_NAME}"
HOST="${HOST:-0.0.0.0}"
PORT="${PORT:-8000}"
TP="${TP:-8}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-4096}"
MAX_NUM_SEQS="${MAX_NUM_SEQS:-4}"
MAX_NUM_BATCHED_TOKENS="${MAX_NUM_BATCHED_TOKENS:-4096}"
GPU_UTIL="${GPU_UTIL:-0.90}"
KV_CACHE_DTYPE="${KV_CACHE_DTYPE:-fp8}"
BLOCK_SIZE="${BLOCK_SIZE:-128}"
ENABLE_EP="${ENABLE_EP:-1}"
DISABLE_CUSTOM_AR="${DISABLE_CUSTOM_AR:-1}"
APPLY_M3_PATCHES="${APPLY_M3_PATCHES:-1}"
# Default ON for HTTP smoke: AsyncLLM KV/graph init has hit CUDA IMA on M3
# (same class as BUGS_AND_FIXES.md cudagraph capture). Offline LLM.generate
# for cyankiwi can pass with graphs; HTTP path is flakier. Set ENFORCE_EAGER=0
# only after a clean free_gpus and a known-good graphs-on bring-up.
ENFORCE_EAGER="${ENFORCE_EAGER:-1}"
LOG="${LOG:-/mnt/nfs/hoangduy/logs/serve-$(basename "$SERVED_NAME" | tr '/' '-')-http-smoke.log}"
PID_FILE="${PID_FILE:-/mnt/nfs/hoangduy/logs/serve-$(basename "$SERVED_NAME" | tr '/' '-')-http-smoke.pid}"

test -f "$MODEL_CKPT/config.json" || {
  echo "ERROR: missing config.json at $MODEL_CKPT"
  exit 1
}

# Fail fast if someone pointed CKPT at Nemotron / non-M3 while we attach
# minimax_m3 parsers (the failure mode that produced this bug report).
python3 - "$MODEL_CKPT" <<'PY'
import json, sys
from pathlib import Path
cfg = json.loads(Path(sys.argv[1], "config.json").read_text())
archs = cfg.get("architectures") or []
mt = str(cfg.get("model_type") or "")
blob = " ".join(archs) + " " + mt
if "minimax" not in blob.lower() and "MiniMax" not in "".join(archs):
    print(
        f"ERROR: checkpoint does not look like MiniMax-M3\n"
        f"  path: {sys.argv[1]}\n"
        f"  architectures={archs!r} model_type={mt!r}\n"
        f"  Refusing to attach --reasoning-parser minimax_m3 "
        f"(that is the Nemotron-weights + MiniMax-parser footgun).\n"
        f"  Fix: unset MODEL_CKPT; use CKPT=.../cyankiwi/MiniMax-M3-AWQ-INT4",
        file=sys.stderr,
    )
    sys.exit(2)
print(f"checkpoint ok: architectures={archs} model_type={mt}")
PY

if [[ -f "$PID_FILE" ]]; then
  old_pid="$(cat "$PID_FILE" 2>/dev/null || true)"
  if [[ -n "$old_pid" ]] && kill -0 "$old_pid" 2>/dev/null; then
    echo "Serve already running (pid=$old_pid). tail -f $LOG"
    exit 0
  fi
fi

if [[ "${SKIP_GPU_PREFLIGHT:-0}" != "1" ]]; then
  MIN_FREE_GIB="${MIN_FREE_GIB:-70}" bash "$SCRIPT_DIR/free_gpus.sh" || {
    echo "ERROR: GPUs are not free; refusing to start serve."
    exit 1
  }
fi

# Compressed-tensors / Marlin M3 path: site-packages patches must exist before
# Worker_TP* spawn (same requirement as offline serve-verify).
if [[ "$APPLY_M3_PATCHES" == "1" || "$APPLY_M3_PATCHES" == "true" ]]; then
  python "$SCRIPT_DIR/patch_vllm_m3_serve.py" || {
    echo "WARNING: patch_vllm_m3_serve.py failed; continuing (may be fine for non-M3)."
  }
fi

ARGS=(
  serve "$MODEL_CKPT"
  --served-model-name "$SERVED_NAME"
  --host "$HOST"
  --port "$PORT"
  --trust-remote-code
  --tensor-parallel-size "$TP"
  --kv-cache-dtype "$KV_CACHE_DTYPE"
  --max-model-len "$MAX_MODEL_LEN"
  --max-num-seqs "$MAX_NUM_SEQS"
  --max-num-batched-tokens "$MAX_NUM_BATCHED_TOKENS"
  --gpu-memory-utilization "$GPU_UTIL"
  --block-size "$BLOCK_SIZE"
  --tool-call-parser minimax_m3
  --reasoning-parser minimax_m3
  --enable-auto-tool-choice
)

if [[ "$ENABLE_EP" == "1" || "$ENABLE_EP" == "true" ]]; then
  ARGS+=(--enable-expert-parallel)
fi
if [[ "$DISABLE_CUSTOM_AR" == "1" || "$DISABLE_CUSTOM_AR" == "true" ]]; then
  ARGS+=(--disable-custom-all-reduce)
fi
if [[ "$ENFORCE_EAGER" == "1" || "$ENFORCE_EAGER" == "true" ]]; then
  ARGS+=(--enforce-eager)
fi

# Extra raw flags, space-separated (e.g. EXTRA_VLLM_ARGS='--language-model-only').
if [[ -n "${EXTRA_VLLM_ARGS:-}" ]]; then
  # shellcheck disable=SC2206
  ARGS+=($EXTRA_VLLM_ARGS)
fi

echo "host=$(hostname) starting HTTP vLLM smoke serve"
echo "  checkpoint:   $MODEL_CKPT"
echo "  served-name:  $SERVED_NAME"
echo "  port:         $PORT"
echo "  max-model-len:$MAX_MODEL_LEN"
echo "  enforce_eager:$ENFORCE_EAGER"
echo "  log:          $LOG"
python -c "import vllm; print('vllm', vllm.__version__)" 2>/dev/null || echo "vllm: not importable"
nvidia-smi --query-gpu=index,name,memory.total,memory.free --format=csv 2>/dev/null || true

nohup setsid vllm "${ARGS[@]}" >"$LOG" 2>&1 &
echo $! >"$PID_FILE"
echo "serve pid=$(cat "$PID_FILE")"
echo "  tail -f $LOG"
echo "  bash pipeline/slurm/smoke_chat_completions.sh"
echo "  kill \$(cat $PID_FILE)"
