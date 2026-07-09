#!/usr/bin/env bash
# Nemotron-style HTTP smoke serve: ``vllm serve`` + OpenAI chat/completions.
#
# Unlike ``run_serve_minimax_m3_detached.sh`` (offline ``LLM.generate`` via
# ``pipeline.run --stage serve``), this matches the usable Nemotron Ultra flow:
# long-lived HTTP server, then curl ``/v1/chat/completions``.
#
# Default target: cyankiwi MiniMax-M3 AWQ-INT4 on 8 GPUs.
#
# Root cause note (HTTP IMA while offline cyankiwi PASS at max_model_len=8192):
#   Offline ``LLM()`` and HTTP ``vllm serve`` are NOT the same envelope.
#   The HTTP smoke previously inherited Nemotron knobs (max_num_seqs /
#   max_num_batched_tokens) and skipped the offline preflight that
#   ``serve_verify`` always runs (config patch + VL processor artifacts +
#   ``--language-model-only`` for text-only M3). Those diffs — not 4096 vs
#   2048 — are why HTTP failed after offline already worked. See
#   BUGS_AND_FIXES.md "HTTP vllm serve vs offline LLM".
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
export PYTHONPATH="${PYTHONPATH:+$PYTHONPATH:}$REPO_ROOT"
export PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM=false
export FLASHINFER_USE_CUDA_NORM=1
export FLASHINFER_WORKSPACE_DIR="${FLASHINFER_WORKSPACE_DIR:-${HOME}/cache/flashinfer}"
export VLLM_WORKER_MULTIPROC_METHOD=spawn
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
# Nemotron leftovers — meaningless / harmful for MiniMax HTTP smoke.
unset VLLM_USE_FLASHINFER_MOE_FP4 2>/dev/null || true
# Do NOT force VLLM_MEMORY_PROFILER_ESTIMATE_CUDAGRAPHS=1 (Nemotron default).
# Offline cyankiwi never set it; leave unset unless the caller exports it.
mkdir -p "$FLASHINFER_WORKSPACE_DIR" /mnt/nfs/hoangduy/logs

DEFAULT_CKPT="/mnt/nfs/hoangduy/hf_assets/cyankiwi/MiniMax-M3-AWQ-INT4"
DEFAULT_NAME="cyankiwi/MiniMax-M3-AWQ-INT4"
# Same source used by offline serve_verify for config / processor restore.
DEFAULT_MODEL_ID="/mnt/nfs/hoangduy/hf_assets/MiniMaxAI/MiniMax-M3"

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
MODEL_ID="${MODEL_ID:-$DEFAULT_MODEL_ID}"
HOST="${HOST:-0.0.0.0}"
PORT="${PORT:-8000}"
TP="${TP:-8}"
# Match the offline cyankiwi PASS (MAX_MODEL_LEN=8192, GPU_UTIL=0.9), not the
# W4AFP8 debug envelope. HTTP previously failed for other reasons (see header).
MAX_MODEL_LEN="${MAX_MODEL_LEN:-8192}"
GPU_UTIL="${GPU_UTIL:-0.9}"
KV_CACHE_DTYPE="${KV_CACHE_DTYPE:-fp8}"
BLOCK_SIZE="${BLOCK_SIZE:-128}"
ENABLE_EP="${ENABLE_EP:-1}"
DISABLE_CUSTOM_AR="${DISABLE_CUSTOM_AR:-1}"
APPLY_M3_PATCHES="${APPLY_M3_PATCHES:-1}"
# Text-only smoke: skip VL multimodal budget (official MiniMax recipes use this).
# Offline LLM() does not allocate the same MM path as HTTP AsyncLLM.
LANGUAGE_MODEL_ONLY="${LANGUAGE_MODEL_ONLY:-1}"
# Optional Nemotron-style batching. Empty = omit (match offline LLM() defaults).
# Setting these was a suspected HTTP-vs-offline divergence for graph capture.
MAX_NUM_SEQS="${MAX_NUM_SEQS:-}"
MAX_NUM_BATCHED_TOKENS="${MAX_NUM_BATCHED_TOKENS:-}"
# Graphs on by default (offline cyankiwi LLM() PASS at 8192). HTTP AsyncLLM has
# still IMA'd at capture with patches live — ENFORCE_EAGER=1 unblocks chat smoke
# while we name the kernel (DEBUG_CUDAGRAPH=1). Do not treat eager as the fix.
ENFORCE_EAGER="${ENFORCE_EAGER:-0}"
# Same preflight serve_verify runs before LLM() — required for raw vllm serve.
PATCH_CKPT_CONFIG="${PATCH_CKPT_CONFIG:-1}"
# When 1: force sync CUDA + DSA so the IMA names the real kernel (not empty_cache).
DEBUG_CUDAGRAPH="${DEBUG_CUDAGRAPH:-0}"
LOG="${LOG:-/mnt/nfs/hoangduy/logs/serve-$(basename "$SERVED_NAME" | tr '/' '-')-http-smoke.log}"
PID_FILE="${PID_FILE:-/mnt/nfs/hoangduy/logs/serve-$(basename "$SERVED_NAME" | tr '/' '-')-http-smoke.pid}"

if [[ "$DEBUG_CUDAGRAPH" == "1" || "$DEBUG_CUDAGRAPH" == "true" ]]; then
  export CUDA_LAUNCH_BLOCKING=1
  export TORCH_USE_CUDA_DSA=1
  export PYTHONFAULTHANDLER=1
  echo "[http-smoke] DEBUG_CUDAGRAPH=1 — CUDA_LAUNCH_BLOCKING=1 TORCH_USE_CUDA_DSA=1"
fi

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

# Mirror serve_verify preflight: restore hidden_act=swigluoai + VL processor
# files. Raw ``vllm serve`` does not do this; offline LLM() always does.
if [[ "$PATCH_CKPT_CONFIG" == "1" || "$PATCH_CKPT_CONFIG" == "true" ]]; then
  python - "$MODEL_CKPT" "$MODEL_ID" <<'PY'
import sys
from pathlib import Path

ckpt = Path(sys.argv[1])
source = sys.argv[2]
try:
    from pipeline.minimax_m3_config import ensure_minimax_m3_vllm_serve_config
    from pipeline.vl_artifacts import ensure_vl_processor_artifacts
except ImportError as exc:
    print(f"WARNING: cannot import pipeline helpers ({exc}); skip config patch")
    sys.exit(0)

cfg_patches = ensure_minimax_m3_vllm_serve_config(ckpt, source)
if cfg_patches:
    print(f"[http-smoke] patched checkpoint config: {cfg_patches}")
else:
    print("[http-smoke] checkpoint config already vLLM-ready")
added = ensure_vl_processor_artifacts(ckpt, source, trust_remote_code=True)
if added:
    print(f"[http-smoke] copied VL processor artifacts: {added}")
else:
    print("[http-smoke] VL processor artifacts present")
PY
fi

# Compressed-tensors / Marlin M3 path: site-packages patches must exist before
# Worker_TP* spawn (same requirement as offline serve-verify).
# Fail loud if patches are missing — otherwise we re-debug a "fixed" bug under
# graphs-on while the real issue is an unpatched venv.
if [[ "$APPLY_M3_PATCHES" == "1" || "$APPLY_M3_PATCHES" == "true" ]]; then
  if ! python "$SCRIPT_DIR/patch_vllm_m3_serve.py" --check; then
    echo "[http-smoke] patches incomplete; applying..."
    python "$SCRIPT_DIR/patch_vllm_m3_serve.py" || {
      echo "ERROR: patch_vllm_m3_serve.py failed"
      exit 1
    }
    python "$SCRIPT_DIR/patch_vllm_m3_serve.py" --check || {
      echo "ERROR: vLLM M3 patches still incomplete after apply"
      exit 1
    }
  fi
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
  --gpu-memory-utilization "$GPU_UTIL"
  --block-size "$BLOCK_SIZE"
  --tool-call-parser minimax_m3
  --reasoning-parser minimax_m3
  --enable-auto-tool-choice
)

# Only pass batching knobs when explicitly set (empty = offline LLM defaults).
if [[ -n "$MAX_NUM_SEQS" ]]; then
  ARGS+=(--max-num-seqs "$MAX_NUM_SEQS")
fi
if [[ -n "$MAX_NUM_BATCHED_TOKENS" ]]; then
  ARGS+=(--max-num-batched-tokens "$MAX_NUM_BATCHED_TOKENS")
fi

if [[ "$ENABLE_EP" == "1" || "$ENABLE_EP" == "true" ]]; then
  ARGS+=(--enable-expert-parallel)
fi
if [[ "$DISABLE_CUSTOM_AR" == "1" || "$DISABLE_CUSTOM_AR" == "true" ]]; then
  ARGS+=(--disable-custom-all-reduce)
fi
if [[ "$LANGUAGE_MODEL_ONLY" == "1" || "$LANGUAGE_MODEL_ONLY" == "true" ]]; then
  ARGS+=(--language-model-only)
fi
if [[ "$ENFORCE_EAGER" == "1" || "$ENFORCE_EAGER" == "true" ]]; then
  ARGS+=(--enforce-eager)
fi

# Extra raw flags, space-separated.
if [[ -n "${EXTRA_VLLM_ARGS:-}" ]]; then
  # shellcheck disable=SC2206
  ARGS+=($EXTRA_VLLM_ARGS)
fi

echo "host=$(hostname) starting HTTP vLLM smoke serve"
echo "  checkpoint:          $MODEL_CKPT"
echo "  served-name:         $SERVED_NAME"
echo "  port:                $PORT"
echo "  max-model-len:       $MAX_MODEL_LEN"
echo "  gpu-util:            $GPU_UTIL"
echo "  max-num-seqs:        ${MAX_NUM_SEQS:-<vllm default>}"
echo "  max-num-batched-tok: ${MAX_NUM_BATCHED_TOKENS:-<vllm default>}"
echo "  language-model-only: $LANGUAGE_MODEL_ONLY"
echo "  enforce_eager:       $ENFORCE_EAGER"
echo "  debug_cudagraph:     $DEBUG_CUDAGRAPH"
echo "  log:                 $LOG"
python -c "import vllm; print('vllm', vllm.__version__)" 2>/dev/null || echo "vllm: not importable"
nvidia-smi --query-gpu=index,name,memory.total,memory.free --format=csv 2>/dev/null || true

nohup setsid vllm "${ARGS[@]}" >"$LOG" 2>&1 &
echo $! >"$PID_FILE"
echo "serve pid=$(cat "$PID_FILE")"
echo "  tail -f $LOG"
echo "  bash pipeline/slurm/smoke_chat_completions.sh"
echo "  kill \$(cat $PID_FILE)"
