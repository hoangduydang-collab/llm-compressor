#!/usr/bin/env bash
# Nemotron-style HTTP smoke serve: ``vllm serve`` + OpenAI chat/completions.
#
# Unlike ``run_serve_minimax_m3_detached.sh`` (offline ``LLM.generate`` via
# ``pipeline.run --stage serve``), this matches the usable Nemotron Ultra flow:
# long-lived HTTP server, then curl ``/v1/chat/completions``.
#
# Default target: cyankiwi MiniMax-M3 AWQ-INT4 on 8 GPUs.
#
# Root cause note (HTTP graphs-on IMA; see BUGS_AND_FIXES.md):
#   Async CUDA + shared-expert aux stream → flaky IMA during graph capture.
#   Production default for MiniMax-M3 only:
#     VLLM_DISABLE_SHARED_EXPERTS_STREAM=1  (h125 matrix: 3/3 ready+chat)
#     DEBUG_CUDAGRAPH=0                     (async CUDA; standard practice)
#   DEBUG_CUDAGRAPH=1 remains a diagnostic opt-in (masks the race; not a fix).
#   Other models must not inherit this stream disablement.
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

M3_W4A8_BACKEND="${M3_W4A8_BACKEND:-cutlass}"
case "$M3_W4A8_BACKEND" in
  cutlass) ;;
  humming) ;;
  *)
    echo "ERROR: M3_W4A8_BACKEND must be cutlass or humming; got: $M3_W4A8_BACKEND" >&2
    exit 2
    ;;
esac

case " ${EXTRA_VLLM_ARGS:-} " in
  *" --quantization "*|*" --quantization="*)
    echo "ERROR: EXTRA_VLLM_ARGS must not set --quantization; use M3_W4A8_BACKEND" >&2
    exit 2
    ;;
esac

if [[ "$M3_W4A8_BACKEND" == "humming" ]]; then
  case "${VLLM_HUMMING_USE_F16_ACCUM:-0}" in
    ""|0|false|False) export VLLM_HUMMING_USE_F16_ACCUM=0 ;;
    *)
      echo "ERROR: first Humming qualification requires FP32 accumulation" >&2
      exit 2
      ;;
  esac
  case "${VLLM_HUMMING_MOE_GEMM_TYPE:-indexed}" in
    ""|indexed) export VLLM_HUMMING_MOE_GEMM_TYPE=indexed ;;
    *)
      echo "ERROR: first Humming qualification requires indexed MoE GEMM" >&2
      exit 2
      ;;
  esac
fi

# PRINT_EFFECTIVE_CONFIG dry-runs may run off-cluster (no NFS mounts). Skip
# env/venv sourcing when those paths are absent so local DRY_RUN validation works.
_PRINT_CFG=0
if [[ "${PRINT_EFFECTIVE_CONFIG:-0}" == "1" || "${PRINT_EFFECTIVE_CONFIG:-}" == "true" ]]; then
  _PRINT_CFG=1
fi

if [[ -f /mnt/nfs/hoangduy/env.sh ]]; then
  # shellcheck disable=SC1091
  source /mnt/nfs/hoangduy/env.sh
elif [[ "$_PRINT_CFG" -eq 1 ]]; then
  echo "[http-smoke] WARNING: /mnt/nfs/hoangduy/env.sh missing (PRINT_EFFECTIVE_CONFIG dry-run)"
else
  echo "ERROR: missing /mnt/nfs/hoangduy/env.sh"
  exit 1
fi

if [[ -f /mnt/nfs/hoangduy/venvs/quant/bin/activate ]]; then
  # shellcheck disable=SC1091
  source /mnt/nfs/hoangduy/venvs/quant/bin/activate
elif [[ "$_PRINT_CFG" -eq 1 ]]; then
  echo "[http-smoke] WARNING: quant venv missing (PRINT_EFFECTIVE_CONFIG dry-run)"
else
  echo "ERROR: missing /mnt/nfs/hoangduy/venvs/quant/bin/activate"
  exit 1
fi

export HOME="${HOME:-/mnt/nfs/hoangduy}"
export WORK_ROOT="${WORK_ROOT:-/mnt/nfs/hoangduy}"
export PYTHONPATH="${PYTHONPATH:+$PYTHONPATH:}$REPO_ROOT"
export PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM=false
export FLASHINFER_USE_CUDA_NORM=1
export FLASHINFER_WORKSPACE_DIR="${FLASHINFER_WORKSPACE_DIR:-${HOME}/cache/flashinfer}"
export VLLM_WORKER_MULTIPROC_METHOD=spawn
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
if [[ "$M3_W4A8_BACKEND" == "humming" ]]; then
  _HUMMING_M3_CACHE_ROOT="${HUMMING_M3_W4A8_CACHE_ROOT:-$HOME/.humming}"
  export HUMMING_CACHE_DIR="$_HUMMING_M3_CACHE_ROOT/cache-m3-gptq-w4a8-v1"
  # Humming JIT builds a helper binary that links libnvrtc with -Wl,-rpath, but
  # NVRTC dlopen()s libnvrtc-builtins.so.<ver> at runtime by plain soname, which
  # does not consult that rpath. Without the lib dir on LD_LIBRARY_PATH every
  # kernel compile dies with NVRTC_ERROR_BUILTIN_OPERATION_FAILURE during
  # process_weights_after_loading. Derived from the installed nvidia wheel so it
  # tracks the CUDA major in use rather than hardcoding a version.
  _NVRTC_LIB_DIR="$(python - <<'PY' 2>/dev/null || true
import glob
import os
import site

for base in site.getsitepackages():
    for hit in sorted(glob.glob(os.path.join(base, "nvidia", "*", "lib"))):
        if glob.glob(os.path.join(hit, "libnvrtc-builtins.so*")):
            print(hit)
            raise SystemExit(0)
PY
)"
  if [[ -n "$_NVRTC_LIB_DIR" ]]; then
    case ":${LD_LIBRARY_PATH:-}:" in
      *":$_NVRTC_LIB_DIR:"*) ;;
      *) export LD_LIBRARY_PATH="$_NVRTC_LIB_DIR${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}" ;;
    esac
  else
    echo "ERROR: humming backend selected but libnvrtc-builtins.so was not found;" >&2
    echo "       NVRTC JIT would fail at weight repack. Aborting." >&2
    exit 1
  fi
fi
# MiniMax-M3 shared-expert aux-stream overlap: ON by default since 2026-07-24.
# The old stream-off workaround masked a fork bug — BreakableCUDAGraph._capture
# dropped torch.cuda.graph's pre-cleanup device synchronize, so warmup work on
# the aux stream raced empty_cache's cuMemUnmap during the capture ladder (IMA
# at 43-48/51). LLMC_M3_CAPTURE_SYNC=sync restores the upstream ordering
# (patch_vllm_m3_serve.py). Validation: 12/12 + 2/2 clean captures, conc-1
# TPOT 8.454 -> 7.828 ms (-7.4%), greedy smokes clean (BUGS_AND_FIXES.md).
# Override with VLLM_DISABLE_SHARED_EXPERTS_STREAM=1 to fall back to stream-off.
export VLLM_DISABLE_SHARED_EXPERTS_STREAM="${VLLM_DISABLE_SHARED_EXPERTS_STREAM:-0}"
export LLMC_M3_CAPTURE_SYNC="${LLMC_M3_CAPTURE_SYNC:-sync}"
# Nemotron leftovers — meaningless / harmful for MiniMax HTTP smoke.
unset VLLM_USE_FLASHINFER_MOE_FP4 2>/dev/null || true
# Do NOT force VLLM_MEMORY_PROFILER_ESTIMATE_CUDAGRAPHS=1 (Nemotron default).
# Offline cyankiwi never set it; leave unset unless the caller exports it.
if [[ "$_PRINT_CFG" -eq 0 ]]; then
  CACHE_DIRS=("$FLASHINFER_WORKSPACE_DIR" /mnt/nfs/hoangduy/logs)
  if [[ "$M3_W4A8_BACKEND" == "humming" ]]; then
    CACHE_DIRS+=("$HUMMING_CACHE_DIR")
  fi
  mkdir -p "${CACHE_DIRS[@]}"
fi

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
# Graphs on by default. ENFORCE_EAGER=1 is escape hatch (skips capture entirely).
ENFORCE_EAGER="${ENFORCE_EAGER:-0}"
# Same preflight serve_verify runs before LLM() — required for raw vllm serve.
PATCH_CKPT_CONFIG="${PATCH_CKPT_CONFIG:-1}"
# Async CUDA is the production default (with shared-expert stream disabled).
# DEBUG_CUDAGRAPH=1 is diagnostic-only: forces CUDA_LAUNCH_BLOCKING and masks
# the race; never treat a masked pass as a root-cause fix.
DEBUG_CUDAGRAPH="${DEBUG_CUDAGRAPH:-0}"
LOG="${LOG:-/mnt/nfs/hoangduy/logs/serve-$(basename "$SERVED_NAME" | tr '/' '-')-http-smoke.log}"
PID_FILE="${PID_FILE:-/mnt/nfs/hoangduy/logs/serve-$(basename "$SERVED_NAME" | tr '/' '-')-http-smoke.pid}"

if [[ "$DEBUG_CUDAGRAPH" == "1" || "$DEBUG_CUDAGRAPH" == "true" ]]; then
  export CUDA_LAUNCH_BLOCKING=1
  export TORCH_USE_CUDA_DSA=1
  export PYTHONFAULTHANDLER=1
  echo "[http-smoke] DEBUG_CUDAGRAPH=1 — CUDA_LAUNCH_BLOCKING=1 TORCH_USE_CUDA_DSA=1"
  echo "             (diagnostic only: masks HTTP async cudagraph race; not a fix)"
fi

test -f "$MODEL_CKPT/config.json" || {
  if [[ "${PRINT_EFFECTIVE_CONFIG:-0}" == "1" || "${PRINT_EFFECTIVE_CONFIG:-}" == "true" ]]; then
    echo "WARNING: missing config.json at $MODEL_CKPT (ignored for PRINT_EFFECTIVE_CONFIG)"
  else
    echo "ERROR: missing config.json at $MODEL_CKPT"
    exit 1
  fi
}

# Fail fast if someone pointed CKPT at Nemotron / non-M3 while we attach
# minimax_m3 parsers (the failure mode that produced this bug report).
if [[ -f "$MODEL_CKPT/config.json" ]]; then
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
fi

if [[ -f "$PID_FILE" ]]; then
  old_pid="$(cat "$PID_FILE" 2>/dev/null || true)"
  if [[ -n "$old_pid" ]] && kill -0 "$old_pid" 2>/dev/null; then
    echo "Serve already running (pid=$old_pid). tail -f $LOG"
    exit 0
  fi
fi

# PRINT_EFFECTIVE_CONFIG is read-only observability (RCA matrix dry-run): skip
# GPU kill/preflight and site-packages patch apply — we never launch.
if [[ "${PRINT_EFFECTIVE_CONFIG:-0}" == "1" || "${PRINT_EFFECTIVE_CONFIG:-}" == "true" ]]; then
  SKIP_GPU_PREFLIGHT="${SKIP_GPU_PREFLIGHT:-1}"
  APPLY_M3_PATCHES=0
  PATCH_CKPT_CONFIG=0
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
  PATCH_ARGS=()
  if [[ "$M3_W4A8_BACKEND" == "humming" ]]; then
    PATCH_ARGS+=(--humming)
  fi
  if ! python "$SCRIPT_DIR/patch_vllm_m3_serve.py" --check "${PATCH_ARGS[@]}"; then
    echo "[http-smoke] patches incomplete; applying..."
    python "$SCRIPT_DIR/patch_vllm_m3_serve.py" "${PATCH_ARGS[@]}" || {
      echo "ERROR: patch_vllm_m3_serve.py failed"
      exit 1
    }
    python "$SCRIPT_DIR/patch_vllm_m3_serve.py" --check "${PATCH_ARGS[@]}" || {
      echo "ERROR: vLLM M3 patches still incomplete after apply"
      exit 1
    }
  fi
fi

# r7 gate-alpha checkpoints: inject the dormant per-channel-swiglu support and
# hand the sidecar to the workers. Fail loud — serving an r7 checkpoint with
# the scalar swiglu is a silent function change (see m3_gate_alpha_serve_patch).
if [[ -n "${M3_GATE_ALPHA_SIDECAR:-}" ]]; then
  [[ -f "$M3_GATE_ALPHA_SIDECAR" ]] || {
    echo "ERROR: M3_GATE_ALPHA_SIDECAR not found: $M3_GATE_ALPHA_SIDECAR"
    exit 1
  }
  python "$SCRIPT_DIR/patch_vllm_m3_serve.py" --gate-alpha || {
    echo "ERROR: gate-alpha injection failed"
    exit 1
  }
  export M3_GATE_ALPHA_SIDECAR
  echo "[http-smoke] r7 gate-alpha enabled: $M3_GATE_ALPHA_SIDECAR"
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

if [[ "$M3_W4A8_BACKEND" == "humming" ]]; then
  ARGS+=(--quantization humming)
fi

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
echo "  w4a8-backend:        $M3_W4A8_BACKEND"
echo "  enforce_eager:       $ENFORCE_EAGER"
echo "  debug_cudagraph:     $DEBUG_CUDAGRAPH"
echo "  disable_shared_experts_stream: $VLLM_DISABLE_SHARED_EXPERTS_STREAM"
echo "  capture_sync:        $LLMC_M3_CAPTURE_SYNC"
echo "  log:                 $LOG"

HUMMING_PREFLIGHT="${LOG}.humming-preflight.json"
if [[ "$M3_W4A8_BACKEND" == "humming" && "$_PRINT_CFG" -eq 0 ]]; then
  python -m pipeline.m3_humming_w4a8 preflight \
    --checkpoint "$MODEL_CKPT" \
    --out "$HUMMING_PREFLIGHT" || {
      echo "ERROR: Humming W4A8 preflight failed: $HUMMING_PREFLIGHT"
      exit 1
    }
fi

# Read-only observability for the RCA matrix: print effective env + argv, then exit
# without launching. Does not change default flags or patch behavior.
if [[ "${PRINT_EFFECTIVE_CONFIG:-0}" == "1" || "${PRINT_EFFECTIVE_CONFIG:-}" == "true" ]]; then
  echo "[http-smoke] PRINT_EFFECTIVE_CONFIG=1 — effective configuration (no launch)"
  echo "EFFECTIVE_ENV:"
  echo "  CKPT=$MODEL_CKPT"
  echo "  SERVED_NAME=$SERVED_NAME"
  echo "  HOST=$HOST"
  echo "  PORT=$PORT"
  echo "  TP=$TP"
  echo "  MAX_MODEL_LEN=$MAX_MODEL_LEN"
  echo "  GPU_UTIL=$GPU_UTIL"
  echo "  KV_CACHE_DTYPE=$KV_CACHE_DTYPE"
  echo "  BLOCK_SIZE=$BLOCK_SIZE"
  echo "  ENABLE_EP=$ENABLE_EP"
  echo "  M3_W4A8_BACKEND=$M3_W4A8_BACKEND"
  echo "  DISABLE_CUSTOM_AR=$DISABLE_CUSTOM_AR"
  echo "  LANGUAGE_MODEL_ONLY=$LANGUAGE_MODEL_ONLY"
  echo "  ENFORCE_EAGER=$ENFORCE_EAGER"
  echo "  DEBUG_CUDAGRAPH=$DEBUG_CUDAGRAPH"
  echo "  CUDA_LAUNCH_BLOCKING=${CUDA_LAUNCH_BLOCKING:-<unset>}"
  echo "  TORCH_USE_CUDA_DSA=${TORCH_USE_CUDA_DSA:-<unset>}"
  echo "  VLLM_DISABLE_SHARED_EXPERTS_STREAM=$VLLM_DISABLE_SHARED_EXPERTS_STREAM"
  echo "  LLMC_M3_CAPTURE_SYNC=$LLMC_M3_CAPTURE_SYNC"
  echo "  VLLM_USE_BREAKABLE_CUDAGRAPH=${VLLM_USE_BREAKABLE_CUDAGRAPH:-<unset>}"
  echo "  APPLY_M3_PATCHES=$APPLY_M3_PATCHES"
  echo "  PATCH_CKPT_CONFIG=$PATCH_CKPT_CONFIG"
  if [[ "$M3_W4A8_BACKEND" == "humming" ]]; then
    echo "  VLLM_HUMMING_USE_F16_ACCUM=$VLLM_HUMMING_USE_F16_ACCUM"
    echo "  VLLM_HUMMING_MOE_GEMM_TYPE=$VLLM_HUMMING_MOE_GEMM_TYPE"
    echo "  HUMMING_CACHE_DIR=$HUMMING_CACHE_DIR"
    echo "  HUMMING_NVRTC_LIB_DIR=$_NVRTC_LIB_DIR"
    echo "  HUMMING_PREFLIGHT=$HUMMING_PREFLIGHT"
  fi
  echo "  LOG=$LOG"
  echo "  PID_FILE=$PID_FILE"
  echo "  EXTRA_VLLM_ARGS=${EXTRA_VLLM_ARGS:-}"
  # shell-escaped argv for reproducibility
  printf 'EFFECTIVE_ARGV: vllm'
  for a in "${ARGS[@]}"; do
    printf ' %q' "$a"
  done
  printf '\n'
  exit 0
fi

python -c "import vllm; print('vllm', vllm.__version__)" 2>/dev/null || echo "vllm: not importable"
nvidia-smi --query-gpu=index,name,memory.total,memory.free --format=csv 2>/dev/null || true

nohup setsid vllm "${ARGS[@]}" >"$LOG" 2>&1 &
echo $! >"$PID_FILE"
echo "serve pid=$(cat "$PID_FILE")"
echo "  tail -f $LOG"
echo "  bash pipeline/slurm/smoke_chat_completions.sh"
echo "  kill \$(cat $PID_FILE)"
