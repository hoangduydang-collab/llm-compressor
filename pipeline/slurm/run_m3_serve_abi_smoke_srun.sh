#!/usr/bin/env bash
# Single-node serve ABI smoke for an M3 quant checkpoint (tmux controller).
#
# Wraps run_vllm_http_serve_smoke.sh for srun use: that script detaches
# ``vllm serve`` and exits, which under srun would end the step and kill the
# server. This controller serves, polls readiness, runs the chat smoke, greps
# the fail-closed markers, then shuts the server down and reports
# ABI_SMOKE_RC=<rc>.
#
#   CKPT=<checkpoint-vllm-w123> TAG=r8 \
#     bash pipeline/slurm/run_m3_serve_abi_smoke_srun.sh
#
# r7 gate-alpha checkpoints: also set
#   M3_GATE_ALPHA_SIDECAR=$CKPT/gate_smooth_scale_sidecar.pt
# The smoke then additionally REQUIRES the worker bind marker
# ("M3 gate-alpha: bound N/N MoE layers") in the serve log — a load that
# silently skipped the per-channel swiglu is a FAIL, not a pass.
set -uo pipefail

REPO=/mnt/nfs/hoangduy/projects/llm-compressor
CKPT="${CKPT:?set CKPT to the checkpoint-vllm-w123 to smoke}"
TAG="${TAG:?set TAG (e.g. r7, r8) for log naming}"
PORT="${PORT:-8010}"
SERVED_NAME="${SERVED_NAME:-m3-abi-smoke-$TAG}"
READY_MAX_S="${READY_MAX_S:-3600}"
LOG_DIR=/mnt/nfs/hoangduy/logs/m3-serve-abi-smoke
mkdir -p "$LOG_DIR"
TS=$(date -u +%Y%m%dT%H%M%SZ)
SRV_LOG="$LOG_DIR/$TS-$TAG-serve.log"
PID_FILE="$LOG_DIR/$TS-$TAG-serve.pid"

node_main() {
  set -uo pipefail
  cd "$REPO"
  rc=1
  CKPT="$CKPT" SERVED_NAME="$SERVED_NAME" PORT="$PORT" \
  LOG="$SRV_LOG" PID_FILE="$PID_FILE" \
    bash pipeline/slurm/run_vllm_http_serve_smoke.sh || {
    echo "[abi-smoke] serve launcher failed"
    echo "ABI_SMOKE_RC=1"
    return 1
  }

  echo "[abi-smoke] polling readiness (max ${READY_MAX_S}s)"
  waited=0
  until curl -sf "http://127.0.0.1:$PORT/health" >/dev/null 2>&1; do
    if ! kill -0 "$(cat "$PID_FILE")" 2>/dev/null; then
      echo "[abi-smoke] server process died during load"
      tail -40 "$SRV_LOG"
      echo "ABI_SMOKE_RC=1"
      return 1
    fi
    sleep 15
    waited=$((waited + 15))
    if [ "$waited" -ge "$READY_MAX_S" ]; then
      echo "[abi-smoke] readiness timeout"
      echo "ABI_SMOKE_RC=1"
      return 1
    fi
  done
  echo "[abi-smoke] READY after ${waited}s"

  if [ -n "${M3_GATE_ALPHA_SIDECAR:-}" ]; then
    if grep -q "M3 gate-alpha: bound" "$SRV_LOG"; then
      grep "M3 gate-alpha" "$SRV_LOG" | tail -3
    else
      echo "[abi-smoke] FAIL: sidecar set but no gate-alpha bind marker in log"
      echo "ABI_SMOKE_RC=1"
      return 1
    fi
  fi

  if MODEL="$SERVED_NAME" PORT="$PORT" PROMPT="What is 2+2? Answer briefly." \
     MAX_TOKENS=64 bash pipeline/slurm/smoke_chat_completions.sh; then
    rc=0
  else
    echo "[abi-smoke] chat smoke FAILED"
    rc=1
  fi

  # Longer generation probe: quant-garbage checkpoints often pass a 1-token
  # factoid but degenerate over hundreds of tokens.
  if [ "$rc" = 0 ]; then
    if ! MODEL="$SERVED_NAME" PORT="$PORT" \
       PROMPT="Explain, step by step, why the sky is blue." MAX_TOKENS=512 \
       bash pipeline/slurm/smoke_chat_completions.sh; then
      echo "[abi-smoke] long-generation smoke FAILED"
      rc=1
    fi
  fi

  kill "$(cat "$PID_FILE")" 2>/dev/null
  echo "ABI_SMOKE_RC=$rc"
  return "$rc"
}

if [ "${1:-}" = "--node" ]; then
  node_main
  exit $?
fi

if [ -n "${SLURM_JOB_ID:-}" ]; then
  echo "ERROR: run this controller outside an allocation; it owns top-level srun" >&2
  exit 2
fi

NODE_OPT=()
[ -n "${NODELIST:-}" ] && NODE_OPT=(--nodelist="$NODELIST")
[ -n "${SMOKE_PARTITION:-}" ] && NODE_OPT+=(--partition="$SMOKE_PARTITION")
srun --exclusive "${NODE_OPT[@]}" --nodes=1 --ntasks=1 --gres=gpu:8 \
     --cpus-per-task=192 --time=4:00:00 --kill-on-bad-exit=1 \
     --job-name="m3-abi-smoke-$TAG" --export=ALL \
     bash "$REPO/pipeline/slurm/run_m3_serve_abi_smoke_srun.sh" --node
rc=$?
echo "CONTROLLER_RC=$rc"
