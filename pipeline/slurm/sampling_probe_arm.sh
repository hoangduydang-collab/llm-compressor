#!/usr/bin/env bash
# One arm of the sampling-sensitivity probe (see pipeline/sampling_probe.py).
# Two modes:
#   MODE=local  : serve $CKPT on $PORT (single node, smoke serve), wait ready,
#                 run the probe driver against localhost, stop serve.
#   MODE=remote : do NOT serve; wait for $ROOT/bf16/ready, read endpoint-ip,
#                 run the probe driver against the shared 2-node BF16 endpoint.
# The driver re-runs the tok64k GPQA exhausted docs (+ a terminated control)
# under greedy (temp 0) and sampled (temp 1.0 / top_p 0.95) at a FIXED 32k
# budget, writing $ROOT/sampling/$ARM.jsonl.
#
# Env: ROOT ARM MODE SAMPLES_GLOB  (local also: CKPT PORT)  [N_SAMPLES etc pass through]
set -uo pipefail

ROOT=${ROOT:?}; ARM=${ARM:?}; MODE=${MODE:?}; SAMPLES_GLOB=${SAMPLES_GLOB:?}
REPO=/mnt/nfs/hoangduy/projects/llm-compressor
BVENV=/mnt/nfs/hoangduy/venvs/benchmarks
SERVED_NAME=MiniMaxAI/MiniMax-M3
TOKENIZER=/mnt/nfs/hoangduy/hf_assets/MiniMaxAI/MiniMax-M3
MAX_MODEL_LEN=${MAX_MODEL_LEN:-40960}   # short GPQA prompts + 32k gen budget

C=$ROOT/probe-$ARM
mkdir -p "$C" "$ROOT/sampling"
note() { echo "[probe-$ARM $(date -u +%H:%M:%S)] $1" | tee -a "$C/client.log"; }

export HF_HOME=/mnt/nfs/hoangduy/cache/huggingface HF_DATASETS_OFFLINE=1 HF_HUB_OFFLINE=1
note "host=$(hostname) mode=$MODE"

if [ "$MODE" = local ]; then
  CKPT=${CKPT:?}; PORT=${PORT:?}
  note "serve $CKPT on $PORT (max_model_len=$MAX_MODEL_LEN)"
  CKPT="$CKPT" SERVED_NAME="$SERVED_NAME" MODEL_ID="$TOKENIZER" PORT="$PORT" \
    MAX_MODEL_LEN="$MAX_MODEL_LEN" LOG="$C/serve.log" PID_FILE="$C/serve.pid" \
    bash "$REPO/pipeline/slurm/run_vllm_http_serve_smoke.sh" >>"$C/client.log" 2>&1
  rc=$?; [ "$rc" = 0 ] || { note "serve start rc=$rc"; exit 1; }
  ready=1
  for _ in $(seq 1 540); do
    curl -sf "http://localhost:$PORT/v1/models" -o "$C/models.json" 2>/dev/null && { ready=0; break; }
    kill -0 "$(cat "$C/serve.pid" 2>/dev/null)" 2>/dev/null || { note "serve died"; break; }
    sleep 10
  done
  [ "$ready" = 0 ] || { tail -60 "$C/serve.log" | tee -a "$C/client.log"; exit 1; }
  BASE_URL="http://localhost:$PORT"
else
  note "waiting for BF16 endpoint (max 2h)"
  bf16=1
  for _ in $(seq 1 720); do [ -f "$ROOT/bf16/ready" ] && { bf16=0; break; }; sleep 10; done
  [ "$bf16" = 0 ] || { note "bf16 never ready"; exit 1; }
  BASE_URL="http://$(cat "$ROOT/bf16/endpoint-ip"):${BF16_PORT:-8001}"
  curl -sf "$BASE_URL/v1/models" -o "$C/models.json" || { note "bf16 unreachable"; exit 1; }
fi

note "run probe driver against $BASE_URL"
SAMPLES_GLOB="$SAMPLES_GLOB" BASE_URL="$BASE_URL" SERVED_NAME="$SERVED_NAME" \
  OUT="$ROOT/sampling/$ARM.jsonl" \
  N_SAMPLES="${N_SAMPLES:-5}" MAX_TOKENS="${MAX_TOKENS:-32768}" \
  N_CONTROL="${N_CONTROL:-25}" CONCURRENCY="${CONCURRENCY:-24}" \
  REQUEST_TIMEOUT_S="${REQUEST_TIMEOUT_S:-3600}" \
  "$BVENV/bin/python" "$REPO/pipeline/sampling_probe.py" >>"$C/driver.log" 2>&1
rc=$?; note "driver rc=$rc"; tail -5 "$C/driver.log" | tee -a "$C/client.log"

if [ "$MODE" = local ]; then
  note "stop serve"
  kill "$(cat "$C/serve.pid" 2>/dev/null)" 2>/dev/null || true
  sleep 15
  kill -9 -"$(cat "$C/serve.pid" 2>/dev/null)" 2>/dev/null || true
fi
exit "$rc"
