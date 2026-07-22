#!/usr/bin/env bash
# One arm of the official PERFORMANCE eval run (aiperf suite in the benchmarks
# repo: preflight gate + AA reasoning + agentic warm/cold; nonreasoning
# self-skips — M3 profiles lack that capability). Two modes:
#   MODE=local  : serve $CKPT on $PORT (single node, smoke serve), wait ready,
#                 run the perf suite against localhost, stop serve.
#   MODE=remote : do NOT serve; wait for $ROOT/bf16/ready, read endpoint-ip,
#                 run the perf suite against the shared 2-node BF16 endpoint.
#
# The suite writes results under $BENCH/results/$PROFILE/vllm/perf/*/$RUN_TS.
# Env: ROOT ARM MODE PROFILE  (local also: CKPT PORT)  RUN_TS shared from the
# controller so all arms land under one timestamp.
set -uo pipefail

ROOT=${ROOT:?}; ARM=${ARM:?}; MODE=${MODE:?}; PROFILE=${PROFILE:?}
REPO=/mnt/nfs/hoangduy/projects/llm-compressor
BENCH=/mnt/nfs/hoangduy/projects/benchmarks
PERF_VENV=/mnt/nfs/hoangduy/venvs/perf     # aiperf 0.8.0 + analyzers' deps
SERVED_NAME=MiniMaxAI/MiniMax-M3
TOKENIZER=/mnt/nfs/hoangduy/hf_assets/MiniMaxAI/MiniMax-M3
# 40960 covers the deepest agentic warm turn (~18.5k prefix) + reasoning 1k+8k.
MAX_MODEL_LEN=${MAX_MODEL_LEN:-40960}

C=$ROOT/perf-$ARM
mkdir -p "$C"
note() { echo "[perf-$ARM $(date -u +%H:%M:%S)] $1" | tee -a "$C/client.log"; }

export HF_HOME=/mnt/nfs/hoangduy/cache/huggingface HF_DATASETS_OFFLINE=1 HF_HUB_OFFLINE=1
export PATH="$PERF_VENV/bin:$PATH"
export PYBIN="$PERF_VENV/bin/python"
note "host=$(hostname) mode=$MODE profile=$PROFILE run_ts=${RUN_TS:-unset}"

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
  export BASE_URL="http://localhost:$PORT"
else
  note "waiting for BF16 endpoint (max 2h)"
  bf16=1
  for _ in $(seq 1 720); do [ -f "$ROOT/bf16/ready" ] && { bf16=0; break; }; sleep 10; done
  [ "$bf16" = 0 ] || { note "bf16 never ready"; exit 1; }
  export BASE_URL="http://$(cat "$ROOT/bf16/endpoint-ip"):${BF16_PORT:-8001}"
  curl -sf "$BASE_URL/v1/models" -o "$C/models.json" || { note "bf16 unreachable"; exit 1; }
fi

# run_performance.sh runs the preflight gate itself (aborts on fail) and with
# PERF_STRICT=1 propagates workflow failures as a nonzero exit. (run_all.sh
# would swallow that rc — do not use it here.)
# AGENTIC_ONLY=1: refresh mode — preflight gate, then ONLY the agentic
# workload (used to re-measure under a new AG_* shape without repeating the
# ~2h reasoning sweep).
cd "$BENCH"
if [ "${AGENTIC_ONLY:-0}" = 1 ]; then
  note "AGENTIC-ONLY refresh (preflight + agentic) against $BASE_URL"
  PROFILE="$PROFILE" ENGINE=vllm bash performance/scripts/preflight.sh >"$C/suite.log" 2>&1 &&
    PROFILE="$PROFILE" ENGINE=vllm bash performance/workloads/run_perf_agentic.sh >>"$C/suite.log" 2>&1
  rc=$?
else
  note "perf suite (preflight + workflows) against $BASE_URL"
  PROFILE="$PROFILE" ENGINE=vllm PERF_STRICT=1 \
    bash performance/scripts/run_performance.sh >"$C/suite.log" 2>&1
  rc=$?
fi
note "suite rc=$rc"
tail -8 "$C/suite.log" | tee -a "$C/client.log"
echo "$BENCH/results/$PROFILE/vllm/perf" > "$C/results.path"

if [ "$MODE" = local ]; then
  note "stop serve"
  kill "$(cat "$C/serve.pid" 2>/dev/null)" 2>/dev/null || true
  sleep 15
  kill -9 -"$(cat "$C/serve.pid" 2>/dev/null)" 2>/dev/null || true
fi
exit "$rc"
