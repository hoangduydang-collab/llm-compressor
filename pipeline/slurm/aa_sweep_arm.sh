#!/usr/bin/env bash
# One arm of the AA-style perf sweep (benchmarks performance/aa/run_aa_sweep.py)
# against a locally-served checkpoint. Modeled on perf_eval_arm.sh MODE=local --
# same smoke serve + readiness wait + Humming attestation -- but instead of the
# suite-native workflows it runs the AA matrix: input length (1k/10k) x
# concurrency (1/10), temp 0.6, thinking on, NATURAL output length (the AA
# answer-token floor is checked, not forced).
#
# Results land under the same per-arm profile namespace the suite uses:
#   $BENCH/results/minimax-m3-inhouse-$ARM/self-hosted/perf/aa-sweep/$RUN_TS
# Env: ROOT ARM CKPT PORT RUN_TS (+ the humming arm env from the controller:
# PYTHONPATH=<site>:<repo>, VLLM_HUMMING_MOE_GEMM_TYPE, M3_W4A8_BACKEND, ...).
set -uo pipefail

ROOT=${ROOT:?}; ARM=${ARM:?}; CKPT=${CKPT:?}; PORT=${PORT:?}; RUN_TS=${RUN_TS:?}
REPO=/mnt/nfs/hoangduy/projects/llm-compressor
BENCH=/mnt/nfs/hoangduy/projects/benchmarks
PERF_VENV=/mnt/nfs/hoangduy/venvs/perf
SERVED_NAME=MiniMaxAI/MiniMax-M3
TOKENIZER=/mnt/nfs/hoangduy/hf_assets/MiniMaxAI/MiniMax-M3
MAX_MODEL_LEN=${MAX_MODEL_LEN:-40960}

C=$ROOT/aa-$ARM
mkdir -p "$C"
note() { echo "[aa-$ARM $(date -u +%H:%M:%S)] $1" | tee -a "$C/client.log"; }
stop_local_serve() {
  note "stop serve"
  kill "$(cat "$C/serve.pid" 2>/dev/null)" 2>/dev/null || true
  sleep 15
  kill -9 -"$(cat "$C/serve.pid" 2>/dev/null)" 2>/dev/null || true
}

export HF_HOME=/mnt/nfs/hoangduy/cache/huggingface HF_DATASETS_OFFLINE=1 HF_HUB_OFFLINE=1
export PATH="$PERF_VENV/bin:$PATH"
note "host=$(hostname) arm=$ARM run_ts=$RUN_TS"

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

if [ "${M3_W4A8_BACKEND:-cutlass}" = humming ]; then
  note "attest Humming backend before sweep"
  (
    cd "$REPO" &&
    python -m pipeline.m3_humming_w4a8 attest \
      --preflight "$C/serve.log.humming-preflight.json" \
      --log "$C/serve.log" \
      --out "$C/backend-attestation.json"
  ) >>"$C/client.log" 2>&1
  rc=$?
  if [ "$rc" != 0 ]; then
    note "Humming backend attestation failed rc=$rc"
    stop_local_serve
    exit 1
  fi
fi

note "AA sweep (self-hosted, inputs 1k,10k, conc 1,10) against http://localhost:$PORT"
cd "$BENCH"
"$PERF_VENV/bin/python" -m performance.aa.run_aa_sweep \
  --deployment self-hosted \
  --model "$SERVED_NAME" \
  --tokenizer "$TOKENIZER" \
  --base-url "http://localhost:$PORT" \
  --profile "minimax-m3-inhouse-$ARM" \
  --timestamp "$RUN_TS" \
  --aiperf-bin "$PERF_VENV/bin/aiperf" \
  >"$C/aa-sweep.log" 2>&1
rc=$?
note "aa sweep rc=$rc"
tail -8 "$C/aa-sweep.log" | tee -a "$C/client.log"
echo "$BENCH/results/minimax-m3-inhouse-$ARM/self-hosted/perf/aa-sweep/$RUN_TS" > "$C/results.path"

stop_local_serve
exit "$rc"
