#!/usr/bin/env bash
# One arm of the two-axis perf data-completion window (M3_TWO_AXIS_PERF_PLAN.md):
# serve ONCE, then run the requested workloads against that single server.
#   WORKLOADS=suite,aa (default) : suite-native primary path, then AA sweep
#   WORKLOADS=aa                 : AA sweep only (axis-1 kernel arms)
# Modes as perf_eval_arm.sh:
#   MODE=local  : smoke-serve $CKPT on $PORT on this node.
#   MODE=remote : wait for the controller's shared 2-node BF16 endpoint.
# The AA matrix ($AA_INPUTS x $AA_CONC, default 1k,10k,100k x 1,10) needs
# ~117k context for the 100k cells, hence MAX_MODEL_LEN=131072 default. If the
# serve cannot come up there and FALLBACK_MML is set (mxfp8/cyankiwi arms),
# retry once at the fallback ctx and drop the 100k cells (recorded in
# aa-inputs.effective).
set -uo pipefail

ROOT=${ROOT:?}; ARM=${ARM:?}; MODE=${MODE:?}; RUN_TS=${RUN_TS:?}
WORKLOADS=${WORKLOADS:-suite,aa}
REPO=/mnt/nfs/hoangduy/projects/llm-compressor
BENCH=/mnt/nfs/hoangduy/projects/benchmarks
PERF_VENV=/mnt/nfs/hoangduy/venvs/perf     # aiperf 0.8.0 + analyzers' deps
SERVED_NAME=${SERVED_NAME:-MiniMaxAI/MiniMax-M3}
TOKENIZER=/mnt/nfs/hoangduy/hf_assets/MiniMaxAI/MiniMax-M3
MAX_MODEL_LEN=${MAX_MODEL_LEN:-131072}
FALLBACK_MML=${FALLBACK_MML:-}
AA_INPUTS=${AA_INPUTS:-1k,10k,100k}
AA_CONC=${AA_CONC:-1,10}
AA_PROFILE=${AA_PROFILE:-minimax-m3-inhouse-$ARM}

C=$ROOT/arm-$ARM
mkdir -p "$C"
note() { echo "[arm-$ARM $(date -u +%H:%M:%S)] $1" | tee -a "$C/client.log"; }
stop_local_serve() {
  note "stop serve"
  kill "$(cat "$C/serve.pid" 2>/dev/null)" 2>/dev/null || true
  sleep 15
  kill -9 -"$(cat "$C/serve.pid" 2>/dev/null)" 2>/dev/null || true
}

export HF_HOME=/mnt/nfs/hoangduy/cache/huggingface HF_DATASETS_OFFLINE=1 HF_HUB_OFFLINE=1
export PATH="$PERF_VENV/bin:$PATH"
export PYBIN="$PERF_VENV/bin/python"
note "host=$(hostname) mode=$MODE workloads=$WORKLOADS run_ts=$RUN_TS mml=$MAX_MODEL_LEN"

start_serve_and_wait() {  # $1 = max_model_len; 0 = ready
  local mml=$1
  note "serve $CKPT on $PORT (max_model_len=$mml)"
  CKPT="$CKPT" SERVED_NAME="$SERVED_NAME" MODEL_ID="$TOKENIZER" PORT="$PORT" \
    MAX_MODEL_LEN="$mml" LOG="$C/serve.log" PID_FILE="$C/serve.pid" \
    bash "$REPO/pipeline/slurm/run_vllm_http_serve_smoke.sh" >>"$C/client.log" 2>&1
  local rc=$?; [ "$rc" = 0 ] || { note "serve start rc=$rc"; return 1; }
  for _ in $(seq 1 540); do
    curl -sf "http://localhost:$PORT/v1/models" -o "$C/models.json" 2>/dev/null && return 0
    kill -0 "$(cat "$C/serve.pid" 2>/dev/null)" 2>/dev/null || { note "serve died"; return 1; }
    sleep 10
  done
  note "serve never became ready"
  return 1
}

if [ "$MODE" = local ]; then
  CKPT=${CKPT:?}; PORT=${PORT:?}
  if ! start_serve_and_wait "$MAX_MODEL_LEN"; then
    tail -60 "$C/serve.log" | tee -a "$C/client.log"
    if [ -n "$FALLBACK_MML" ]; then
      note "FALLBACK: retry at max_model_len=$FALLBACK_MML; 100k AA cells dropped"
      stop_local_serve
      AA_INPUTS=$(echo "$AA_INPUTS" | tr ',' '\n' | grep -v '^100k$' | paste -sd,)
      mv "$C/serve.log" "$C/serve.log.failed-$MAX_MODEL_LEN" 2>/dev/null || true
      start_serve_and_wait "$FALLBACK_MML" || {
        tail -60 "$C/serve.log" | tee -a "$C/client.log"; exit 1; }
    else
      exit 1
    fi
  fi
  if [ "${M3_W4A8_BACKEND:-cutlass}" = humming ]; then
    note "attest Humming backend before benchmarks"
    (
      cd "$REPO" &&
      python -m pipeline.m3_humming_w4a8 attest \
        --preflight "$C/serve.log.humming-preflight.json" \
        --log "$C/serve.log" \
        --out "$C/backend-attestation.json"
    ) >>"$C/client.log" 2>&1
    rc=$?
    if [ "$rc" != 0 ]; then
      note "Humming backend attestation failed rc=$rc (fail closed, no fallback)"
      stop_local_serve
      exit 1
    fi
  fi
  export BASE_URL="http://localhost:$PORT"
else
  note "waiting for BF16 endpoint (max 2h)"
  bf16=1
  for _ in $(seq 1 720); do [ -f "$ROOT/bf16/ready" ] && { bf16=0; break; }; sleep 10; done
  [ "$bf16" = 0 ] || { note "bf16 never ready"; exit 1; }
  export BASE_URL="http://$(cat "$ROOT/bf16/endpoint-ip"):${BF16_PORT:-8001}"
  curl -sf "$BASE_URL/v1/models" -o "$C/models.json" || { note "bf16 unreachable"; exit 1; }
fi

rc_all=0

case ",$WORKLOADS," in *,suite,*)
  PROFILE=${PROFILE:?suite workload needs PROFILE}
  # run_performance.sh runs the preflight gate itself; PERF_STRICT=1 makes any
  # workflow failure a nonzero exit (run_all.sh would swallow it — never use it).
  note "primary suite (preflight + workflows) against $BASE_URL profile=$PROFILE"
  ( cd "$BENCH" &&
    PROFILE="$PROFILE" ENGINE=vllm PERF_STRICT=1 \
      bash performance/scripts/run_performance.sh ) >"$C/suite.log" 2>&1
  rc=$?
  note "suite rc=$rc"
  tail -8 "$C/suite.log" | tee -a "$C/client.log"
  echo "$BENCH/results/$PROFILE/vllm/perf" > "$C/suite-results.path"
  [ "$rc" = 0 ] || rc_all=1     # still run AA below: more evidence, same serve
;; esac

echo "$AA_INPUTS" > "$C/aa-inputs.effective"
note "AA sweep (inputs $AA_INPUTS x conc $AA_CONC) against $BASE_URL profile=$AA_PROFILE"
( cd "$BENCH" &&
  "$PERF_VENV/bin/python" -m performance.aa.run_aa_sweep \
    --deployment self-hosted \
    --model "$SERVED_NAME" \
    --tokenizer "$TOKENIZER" \
    --base-url "$BASE_URL" \
    --inputs "$AA_INPUTS" \
    --concurrency "$AA_CONC" \
    --profile "$AA_PROFILE" \
    --timestamp "$RUN_TS" \
    --aiperf-bin "$PERF_VENV/bin/aiperf" ) >"$C/aa-sweep.log" 2>&1
rc=$?
note "aa sweep rc=$rc"
tail -8 "$C/aa-sweep.log" | tee -a "$C/client.log"
echo "$BENCH/results/$AA_PROFILE/self-hosted/perf/aa-sweep/$RUN_TS" > "$C/aa-results.path"
[ "$rc" = 0 ] || rc_all=1

if [ "$MODE" = local ]; then
  stop_local_serve
fi
note "arm done rc=$rc_all"
exit "$rc_all"
