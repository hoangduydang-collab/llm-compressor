#!/usr/bin/env bash
# One arm of the EAGLE3 speculative-decoding A/B (design: M3_SPECDEC_EAGLE3_PLAN.md).
#
# Target + kernel + topology are FIXED and identical to the 20260726T132617Z
# window's gptq-hum-idx-0110 arm (in-house GPTQ W4AFP8 ABI overlay, Humming
# indexed 0.1.10, TP8/EP8, MAX_MODEL_LEN=131072). The ONLY per-arm difference:
#   SPEC_K=0   control -- no --speculative-config at all
#   SPEC_K=n   eagle3 drafter with num_speculative_tokens=n
#
# Workload is the AA-style sweep only (natural output length: no ignore_eos, so
# acceptance is measured on real generations rather than forced continuations).
# Structure follows two_axis_perf_arm.sh; that script is left untouched so the
# window's provenance stays intact.
set -uo pipefail

ROOT=${ROOT:?}; ARM=${ARM:?}; RUN_TS=${RUN_TS:?}
CKPT=${CKPT:?}; PORT=${PORT:?}; SPEC_K=${SPEC_K:?}
DRAFTER=${DRAFTER:-/mnt/nfs/hoangduy/hf_assets/Inferact/MiniMax-M3-EAGLE3}
REPO=/mnt/nfs/hoangduy/projects/llm-compressor
BENCH=/mnt/nfs/hoangduy/projects/benchmarks
PERF_VENV=/mnt/nfs/hoangduy/venvs/perf      # aiperf 0.8.0 + analyzer deps
SERVED_NAME=${SERVED_NAME:-MiniMaxAI/MiniMax-M3}
TOKENIZER=/mnt/nfs/hoangduy/hf_assets/MiniMaxAI/MiniMax-M3
MAX_MODEL_LEN=${MAX_MODEL_LEN:-131072}
AA_INPUTS=${AA_INPUTS:-1k,10k}
AA_CONC=${AA_CONC:-1,10}
AA_PROFILE=${AA_PROFILE:-minimax-m3-specdec-$ARM}

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
note "host=$(hostname) spec_k=$SPEC_K run_ts=$RUN_TS mml=$MAX_MODEL_LEN aa=$AA_INPUTS x $AA_CONC"

# --- speculative config (the one variable of this experiment) -----------------
# Compact JSON on purpose: run_vllm_http_serve_smoke.sh word-splits
# EXTRA_VLLM_ARGS, so the payload must contain no spaces.
if [ "$SPEC_K" -gt 0 ]; then
  test -f "$DRAFTER/config.json" || { note "ABORT drafter missing: $DRAFTER"; exit 1; }
  export EXTRA_VLLM_ARGS="--speculative-config {\"method\":\"eagle3\",\"model\":\"$DRAFTER\",\"num_speculative_tokens\":$SPEC_K,\"attention_backend\":\"FLASH_ATTN\"}"
  note "spec-dec ON: eagle3 k=$SPEC_K drafter=$DRAFTER"
else
  note "control arm: no speculative config"
fi
printf '%s\n' "${EXTRA_VLLM_ARGS:-<none>}" > "$C/extra-vllm-args.txt"

start_serve_and_wait() {
  note "serve $CKPT on $PORT (max_model_len=$MAX_MODEL_LEN)"
  CKPT="$CKPT" SERVED_NAME="$SERVED_NAME" MODEL_ID="$TOKENIZER" PORT="$PORT" \
    MAX_MODEL_LEN="$MAX_MODEL_LEN" LOG="$C/serve.log" PID_FILE="$C/serve.pid" \
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

if ! start_serve_and_wait; then
  tail -80 "$C/serve.log" | tee -a "$C/client.log"
  exit 1
fi
export BASE_URL="http://localhost:$PORT"

# --- gate 1: Humming attestation (fail closed, no CUTLASS fallback) ----------
note "attest Humming backend"
( cd "$REPO" &&
  python -m pipeline.m3_humming_w4a8 attest \
    --preflight "$C/serve.log.humming-preflight.json" \
    --log "$C/serve.log" \
    --out "$C/backend-attestation.json"
) >>"$C/client.log" 2>&1
rc=$?
if [ "$rc" != 0 ]; then
  note "Humming attestation failed rc=$rc (fail closed)"
  stop_local_serve; exit 1
fi

# --- gate 2: spec-dec is really enabled --------------------------------------
# A "no gain" verdict produced by silently-disabled spec-dec is the one wrong
# conclusion this experiment must never emit, so it is a hard gate.
if [ "$SPEC_K" -gt 0 ]; then
  if grep -qi "num_speculative_tokens" "$C/serve.log"; then
    grep -i "speculative\|eagle" "$C/serve.log" | head -40 > "$C/spec-boot.log"
    note "spec config present in serve.log (see spec-boot.log)"
  else
    note "ABORT: serve.log shows no speculative config"
    stop_local_serve; exit 1
  fi
fi

# --- greedy-equivalence probe (correctness, cheap) ---------------------------
note "greedy probe (temp 0, 8 prompts)"
"$PYBIN" "$REPO/pipeline/specdec_greedy_probe.py" \
  --base-url "$BASE_URL" --model "$SERVED_NAME" --arm "$ARM" \
  --out "$C/greedy-probe.json" >>"$C/client.log" 2>&1 \
  || note "WARN greedy probe returned nonzero (see client.log)"

# --- AA-style sweep ----------------------------------------------------------
curl -sf "$BASE_URL/metrics" -o "$C/metrics-pre-aa.txt" 2>/dev/null || true
note "AA sweep (inputs $AA_INPUTS x conc $AA_CONC) against $BASE_URL"
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
rc_aa=$?
note "aa sweep rc=$rc_aa"
tail -8 "$C/aa-sweep.log" | tee -a "$C/client.log"
echo "$BENCH/results/$AA_PROFILE/self-hosted/perf/aa-sweep/$RUN_TS" > "$C/aa-results.path"
curl -sf "$BASE_URL/metrics" -o "$C/metrics-post-aa.txt" 2>/dev/null || true

# --- acceptance evidence -----------------------------------------------------
# vLLM's SpecDecodingLogging emits "Mean acceptance length: X, ... Per-position
# acceptance rate: a, b, c, ..." on the engine log cadence.
grep -i "acceptance\|SpecDecoding" "$C/serve.log" > "$C/spec-metrics.log" 2>/dev/null || true
if [ "$SPEC_K" -gt 0 ]; then
  if [ -s "$C/spec-metrics.log" ]; then
    note "acceptance lines captured: $(wc -l < "$C/spec-metrics.log")"
    tail -3 "$C/spec-metrics.log" | tee -a "$C/client.log"
  else
    note "WARN no acceptance lines in serve.log -- check /metrics deltas instead"
  fi
fi

stop_local_serve
note "arm done rc=$rc_aa"
exit "$rc_aa"
