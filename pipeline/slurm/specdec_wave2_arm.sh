#!/usr/bin/env bash
# One arm of EAGLE3 spec-dec wave 2 (design: M3_SPECDEC_EAGLE3_PLAN.md).
#
# Same fixed target/kernel/topology as wave 1; the only serve-level variable is
# SPEC_K (0 = control, 3 = eagle3 num_speculative_tokens=3). Wave 2 splits the
# workload across THREE phases, each on its own pair of serves so the phases run
# on parallel hardware -- they must NOT share one server, because a conc-64 load
# running beside a conc-1 latency cell would confound both.
#
#   PHASE=natural  ShareGPT real prompts, natural output, temp 0.6 then 0,
#                  conc 1 and 10          -> the production multiplier + how much
#                                            of wave 1's gap was temperature
#   PHASE=load     suite reasoning (1k in / 8k pinned out), temp 0.6,
#                  conc 16/32/64          -> where spec-dec stops paying
#   PHASE=lowconc  same shape, conc 1/4   -> like-for-like vs the two-axis tables
#
# Acceptance is captured per cell from /metrics deltas (cumulative counters), not
# only from the log cadence, so each cell gets its own acceptance number.
set -uo pipefail

ROOT=${ROOT:?}; ARM=${ARM:?}; RUN_TS=${RUN_TS:?}; PHASE=${PHASE:?}
CKPT=${CKPT:?}; PORT=${PORT:?}; SPEC_K=${SPEC_K:?}
DRAFTER=${DRAFTER:-/mnt/nfs/hoangduy/hf_assets/Inferact/MiniMax-M3-EAGLE3}
REPO=/mnt/nfs/hoangduy/projects/llm-compressor
BENCH=/mnt/nfs/hoangduy/projects/benchmarks
PERF_VENV=/mnt/nfs/hoangduy/venvs/perf
# aiperf caches public datasets at $CWD/.cache/aiperf/datasets -- ShareGPT is
# pre-staged inside the workspace (gitignored) so the arms need no network.
DATASET_DIR=${DATASET_DIR:-$REPO/artifacts/aiperf-datasets}
SERVED_NAME=${SERVED_NAME:-MiniMaxAI/MiniMax-M3}
TOKENIZER=/mnt/nfs/hoangduy/hf_assets/MiniMaxAI/MiniMax-M3
MAX_MODEL_LEN=${MAX_MODEL_LEN:-131072}
NAT_MAX_TOKENS=${NAT_MAX_TOKENS:-2048}

C=$ROOT/arm-$ARM
mkdir -p "$C" "$C/metrics"
note() { echo "[arm-$ARM $(date -u +%H:%M:%S)] $1" | tee -a "$C/client.log"; }
stop_local_serve() {
  note "stop serve"
  kill "$(cat "$C/serve.pid" 2>/dev/null)" 2>/dev/null || true
  sleep 15
  kill -9 -"$(cat "$C/serve.pid" 2>/dev/null)" 2>/dev/null || true
}
snap() { curl -sf "http://localhost:$PORT/metrics" -o "$C/metrics/$1.txt" 2>/dev/null || true; }

export HF_HOME=/mnt/nfs/hoangduy/cache/huggingface HF_DATASETS_OFFLINE=1 HF_HUB_OFFLINE=1
export PATH="$PERF_VENV/bin:$PATH"
export PYBIN="$PERF_VENV/bin/python"
note "host=$(hostname) phase=$PHASE spec_k=$SPEC_K run_ts=$RUN_TS"

if [ "$SPEC_K" -gt 0 ]; then
  test -f "$DRAFTER/config.json" || { note "ABORT drafter missing: $DRAFTER"; exit 1; }
  export EXTRA_VLLM_ARGS="--speculative-config {\"method\":\"eagle3\",\"model\":\"$DRAFTER\",\"num_speculative_tokens\":$SPEC_K,\"attention_backend\":\"FLASH_ATTN\"}"
  SPEC_LABEL="eagle3-k$SPEC_K"
else
  SPEC_LABEL="none"
fi
printf '%s\n' "${EXTRA_VLLM_ARGS:-<none>}" > "$C/extra-vllm-args.txt"

note "serve $CKPT on $PORT (max_model_len=$MAX_MODEL_LEN)"
CKPT="$CKPT" SERVED_NAME="$SERVED_NAME" MODEL_ID="$TOKENIZER" PORT="$PORT" \
  MAX_MODEL_LEN="$MAX_MODEL_LEN" LOG="$C/serve.log" PID_FILE="$C/serve.pid" \
  bash "$REPO/pipeline/slurm/run_vllm_http_serve_smoke.sh" >>"$C/client.log" 2>&1 \
  || { note "serve start rc=$?"; tail -60 "$C/serve.log" 2>/dev/null | tee -a "$C/client.log"; exit 1; }
ready=1
for _ in $(seq 1 540); do
  curl -sf "http://localhost:$PORT/v1/models" -o "$C/models.json" 2>/dev/null && { ready=0; break; }
  kill -0 "$(cat "$C/serve.pid" 2>/dev/null)" 2>/dev/null || { note "serve died"; break; }
  sleep 10
done
[ "$ready" = 0 ] || { tail -80 "$C/serve.log" | tee -a "$C/client.log"; exit 1; }
export BASE_URL="http://localhost:$PORT"

# --- gates (same fail-closed set as wave 1) ----------------------------------
( cd "$REPO" && python -m pipeline.m3_humming_w4a8 attest \
    --preflight "$C/serve.log.humming-preflight.json" --log "$C/serve.log" \
    --out "$C/backend-attestation.json" ) >>"$C/client.log" 2>&1 \
  || { note "Humming attestation failed (fail closed)"; stop_local_serve; exit 1; }
if [ "$SPEC_K" -gt 0 ]; then
  grep -qi "num_speculative_tokens" "$C/serve.log" \
    || { note "ABORT: serve.log shows no speculative config"; stop_local_serve; exit 1; }
  grep -i "speculative\|eagle" "$C/serve.log" | head -40 > "$C/spec-boot.log"
fi

rc_all=0

# --- phase: natural (ShareGPT real prompts, natural output) -------------------
if [ "$PHASE" = natural ]; then
  OUT=$C/natural
  # Local tokenizer paths have tripped aiperf's HF validation on dataset paths
  # before (see benchmarks run_perf_agentic.sh mooncake note); fall back to the
  # builtin tokenizer if that happens. Both arms use whichever wins, so the A/B
  # stays valid -- only absolute token counts would shift.
  TOK="$TOKENIZER"
  for temp in 0.6 0; do
    tag="t$(echo "$temp" | tr -d '.')"
    for conc in 1 10; do
      n=$([ "$conc" = 1 ] && echo 40 || echo 100)
      note "natural temp=$temp conc=$conc requests=$n tokenizer=$TOK"
      snap "natural-$tag-c$conc-pre"
      ( cd "$DATASET_DIR" && "$PERF_VENV/bin/aiperf" profile \
          --model "$SERVED_NAME" --url "$BASE_URL" --endpoint-type chat --streaming \
          --tokenizer "$TOK" --public-dataset sharegpt \
          --extra-inputs "{\"temperature\":$temp,\"max_tokens\":$NAT_MAX_TOKENS,\"chat_template_kwargs\":{\"enable_thinking\":true}}" \
          --concurrency "$conc" --request-count "$n" --warmup-request-count "$conc" \
          --artifact-dir "$OUT/$tag/conc_$conc" ) >>"$C/natural-$tag-c$conc.log" 2>&1
      rc=$?
      if [ "$rc" != 0 ] && [ "$TOK" != builtin ]; then
        note "aiperf rc=$rc with local tokenizer; retrying this cell with builtin"
        TOK=builtin
        ( cd "$DATASET_DIR" && "$PERF_VENV/bin/aiperf" profile \
            --model "$SERVED_NAME" --url "$BASE_URL" --endpoint-type chat --streaming \
            --tokenizer builtin --public-dataset sharegpt \
            --extra-inputs "{\"temperature\":$temp,\"max_tokens\":$NAT_MAX_TOKENS,\"chat_template_kwargs\":{\"enable_thinking\":true}}" \
            --concurrency "$conc" --request-count "$n" --warmup-request-count "$conc" \
            --artifact-dir "$OUT/$tag/conc_$conc" ) >>"$C/natural-$tag-c$conc.log" 2>&1
        rc=$?
      fi
      snap "natural-$tag-c$conc-post"
      note "natural temp=$temp conc=$conc rc=$rc"
      [ "$rc" = 0 ] || rc_all=1
    done
    "$PYBIN" "$BENCH/performance/workloads/analyze_perf.py" --run-dir "$OUT/$tag" \
      --mode "sharegpt_$tag" --label "$ARM" --precision W4AFP8 --gpu 8xH100 \
      --num-gpus 8 --spec-decode "$SPEC_LABEL" >>"$C/analyze-$tag.log" 2>&1 \
      || note "WARN analyze_perf failed for $tag"
  done
  echo "$TOK" > "$C/natural-tokenizer.txt"
fi

# --- phases: load / lowconc (suite reasoning, pinned 8k output) --------------
if [ "$PHASE" = load ] || [ "$PHASE" = lowconc ]; then
  CONCS=$([ "$PHASE" = load ] && echo "16 32 64" || echo "1 4")
  for conc in $CONCS; do
    note "reasoning conc=$conc (1k in / 8k pinned out, temp 0.6)"
    snap "reasoning-c$conc-pre"
    ( cd "$BENCH" &&
      PROFILE=minimax-m3-inhouse ENGINE=vllm \
      M3_ARM="specdec-w2-$PHASE-k$SPEC_K" MODEL_PATH="$CKPT" \
      QUANT_RECIPE="gptq-w4afp8-humming-indexed-spec-$SPEC_LABEL" \
      BASE_URL="$BASE_URL" RUN_TS="$RUN_TS" CONC_REASONING="$conc" \
      bash performance/workloads/run_perf_reasoning.sh ) >>"$C/reasoning-c$conc.log" 2>&1
    rc=$?
    snap "reasoning-c$conc-post"
    note "reasoning conc=$conc rc=$rc"
    [ "$rc" = 0 ] || rc_all=1
  done
  echo "$BENCH/results/minimax-m3-inhouse-specdec-w2-$PHASE-k$SPEC_K/vllm/perf/reasoning/$RUN_TS" \
    > "$C/reasoning-results.path"
fi

grep -i "acceptance\|SpecDecoding" "$C/serve.log" > "$C/spec-metrics.log" 2>/dev/null || true
stop_local_serve
note "arm done rc=$rc_all"
exit "$rc_all"
