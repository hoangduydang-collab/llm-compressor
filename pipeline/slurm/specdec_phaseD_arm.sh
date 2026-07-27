#!/usr/bin/env bash
# One arm of EAGLE3 spec-dec phase D -- SPEED-Bench length x entropy sweep.
# Design: M3_SPECDEC_EAGLE3_PLAN.md ("Phase D").
#
# Question: does the conc-1 speedup change with LONGER natural prompts? Wave 1
# measured the length axis only on synthetic random tokens (1.72x at 1k, 1.75x at
# 10k -- flat), and phase A measured natural prompts only at ~227 tokens (1.81x).
# Neither can show the mechanism that would actually move the number: when the
# output copies/quotes a long natural context, drafting gets easy.
#
# Instrument: nvidia/SPEED-Bench, built for exactly this ("across diverse semantic
# domains and realistic serving regimes ... acceptance-rate characteristics and
# end-to-end throughput"). Its throughput splits are fixed-ISL buckets (1k/8k/32k)
# crossed with entropy tiers, so length and content vary independently:
#   low_entropy  = code and sorting        -> copy-heavy, spec-dec best case
#   high_entropy = creative writing        -> spec-dec worst case
# The `mixed` tier is dropped: 512/512 masked in the public release.
#
# Prompts are pre-staged clean subsets (pipeline/stage_speedbench.py) because
# aiperf's SpeedBenchLoader does not filter the release's masked placeholders.
# Output is NATURAL (no ignore_eos) -- the pinned-8k shape inflates acceptance
# +33% (measured in wave 2 phase B/C) and would answer the wrong question.
set -uo pipefail

ROOT=${ROOT:?}; ARM=${ARM:?}; RUN_TS=${RUN_TS:?}
CKPT=${CKPT:?}; PORT=${PORT:?}; SPEC_K=${SPEC_K:?}
DRAFTER=${DRAFTER:-/mnt/nfs/hoangduy/hf_assets/Inferact/MiniMax-M3-EAGLE3}
REPO=/mnt/nfs/hoangduy/projects/llm-compressor
BENCH=/mnt/nfs/hoangduy/projects/benchmarks
PERF_VENV=/mnt/nfs/hoangduy/venvs/perf
SB_DIR=${SB_DIR:-$REPO/artifacts/aiperf-datasets/speedbench}
SERVED_NAME=${SERVED_NAME:-MiniMaxAI/MiniMax-M3}
TOKENIZER=/mnt/nfs/hoangduy/hf_assets/MiniMaxAI/MiniMax-M3
MAX_MODEL_LEN=${MAX_MODEL_LEN:-131072}
MAX_TOKENS=${MAX_TOKENS:-2048}
TEMP=${TEMP:-0.6}

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
note "host=$(hostname) spec_k=$SPEC_K run_ts=$RUN_TS temp=$TEMP max_tokens=$MAX_TOKENS"

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

# --- gates (same fail-closed set as waves 1 and 2) ----------------------------
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
OUT=$C/speedbench

# $1 cell (matches <bucket>-<tier>.jsonl)  $2 concurrency  $3 request count
run_cell() {
  local cell=$1 conc=$2 n=$3
  local file="$SB_DIR/$cell.jsonl"
  test -s "$file" || { note "ABORT: missing staged prompts $file"; rc_all=1; return; }
  note "speedbench cell=$cell conc=$conc requests=$n"
  snap "sb-$cell-c$conc-pre"
  # --random-seed is fixed so both arms draw the SAME prompts in the same order.
  "$PERF_VENV/bin/aiperf" profile \
      --model "$SERVED_NAME" --url "$BASE_URL" --endpoint-type chat --streaming \
      --tokenizer "$TOKENIZER" \
      --custom-dataset-type single_turn --input-file "$file" \
      --extra-inputs "{\"temperature\":$TEMP,\"max_tokens\":$MAX_TOKENS,\"chat_template_kwargs\":{\"enable_thinking\":true}}" \
      --random-seed 42 \
      --concurrency "$conc" --request-count "$n" --warmup-request-count "$conc" \
      --artifact-dir "$OUT/$cell/conc_$conc" >>"$C/sb-$cell-c$conc.log" 2>&1
  local rc=$?
  snap "sb-$cell-c$conc-post"
  note "speedbench cell=$cell conc=$conc rc=$rc"
  [ "$rc" = 0 ] || rc_all=1
}

# Length x entropy at conc 1 (the latency tier this decision is about).
for cell in 1k-low 1k-high 8k-low 8k-high 32k-low 32k-high; do
  run_cell "$cell" 1 40
done
# Same crossing at conc 10; 32k is skipped at load to bound KV and wall clock.
for cell in 1k-low 1k-high 8k-low 8k-high; do
  run_cell "$cell" 10 100
done

for cell in 1k-low 1k-high 8k-low 8k-high 32k-low 32k-high; do
  [ -d "$OUT/$cell" ] || continue
  "$PYBIN" "$BENCH/performance/workloads/analyze_perf.py" --run-dir "$OUT/$cell" \
    --mode "speedbench_$cell" --label "$ARM" --precision W4AFP8 --gpu 8xH100 \
    --num-gpus 8 --spec-decode "$SPEC_LABEL" >>"$C/analyze-$cell.log" 2>&1 \
    || note "WARN analyze_perf failed for $cell"
done

grep -i "acceptance\|SpecDecoding" "$C/serve.log" > "$C/spec-metrics.log" 2>/dev/null || true
stop_local_serve
note "arm done rc=$rc_all"
exit "$rc_all"
