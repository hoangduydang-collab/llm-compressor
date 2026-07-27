#!/usr/bin/env bash
# One arm of the EAGLE3 drafter-compatibility test -- M3_SPECDEC_EAGLE3_PLAN.md
# ("Phase E: drafter x target format").
#
# Question: is the Inferact EAGLE3 drafter as compatible with OUR aggressive 4-bit
# W4AFP8 target as it is with the vendor's mild 8-bit MXFP8 target? The drafter was
# trained against the original model and consumes the TARGET's hidden states, so a
# quantization that perturbs those states can cost acceptance twice over -- once by
# shifting the verify distribution, once by feeding the drafter off-distribution
# input. Acceptance length and per-position rates measure exactly that.
#
# Everything but the weight format is held constant, matching the phase D window
# flag-for-flag so its W4AFP8 8k cells serve as the comparison arm: same staged
# SPEED-Bench prompts (hash-gated, same bytes), same --random-seed 42, temp 0.6,
# max_tokens 2048, TP8/EP8 on one node, max_model_len 131072, kv_cache_dtype=fp8,
# block_size 128, gpu_util 0.9, disable_custom_all_reduce, language_model_only.
# Verified against phase D's serve banner: the only non-default args that differ
# are the checkpoint and `quantization: humming` (absent on the native MXFP8 path).
#
# MXFP8 serves fine at 131072 -- the two-axis window's FALLBACK_MML=40960 was
# declared but never fired (no FALLBACK line, no serve.log.failed-131072), so
# there is no reason to lower either arm's context budget.
set -uo pipefail

ROOT=${ROOT:?}; ARM=${ARM:?}; RUN_TS=${RUN_TS:?}
CKPT=${CKPT:?}; PORT=${PORT:?}; SPEC_K=${SPEC_K:?}
FORMAT=${FORMAT:?}            # w4a8 | mxfp8   (label + precision reported)
BACKEND=${BACKEND:-cutlass}   # humming for W4AFP8; cutlass/native for MXFP8
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

case "$FORMAT" in
  w4a8)  PRECISION=W4AFP8 ;;
  mxfp8) PRECISION=MXFP8 ;;
  *)     PRECISION=$FORMAT ;;
esac

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
note "host=$(hostname) format=$FORMAT backend=$BACKEND spec_k=$SPEC_K mml=$MAX_MODEL_LEN temp=$TEMP"

if [ "$SPEC_K" -gt 0 ]; then
  test -f "$DRAFTER/config.json" || { note "ABORT drafter missing: $DRAFTER"; exit 1; }
  export EXTRA_VLLM_ARGS="--speculative-config {\"method\":\"eagle3\",\"model\":\"$DRAFTER\",\"num_speculative_tokens\":$SPEC_K,\"attention_backend\":\"FLASH_ATTN\"}"
  SPEC_LABEL="eagle3-k$SPEC_K"
else
  SPEC_LABEL="none"
fi
printf '%s\n' "${EXTRA_VLLM_ARGS:-<none>}" > "$C/extra-vllm-args.txt"

note "serve $CKPT on $PORT"
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

# --- gates (fail closed) ------------------------------------------------------
# Humming attestation applies only to the W4AFP8 arms; MXFP8 uses vLLM's native
# path and has no Humming kernel to attest.
if [ "$BACKEND" = humming ]; then
  ( cd "$REPO" && python -m pipeline.m3_humming_w4a8 attest \
      --preflight "$C/serve.log.humming-preflight.json" --log "$C/serve.log" \
      --out "$C/backend-attestation.json" ) >>"$C/client.log" 2>&1 \
    || { note "Humming attestation failed (fail closed)"; stop_local_serve; exit 1; }
else
  # Assert the native MXFP8 path actually engaged rather than silently upcasting.
  grep -qi "mxfp8" "$C/serve.log" \
    || { note "ABORT: serve.log shows no mxfp8 quant path"; stop_local_serve; exit 1; }
  grep -i "mxfp8\|quantization" "$C/serve.log" | head -40 > "$C/quant-boot.log"
fi
if [ "$SPEC_K" -gt 0 ]; then
  grep -qi "num_speculative_tokens" "$C/serve.log" \
    || { note "ABORT: serve.log shows no speculative config"; stop_local_serve; exit 1; }
  grep -i "speculative\|eagle" "$C/serve.log" | head -40 > "$C/spec-boot.log"
fi

rc_all=0
OUT=$C/speedbench

# $1 cell  $2 concurrency  $3 request count
run_cell() {
  local cell=$1 conc=$2 n=$3
  local file="$SB_DIR/$cell.jsonl"
  test -s "$file" || { note "ABORT: missing staged prompts $file"; rc_all=1; return; }
  note "cell=$cell conc=$conc requests=$n"
  snap "sb-$cell-c$conc-pre"
  # Identical seed to phase D so every arm draws the SAME prompts in the same order.
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
  note "cell=$cell conc=$conc rc=$rc"
  [ "$rc" = 0 ] || rc_all=1
}

# 8k only, both entropy tiers, conc 1 and 10 -- the scoped ask.
for cell in 8k-low 8k-high; do run_cell "$cell" 1 40; done
for cell in 8k-low 8k-high; do run_cell "$cell" 10 100; done

for cell in 8k-low 8k-high; do
  [ -d "$OUT/$cell" ] || continue
  "$PYBIN" "$BENCH/performance/workloads/analyze_perf.py" --run-dir "$OUT/$cell" \
    --mode "sbfmt_$cell" --label "$ARM" --precision "$PRECISION" --gpu 8xH100 \
    --num-gpus 8 --spec-decode "$SPEC_LABEL" >>"$C/analyze-$cell.log" 2>&1 \
    || note "WARN analyze_perf failed for $cell"
done

grep -i "acceptance\|SpecDecoding" "$C/serve.log" > "$C/spec-metrics.log" 2>/dev/null || true
stop_local_serve
note "arm done rc=$rc_all"
exit "$rc_all"
