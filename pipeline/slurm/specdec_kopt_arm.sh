#!/usr/bin/env bash
# One arm of EAGLE3 spec-dec phase H -- optimal draft depth per entropy tier.
#
# Phase G found the optimum is workload-dependent and that neither group's optimum
# was bracketed by k in {3,4,5}: at 8k-low, ITL fell monotonically 3.280 -> 3.120 ->
# 2.965 ms with the marginal gain NOT decaying (-4.9%, -5.0%), so the knee is above
# k=5; at 8k-high, k=3 already won and deeper k lost (+1.9%, +6.8%). This phase
# brackets each knee from the other side:
#
#   low  group: k = 5, 6, 7   on 8k-low   (expect a knee, or confirm still rising)
#   high group: k = 1, 2, 3   on 8k-high  (expect k=1 or 2 to win)
#
# DESIGN -- one entropy tier per node, ALL of its k values on that node, serially.
# Phase G's INT4-vs-bf16 A/B was same-node by construction, but its k-trend was not:
# k=3/4/5 ran on gpu-h107/h123/h108, so node variance is folded into the -4.9%/-5.0%
# marginal figures. Choosing k is exactly a k-comparison, so here the k axis is the
# one that must not cross a node.
#
# The serial-sweep threat is drift instead of node variance, so it is measured rather
# than assumed: after the last k, the arm re-serves the FIRST spec k and re-runs both
# of its cells as `<cell>-repeat`. That is a full end-of-window re-measurement, on a
# fresh engine, of a point measured at the start of the window; any monotone k-trend
# has to be larger than the gap it reveals.
#
# Each arm also carries its own k=0 control, measured first on a fresh engine, so
# speedups are within-window and no phase D/G number is needed to state them.
#
# Drafter: the derived INT4 artifact (pipeline/prepare_int4_drafter.py) -- phase G
# showed INT4 costs no acceptance (3.134 vs 3.106 at k=3, higher at every position)
# while cutting ITL, so it is the right drafter to tune depth on.
#
# Prompts are the same staged SPEED-Bench bytes as phases D-G (controller gates the
# sha256), same --random-seed 42, same request counts. Only k varies.
set -uo pipefail

ROOT=${ROOT:?}; ARM=${ARM:?}; RUN_TS=${RUN_TS:?}
CKPT=${CKPT:?}; PORT_BASE=${PORT_BASE:?}
CELL=${CELL:?}                 # 8k-low | 8k-high
KS=${KS:?}                     # e.g. "0 5 6 7"; k=0 is the control
REPEAT_K=${REPEAT_K:?}         # re-served last, drift control
DRAFTER=${DRAFTER:?}
REPO=/mnt/nfs/hoangduy/projects/llm-compressor
BENCH=/mnt/nfs/hoangduy/projects/benchmarks
PERF_VENV=/mnt/nfs/hoangduy/venvs/perf
SB_DIR=${SB_DIR:-$REPO/artifacts/aiperf-datasets/speedbench}
SERVED_NAME=${SERVED_NAME:-MiniMaxAI/MiniMax-M3}
TOKENIZER=/mnt/nfs/hoangduy/hf_assets/MiniMaxAI/MiniMax-M3
MAX_MODEL_LEN=${MAX_MODEL_LEN:-131072}
MAX_TOKENS=${MAX_TOKENS:-2048}
TEMP=${TEMP:-0.6}
LOADED_GIB_MAX=${LOADED_GIB_MAX:-29.05}

C=$ROOT/arm-$ARM
mkdir -p "$C"
note() { echo "[arm-$ARM $(date -u +%H:%M:%S)] $1" | tee -a "$C/client.log"; }

export HF_HOME=/mnt/nfs/hoangduy/cache/huggingface HF_DATASETS_OFFLINE=1 HF_HUB_OFFLINE=1
export PATH="$PERF_VENV/bin:$PATH"
export PYBIN="$PERF_VENV/bin/python"
note "host=$(hostname) cell=$CELL ks='$KS' repeat_k=$REPEAT_K run_ts=$RUN_TS"

rc_all=0
CUR=""; CUR_PORT=0

stop_serve() {
  [ -n "$CUR" ] || return 0
  note "stop serve ($(basename "$CUR"))"
  kill "$(cat "$CUR/serve.pid" 2>/dev/null)" 2>/dev/null || true
  sleep 15
  kill -9 -"$(cat "$CUR/serve.pid" 2>/dev/null)" 2>/dev/null || true
  sleep 5
}

wait_gpus_free() {
  for _ in $(seq 1 "${1:-60}"); do
    local bad
    bad=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits 2>/dev/null \
          | awk '{if ($1 < 70000) n++} END {print n+0}')
    [ "$bad" = 0 ] && { note "GPUs free"; return 0; }
    sleep 10
  done
  note "WARN GPUs still busy after wait"
  return 1
}

# $1 label  $2 k  $3 port
serve_config() {
  local label=$1 k=$2 port=$3
  CUR=$C/$label; CUR_PORT=$port
  mkdir -p "$CUR" "$CUR/metrics"
  if [ "$k" -gt 0 ]; then
    test -f "$DRAFTER/config.json" || { note "ABORT drafter missing: $DRAFTER"; return 1; }
    export EXTRA_VLLM_ARGS="--speculative-config {\"method\":\"eagle3\",\"model\":\"$DRAFTER\",\"num_speculative_tokens\":$k,\"attention_backend\":\"FLASH_ATTN\"}"
  else
    unset EXTRA_VLLM_ARGS
  fi
  printf '%s\n' "${EXTRA_VLLM_ARGS:-<none>}" > "$CUR/extra-vllm-args.txt"

  note "serve $label (k=$k) on $port"
  CKPT="$CKPT" SERVED_NAME="$SERVED_NAME" MODEL_ID="$TOKENIZER" PORT="$port" \
    MAX_MODEL_LEN="$MAX_MODEL_LEN" LOG="$CUR/serve.log" PID_FILE="$CUR/serve.pid" \
    bash "$REPO/pipeline/slurm/run_vllm_http_serve_smoke.sh" >>"$C/client.log" 2>&1 \
    || { note "$label serve start rc=$?"; tail -60 "$CUR/serve.log" 2>/dev/null | tee -a "$C/client.log"; return 1; }

  local ready=1
  for _ in $(seq 1 540); do
    curl -sf "http://localhost:$port/v1/models" -o "$CUR/models.json" 2>/dev/null && { ready=0; break; }
    kill -0 "$(cat "$CUR/serve.pid" 2>/dev/null)" 2>/dev/null || { note "$label serve died"; break; }
    sleep 10
  done
  if [ "$ready" != 0 ]; then
    note "$label never became ready"
    grep -nE "Traceback|AttributeError|KeyError|Error" "$CUR/serve.log" | tail -40 | tee -a "$C/client.log"
    return 1
  fi
  export BASE_URL="http://localhost:$port"
  return 0
}

# $1 label  $2 k
gate_config() {
  local label=$1 k=$2
  ( cd "$REPO" && python -m pipeline.m3_humming_w4a8 attest \
      --preflight "$CUR/serve.log.humming-preflight.json" --log "$CUR/serve.log" \
      --out "$CUR/backend-attestation.json" ) >>"$C/client.log" 2>&1 \
    || { note "$label: Humming attestation failed (fail closed)"; return 1; }

  if [ "$k" -gt 0 ]; then
    grep -q "'num_speculative_tokens': $k" "$CUR/serve.log" \
      || { note "$label ABORT: serve.log does not confirm k=$k"; return 1; }
    # Same drafter wiring as phase G: target embedding shared, drafter's own INT4
    # lm_head kept. If either flips, the arm is not measuring the same drafter.
    grep -q "embed_tokens identical to the target model" "$CUR/serve.log" \
      || { note "$label ABORT: target embedding not shared with drafter"; return 1; }
    grep -q "distinct lm_head weights" "$CUR/serve.log" \
      || { note "$label ABORT: drafter lm_head was shared, not its own"; return 1; }
    local gib
    gib=$(grep -m1 -oE "Model loading took [0-9.]+ GiB" "$CUR/serve.log" | grep -oE "[0-9.]+")
    printf '%s\n' "${gib:-unknown}" > "$CUR/model-loading-gib.txt"
    [ -n "$gib" ] || { note "$label ABORT: no 'Model loading took' line"; return 1; }
    awk -v g="$gib" -v m="$LOADED_GIB_MAX" 'BEGIN{exit !(g < m)}' \
      || { note "$label ABORT: loaded $gib GiB >= $m -- INT4 drafter not resident"; return 1; }
    note "$label: k=$k confirmed, drafter INT4 resident (${gib} GiB)"
    grep -i "speculative\|Detected EAGLE" "$CUR/serve.log" | head -40 > "$CUR/spec-boot.log"
  else
    grep -q "num_speculative_tokens" "$CUR/serve.log" \
      && { note "$label ABORT: control arm shows a speculative config"; return 1; }
    note "$label: k=0 control confirmed (no speculative config)"
  fi
  return 0
}

snap() { curl -sf "http://localhost:$CUR_PORT/metrics" -o "$CUR/metrics/$1.txt" 2>/dev/null || true; }

# $1 conc  $2 request count  $3 output cell name
run_cell() {
  local conc=$1 n=$2 out_cell=$3
  local file="$SB_DIR/$CELL.jsonl"
  test -s "$file" || { note "ABORT: missing staged prompts $file"; rc_all=1; return; }
  note "cell=$out_cell conc=$conc requests=$n"
  snap "sb-$out_cell-c$conc-pre"
  "$PERF_VENV/bin/aiperf" profile \
      --model "$SERVED_NAME" --url "$BASE_URL" --endpoint-type chat --streaming \
      --tokenizer "$TOKENIZER" \
      --custom-dataset-type single_turn --input-file "$file" \
      --extra-inputs "{\"temperature\":$TEMP,\"max_tokens\":$MAX_TOKENS,\"chat_template_kwargs\":{\"enable_thinking\":true}}" \
      --random-seed 42 \
      --concurrency "$conc" --request-count "$n" --warmup-request-count "$conc" \
      --artifact-dir "$CUR/speedbench/$out_cell/conc_$conc" >>"$CUR/sb-$out_cell-c$conc.log" 2>&1
  local rc=$?
  snap "sb-$out_cell-c$conc-post"
  note "cell=$out_cell conc=$conc rc=$rc"
  [ "$rc" = 0 ] || rc_all=1
}

analyze() {
  local label=$1 k=$2 out_cell=$3
  [ -d "$CUR/speedbench/$out_cell" ] || return 0
  "$PYBIN" "$BENCH/performance/workloads/analyze_perf.py" \
    --run-dir "$CUR/speedbench/$out_cell" --mode "speedbench_$out_cell" \
    --label "$ARM-$label" --precision W4AFP8 --gpu 8xH100 --num-gpus 8 \
    --spec-decode "$([ "$k" -gt 0 ] && echo "eagle3-int4-k$k" || echo none)" \
    >>"$CUR/analyze-$out_cell.log" 2>&1 || note "WARN analyze_perf failed for $out_cell"
}

# $1 label  $2 k  $3 port  $4 out_cell
run_one() {
  local label=$1 k=$2 port=$3 out_cell=$4
  if serve_config "$label" "$k" "$port"; then
    if gate_config "$label" "$k"; then
      run_cell 1 40 "$out_cell"
      run_cell 10 100 "$out_cell"
      analyze "$label" "$k" "$out_cell"
      grep -i "acceptance\|SpecDecoding" "$CUR/serve.log" > "$CUR/spec-metrics.log" 2>/dev/null || true
    else
      note "$label gates failed -- no cells run"; rc_all=1
    fi
  else
    note "$label serve failed"; rc_all=1
  fi
  stop_serve
  wait_gpus_free 60
}

port=$PORT_BASE
for k in $KS; do
  run_one "k$k" "$k" "$port" "$CELL"
  port=$((port + 1))
done

# Drift control: re-serve the first spec k and re-measure its conc-1 cell.
note "drift control: re-serving k=$REPEAT_K"
run_one "k${REPEAT_K}-repeat" "$REPEAT_K" "$port" "$CELL-repeat"

note "arm done rc=$rc_all"
exit "$rc_all"
