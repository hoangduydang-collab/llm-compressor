#!/usr/bin/env bash
# One arm of EAGLE3 spec-dec phase G -- INT4 drafter vs bf16 drafter at fixed k.
#
# QUESTION: can we buy back drafting overhead by quantizing the DRAFTER (not the
# target)? Phase D measured, on the W4AFP8 target at 8k-low conc 1, a k=0 step of
# 7.313 ms and a k=3 step of 10.49 ms -- so drafting costs 3.18 ms per step, ~1.06 ms
# per draft token, and that overhead is what caps the speedup at 2.17x. The drafter
# reads 2.03 B parameters per forward (attn + MLP + fc + lm_head; the embedding is
# a gather, and vLLM shares the target's table anyway), which at TP8 is 254 M params
# per rank: 508 MB per forward in bf16, 131 MB in INT4 (127 MB packed + ~4 MB of
# group-128 scales), 254 MB in FP8. Arithmetic intensity is ~2 FLOP/byte at these
# batch sizes, three orders below H100's compute/bandwidth ratio, so a drafter
# forward is bandwidth-bound and its time tracks bytes read. At ~2.5 TB/s that is
# 0.20 ms (bf16) vs 0.05 ms (INT4) per forward -> ~0.45 ms/step at k=3, against a
# measured drafting overhead of 3.18 ms/step. So the prize is real but bounded:
# low-single-digit percent end to end, and only if the kernels reach the bandwidth
# roof rather than going launch- or all-reduce-bound. The risk is on the other side:
# RTN INT4 on a 1-layer drafter could cost acceptance, which would make it a net loss.
#
# DESIGN -- both drafters on the SAME NODE, back to back, in ONE allocation.
# The expected effect is a few percent, which is smaller than the node-to-node and
# window-to-window spread we have already seen in this study. So each arm serves the
# INT4 drafter, runs the cell grid, tears the engine down, serves the bf16 drafter,
# and runs the identical grid. The A/B never crosses a node, an allocation, or a
# checkpoint of the target. Reused-from-phase-D numbers are NOT the comparison; the
# in-arm bf16 config is.
#
# Both configs draw the SAME prompts: the same staged SPEED-Bench files (sha256-gated
# by the controller against phases D/E), the same --random-seed 42, the same request
# counts, the same sampling. Nothing about the request stream differs between the two
# halves of an arm.
#
# INT4 first, bf16 second, in every arm. Consistent ordering keeps the k-trend of the
# delta interpretable; the bf16 half re-runs its first cell at the end (cell
# "8k-low-repeat") to bound within-serve drift, which is the noise floor the delta
# has to clear.
#
# The INT4 drafter is a DERIVED artifact (pipeline/prepare_int4_drafter.py): the
# published Sebesky/MiniMax-M3-EAGLE3-RTN-INT4 additionally quantizes embed_tokens,
# which vLLM's unguarded `self.model.model.embed_tokens.weight` access in
# _maybe_share_embeddings cannot load -- and which could not have helped anyway,
# since that tensor is deleted and replaced by the target's table. The derivation
# restores the bf16 embedding byte-for-byte and changes nothing else. The published
# artifact is tested separately by PROBE_ONLY=1.
set -uo pipefail

ROOT=${ROOT:?}; ARM=${ARM:?}; RUN_TS=${RUN_TS:?}
CKPT=${CKPT:?}; PORT=${PORT:?}; SPEC_K=${SPEC_K:?}
DRAFTER_INT4=${DRAFTER_INT4:?}
DRAFTER_FP=${DRAFTER_FP:-/mnt/nfs/hoangduy/hf_assets/Inferact/MiniMax-M3-EAGLE3}
PROBE_ONLY=${PROBE_ONLY:-0}
REPO=/mnt/nfs/hoangduy/projects/llm-compressor
BENCH=/mnt/nfs/hoangduy/projects/benchmarks
PERF_VENV=/mnt/nfs/hoangduy/venvs/perf
SB_DIR=${SB_DIR:-$REPO/artifacts/aiperf-datasets/speedbench}
SERVED_NAME=${SERVED_NAME:-MiniMaxAI/MiniMax-M3}
TOKENIZER=/mnt/nfs/hoangduy/hf_assets/MiniMaxAI/MiniMax-M3
MAX_MODEL_LEN=${MAX_MODEL_LEN:-131072}
MAX_TOKENS=${MAX_TOKENS:-2048}
TEMP=${TEMP:-0.6}
# fp reference from phase D arm-phaseD-k3 (same target ckpt, TP8, gpu_util 0.9):
# "Model loading took 29.26 GiB". The INT4 drafter must land measurably below it.
LOADED_GIB_MAX=${LOADED_GIB_MAX:-29.05}

C=$ROOT/arm-$ARM
mkdir -p "$C"
note() { echo "[arm-$ARM $(date -u +%H:%M:%S)] $1" | tee -a "$C/client.log"; }

export HF_HOME=/mnt/nfs/hoangduy/cache/huggingface HF_DATASETS_OFFLINE=1 HF_HUB_OFFLINE=1
export PATH="$PERF_VENV/bin:$PATH"
export PYBIN="$PERF_VENV/bin/python"
note "host=$(hostname) spec_k=$SPEC_K run_ts=$RUN_TS probe_only=$PROBE_ONLY"
note "int4=$DRAFTER_INT4"
note "fp=$DRAFTER_FP"

rc_all=0
CUR=""            # per-config dir, set by serve_config
CUR_PORT=0

stop_serve() {
  [ -n "$CUR" ] || return 0
  note "stop serve ($CUR)"
  kill "$(cat "$CUR/serve.pid" 2>/dev/null)" 2>/dev/null || true
  sleep 15
  kill -9 -"$(cat "$CUR/serve.pid" 2>/dev/null)" 2>/dev/null || true
  sleep 5
}

# The second serve runs run_vllm_http_serve_smoke.sh's own free-GPU preflight
# (MIN_FREE_GIB=70), so the first engine's memory has to be back before we start it.
wait_gpus_free() {
  local tries=${1:-60}
  for _ in $(seq 1 "$tries"); do
    local bad
    bad=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits 2>/dev/null \
          | awk '{if ($1 < 70000) n++} END {print n+0}')
    [ "$bad" = 0 ] && { note "GPUs free"; return 0; }
    sleep 10
  done
  note "WARN GPUs still busy after wait"
  return 1
}

# $1 label  $2 drafter path  $3 port  -> sets CUR; returns non-zero if unusable
serve_config() {
  local label=$1 drafter=$2 port=$3
  CUR=$C/$label; CUR_PORT=$port
  mkdir -p "$CUR" "$CUR/metrics"

  test -f "$drafter/config.json" || { note "ABORT drafter missing: $drafter"; return 1; }
  export EXTRA_VLLM_ARGS="--speculative-config {\"method\":\"eagle3\",\"model\":\"$drafter\",\"num_speculative_tokens\":$SPEC_K,\"attention_backend\":\"FLASH_ATTN\"}"
  printf '%s\n' "$EXTRA_VLLM_ARGS" > "$CUR/extra-vllm-args.txt"
  printf '%s\n' "$drafter" > "$CUR/drafter-path.txt"
  cp "$drafter/config.json" "$CUR/drafter-config.json" 2>/dev/null || true

  note "serve $label on $port (drafter=$drafter mml=$MAX_MODEL_LEN)"
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
    grep -nE "Traceback|AttributeError|Error|error" "$CUR/serve.log" | tail -40 | tee -a "$C/client.log"
    tail -80 "$CUR/serve.log" | tee -a "$C/client.log"
    return 1
  fi
  export BASE_URL="http://localhost:$port"
  return 0
}

# $1 label  $2 kind (int4|fp)  -> fail-closed gates on the running server
gate_config() {
  local label=$1 kind=$2
  ( cd "$REPO" && python -m pipeline.m3_humming_w4a8 attest \
      --preflight "$CUR/serve.log.humming-preflight.json" --log "$CUR/serve.log" \
      --out "$CUR/backend-attestation.json" ) >>"$C/client.log" 2>&1 \
    || { note "$label: Humming attestation failed (fail closed)"; return 1; }

  grep -qi "num_speculative_tokens" "$CUR/serve.log" \
    || { note "$label ABORT: serve.log shows no speculative config"; return 1; }
  grep -q "'num_speculative_tokens': $SPEC_K" "$CUR/serve.log" \
    || { note "$label ABORT: serve.log does not confirm k=$SPEC_K"; return 1; }
  grep -i "speculative\|eagle\|Detected EAGLE" "$CUR/serve.log" | head -60 > "$CUR/spec-boot.log"

  # The whole point of the derived INT4 artifact is that the embedding still shares
  # (so the only difference from bf16 is the precision of what the drafter COMPUTES
  # with) while the lm_head stays the drafter's own INT4 copy. Assert both, for both
  # configs -- if they diverge, the A/B is not measuring what it claims to.
  grep -q "embed_tokens identical to the target model" "$CUR/serve.log" \
    || { note "$label ABORT: target embedding not shared with drafter"; return 1; }
  grep -q "distinct lm_head weights" "$CUR/serve.log" \
    || { note "$label ABORT: drafter lm_head was shared, not its own"; return 1; }

  local gib
  gib=$(grep -m1 -oE "Model loading took [0-9.]+ GiB" "$CUR/serve.log" | grep -oE "[0-9.]+")
  printf '%s\n' "${gib:-unknown}" > "$CUR/model-loading-gib.txt"
  note "$label: model loading took ${gib:-?} GiB"
  if [ "$kind" = int4 ]; then
    [ -n "$gib" ] || { note "$label ABORT: no 'Model loading took' line to verify INT4 residency"; return 1; }
    awk -v g="$gib" -v m="$LOADED_GIB_MAX" 'BEGIN{exit !(g < m)}' \
      || { note "$label ABORT: loaded $gib GiB >= $m GiB -- INT4 drafter weights not resident"; return 1; }
  fi
  return 0
}

snap() { curl -sf "http://localhost:$CUR_PORT/metrics" -o "$CUR/metrics/$1.txt" 2>/dev/null || true; }

# $1 cell  $2 concurrency  $3 request count  $4 output cell name (defaults to $1)
run_cell() {
  local cell=$1 conc=$2 n=$3 out_cell=${4:-$1}
  local file="$SB_DIR/$cell.jsonl"
  test -s "$file" || { note "ABORT: missing staged prompts $file"; rc_all=1; return; }
  note "cell=$out_cell conc=$conc requests=$n"
  snap "sb-$out_cell-c$conc-pre"
  # Identical prompt stream to phases D/E and to the other half of this arm.
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

# $1 label  $2 with_repeat(0|1)
run_grid() {
  local label=$1 with_repeat=$2
  for cell in 8k-low 8k-high; do run_cell "$cell" 1 40; done
  for cell in 8k-low 8k-high; do run_cell "$cell" 10 100; done
  # Drift control: same cell, same serve, ~90 min later. Bounds the noise floor.
  [ "$with_repeat" = 1 ] && run_cell 8k-low 1 40 8k-low-repeat
  for cell in 8k-low 8k-high 8k-low-repeat; do
    [ -d "$CUR/speedbench/$cell" ] || continue
    "$PYBIN" "$BENCH/performance/workloads/analyze_perf.py" \
      --run-dir "$CUR/speedbench/$cell" --mode "speedbench_$cell" \
      --label "$ARM-$label" --precision W4AFP8 --gpu 8xH100 --num-gpus 8 \
      --spec-decode "eagle3-k$SPEC_K-$label" >>"$CUR/analyze-$cell.log" 2>&1 \
      || note "WARN analyze_perf failed for $cell"
  done
  grep -i "acceptance\|SpecDecoding" "$CUR/serve.log" > "$CUR/spec-metrics.log" 2>/dev/null || true
}

# --- PROBE_ONLY: does the AS-PUBLISHED INT4 checkpoint load at all? -------------
# Serve-only, no cells, short budget. Recorded as evidence either way; this arm's
# rc is deliberately 0 on a load failure, because "it does not load" is the result.
if [ "$PROBE_ONLY" = 1 ]; then
  PROBE_DRAFTER=${PROBE_DRAFTER:?}
  note "PROBE: attempting to serve as-published $PROBE_DRAFTER"
  if serve_config published-asis "$PROBE_DRAFTER" "$PORT"; then
    note "PROBE RESULT: as-published INT4 drafter LOADED"
    echo "loaded" > "$C/probe-verdict.txt"
    gate_config published-asis int4 || note "PROBE: loaded but failed gates"
  else
    note "PROBE RESULT: as-published INT4 drafter FAILED to load"
    echo "failed" > "$C/probe-verdict.txt"
    grep -nE "Traceback|AttributeError|KeyError|ValueError|embed_tokens" \
      "$CUR/serve.log" 2>/dev/null | tail -60 > "$C/probe-failure-excerpt.txt"
  fi
  stop_serve
  note "probe arm done rc=0 verdict=$(cat "$C/probe-verdict.txt" 2>/dev/null)"
  exit 0
fi

# --- A/B: INT4 drafter, then bf16 drafter, same node, same allocation -----------
if serve_config int4 "$DRAFTER_INT4" "$PORT"; then
  if gate_config int4 int4; then
    run_grid int4 0
  else
    note "int4 gates failed -- no cells run"
    rc_all=1
  fi
else
  note "int4 serve failed"
  rc_all=1
fi
stop_serve
wait_gpus_free 60

if serve_config fp "$DRAFTER_FP" "$((PORT + 100))"; then
  if gate_config fp fp; then
    run_grid fp 1
  else
    note "fp gates failed -- no cells run"
    rc_all=1
  fi
else
  note "fp serve failed"
  rc_all=1
fi
stop_serve

note "arm done rc=$rc_all"
exit "$rc_all"
