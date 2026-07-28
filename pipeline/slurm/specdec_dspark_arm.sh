#!/usr/bin/env bash
# Axis 1 (draft depth k) for the NVIDIA DSpark drafter on the M3 W4A8 target.
#
# WHY A SEPARATE ARM FROM specdec_unified_arm.sh
# ----------------------------------------------
# Three of that arm's mechanisms are wrong here, and two of them would pass wrongly
# rather than fail closed:
#   * Axis 2 (drafter W4A16 kernel) does not exist for this drafter -- DSpark ships
#     bf16/f32 with no quantized linears, so there is no WNA16 kernel to choose and
#     no kernel-variant env lever. Axis 3 (drafter INT4 vs bf16) likewise has no
#     INT4 counterpart. Only k is swept.
#   * The EAGLE3 drafter-wiring greps ("embed_tokens identical to the target",
#     "distinct lm_head weights") come from our EAGLE3 lm_head patch and describe
#     EAGLE3's weight sharing. DSpark shares the target embedding and carries its
#     own bf16 lm_head, so those lines are recorded, never fatal.
#   * The loaded-GiB band was calibrated on phase G's 28.78/29.26 GiB EAGLE3 loads.
#     This drafter is a different size, so a tight band would abort every serve.
#
# THE GATE THAT MATTERS MOST -- aux hidden-state layers
# ----------------------------------------------------
# DSpark consumes SIX mean-pooled aux hidden states; ours are declared as
# dflash_config.target_layer_ids = [1,12,23,35,46,57] and vLLM converts them with a
# +1 offset to (2,13,24,36,47,58). If that resolution ever fails, vLLM does NOT
# error -- it silently falls back to the TARGET's default 3-layer EAGLE3 set
# (gpu_model_runner.py: `aux_layers = self.model.get_eagle3_default_aux_hidden_state_layers()`).
# The drafter would then be fed the wrong tensors, and the failure surfaces only as
# low acceptance with entirely plausible ITL. That is exactly the class of defect
# that cost the in-house GLM-5.2 DSpark study a wasted window (acceptance 1.27
# instead of ~3.9, see /mnt/nfs/alexyang/speculative_decoding/DSPARK_REPORT.md).
# So we assert BOTH that the log took the from-config path AND the six exact ids.
#
# SMOKE-FIRST ORDERING
# --------------------
# DSpark has never been booted against this target, so the first serve is the
# published reference config (k=8) and its acceptance gate is ARM-FATAL: if the
# layout is wrong, every later serve would measure the same broken drafter, so we
# stop and keep the allocation for diagnosis instead of burning ~4 h. Later serves
# use the same gate at WARN level, since by then the layout is proven and a single
# odd cell should not kill the window.
#
# Replicate order interleaves k=8 and k=6 and brackets the window with k=0 controls,
# so a monotone drift cannot alias into the k-trend.
set -uo pipefail

ROOT=${ROOT:?}; ARM=${ARM:?}; RUN_TS=${RUN_TS:?}
CKPT=${CKPT:?}; PORT_BASE=${PORT_BASE:?}
SERVES=${SERVES:?}                       # newline-separated: label;k;cells
ONLY=${ONLY:-}                           # optional comma-separated label subset
DRAFTER=${DRAFTER:?}                     # nvidia/MiniMax-M3-DSpark checkout
SMOKE_LABEL=${SMOKE_LABEL:-}             # label whose acceptance gate is arm-fatal
REPO=/mnt/nfs/hoangduy/projects/llm-compressor
BENCH=/mnt/nfs/hoangduy/projects/benchmarks
PERF_VENV=/mnt/nfs/hoangduy/venvs/perf
SB_DIR=${SB_DIR:-$REPO/artifacts/aiperf-datasets/speedbench}
SERVED_NAME=${SERVED_NAME:-MiniMaxAI/MiniMax-M3}
TOKENIZER=/mnt/nfs/hoangduy/hf_assets/MiniMaxAI/MiniMax-M3
MAX_MODEL_LEN=${MAX_MODEL_LEN:-131072}
MAX_TOKENS=${MAX_TOKENS:-2048}
TEMP=${TEMP:-0.6}
N_C1=${N_C1:-30}
N_C10=${N_C10:-80}
BACKEND=${BACKEND:-humming}              # target W4A8 kernel; fixed for axis 1

# Acceptance floor. NVIDIA publishes 3.19 (SPEED-Bench throughput-32k low_entropy)
# and 5.26 (qualitative coding) at block 8; the layout-mismatch failure mode lands
# near 1.3. 2.2 sits well clear of both -- it is a "the drafter is wired correctly"
# gate, not a performance target, and it must not fire on a genuinely modest k.
ACC_MIN=${ACC_MIN:-2.2}
ACC_MIN_K=${ACC_MIN_K:-6}                # only assert at k >= this

# Loaded-GiB band. Deliberately WIDE: the EAGLE3 bands do not transfer and this arm
# is the first to measure this drafter's footprint. The k=0 controls in the same
# window provide the real reference, computed after the fact.
GIB_MIN=${GIB_MIN:-27.5}
GIB_MAX=${GIB_MAX:-32.5}

# The six aux layers vLLM must report, derived offline from the drafter config:
# dflash_config.target_layer_ids [1,12,23,35,46,57] + 1 each.
AUX_EXPECT=${AUX_EXPECT:-"2, 13, 24, 36, 47, 58"}

C=$ROOT/arm-$ARM
mkdir -p "$C"
note() { echo "[arm-$ARM $(date -u +%H:%M:%S)] $1" | tee -a "$C/client.log"; }

export HF_HOME=/mnt/nfs/hoangduy/cache/huggingface HF_DATASETS_OFFLINE=1 HF_HUB_OFFLINE=1
export PATH="$PERF_VENV/bin:$PATH"
export PYBIN="$PERF_VENV/bin/python"

note "host=$(hostname) run_ts=$RUN_TS backend=$BACKEND n_c1=$N_C1 n_c10=$N_C10"
note "drafter=$DRAFTER smoke=${SMOKE_LABEL:-<none>} only='${ONLY:-<all>}'"
printf 'host=%s\nrun_ts=%s\nbackend=%s\ndrafter=%s\nn_c1=%s\nn_c10=%s\nsmoke=%s\nonly=%s\n' \
  "$(hostname)" "$RUN_TS" "$BACKEND" "$DRAFTER" "$N_C1" "$N_C10" \
  "${SMOKE_LABEL:-<none>}" "${ONLY:-<all>}" > "$C/arm-identity.txt"

rc_all=0
CUR=""; CUR_PORT=0

enabled() {
  [ -z "$ONLY" ] && return 0
  case ",$ONLY," in *",$1,"*) return 0;; *) return 1;; esac
}

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
    # No attention_backend pin. EAGLE3 forced FLASH_ATTN, but this drafter is
    # non-causal with a 1024 sliding window (dflash_config.causal=false,
    # use_swa=true), so we let vLLM select and record what it chose rather than
    # forcing a backend whose non-causal support we have not verified.
    export EXTRA_VLLM_ARGS="--speculative-config {\"method\":\"dspark\",\"model\":\"$DRAFTER\",\"num_speculative_tokens\":$k}"
  else
    unset EXTRA_VLLM_ARGS
  fi

  printf '%s\n' "${EXTRA_VLLM_ARGS:-<none>}" > "$CUR/extra-vllm-args.txt"
  printf 'k=%s\nbackend=%s\ndrafter=dspark\ndrafter_path=%s\nkernel=default\nmethod=%s\n' \
    "$k" "$BACKEND" "$DRAFTER" "$([ "$k" -gt 0 ] && echo dspark || echo none)" \
    > "$CUR/cell-config.txt"

  note "serve $label (k=$k backend=$BACKEND drafter=dspark) on $port"
  M3_W4A8_BACKEND="$BACKEND" \
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

  # --- target W4A8 backend attestation (same contract as the EAGLE3 arm) ---
  if [ "$BACKEND" = humming ]; then
    ( cd "$REPO" && python -m pipeline.m3_humming_w4a8 attest \
        --preflight "$CUR/serve.log.humming-preflight.json" --log "$CUR/serve.log" \
        --out "$CUR/backend-attestation.json" ) >>"$C/client.log" 2>&1 \
      || { note "$label ABORT: Humming attestation failed (fail closed)"; return 1; }
    # Surface the provisional-vLLM advisory per serve, so it is visible in the run
    # log and not only buried in the JSON.
    if grep -q "VLLM_VERSION_PROVISIONAL" "$CUR/backend-attestation.json" 2>/dev/null; then
      note "$label: Humming attested WITH ADVISORY VLLM_VERSION_PROVISIONAL (0.26.0 not yet qualified)"
    else
      note "$label: Humming backend attested"
    fi
  else
    grep -qE "quantization[ =']+humming" "$CUR/serve.log" \
      && { note "$label ABORT: cutlass arm shows humming quantization"; return 1; }
    note "$label: CUTLASS backend confirmed"
  fi

  if [ "$k" = 0 ]; then
    grep -q "num_speculative_tokens" "$CUR/serve.log" \
      && { note "$label ABORT: k=0 control shows a speculative config"; return 1; }
    local gib0
    gib0=$(grep -m1 -oE "Model loading took [0-9.]+ GiB" "$CUR/serve.log" | grep -oE "[0-9.]+")
    printf '%s\n' "${gib0:-unknown}" > "$CUR/model-loading-gib.txt"
    note "$label: k=0 control confirmed (no speculative config, ${gib0:-?} GiB)"
    return 0
  fi

  # --- k and method ---
  grep -q "'num_speculative_tokens': $k" "$CUR/serve.log" \
    || { note "$label ABORT: serve.log does not confirm k=$k"; return 1; }
  grep -qE "'method': 'dspark'|method=dspark" "$CUR/serve.log" \
    || { note "$label ABORT: serve.log does not confirm method=dspark"; return 1; }
  grep -iE "speculative|dspark|DSpark|auxiliary layers" "$CUR/serve.log" | head -60 > "$CUR/spec-boot.log"

  # --- THE aux-layer gate: from-config path AND the six exact ids ---
  # Both runner implementations log one of these two phrasings; a fallback to the
  # target's 3-layer default logs "from model" / no "from ... config" at all.
  local auxline
  auxline=$(grep -ohE "Using (Eagle3 )?auxiliary layers from (speculative )?config: \(?[0-9, ]+\)?" \
              "$CUR/serve.log" | tail -1)
  printf '%s\n' "${auxline:-<no from-config aux line>}" > "$CUR/aux-layers.txt"
  grep -ohE "Using (Eagle3 )?auxiliary layers from model.*" "$CUR/serve.log" \
      >> "$CUR/aux-layers.txt" 2>/dev/null || true
  if [ -z "$auxline" ]; then
    note "$label ABORT: no 'auxiliary layers from config' line -- vLLM fell back to"
    note "$label          the target's default 3-layer EAGLE3 set; the drafter would"
    note "$label          be fed the wrong tensors and only acceptance would show it."
    return 1
  fi
  case "$auxline" in
    *"$AUX_EXPECT"*) note "$label: aux layers from config = ($AUX_EXPECT)" ;;
    *) note "$label ABORT: aux layers '$auxline' != expected ($AUX_EXPECT)"; return 1 ;;
  esac

  # --- footprint, recorded against a deliberately wide band ---
  local gib
  gib=$(grep -m1 -oE "Model loading took [0-9.]+ GiB" "$CUR/serve.log" | grep -oE "[0-9.]+")
  printf '%s\n' "${gib:-unknown}" > "$CUR/model-loading-gib.txt"
  [ -n "$gib" ] || { note "$label ABORT: no 'Model loading took' line"; return 1; }
  awk -v g="$gib" -v lo="$GIB_MIN" -v hi="$GIB_MAX" 'BEGIN{exit !(g >= lo && g < hi)}' \
    || { note "$label ABORT: loaded $gib GiB outside [$GIB_MIN,$GIB_MAX)"; return 1; }
  note "$label: k=$k confirmed, drafter resident ($gib GiB)"

  # --- recorded, never fatal: this arm is the first to observe any of these ---
  grep -ohE "Using [A-Za-z]+LinearKernel for [A-Za-z]+" "$CUR/serve.log" \
    | sort | uniq -c | sort -rn > "$CUR/kernel-census.txt" 2>/dev/null || true
  grep -ohE "draft.*attention backend.*|Using .*Backend for draft|attn_backend=[A-Za-z_]+" \
    "$CUR/serve.log" | sort -u > "$CUR/draft-attn-backend.txt" 2>/dev/null || true
  { grep -c "embed_tokens identical to the target model" "$CUR/serve.log"
    grep -c "distinct lm_head weights" "$CUR/serve.log"; } \
    > "$CUR/drafter-wiring.txt" 2>/dev/null || true
  return 0
}

snap() { curl -sf "http://localhost:$CUR_PORT/metrics" -o "$CUR/metrics/$1.txt" 2>/dev/null || true; }

# $1 conc  $2 request count  $3 cell
run_cell() {
  local conc=$1 n=$2 cell=$3
  local file="$SB_DIR/$cell.jsonl"
  test -s "$file" || { note "ABORT: missing staged prompts $file"; rc_all=1; return; }
  note "cell=$cell conc=$conc requests=$n"
  snap "sb-$cell-c$conc-pre"
  "$PERF_VENV/bin/aiperf" profile \
      --model "$SERVED_NAME" --url "$BASE_URL" --endpoint-type chat --streaming \
      --tokenizer "$TOKENIZER" \
      --custom-dataset-type single_turn --input-file "$file" \
      --extra-inputs "{\"temperature\":$TEMP,\"max_tokens\":$MAX_TOKENS,\"chat_template_kwargs\":{\"enable_thinking\":true}}" \
      --random-seed 42 \
      --concurrency "$conc" --request-count "$n" --warmup-request-count "$conc" \
      --artifact-dir "$CUR/speedbench/$cell/conc_$conc" >>"$CUR/sb-$cell-c$conc.log" 2>&1
  local rc=$?
  snap "sb-$cell-c$conc-post"
  note "cell=$cell conc=$conc rc=$rc"
  [ "$rc" = 0 ] || rc_all=1
}

# Returns 2 when the smoke serve fails the floor, which the main loop turns into an
# early arm exit. Any other serve only warns.
acc_gate() {
  local label=$1 k=$2 cell=$3
  [ "$cell" = 8k-low ] || return 0
  [ "$k" -ge "$ACC_MIN_K" ] || return 0
  local pre=$CUR/metrics/sb-$cell-c1-pre.txt post=$CUR/metrics/sb-$cell-c1-post.txt
  [ -s "$pre" ] && [ -s "$post" ] || { note "$label WARN: no metrics for acceptance gate"; return 0; }
  local acc
  acc=$("$PYBIN" - "$pre" "$post" <<'PY'
import re, sys
def g(p, name):
    pat = re.compile(rf"^vllm:{name}(?:\{{.*?\}})?\s+([0-9.eE+]+)$", re.M)
    return sum(float(v) for v in pat.findall(open(p).read())) or None
pre, post = sys.argv[1], sys.argv[2]
d0, a0 = g(pre, "spec_decode_num_drafts_total"), g(pre, "spec_decode_num_accepted_tokens_total")
d1, a1 = g(post, "spec_decode_num_drafts_total"), g(post, "spec_decode_num_accepted_tokens_total")
print("" if None in (d0, a0, d1, a1) or d1 == d0 else f"{1 + (a1 - a0) / (d1 - d0):.4f}")
PY
)
  printf '%s\n' "${acc:-unknown}" > "$CUR/accepted-$cell-c1.txt"
  [ -n "$acc" ] || { note "$label WARN: acceptance not computable"; return 0; }
  if awk -v a="$acc" -v m="$ACC_MIN" 'BEGIN{exit !(a >= m)}'; then
    note "$label: accepted length $acc (>= $ACC_MIN)"
    return 0
  fi
  rc_all=1
  if [ "$label" = "$SMOKE_LABEL" ]; then
    note "$label SMOKE FAILED: accepted $acc < $ACC_MIN"
    note "  This is the layout-mismatch signature. Every later serve would measure the"
    note "  same broken drafter, so the arm stops here and leaves the node allocated."
    note "  Check: $CUR/aux-layers.txt, $CUR/spec-boot.log, $CUR/serve.log"
    return 2
  fi
  note "$label WARN: accepted $acc < $ACC_MIN (recorded; layout already proven by smoke)"
  return 0
}

analyze() {
  local label=$1 k=$2 cell=$3
  [ -d "$CUR/speedbench/$cell" ] || return 0
  "$PYBIN" "$BENCH/performance/workloads/analyze_perf.py" \
    --run-dir "$CUR/speedbench/$cell" --mode "speedbench_$cell" \
    --label "$ARM-$label" --precision W4AFP8 --gpu 8xH100 --num-gpus 8 \
    --spec-decode "$([ "$k" -gt 0 ] && echo "dspark-k$k" || echo none)" \
    >>"$CUR/analyze-$cell.log" 2>&1 || note "WARN analyze_perf failed for $cell"
}

# ---- main loop over the controller's ordered serve list ----
port=$PORT_BASE
n_done=0; n_skip=0; n_fail=0
smoke_abort=0
while IFS= read -r spec; do
  spec=$(printf '%s' "$spec" | sed 's/^[[:space:]]*//;s/[[:space:]]*$//')
  [ -n "$spec" ] || continue
  case $spec in \#*) continue ;; esac

  IFS=';' read -r label k cells <<<"$spec"
  if ! enabled "$label"; then
    note "skip $label (not in ONLY)"; n_skip=$((n_skip + 1)); port=$((port + 1)); continue
  fi

  if serve_config "$label" "$k" "$port"; then
    if gate_config "$label" "$k"; then
      for cell in $cells; do
        run_cell 1  "$N_C1"  "$cell"
        run_cell 10 "$N_C10" "$cell"
        acc_gate "$label" "$k" "$cell"; ag=$?
        analyze "$label" "$k" "$cell"
        if [ "$ag" = 2 ]; then smoke_abort=1; break; fi
      done
      n_done=$((n_done + 1))
    else
      note "$label gates failed -- no cells run"; rc_all=1; n_fail=$((n_fail + 1))
      [ "$label" = "$SMOKE_LABEL" ] && smoke_abort=1
    fi
  else
    note "$label serve failed"; rc_all=1; n_fail=$((n_fail + 1))
    [ "$label" = "$SMOKE_LABEL" ] && smoke_abort=1
  fi
  stop_serve
  wait_gpus_free 60
  port=$((port + 1))
  printf '%s done=%s skip=%s fail=%s\n' "$label" "$n_done" "$n_skip" "$n_fail" >> "$C/progress.txt"
  if [ "$smoke_abort" = 1 ]; then
    note "STOPPING EARLY: smoke serve '$SMOKE_LABEL' did not qualify the DSpark wiring."
    break
  fi
done <<< "$SERVES"

printf 'done=%s skip=%s fail=%s rc=%s smoke_abort=%s finished=%s\n' \
  "$n_done" "$n_skip" "$n_fail" "$rc_all" "$smoke_abort" \
  "$(date -u +%Y-%m-%dT%H:%M:%SZ)" | tee "$C/arm-done.txt" | tee -a "$C/client.log"
note "arm done rc=$rc_all"
exit "$rc_all"
