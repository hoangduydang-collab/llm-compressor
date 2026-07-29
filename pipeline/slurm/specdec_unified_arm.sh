#!/usr/bin/env bash
# Unified one-node EAGLE3 spec-dec arm: barebone kernel leg + all three tuning axes.
#
# WHY THIS ARM EXISTS
# -------------------
# Phases D-I.2 answered the three tuning questions but spread the answers over six
# windows and five nodes (h107/h108/h113/h114/h123). Node variance in this study runs
# 1-2%, which is the same size as two of the three effects, so the cross-axis story
# was only as credible as its weakest join. This arm re-measures EVERYTHING on ONE
# node in ONE allocation, and adds the barebone kernel leg the story was missing.
#
# It also fixes a workload defect in the intended narrative. The published
# "95 -> 137 tok/s" CUTLASS->Humming gain is from the PINNED reasoning shape
# (1k in / 8k forced out, ignore_eos), while "137 -> 340" is SPEED-Bench with natural
# stopping. Wave 2 measured that ignore_eos inflates acceptance +33%, so chaining the
# two would multiply a pinned-shape kernel gain by a natural-shape spec-dec gain. Here
# the CUTLASS leg is re-measured on SPEED-Bench so all rungs share one workload.
#
# DESIGN -- an explicit ordered serve list, supplied by the controller
# -------------------------------------------------------------------
# Every serve is one spec: label;k;backend;drafter;kernel;cells
# The controller owns the ORDER, which carries the statistics:
#   * A/B pairs are INTERLEAVED, never blocked, so a monotone drift through the
#     6-hour window cannot alias into an effect (CUTLASS/Humming at k=0; bf16/INT4
#     at k=5).
#   * replicate counts are sized from the measured cross-engine floors --
#     conc-1 sd 1.02%, conc-10 sd 0.16%, se_diff = sd*sqrt(2/n). n=3 gives 0.83% at
#     conc 1, which resolves the ~2.2-2.5% axis-1/axis-3 effects; conc-10 effects are
#     6-12 sd at n=1 and need no replication.
# `ONLY` subsets the list by label so a died allocation resumes instead of restarting.
#
# THREE GATE TRAPS this arm handles that a naive reuse of specdec_kopt_arm.sh would
# fail closed on (or worse, pass wrongly):
#   1. The Humming attestation is backend-CONDITIONAL. CUTLASS serves get the converse
#      gate instead: assert `--quantization humming` is absent.
#   2. The loaded-GiB gate is DIRECTIONAL. INT4 drafter = 28.78 GiB, bf16 = 29.26 GiB,
#      so a single "< 29.05" bound would abort every axis-3 bf16 serve.
#   3. A bf16 drafter has NO quantized linears, so the WNA16 kernel-set gate must be
#      skipped for it -- an empty kernel set is correct there, not a failure.
#
# Prompts are the same staged SPEED-Bench bytes as phases D-I.2 (controller gates the
# sha256), same --random-seed 42. Request counts are 30/80 rather than phase H's
# 40/100: within-cell request scatter is not the limiting noise here (cross-engine is),
# and the cut is what fits 26 serves into the agreed budget.
set -uo pipefail

ROOT=${ROOT:?}; ARM=${ARM:?}; RUN_TS=${RUN_TS:?}
CKPT=${CKPT:?}; PORT_BASE=${PORT_BASE:?}
SERVES=${SERVES:?}                       # newline-separated specs
ONLY=${ONLY:-}                           # optional comma-separated label subset
DRAFTER_INT4=${DRAFTER_INT4:?}
DRAFTER_BF16=${DRAFTER_BF16:?}
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
ACC_MIN=${ACC_MIN:-3.0}                  # low-entropy acceptance sanity (k>=5)
# Drafter identity bounds, from phase G's measured loads (28.78 vs 29.26 GiB).
GIB_INT4_MAX=${GIB_INT4_MAX:-29.05}
GIB_BF16_MIN=${GIB_BF16_MIN:-29.05}
GIB_BF16_MAX=${GIB_BF16_MAX:-29.60}

C=$ROOT/arm-$ARM
mkdir -p "$C"
note() { echo "[arm-$ARM $(date -u +%H:%M:%S)] $1" | tee -a "$C/client.log"; }

export HF_HOME=/mnt/nfs/hoangduy/cache/huggingface HF_DATASETS_OFFLINE=1 HF_HUB_OFFLINE=1
export PATH="$PERF_VENV/bin:$PATH"
export PYBIN="$PERF_VENV/bin/python"

note "host=$(hostname) run_ts=$RUN_TS n_c1=$N_C1 n_c10=$N_C10"
note "only='${ONLY:-<all>}'"
printf 'host=%s\nrun_ts=%s\nn_c1=%s\nn_c10=%s\nonly=%s\n' \
  "$(hostname)" "$RUN_TS" "$N_C1" "$N_C10" "${ONLY:-<all>}" > "$C/arm-identity.txt"

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

# Map a kernel-variant name to the env levers that produce it, and to the WNA16
# kernel set the serve MUST report. Phase I/I.2 established both sides.
#   default      Machete x8 + Marlin lm_head   (vLLM's own choice; 25008 % 128 == 48)
#   hum-lmhead   Machete x8 + Humming lm_head  (needs patch_vllm_humming_lmhead.py)
#   hum-all      Humming x9
#   machete-all  Machete x9                    (draft vocab padded 200064 -> 200704)
kernel_env() {
  case $1 in
    default)     unset VLLM_DISABLED_KERNELS; unset LLMC_EAGLE3_LMHEAD_PAD ;;
    hum-lmhead)  export VLLM_DISABLED_KERNELS=MarlinLinearKernel; unset LLMC_EAGLE3_LMHEAD_PAD ;;
    hum-all)     export VLLM_DISABLED_KERNELS=MarlinLinearKernel,MacheteLinearKernel; unset LLMC_EAGLE3_LMHEAD_PAD ;;
    machete-all) unset VLLM_DISABLED_KERNELS; export LLMC_EAGLE3_LMHEAD_PAD=1024 ;;
    *) note "ABORT unknown kernel variant: $1"; return 1 ;;
  esac
  return 0
}
kernel_expect() {
  case $1 in
    default)     echo "Machete,Marlin" ;;
    hum-lmhead)  echo "Humming,Machete" ;;
    hum-all)     echo "Humming" ;;
    machete-all) echo "Machete" ;;
  esac
}

# $1 label  $2 k  $3 backend  $4 drafter  $5 kernel  $6 port
serve_config() {
  local label=$1 k=$2 backend=$3 drafter=$4 kernel=$5 port=$6
  CUR=$C/$label; CUR_PORT=$port
  mkdir -p "$CUR" "$CUR/metrics"

  local dpath=""
  case $drafter in
    int4) dpath=$DRAFTER_INT4 ;;
    bf16) dpath=$DRAFTER_BF16 ;;
    none) dpath="" ;;
    *) note "ABORT unknown drafter: $drafter"; return 1 ;;
  esac

  if [ "$k" -gt 0 ]; then
    test -f "$dpath/config.json" || { note "ABORT drafter missing: $dpath"; return 1; }
    export EXTRA_VLLM_ARGS="--speculative-config {\"method\":\"eagle3\",\"model\":\"$dpath\",\"num_speculative_tokens\":$k,\"attention_backend\":\"FLASH_ATTN\"}"
  else
    unset EXTRA_VLLM_ARGS
  fi
  kernel_env "$kernel" || return 1

  printf '%s\n' "${EXTRA_VLLM_ARGS:-<none>}" > "$CUR/extra-vllm-args.txt"
  printf 'k=%s\nbackend=%s\ndrafter=%s\ndrafter_path=%s\nkernel=%s\nVLLM_DISABLED_KERNELS=%s\nLLMC_EAGLE3_LMHEAD_PAD=%s\n' \
    "$k" "$backend" "$drafter" "${dpath:-<none>}" "$kernel" \
    "${VLLM_DISABLED_KERNELS:-<unset>}" "${LLMC_EAGLE3_LMHEAD_PAD:-<unset>}" > "$CUR/cell-config.txt"

  note "serve $label (k=$k backend=$backend drafter=$drafter kernel=$kernel) on $port"
  M3_W4A8_BACKEND="$backend" \
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

# $1 label  $2 k  $3 backend  $4 drafter  $5 kernel
gate_config() {
  local label=$1 k=$2 backend=$3 drafter=$4 kernel=$5

  # --- TRAP 1: backend gate is conditional, and CUTLASS gets the converse check ---
  if [ "$backend" = humming ]; then
    ( cd "$REPO" && python -m pipeline.m3_humming_w4a8 attest \
        --preflight "$CUR/serve.log.humming-preflight.json" --log "$CUR/serve.log" \
        --out "$CUR/backend-attestation.json" ) >>"$C/client.log" 2>&1 \
      || { note "$label ABORT: Humming attestation failed (fail closed)"; return 1; }
    note "$label: Humming backend attested"
  else
    grep -qE "quantization[ =']+humming" "$CUR/serve.log" \
      && { note "$label ABORT: cutlass arm shows humming quantization"; return 1; }
    grep -ohE "quantization=[A-Za-z0-9_-]+" "$CUR/serve.log" | head -3 > "$CUR/quant-method.txt"
    grep -ci "humming" "$CUR/serve.log" > "$CUR/humming-mentions.txt" 2>/dev/null || true
    note "$label: CUTLASS backend confirmed (no humming quantization in log)"
  fi

  if [ "$k" -gt 0 ]; then
    grep -q "'num_speculative_tokens': $k" "$CUR/serve.log" \
      || { note "$label ABORT: serve.log does not confirm k=$k"; return 1; }
    grep -i "speculative\|Detected EAGLE" "$CUR/serve.log" | head -40 > "$CUR/spec-boot.log"

    # --- TRAP 2: the loaded-GiB bound is directional, not a single ceiling ---
    local gib
    gib=$(grep -m1 -oE "Model loading took [0-9.]+ GiB" "$CUR/serve.log" | grep -oE "[0-9.]+")
    printf '%s\n' "${gib:-unknown}" > "$CUR/model-loading-gib.txt"
    [ -n "$gib" ] || { note "$label ABORT: no 'Model loading took' line"; return 1; }
    if [ "$drafter" = int4 ]; then
      awk -v g="$gib" -v m="$GIB_INT4_MAX" 'BEGIN{exit !(g < m)}' \
        || { note "$label ABORT: loaded $gib GiB >= $m -- INT4 drafter not resident"; return 1; }
      note "$label: k=$k confirmed, INT4 drafter resident ($gib GiB)"
    else
      awk -v g="$gib" -v lo="$GIB_BF16_MIN" -v hi="$GIB_BF16_MAX" \
        'BEGIN{exit !(g >= lo && g < hi)}' \
        || { note "$label ABORT: loaded $gib GiB outside bf16 band [$GIB_BF16_MIN,$GIB_BF16_MAX)"; return 1; }
      note "$label: k=$k confirmed, bf16 drafter resident ($gib GiB)"
    fi

    # Drafter wiring. Fatal for INT4 (phase G proved both lines appear); recorded but
    # non-fatal for bf16, where this arm is the first to assert them -- an unproven
    # gate must not be able to burn a 15-minute serve.
    local wiring=0
    grep -q "embed_tokens identical to the target model" "$CUR/serve.log" || wiring=1
    grep -q "distinct lm_head weights" "$CUR/serve.log" || wiring=$((wiring + 2))
    printf 'wiring_flags=%s (1=embed-not-shared 2=lm_head-shared)\n' "$wiring" > "$CUR/drafter-wiring.txt"
    if [ "$wiring" != 0 ]; then
      if [ "$drafter" = int4 ]; then
        note "$label ABORT: drafter wiring changed (flags=$wiring)"; return 1
      fi
      note "$label WARN: bf16 drafter wiring flags=$wiring (recorded, not fatal)"
    fi

    # --- TRAP 3: a bf16 drafter has no quantized linears, so no WNA16 kernel set ---
    grep -ohE "Using [A-Za-z]+LinearKernel for [A-Za-z]+" "$CUR/serve.log" \
      | sort | uniq -c | sort -rn > "$CUR/kernel-census.txt" 2>/dev/null || true
    if [ "$drafter" = int4 ]; then
      local got want
      got=$(grep -ohE "Using ([A-Za-z]+)LinearKernel for CompressedTensorsWNA16" "$CUR/serve.log" \
            | sed -E 's/Using ([A-Za-z]+)LinearKernel.*/\1/' | sort -u | paste -sd, -)
      want=$(kernel_expect "$kernel")
      printf '%s\n' "${got:-<none>}" > "$CUR/wna16-kernels.txt"
      [ "$got" = "$want" ] \
        || { note "$label ABORT: WNA16 kernels '$got' != expected '$want'"; return 1; }
      note "$label: WNA16 kernel set = $got (as designed for '$kernel')"
      # Marlin pads the 25008-wide lm_head and warns per forward; the warning must be
      # present iff Marlin actually holds that layer.
      local padwarn=absent
      grep -q "padded.*sliced on every forward\|Marlin.*pad" "$CUR/serve.log" && padwarn=present
      printf '%s\n' "$padwarn" > "$CUR/marlin-pad-warning.txt"
      case "$kernel:$padwarn" in
        default:present|hum-lmhead:absent|hum-all:absent|machete-all:absent) ;;
        *) note "$label WARN: marlin pad warning $padwarn unexpected for '$kernel'" ;;
      esac
    else
      printf '<n/a: bf16 drafter has no quantized linears>\n' > "$CUR/wna16-kernels.txt"
      note "$label: bf16 drafter -- WNA16 kernel gate skipped by design"
    fi
  else
    grep -q "num_speculative_tokens" "$CUR/serve.log" \
      && { note "$label ABORT: k=0 control shows a speculative config"; return 1; }
    local gib
    gib=$(grep -m1 -oE "Model loading took [0-9.]+ GiB" "$CUR/serve.log" | grep -oE "[0-9.]+")
    printf '%s\n' "${gib:-unknown}" > "$CUR/model-loading-gib.txt"
    note "$label: k=0 control confirmed (no speculative config, ${gib:-?} GiB)"
  fi
  return 0
}

snap() { curl -sf "http://localhost:$CUR_PORT/metrics" -o "$CUR/metrics/$1.txt" 2>/dev/null || true; }

# $1 conc  $2 request count  $3 cell (== staged prompt basename and output dir)
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

# Acceptance is the numerics check: a broken drafter kernel or a mis-served lm_head
# collapses it toward 1.0 while ITL still looks plausible. Only asserted on the
# low-entropy tier at k>=5, where phases G-I.2 put the floor at 3.74-3.92.
acc_gate() {
  local label=$1 k=$2 cell=$3
  [ "$cell" = 8k-low ] || return 0
  [ "$k" -ge 5 ] || return 0
  local pre=$CUR/metrics/sb-$cell-c1-pre.txt post=$CUR/metrics/sb-$cell-c1-post.txt
  # An unverifiable k>=5 cell is not a pass. This gate is the ONLY numerics check in
  # the arm -- a broken drafter kernel or a mis-served lm_head collapses acceptance
  # toward 1.0 while ITL still looks plausible -- so "could not check" must be
  # consequential rather than a warning.
  # NOTE: the main loop calls acc_gate without reading $?, so `rc_all=1` (window exit
  # code) plus the note is the effective signal; the return value is conventional
  # only. This arm has no smoke serve, so there is nothing to abort early on.
  if ! { [ -s "$pre" ] && [ -s "$post" ]; }; then
    note "$label ABORT-LEVEL: no metrics for acceptance gate -- cannot verify numerics"
    rc_all=1
    return 1
  fi
  local acc
  acc=$("$PYBIN" - "$pre" "$post" <<'PY'
import re, sys
def g(p, name):
    pat = re.compile(rf"^vllm:{name}(?:\{{.*?\}})?\s+([0-9.eE+]+)$", re.M)
    vals = pat.findall(open(p).read())
    # NOT `sum(...) or None`: a fresh serve's spec-dec counters are a legitimate
    # 0.0, and `0.0 or None` is None -- indistinguishable from "metric absent".
    # That silently disabled this whole gate for every k>=5 cell (it reported
    # "acceptance not computable" and returned 0), so the numerics floor enforced
    # nothing for an entire window. Only an EMPTY match list means absent.
    return sum(float(v) for v in vals) if vals else None
pre, post = sys.argv[1], sys.argv[2]
d0, a0 = g(pre, "spec_decode_num_drafts_total"), g(pre, "spec_decode_num_accepted_tokens_total")
d1, a1 = g(post, "spec_decode_num_drafts_total"), g(post, "spec_decode_num_accepted_tokens_total")
print("" if None in (d0, a0, d1, a1) or d1 == d0 else f"{1 + (a1 - a0) / (d1 - d0):.4f}")
PY
)
  printf '%s\n' "${acc:-unknown}" > "$CUR/accepted-$cell-c1.txt"
  if [ -z "$acc" ]; then
    note "$label ABORT-LEVEL: acceptance not computable (spec-dec counters absent or drafts==0)"
    rc_all=1
    return 1
  fi
  # `m` is an AWK variable, not a shell one: referring to "$m" in these shell notes
  # is an unbound-variable fatal under `set -u`. That regression killed a whole
  # window from the gate's SUCCESS path (job 13429, 2026-07-29) -- the floor passed
  # at 4.62 and the arm then died on the congratulation. Use "$ACC_MIN" here.
  awk -v a="$acc" -v m="$ACC_MIN" 'BEGIN{exit !(a >= m)}' \
    || { note "$label ABORT-LEVEL: accepted $acc < $ACC_MIN -- numerics suspect"; rc_all=1; return 1; }
  note "$label: accepted length $acc (>= $ACC_MIN)"
  return 0
}

analyze() {
  local label=$1 k=$2 cell=$3
  [ -d "$CUR/speedbench/$cell" ] || return 0
  "$PYBIN" "$BENCH/performance/workloads/analyze_perf.py" \
    --run-dir "$CUR/speedbench/$cell" --mode "speedbench_$cell" \
    --label "$ARM-$label" --precision W4AFP8 --gpu 8xH100 --num-gpus 8 \
    --spec-decode "$([ "$k" -gt 0 ] && echo "eagle3-k$k" || echo none)" \
    >>"$CUR/analyze-$cell.log" 2>&1 || note "WARN analyze_perf failed for $cell"
}

# ---- main loop over the controller's ordered serve list ----
port=$PORT_BASE
n_done=0; n_skip=0; n_fail=0
while IFS= read -r spec; do
  spec=$(printf '%s' "$spec" | sed 's/^[[:space:]]*//;s/[[:space:]]*$//')
  [ -n "$spec" ] || continue
  case $spec in \#*) continue ;; esac

  IFS=';' read -r label k backend drafter kernel cells <<<"$spec"
  if ! enabled "$label"; then
    note "skip $label (not in ONLY)"; n_skip=$((n_skip + 1)); port=$((port + 1)); continue
  fi

  if serve_config "$label" "$k" "$backend" "$drafter" "$kernel" "$port"; then
    if gate_config "$label" "$k" "$backend" "$drafter" "$kernel"; then
      for cell in $cells; do
        run_cell 1  "$N_C1"  "$cell"
        run_cell 10 "$N_C10" "$cell"
        acc_gate "$label" "$k" "$cell"
        analyze "$label" "$k" "$cell"
      done
      n_done=$((n_done + 1))
    else
      note "$label gates failed -- no cells run"; rc_all=1; n_fail=$((n_fail + 1))
    fi
  else
    note "$label serve failed"; rc_all=1; n_fail=$((n_fail + 1))
  fi
  stop_serve
  wait_gpus_free 60
  port=$((port + 1))
  printf '%s done=%s skip=%s fail=%s\n' "$label" "$n_done" "$n_skip" "$n_fail" >> "$C/progress.txt"
done <<< "$SERVES"

note "arm done rc=$rc_all served=$n_done skipped=$n_skip failed=$n_fail"
printf 'rc=%s served=%s skipped=%s failed=%s\n' "$rc_all" "$n_done" "$n_skip" "$n_fail" > "$C/arm-done.txt"
exit "$rc_all"
