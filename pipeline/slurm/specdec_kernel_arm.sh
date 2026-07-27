#!/usr/bin/env bash
# One arm of EAGLE3 spec-dec phase I -- which W4A16 kernel should the drafter use?
#
# THE FINDING THIS TESTS
# ---------------------
# The drafter is W4A16 compressed-tensors (group 128, symmetric, no g_idx, bf16
# activations). vLLM's choose_mp_linear_kernel walks _POSSIBLE_KERNELS[CUDA] in
# priority order -- Cutlass(W4A8), Machete, AllSpark, Marlin, Humming, Conch,
# Exllama, Triton -- and takes the first whose can_implement passes. On H100 that
# yields a SPLIT for our drafter, confirmed in phase G's serve.log (the log line is
# deduped per kernel name, so 8 Machete + 8 Marlin lines == one of each per rank):
#
#     8 of 9 linears -> MacheteLinearKernel
#     lm_head        -> MarlinLinearKernel
#
# because check_machete_supports_shape needs out_features % 128 == 0 and lm_head's
# per-rank output is 200064/8 = 25008 (% 128 == 48). lm_head is 153.6 M of the
# 254.3 M params read per rank per drafter forward -- 60% of the weight traffic. vLLM
# warns about it directly:
#
#     WARNING marlin_utils.py:237  Marlin requires thread-tile padding ... padded/
#     sliced on every forward; performance may be degraded.
#
# marlin_padded_nk(25008, 6144, 128) -> (25024, 6144): only +16 columns (+0.064% of
# bytes), so the cost is a per-forward pad/slice and an extra launch, NOT bandwidth.
#
# ELIGIBILITY, established by reading every can_implement (not assumed):
#   CutlassW4A8  no -- requires act_type == float8_e4m3fn; drafter is A16
#   Machete      8/9 -- rejects lm_head on the % 128 rule above
#   AllSpark     no -- Ampere-only: check_allspark_supported_dtype_shape has an
#                      explicit `else: return False` for capability >= 90
#   Marlin       all -- zero-pads tiles; currently owns lm_head
#   Humming      all -- rejects ONLY g_idx and zero_points, which we have neither of.
#                      No shape constraint at all, so it takes lm_head at N=25008
#                      unpadded. Sits BELOW Marlin in the priority list and Marlin
#                      always succeeds, so this path has never run on CUDA.
#   Conch        no -- conch-triton-kernels not installed
#   Exllama      no -- float16 activations only; we serve bf16
#   Triton       all -- fallback (N % 8)
#
# THE CELLS (all k=5, 8k-low, conc 1 and 10, same node, serial)
#   A-baseline      no env                                     -> {Machete, Marlin}
#   B-hum-lmhead    VLLM_DISABLED_KERNELS=Marlin*              -> {Machete, Humming}
#   C-hum-all       VLLM_DISABLED_KERNELS=Machete*,Marlin*      -> {Humming}
#   D-machete-all   LLMC_EAGLE3_LMHEAD_PAD=1024                -> {Machete}
#   A-repeat        no env, re-served last                     -> drift control
#
# WHY THE ENV LEVER IS DRAFTER-SCOPED (this is what makes the A/B clean)
# VLLM_DISABLED_KERNELS matches on EXACT class name
# (`kernel.__name__ in envs.VLLM_DISABLED_KERNELS`), and the drafter's 9 linears are
# the only CompressedTensorsWNA16 consumers in the process: the target's lm_head is
# in its checkpoint's `ignore` list (unquantized bf16) and its group_0 declares
# input_activations (W4A8 -> a different scheme entirely). Phase G's log shows no
# CutlassW4A8LinearKernel line at all, corroborating that. Disabling Marlin also does
# not touch MarlinFP8ScaledMMLinearKernel -- different class name.
#
# `--linear-backend` was rejected as the lever: it filters EVERY layer type globally
# (would perturb the target) and _LINEAR_BACKEND_KERNEL_MAP has no "humming" key at
# all, so --linear-backend=humming raises outright.
#
# Cell D uses pipeline/slurm/patch_vllm_eagle3_lmhead_pad.py, which is INERT unless
# LLMC_EAGLE3_LMHEAD_PAD is set (default 64 == DEFAULT_VOCAB_PADDING_SIZE). At 1024
# the draft vocab pads 200064 -> 200704 (25088/rank, % 128 == 0) so Machete takes
# lm_head. Padded logits are masked by LogitsProcessor on both paths, so no padded id
# can be emitted; see that script's docstring for why physically padding the
# checkpoint instead would be unsafe.
#
# Drafter, prompts, seed, request counts: identical to phases G/H (controller gates
# the sha256 of both the derived INT4 artifact and the staged SPEED-Bench bytes).
# Only the kernel assignment varies.
set -uo pipefail

ROOT=${ROOT:?}; ARM=${ARM:?}; RUN_TS=${RUN_TS:?}
CKPT=${CKPT:?}; PORT_BASE=${PORT_BASE:?}
CELL=${CELL:-8k-low}
SPEC_K=${SPEC_K:-5}
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
# k=5 on 8k-low measured 3.915 accepted in phase G and 3.863/3.014ms in phase H. A
# broken lm_head (e.g. corrupted padded logits) collapses acceptance toward 1.0, so
# this catches catastrophic breakage without being brittle about a few percent.
ACC_MIN=${ACC_MIN:-3.0}
# Comma-separated cell subset. Phase I.2 re-runs only A-baseline,B-hum-lmhead,
# C-hum-all after the prepare_humming_layer ParallelLMHead fix
# (pipeline/slurm/patch_vllm_humming_lmhead.py); D and A-repeat are already
# measured and the kernel-is-not-a-lever conclusion does not need them again.
CELLS=${CELLS:-A-baseline,B-hum-lmhead,C-hum-all,D-machete-all,A-repeat}
enabled() { case ",$CELLS," in *",$1,"*) return 0;; *) return 1;; esac; }

C=$ROOT/arm-$ARM
mkdir -p "$C"
note() { echo "[arm-$ARM $(date -u +%H:%M:%S)] $1" | tee -a "$C/client.log"; }

export HF_HOME=/mnt/nfs/hoangduy/cache/huggingface HF_DATASETS_OFFLINE=1 HF_HUB_OFFLINE=1
export PATH="$PERF_VENV/bin:$PATH"
export PYBIN="$PERF_VENV/bin/python"
note "host=$(hostname) cell=$CELL k=$SPEC_K run_ts=$RUN_TS"

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

# $1 label  $2 port  $3 disabled-kernels ("" for none)  $4 lmhead-pad ("" for default)
serve_config() {
  local label=$1 port=$2 disabled=$3 pad=$4
  CUR=$C/$label; CUR_PORT=$port
  mkdir -p "$CUR" "$CUR/metrics"
  test -f "$DRAFTER/config.json" || { note "ABORT drafter missing: $DRAFTER"; return 1; }

  export EXTRA_VLLM_ARGS="--speculative-config {\"method\":\"eagle3\",\"model\":\"$DRAFTER\",\"num_speculative_tokens\":$SPEC_K,\"attention_backend\":\"FLASH_ATTN\"}"
  if [ -n "$disabled" ]; then export VLLM_DISABLED_KERNELS="$disabled"; else unset VLLM_DISABLED_KERNELS; fi
  if [ -n "$pad" ]; then export LLMC_EAGLE3_LMHEAD_PAD="$pad"; else unset LLMC_EAGLE3_LMHEAD_PAD; fi
  {
    printf 'VLLM_DISABLED_KERNELS=%s\n' "${VLLM_DISABLED_KERNELS:-<unset>}"
    printf 'LLMC_EAGLE3_LMHEAD_PAD=%s\n' "${LLMC_EAGLE3_LMHEAD_PAD:-<unset (default 64)>}"
    printf 'EXTRA_VLLM_ARGS=%s\n' "$EXTRA_VLLM_ARGS"
  } > "$CUR/kernel-env.txt"

  note "serve $label on $port (disabled='${disabled:-none}' pad='${pad:-default}')"
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
    grep -nE "Traceback|AttributeError|KeyError|ValueError|Failed to find a kernel|Error" \
      "$CUR/serve.log" | tail -40 | tee -a "$C/client.log"
    return 1
  fi
  export BASE_URL="http://localhost:$port"
  return 0
}

# $1 label  $2 expected kernel set, comma-separated sorted short names (e.g. "Machete,Marlin")
# $3 expect_pad_warning: 0 = must be absent, 1 = must be present, - = don't care
gate_config() {
  local label=$1 want=$2 want_warn=$3
  ( cd "$REPO" && python -m pipeline.m3_humming_w4a8 attest \
      --preflight "$CUR/serve.log.humming-preflight.json" --log "$CUR/serve.log" \
      --out "$CUR/backend-attestation.json" ) >>"$C/client.log" 2>&1 \
    || { note "$label: Humming attestation failed (fail closed)"; return 1; }

  grep -q "'num_speculative_tokens': $SPEC_K" "$CUR/serve.log" \
    || { note "$label ABORT: serve.log does not confirm k=$SPEC_K"; return 1; }
  grep -q "embed_tokens identical to the target model" "$CUR/serve.log" \
    || { note "$label ABORT: target embedding not shared with drafter"; return 1; }
  grep -q "distinct lm_head weights" "$CUR/serve.log" \
    || { note "$label ABORT: drafter lm_head was shared, not its own"; return 1; }

  # Full kernel census: every "Using X for Y" line, so a later diff can prove the
  # env lever never touched a non-WNA16 (i.e. target-model) scheme.
  grep -oE "Using [A-Za-z0-9]+ for [A-Za-z0-9]+" "$CUR/serve.log" \
    | sort | uniq -c | sort -rn > "$CUR/kernel-census.txt"

  # The WNA16 kernel set actually chosen for the drafter.
  local got
  got=$(grep -oE "Using [A-Za-z0-9]+LinearKernel for CompressedTensorsWNA16" "$CUR/serve.log" \
        | sed -e 's/^Using //' -e 's/LinearKernel for CompressedTensorsWNA16$//' \
        | sort -u | paste -sd, -)
  printf '%s\n' "${got:-<none>}" > "$CUR/wna16-kernels.txt"
  if [ "$got" != "$want" ]; then
    note "$label ABORT: WNA16 kernel set is '${got:-none}', expected '$want'"
    grep -nE "cannot implement|disabled by environment|Failed to find a kernel" \
      "$CUR/serve.log" | tail -20 | tee -a "$C/client.log"
    return 1
  fi

  local warn=0
  grep -q "Marlin requires thread-tile padding" "$CUR/serve.log" && warn=1
  printf '%s\n' "$warn" > "$CUR/marlin-pad-warning.txt"
  if [ "$want_warn" != "-" ] && [ "$warn" != "$want_warn" ]; then
    note "$label ABORT: marlin pad warning present=$warn, expected $want_warn"
    return 1
  fi

  local gib
  gib=$(grep -m1 -oE "Model loading took [0-9.]+ GiB" "$CUR/serve.log" | grep -oE "[0-9.]+")
  printf '%s\n' "${gib:-unknown}" > "$CUR/model-loading-gib.txt"
  [ -n "$gib" ] || { note "$label ABORT: no 'Model loading took' line"; return 1; }
  awk -v g="$gib" -v m="$LOADED_GIB_MAX" 'BEGIN{exit !(g < m)}' \
    || { note "$label ABORT: loaded $gib GiB >= $m -- INT4 drafter not resident"; return 1; }

  note "$label: kernels=[$got] pad_warn=$warn drafter INT4 resident (${gib} GiB)"
  grep -i "speculative\|Detected EAGLE" "$CUR/serve.log" | head -40 > "$CUR/spec-boot.log"
  return 0
}

snap() { curl -sf "http://localhost:$CUR_PORT/metrics" -o "$CUR/metrics/$1.txt" 2>/dev/null || true; }

# Accepted length from the pre/post Prometheus snapshots of one cell.
# Cell D changes the lm_head's numerics path (padded shard); if padding ever leaked
# into the emitted logits, acceptance would collapse. Gate on it rather than trust it.
acc_check() {
  local out_cell=$1 conc=$2
  local a
  a=$("$PYBIN" - "$CUR/metrics/sb-$out_cell-c$conc-pre.txt" \
                 "$CUR/metrics/sb-$out_cell-c$conc-post.txt" <<'PY' 2>/dev/null
import re, sys
def m(p, name):
    try: body = open(p).read()
    except OSError: return None
    v = re.findall(rf"^vllm:{name}(?:\{{.*?\}})?\s+([0-9.eE+]+)$", body, re.M)
    return sum(float(x) for x in v) if v else None
pre, post = sys.argv[1], sys.argv[2]
d0, a0 = m(pre, "spec_decode_num_drafts_total"), m(pre, "spec_decode_num_accepted_tokens_total")
d1, a1 = m(post, "spec_decode_num_drafts_total"), m(post, "spec_decode_num_accepted_tokens_total")
print("nan" if None in (d0, a0, d1, a1) or d1 == d0 else f"{1 + (a1 - a0) / (d1 - d0):.4f}")
PY
)
  printf '%s\n' "${a:-nan}" > "$CUR/accepted-$out_cell-c$conc.txt"
  if [ "${a:-nan}" = "nan" ]; then
    note "WARN accepted length unavailable for $out_cell c$conc"
    return 0
  fi
  if awk -v a="$a" -v m="$ACC_MIN" 'BEGIN{exit !(a < m)}'; then
    note "ABORT-CELL: accepted length $a < $ACC_MIN -- drafter numerics look broken"
    rc_all=1
    return 1
  fi
  note "accepted length $out_cell c$conc = $a"
  return 0
}

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
  acc_check "$out_cell" "$conc" || true
}

analyze() {
  local label=$1 out_cell=$2
  [ -d "$CUR/speedbench/$out_cell" ] || return 0
  "$PYBIN" "$BENCH/performance/workloads/analyze_perf.py" \
    --run-dir "$CUR/speedbench/$out_cell" --mode "speedbench_$out_cell" \
    --label "$ARM-$label" --precision W4AFP8 --gpu 8xH100 --num-gpus 8 \
    --spec-decode "eagle3-int4-k$SPEC_K-$label" \
    >>"$CUR/analyze-$out_cell.log" 2>&1 || note "WARN analyze_perf failed for $out_cell"
}

# $1 label  $2 port  $3 disabled  $4 pad  $5 expected kernel set  $6 expect_pad_warning
#                                                                 $7 out_cell
run_one() {
  local label=$1 port=$2 disabled=$3 pad=$4 want=$5 want_warn=$6 out_cell=$7
  if serve_config "$label" "$port" "$disabled" "$pad"; then
    if gate_config "$label" "$want" "$want_warn"; then
      run_cell 1 40 "$out_cell"
      run_cell 10 100 "$out_cell"
      analyze "$label" "$out_cell"
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

note "cells enabled: $CELLS"
p=$PORT_BASE
enabled A-baseline && \
run_one "A-baseline"    "$p" ""                                          ""     "Machete,Marlin"  1 "$CELL"; p=$((p+1))
enabled B-hum-lmhead && \
run_one "B-hum-lmhead"  "$p" "MarlinLinearKernel"                        ""     "Humming,Machete" 0 "$CELL"; p=$((p+1))
enabled C-hum-all && \
run_one "C-hum-all"     "$p" "MacheteLinearKernel,MarlinLinearKernel"    ""     "Humming"         0 "$CELL"; p=$((p+1))
# Cell D runs LAST of the four: it is the only one needing a vLLM source patch, so a
# failure there cannot cost the three env-only cells.
enabled D-machete-all && \
run_one "D-machete-all" "$p" ""                                          "1024" "Machete"         0 "$CELL"; p=$((p+1))

# Drift control: re-serve the baseline on a fresh engine at the end of the window.
# Any kernel effect must exceed the gap this reveals.
if enabled A-repeat; then
  note "drift control: re-serving A-baseline"
  run_one "A-repeat"    "$p" ""                                          ""     "Machete,Marlin"  1 "$CELL-repeat"
fi

note "arm done rc=$rc_all"
exit "$rc_all"
