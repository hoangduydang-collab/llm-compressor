#!/usr/bin/env bash
# Localize the vLLM 0.26.0 MiniMax-M3 illegal-memory-access, k=0 (NO spec-dec).
#
# ============================ WHY THESE THREE ARMS ============================
# Established offline (docs/m3-dspark-blockers-026.md + the D-k0-a serve.log):
#   * The live indexer/attend path on H100 is TRITON, not MSA:
#       "MiniMax M3 indexer: selected Triton (no fmha_sm100) ... sm100=False"
#     so 0.26.0's large nvidia/indexer_msa.py + nvidia/ops/ rewrite is dead code
#     for us and is NOT a candidate.
#   * common/ops/index_topk.py is functionally unchanged on our path (the 0.26.0
#     diff only factors minimax_m3_index_decode into a _score + top-k pair; with
#     score_out=None the behaviour is identical).
#   * ll_bf16 (0.26.0's new dimension-ungated cuteDSL router GEMM) is numerically
#     CORRECT at M3's (K=6144, N=128) for M=1..16 on both dispatch backends
#     -- measured, /mnt/nfs/hoangduy/results/m3-ll-bf16-probe/20260728T083753Z.
#     (compute-sanitizer pass was cut off by the srun wall clock, so a silent
#     out-of-bounds that does not corrupt the result is not excluded.)
#   * What DID change on the live path: MiniMaxM3SparseBackend.get_kv_cache_shape
#     went from the 5-D separated layout
#         (num_blocks, 2, block_size, num_kv_heads, head_size)
#     to a 4-D PACKED layout
#         (num_blocks, num_kv_heads, block_size, 2 * head_size)
#     with matching stride_order changes, and both Triton attend kernels were
#     rewritten (+213 lines) to carry in-kernel fp8 K/V dequant via new
#     k_scale/v_scale pointers. We serve with --kv-cache-dtype fp8, i.e. we take
#     the newly added dequant branch; most M3 users run the default bf16 KV and
#     never touch it.
#   * The crash fires on the FIRST mixed prefill+decode batch (~1 s into the
#     conc-10 cell) after a fully clean conc-1 cell -- so it is deterministic and
#     cheap to reproduce, not a rare shape.
#
# Arms, each a fresh serve on the same node, ~10 min apiece:
#   A stock   fp8 KV, cudagraphs on  -> is the repro deterministic at all?
#   B bf16kv  KV_CACHE_DTYPE=auto    -> is the new in-kernel fp8 dequant to blame?
#   C eager   ENFORCE_EAGER=1        -> kernel bug, or cudagraph replay of the
#                                       mixed-batch (PIECEWISE) graph?
# Both knobs already exist as env vars in run_vllm_http_serve_smoke.sh, so NO
# launcher is edited -- the h114 window is live and re-reads its scripts per serve.
#
# Arm A is the gate: if it does not reproduce, B and C prove nothing and the run
# says so rather than reporting their PASS as a fix.
set -uo pipefail

REPO=/mnt/nfs/hoangduy/projects/llm-compressor
PY=/mnt/nfs/hoangduy/venvs/serve-026/bin/python
RESULTS=/mnt/nfs/hoangduy/results/m3-026-ima-bisect
CKPT="$REPO/artifacts/m3-awq-gptq-prepared/gptq-checkpoint-vllm-w123-abi-overlay"
PROMPTS="$REPO/artifacts/aiperf-datasets/speedbench/8k-low.jsonl"
BURST="$REPO/pipeline/diag/m3_026_ima_burst.py"
SERVE="$REPO/pipeline/slurm/run_vllm_http_serve_smoke.sh"
SERVED_NAME=MiniMaxAI/MiniMax-M3
TOKENIZER="$CKPT"
MAX_MODEL_LEN=131072
BACKEND=humming
NCONC=${NCONC:-10}
MAXTOK=${MAXTOK:-256}

NODE=${NODE:?set NODE, e.g. NODE=gpu-h104}
PORT_BASE=${PORT_BASE:-8160}
ONLY=${ONLY:-}

RUN_TS=$(date -u +%Y%m%dT%H%M%SZ)
ROOT=${ROOT:-$RESULTS/$RUN_TS}
mkdir -p "$ROOT"
fail() { echo "[controller] GATE FAILED: $1" | tee -a "$ROOT/gates.log"; exit 1; }
say()  { echo "[controller] $1" | tee -a "$ROOT/gates.log"; }

say "root=$ROOT node=$NODE venv=serve-026 conc=$NCONC max_tokens=$MAXTOK"

# ======================= PRE-FLIGHT (login node, seconds) =====================
for f in "$PROMPTS" "$BURST" "$SERVE"; do
  [ -s "$f" ] || fail "missing file: $f"
done
[ -d "$CKPT" ] || fail "missing checkpoint: $CKPT"
[ -x "$PY" ]   || fail "missing python: $PY"

sha=$(sha256sum "$PROMPTS" | cut -c1-12)
[ "$sha" = bfcf60739f43 ] || fail "8k-low prompts changed ($sha != bfcf60739f43)"
say "prompts match every EAGLE3/DSpark window (8k-low $sha)"

v=$("$PY" -c 'import vllm; print(vllm.__version__)' 2>&1)
[ "$v" = "0.26.0" ] || fail "expected vLLM 0.26.0 in serve-026, got '$v'"
say "vllm $v"

"$PY" -c 'import requests' 2>/dev/null || fail "serve-026 python lacks requests"
"$PY" -m py_compile "$BURST" || fail "burst client does not compile"
say "burst client compiles"

{
  echo "run_ts   : $RUN_TS"
  echo "node     : $NODE"
  echo "vllm     : $v"
  echo "ckpt     : $CKPT"
  echo "prompts  : $PROMPTS ($sha)"
  echo "conc     : $NCONC"
  echo "max_tok  : $MAXTOK"
  echo "commit   : $(cd "$REPO" && git rev-parse --short HEAD)"
} | tee "$ROOT/provenance.txt"

# =============================== STEP SCRIPT ==================================
cat > "$ROOT/step.sh" <<'STEP'
set -uo pipefail
REPO=/mnt/nfs/hoangduy/projects/llm-compressor
PY=/mnt/nfs/hoangduy/venvs/serve-026/bin/python
C="$ROOT"
note() { echo "[arm $(date -u +%H:%M:%S)] $*" | tee -a "$C/notes.txt"; }

note "node $(hostname)  gpus $(nvidia-smi --query-gpu=count --format=csv,noheader | head -1)"

stop_serve() {
  local cur=$1
  note "stop serve $(basename "$cur")"
  kill "$(cat "$cur/serve.pid" 2>/dev/null)" 2>/dev/null || true
  sleep 15
  kill -9 -"$(cat "$cur/serve.pid" 2>/dev/null)" 2>/dev/null || true
  sleep 5
  for _ in $(seq 1 60); do
    local bad
    bad=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits 2>/dev/null \
          | awk '{if ($1 < 70000) n++} END {print n+0}')
    [ "$bad" = 0 ] && { note "GPUs free"; return 0; }
    sleep 10
  done
  note "WARN GPUs still busy"
}

# $1 label  $2 port  $3 kv_dtype  $4 enforce_eager
run_arm() {
  local label=$1 port=$2 kvd=$3 eager=$4
  local cur=$C/$label
  mkdir -p "$cur"
  printf 'kv_cache_dtype=%s\nenforce_eager=%s\nconc=%s\nmax_tokens=%s\n' \
    "$kvd" "$eager" "$NCONC" "$MAXTOK" > "$cur/arm-config.txt"
  note "=== ARM $label: kv=$kvd eager=$eager port=$port ==="

  M3_W4A8_BACKEND=humming KV_CACHE_DTYPE="$kvd" ENFORCE_EAGER="$eager" \
  CKPT="$CKPT" SERVED_NAME="$SERVED_NAME" MODEL_ID="$TOKENIZER" PORT="$port" \
    MAX_MODEL_LEN="$MAX_MODEL_LEN" LOG="$cur/serve.log" PID_FILE="$cur/serve.pid" \
    bash "$REPO/pipeline/slurm/run_vllm_http_serve_smoke.sh" >>"$C/client.log" 2>&1 \
    || { note "$label serve start rc=$?"; tail -40 "$cur/serve.log" >>"$C/client.log"; echo "SERVE_START_FAIL" > "$cur/verdict.txt"; return 1; }

  local ready=1
  for _ in $(seq 1 540); do
    curl -sf "http://localhost:$port/v1/models" -o "$cur/models.json" 2>/dev/null && { ready=0; break; }
    kill -0 "$(cat "$cur/serve.pid" 2>/dev/null)" 2>/dev/null || { note "$label serve died during load"; break; }
    sleep 10
  done
  if [ "$ready" != 0 ]; then
    note "$label NEVER READY"
    grep -nE "Traceback|Error|assert" "$cur/serve.log" | tail -30 | tee -a "$C/client.log"
    echo "NEVER_READY" > "$cur/verdict.txt"
    stop_serve "$cur"
    return 1
  fi

  # ---- fail-closed: the arm is only evidence if it really is the config we meant.
  local obsv obskv
  obsv=$(grep -m1 -oE "V1 LLM engine \(v[0-9.]+\)" "$cur/serve.log" \
         | grep -oE "[0-9]+\.[0-9]+\.[0-9]+")
  printf '%s\n' "${obsv:-unknown}" > "$cur/vllm-observed.txt"
  if [ "$obsv" != "0.26.0" ]; then
    note "$label ABORT: serve reports vLLM '${obsv:-unknown}', not 0.26.0 -- wrong venv"
    echo "WRONG_VLLM" > "$cur/verdict.txt"
    stop_serve "$cur"
    return 1
  fi
  obskv=$(grep -m1 -oE "kv_cache_dtype=[a-z0-9_]+" "$cur/serve.log" | head -1 | cut -d= -f2)
  printf '%s\n' "${obskv:-unknown}" > "$cur/kvdtype-observed.txt"
  case "$kvd:$obskv" in
    fp8:fp8)   : ;;                       # intended fp8, engine agrees
    auto:auto|auto:bfloat16|auto:torch.bfloat16) : ;;  # intended bf16 KV
    *) note "$label ABORT: asked kv=$kvd but engine reports kv_cache_dtype=$obskv"
       echo "WRONG_KV" > "$cur/verdict.txt"; stop_serve "$cur"; return 1 ;;
  esac
  note "$label verified vLLM $obsv, kv_cache_dtype=$obskv"

  # Record what the engine actually chose, so the arm is self-describing.
  grep -m1 -oE "MiniMax M3 indexer: selected .*" "$cur/serve.log" > "$cur/indexer.txt" 2>/dev/null || true
  grep -m1 -oE "MiniMax M3 sparse attention selected .*" "$cur/serve.log" > "$cur/attend.txt" 2>/dev/null || true
  note "$label indexer: $(cat "$cur/indexer.txt" 2>/dev/null)"
  note "$label attend : $(cat "$cur/attend.txt" 2>/dev/null)"

  "$PY" "$REPO/pipeline/diag/m3_026_ima_burst.py" \
    --base "http://localhost:$port" --model "$SERVED_NAME" \
    --prompts "$REPO/artifacts/aiperf-datasets/speedbench/8k-low.jsonl" \
    -n "$NCONC" --max-tokens "$MAXTOK" >"$cur/burst.log" 2>&1
  local brc=$?
  tail -25 "$cur/burst.log" | tee -a "$C/notes.txt"

  # Capture the crash evidence while it is still on disk.
  grep -n -i -m1 -A3 "illegal memory access" "$cur/serve.log" > "$cur/ima.txt" 2>/dev/null || true
  grep -n -m1 -B2 -A6 "AssertionError\|IndexError\|RuntimeError" "$cur/serve.log" \
    | tail -20 > "$cur/other-exc.txt" 2>/dev/null || true
  if [ -s "$cur/ima.txt" ]; then
    note "$label >>> IMA PRESENT in serve.log"
    echo "CRASH_IMA" > "$cur/verdict.txt"
  elif [ "$brc" != 0 ]; then
    note "$label >>> burst failed rc=$brc but no IMA string"
    echo "CRASH_OTHER" > "$cur/verdict.txt"
  else
    note "$label >>> CLEAN"
    echo "CLEAN" > "$cur/verdict.txt"
  fi

  stop_serve "$cur"
  return 0
}

skip() { case ",$ONLY," in *",$1,"*) return 1;; esac; [ -n "$ONLY" ]; }

skip stock  || run_arm stock  "$((PORT_BASE+0))" fp8  0
skip bf16kv || run_arm bf16kv "$((PORT_BASE+1))" auto 0
skip eager  || run_arm eager  "$((PORT_BASE+2))" fp8  1

note "================ SUMMARY ================"
for a in stock bf16kv eager; do
  [ -d "$C/$a" ] || continue
  note "$(printf '%-8s %s' "$a" "$(cat "$C/$a/verdict.txt" 2>/dev/null || echo NOT_RUN)")"
done

# Fail-closed interpretation: arm A is the control for the whole run.
sv=$(cat "$C/stock/verdict.txt" 2>/dev/null || echo NOT_RUN)
if [ "$sv" != CRASH_IMA ] && [ "$sv" != CRASH_OTHER ]; then
  note "VERDICT: INCONCLUSIVE -- arm 'stock' did not reproduce ($sv);"
  note "         a CLEAN result in bf16kv/eager therefore proves NOTHING."
else
  note "VERDICT: repro confirmed in 'stock'; compare bf16kv/eager above."
fi
STEP

# ============================ ENV FAITHFULNESS ================================
# Copied verbatim from run_specdec_dspark_srun.sh's arm_env, because the bug is
# only meaningful in the configuration that produced it. Several of these are NOT
# conveniences:
#   SERVE_VENV               -> run_vllm_http_serve_smoke.sh defaults to `quant`
#                               (vLLM 0.24.0). Without this override the whole
#                               experiment would test the version that WORKS.
#   LLMC_M3_CAPTURE_SYNC     -> our breakable-cudagraph capture patch's knob.
#   VLLM_HUMMING_*           -> pin the W4A8 MoE GEMM path.
#   LLMC_HUMMING_PROVISIONAL -> unlocks the Humming preflight on 0.26.0 only.
arm_env=(
  "ROOT=$ROOT" "CKPT=$CKPT" "SERVED_NAME=$SERVED_NAME" "TOKENIZER=$TOKENIZER"
  "MAX_MODEL_LEN=$MAX_MODEL_LEN" "NCONC=$NCONC" "MAXTOK=$MAXTOK"
  "PORT_BASE=$PORT_BASE" "ONLY=$ONLY"
  "SERVE_VENV=/mnt/nfs/hoangduy/venvs/serve-026"
  "HUMMING_M3_W4A8_CACHE_ROOT=/mnt/nfs/hoangduy/.humming"
  "VLLM_HUMMING_MOE_GEMM_TYPE=indexed" "VLLM_HUMMING_USE_F16_ACCUM=0"
  "LLMC_M3_CAPTURE_SYNC=sync"
  "LLMC_HUMMING_PROVISIONAL_VLLM=0.26.0"
  "PYTHONPATH=$REPO"
)
printf '%s\n' "${arm_env[@]}" > "$ROOT/arm-env.txt"
say "env recorded to arm-env.txt (SERVE_VENV=serve-026 -- the override that makes this test 0.26.0)"

say "launching srun on $NODE (8 GPUs, 70 min cap)"
env "${arm_env[@]}" \
srun --exclusive --nodes=1 --ntasks=1 --nodelist="$NODE" \
     --gres=gpu:8 --cpus-per-task=192 --time=01:10:00 \
     --kill-on-bad-exit=1 --partition=compute \
     --job-name=m3-026-ima --export=ALL \
     bash "$ROOT/step.sh" 2>&1 | tee "$ROOT/srun.log"
rc=$?

say "srun rc=$rc"
printf 'rc=%s finished=%s\n' "$rc" "$(date -u +%Y-%m-%dT%H:%M:%SZ)" > "$ROOT/done.txt"
echo "$ROOT" > "$RESULTS/latest.txt"
exit $rc
