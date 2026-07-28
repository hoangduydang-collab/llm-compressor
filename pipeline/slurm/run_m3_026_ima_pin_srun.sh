#!/usr/bin/env bash
# Pin the FAULTING kernel of the vLLM 0.26.0 MiniMax-M3 IMA (not the reporting one).
#
# WHY THIS ARM EXISTS
# The bisect run (m3-026-ima-bisect) reproduced the crash in ~5 s with 10 concurrent
# 8k requests, identically at kv_cache_dtype=fp8 and =auto, and the traceback ended in
#     humming/ops/input.py:217  _quant_tensor_kernel[(grid_blocks,)]
#     RuntimeError: Triton Error [CUDA]: an illegal memory access
# That is NOT proof this kernel is at fault. A Triton launch reports `Triton Error
# [CUDA]` when a STICKY error from an earlier asynchronous kernel surfaces at the next
# CUDA API call, so the MoE input-quant launch is only the first synchronous checkpoint
# after the real fault. CUDA_LAUNCH_BLOCKING=1 removes that ambiguity by synchronizing
# every launch, so the raise lands on the kernel that actually faults.
#
# The leading alternative it must distinguish: MiniMaxM3IndexerTritonImpl writes the
# shared topk_indices_buffer as buf[:, :nd] (decode) and buf[:, nd:] (prefill). A MIXED
# prefill+decode batch is the ONLY case where both writes happen in one forward -- pure
# prefill (our clean conc-1 cell) and pure decode each exercise exactly one. Both
# observed crash batches were mixed ({1 decode, 8191 prefill}), which is a
# request-count-dependent discriminator that batch size alone does not explain.
#
# Attribution is sound even with cudagraphs left ON: max_cudagraph_capture_size is 512
# and the faulting batch is 8192 tokens, so it runs eager regardless and every launch in
# it is individually synchronized. Keeping graphs on also keeps this arm identical to
# the configuration that crashes, which matters -- ENFORCE_EAGER is a separate arm.
#
# Cost: ONE serve. CUDA_LAUNCH_BLOCKING makes everything ~10x slower, but the crash
# arrives within seconds of the first burst, so the wall clock is dominated by model load.
set -uo pipefail

REPO=/mnt/nfs/hoangduy/projects/llm-compressor
PY=/mnt/nfs/hoangduy/venvs/serve-026/bin/python
RESULTS=/mnt/nfs/hoangduy/results/m3-026-ima-pin
CKPT="$REPO/artifacts/m3-awq-gptq-prepared/gptq-checkpoint-vllm-w123-abi-overlay"
PROMPTS="$REPO/artifacts/aiperf-datasets/speedbench/8k-low.jsonl"
BURST="$REPO/pipeline/diag/m3_026_ima_burst.py"
SERVED_NAME=MiniMaxAI/MiniMax-M3
MAX_MODEL_LEN=131072
NCONC=${NCONC:-10}
MAXTOK=${MAXTOK:-64}

NODE=${NODE:?set NODE, e.g. NODE=gpu-h104}
PORT=${PORT:-8170}

RUN_TS=$(date -u +%Y%m%dT%H%M%SZ)
ROOT=${ROOT:-$RESULTS/$RUN_TS}
mkdir -p "$ROOT"
fail() { echo "[controller] GATE FAILED: $1" | tee -a "$ROOT/gates.log"; exit 1; }
say()  { echo "[controller] $1" | tee -a "$ROOT/gates.log"; }

say "root=$ROOT node=$NODE port=$PORT conc=$NCONC (CUDA_LAUNCH_BLOCKING=1)"

for f in "$PROMPTS" "$BURST"; do [ -s "$f" ] || fail "missing $f"; done
[ -d "$CKPT" ] || fail "missing checkpoint $CKPT"
sha=$(sha256sum "$PROMPTS" | cut -c1-12)
[ "$sha" = bfcf60739f43 ] || fail "8k-low prompts changed ($sha)"
v=$("$PY" -c 'import vllm; print(vllm.__version__)' 2>&1)
[ "$v" = "0.26.0" ] || fail "expected vLLM 0.26.0, got '$v'"
"$PY" -m py_compile "$BURST" || fail "burst client does not compile"
say "gates passed (vllm $v, prompts $sha)"

cat > "$ROOT/step.sh" <<'STEP'
set -uo pipefail
REPO=/mnt/nfs/hoangduy/projects/llm-compressor
PY=/mnt/nfs/hoangduy/venvs/serve-026/bin/python
C="$ROOT"
note() { echo "[pin $(date -u +%H:%M:%S)] $*" | tee -a "$C/notes.txt"; }
note "node $(hostname)"

cur=$C/blocking
mkdir -p "$cur"

# CUDA_LAUNCH_BLOCKING is exported into the serve's environment, so every worker
# process inherits it and each kernel launch is synchronized.
CUDA_LAUNCH_BLOCKING=1 TORCH_SHOW_CPP_STACKTRACES=1 \
M3_W4A8_BACKEND=humming KV_CACHE_DTYPE=fp8 ENFORCE_EAGER=0 \
CKPT="$CKPT" SERVED_NAME="$SERVED_NAME" MODEL_ID="$CKPT" PORT="$PORT" \
  MAX_MODEL_LEN="$MAX_MODEL_LEN" LOG="$cur/serve.log" PID_FILE="$cur/serve.pid" \
  bash "$REPO/pipeline/slurm/run_vllm_http_serve_smoke.sh" >>"$C/client.log" 2>&1 \
  || { note "serve start rc=$?"; tail -40 "$cur/serve.log" >>"$C/client.log"; echo SERVE_START_FAIL > "$cur/verdict.txt"; exit 1; }

ready=1
for _ in $(seq 1 900); do
  curl -sf "http://localhost:$PORT/v1/models" -o "$cur/models.json" 2>/dev/null && { ready=0; break; }
  kill -0 "$(cat "$cur/serve.pid" 2>/dev/null)" 2>/dev/null || { note "serve died during load"; break; }
  sleep 10
done
[ "$ready" = 0 ] || { note "NEVER READY"; echo NEVER_READY > "$cur/verdict.txt"; exit 1; }

obsv=$(grep -m1 -oE "V1 LLM engine \(v[0-9.]+\)" "$cur/serve.log" | grep -oE "[0-9]+\.[0-9]+\.[0-9]+")
[ "$obsv" = "0.26.0" ] || { note "ABORT: serve is vLLM '$obsv'"; echo WRONG_VLLM > "$cur/verdict.txt"; exit 1; }
grep -q "CUDA_LAUNCH_BLOCKING" "$cur/serve.log" 2>/dev/null \
  && note "serve.log mentions CUDA_LAUNCH_BLOCKING" || true
note "ready: vLLM $obsv"

"$PY" "$REPO/pipeline/diag/m3_026_ima_burst.py" \
  --base "http://localhost:$PORT" --model "$SERVED_NAME" \
  --prompts "$REPO/artifacts/aiperf-datasets/speedbench/8k-low.jsonl" \
  -n "$NCONC" --max-tokens "$MAXTOK" >"$cur/burst.log" 2>&1
brc=$?
tail -15 "$cur/burst.log" | tee -a "$C/notes.txt"

# THE PAYLOAD: with launches synchronized, the deepest non-framework frame is the
# faulting kernel. Extract every candidate so the answer is not a judgement call.
n=$(grep -n -m1 "illegal memory access" "$cur/serve.log" | cut -d: -f1)
if [ -n "$n" ]; then
  sed -n "${n},$((n+220))p" "$cur/serve.log" > "$cur/fault-block.txt"
  grep -oE 'File "[^"]*", line [0-9]+, in [a-zA-Z_]*' "$cur/fault-block.txt" \
    | sed 's|/mnt/nfs/hoangduy/venvs/serve-026/lib/python3.12/site-packages/||' \
    > "$cur/fault-frames.txt"
  # the kernel-launching frames, which is what we came for
  grep -E "minimax_m3|humming|index_topk|sparse_attn|triton" "$cur/fault-frames.txt" \
    | sort -u > "$cur/fault-kernels.txt"
  note "=== faulting frames (CUDA_LAUNCH_BLOCKING=1) ==="
  cat "$cur/fault-kernels.txt" | tee -a "$C/notes.txt"
  echo CRASH_PINNED > "$cur/verdict.txt"
elif [ "$brc" != 0 ]; then
  note "burst failed rc=$brc but no IMA string in serve.log"
  echo CRASH_OTHER > "$cur/verdict.txt"
else
  note ">>> CLEAN under CUDA_LAUNCH_BLOCKING -- the fault is timing/concurrency dependent"
  echo CLEAN_UNDER_BLOCKING > "$cur/verdict.txt"
fi

note "stop serve"
kill "$(cat "$cur/serve.pid" 2>/dev/null)" 2>/dev/null || true
sleep 15
kill -9 -"$(cat "$cur/serve.pid" 2>/dev/null)" 2>/dev/null || true
note "verdict: $(cat "$cur/verdict.txt")"
STEP

arm_env=(
  "ROOT=$ROOT" "CKPT=$CKPT" "SERVED_NAME=$SERVED_NAME"
  "MAX_MODEL_LEN=$MAX_MODEL_LEN" "NCONC=$NCONC" "MAXTOK=$MAXTOK" "PORT=$PORT"
  "SERVE_VENV=/mnt/nfs/hoangduy/venvs/serve-026"
  "HUMMING_M3_W4A8_CACHE_ROOT=/mnt/nfs/hoangduy/.humming"
  "VLLM_HUMMING_MOE_GEMM_TYPE=indexed" "VLLM_HUMMING_USE_F16_ACCUM=0"
  "LLMC_M3_CAPTURE_SYNC=sync"
  "LLMC_HUMMING_PROVISIONAL_VLLM=0.26.0"
  "PYTHONPATH=$REPO"
)
printf '%s\n' "${arm_env[@]}" > "$ROOT/arm-env.txt"

say "launching srun on $NODE (8 GPUs, 50 min cap)"
env "${arm_env[@]}" \
srun --exclusive --nodes=1 --ntasks=1 --nodelist="$NODE" \
     --gres=gpu:8 --cpus-per-task=192 --time=00:50:00 \
     --kill-on-bad-exit=1 --partition=compute \
     --job-name=m3-026-pin --export=ALL \
     bash "$ROOT/step.sh" 2>&1 | tee "$ROOT/srun.log"
rc=$?
say "srun rc=$rc"
printf 'rc=%s finished=%s\n' "$rc" "$(date -u +%Y-%m-%dT%H:%M:%SZ)" > "$ROOT/done.txt"
echo "$ROOT" > "$RESULTS/latest.txt"
exit $rc
