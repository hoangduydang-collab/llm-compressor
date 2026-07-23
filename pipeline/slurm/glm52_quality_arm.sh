#!/usr/bin/env bash
# One GLM-5.2 quality arm: serve on SGLang (single node) -> capability probes
# (fail-closed) -> standalone general-suite orchestrator (7 tasks, greedy, 64k
# budget, usage capture on) -> shutdown. Sequential-arm design (4-node cap):
# arms never overlap; A/B deltas vs glm-5.2-bf16 are rebuilt offline.
#
# Env (required): ARM PROFILE PORT ROOT
# Env (optional): MODEL_PATH QUANT_ARGS TP CTX RUN_ID
#   w4afp8-phala: MODEL_PATH=<phala ckpt> QUANT_ARGS="--quantization w4afp8" TP=8
#   (fp8/bf16 multi-node arms use glm52_quality_arm_multinode.sh instead)
#
# Serve env/flags proven by evals/glm52-w4afp8-smoke-20260722T0958Z (PASS).
set -uo pipefail

ARM=${ARM:?}; PROFILE=${PROFILE:?}; PORT=${PORT:?}; ROOT=${ROOT:?}
MODEL_PATH="${MODEL_PATH:-/mnt/nfs/hoangduy/hf_assets/PhalaCloud/GLM-5.2-W4AFP8}"
QUANT_ARGS="${QUANT_ARGS:---quantization w4afp8}"
TP="${TP:-8}"
CTX="${CTX:-131072}"
RUN_ID="${RUN_ID:-glm52full7}"
# 0.75 + chunked prefill 2048: echo/loglikelihood requests compute logits for
# ALL prompt tokens (vocab 154880 x batch tokens, TP all-gather) — at 0.85 the
# scheduler OOM'd and died (job 13136). 2048-token chunks bound that buffer.
MEM_FRAC="${MEM_FRAC:-0.75}"
CHUNKED_PREFILL="${CHUNKED_PREFILL:-2048}"

REPO=/mnt/nfs/hoangduy/projects/llm-compressor
BENCH=/mnt/nfs/hoangduy/projects/benchmarks
BVENV=/mnt/nfs/hoangduy/venvs/benchmarks

CLIENT=$ROOT/client-$ARM
mkdir -p "$CLIENT"
note() { echo "[$ARM $(date -u +%H:%M:%S)] $1" | tee -a "$CLIENT/client.log"; }
gate() { echo "$1=$2" >> "$CLIENT/gates.txt"; note "gate $1=$2"; }

source /mnt/nfs/hoangduy/env.sh
export HOME="${WORK_ROOT:-/mnt/nfs/hoangduy}"
export HF_HOME=/mnt/nfs/hoangduy/cache/huggingface
export HF_DATASETS_OFFLINE=1 HF_HUB_OFFLINE=1

# DeepGEMM compat (proven combo; system nvcc 12.4 can't build DeepGEMM fp8).
export FLASHINFER_USE_CUDA_NORM=1
export SGLANG_ENABLE_JIT_DEEPGEMM=0
export DG_JIT_NVCC_COMPILER=/mnt/nfs/hoangduy/cuda-12.9/bin/nvcc
export DG_JIT_USE_NVRTC=0
export SGLANG_DG_USE_NVRTC=0
# Node-local triton cache: the shared NFS cache races across nodes/jobs
# (observed: FileNotFoundError on fused_moe_kernel.json, job 13129).
export TRITON_CACHE_DIR=/tmp/triton-cache-${SLURM_JOB_ID:-nojob}
# Inductor's DEFAULT cache (/tmp/torchinductor_$USER) is node-shared across
# jobs and can be poisoned by other jobs' triton versions (job 13141).
export TORCHINDUCTOR_CACHE_DIR=/tmp/inductor-cache-${SLURM_JOB_ID:-nojob}
mkdir -p "$TRITON_CACHE_DIR" "$TORCHINDUCTOR_CACHE_DIR"

note "host=$(hostname) arm=$ARM model=$MODEL_PATH tp=$TP ctx=$CTX port=$PORT run_id=$RUN_ID"

note "step 1: serve on SGLang"
(
  source /mnt/nfs/hoangduy/venvs/sglang-eval/bin/activate
  exec python -m sglang.launch_server \
    --model-path "$MODEL_PATH" \
    $QUANT_ARGS \
    --disable-shared-experts-fusion \
    --tp "$TP" \
    --kv-cache-dtype fp8_e4m3 \
    --reasoning-parser glm45 \
    --tool-call-parser glm47 \
    --context-length "$CTX" \
    --mem-fraction-static "$MEM_FRAC" \
    --chunked-prefill-size "$CHUNKED_PREFILL" \
    --trust-remote-code \
    --host 0.0.0.0 --port "$PORT"
) > "$CLIENT/serve.log" 2>&1 &
SERVER_PID=$!
echo "$SERVER_PID" > "$CLIENT/serve.pid"

healthy=1
for i in $(seq 1 270); do
  kill -0 "$SERVER_PID" 2>/dev/null || { note "server died during startup"; break; }
  curl -sf "http://localhost:$PORT/health_generate" >/dev/null 2>&1 && { healthy=0; break; }
  sleep 10
done
gate serve_healthy "$healthy"
[ "$healthy" = 0 ] || { tail -40 "$CLIENT/serve.log"; exit 1; }
note "server healthy after ~$((i*10))s"

note "step 2: loglikelihood-shape gate (the EXACT lm-eval payload: max_tokens=1 echo logprobs)"
# NOT probe_endpoint's text_offset probe: that sends max_tokens=0, which SGLang
# 400s while vLLM accepts — a probe artifact, not the eval path (job 13128).
curl -s "http://localhost:$PORT/v1/completions" -H 'Content-Type: application/json' \
  -d '{"model":"glm","prompt":"The capital of France is Paris. The capital of Germany is","max_tokens":1,"echo":true,"logprobs":5,"temperature":0.0}' \
  > "$CLIENT/ll-gate.json"
"$BVENV/bin/python" -c "
import json,sys
d=json.load(open('$CLIENT/ll-gate.json'))
lp=d['choices'][0]['logprobs']
assert lp['tokens'] and lp['token_logprobs'], 'echo logprobs missing'
print('[ll-gate] ok:', len(lp['tokens']), 'prompt tokens with logprobs')
"
cap=$?; gate ll_shape "$cap"
if [ "$cap" != 0 ]; then
  note "FATAL: /completions echo+logprobs (lm-eval loglikelihood shape) unsupported"
  cat "$CLIENT/ll-gate.json"
  kill "$SERVER_PID" 2>/dev/null; exit 1
fi
# Record full capability probe output for provenance (non-gating).
( cd "$BENCH" && BASE_URL="http://localhost:$PORT" "$BVENV/bin/python" -m quality.probe_endpoint \
    --profile "configs/glm/$PROFILE" --capabilities ) >"$CLIENT/capabilities.txt" 2>&1 || true

note "step 3: general suite (standalone orchestrator, ${RUN_ID})"
# BASELINE_REF="" -> standalone mode (no live baseline in the sequential design;
# deltas/report rebuilt offline via quality.rebuild_delta after all arms finish).
( cd "$BENCH" && BASE_URL="http://localhost:$PORT" BASELINE_REF="" PATH="$BVENV/bin:$PATH" \
    "$BVENV/bin/python" -m quality.orchestrator \
    --profile "configs/glm/$PROFILE" --out-root "$ROOT/results" --run-id "$RUN_ID" \
    --execute ) >"$CLIENT/general.log" 2>&1
rc=$?; gate general_suite "$rc"
tail -20 "$CLIENT/general.log" | tee -a "$CLIENT/client.log"

note "step 4: shutdown"
kill "$SERVER_PID" 2>/dev/null; sleep 10; kill -9 "$SERVER_PID" 2>/dev/null
note "arm done rc=$rc"
exit $rc
