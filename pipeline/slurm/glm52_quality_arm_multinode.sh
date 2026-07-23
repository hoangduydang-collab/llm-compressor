#!/usr/bin/env bash
# Multi-node GLM-5.2 quality arm (official FP8: 2 nodes TP16; BF16: 4 nodes TP32).
# Runs as N ranks under one srun (--nodes=N --ntasks=N --ntasks-per-node=1).
# Every rank boots sglang.launch_server with --nnodes/--node-rank; the HTTP
# endpoint lives on rank 0, which also runs probes + the general-suite client,
# then writes client-done so the other ranks exit. SMOKE_ONLY=1 stops after a
# single test completion (multi-node load validation without suite spend).
#
# Env (required): ARM PROFILE PORT ROOT MODEL_PATH NNODES TP
# Env (optional): CTX RUN_ID QUANT_ARGS SMOKE_ONLY MEM_FRAC
#
# Cross-node NCCL/gloo must bind the routable fabric (memory: hostnames don't
# route): NCCL_SOCKET_IFNAME=intranet; dist-init-addr uses the intranet IP.
set -uo pipefail

ARM=${ARM:?}; PROFILE=${PROFILE:?}; PORT=${PORT:?}; ROOT=${ROOT:?}
MODEL_PATH=${MODEL_PATH:?}; NNODES=${NNODES:?}; TP=${TP:?}
CTX="${CTX:-131072}"
RUN_ID="${RUN_ID:-glm52full7}"
QUANT_ARGS="${QUANT_ARGS:-}"
SMOKE_ONLY="${SMOKE_ONLY:-0}"
# 0.75 + chunked prefill 2048: echo/loglikelihood requests compute logits for
# ALL prompt tokens (vocab 154880 x batch tokens, TP all-gather) — at 0.85 the
# scheduler OOM'd and died (job 13136). 2048-token chunks bound that buffer.
MEM_FRAC="${MEM_FRAC:-0.75}"
CHUNKED_PREFILL="${CHUNKED_PREFILL:-2048}"
DIST_PORT=$((PORT + 20000))

REPO=/mnt/nfs/hoangduy/projects/llm-compressor
BENCH=/mnt/nfs/hoangduy/projects/benchmarks
BVENV=/mnt/nfs/hoangduy/venvs/benchmarks

rank=${SLURM_PROCID:-0}
CLIENT=$ROOT/client-$ARM
mkdir -p "$CLIENT"
note() { echo "[$ARM r$rank $(date -u +%H:%M:%S)] $1" | tee -a "$CLIENT/client.log"; }
gate() { echo "$1=$2" >> "$CLIENT/gates.txt"; note "gate $1=$2"; }

source /mnt/nfs/hoangduy/env.sh
export HOME="${WORK_ROOT:-/mnt/nfs/hoangduy}"
export HF_HOME=/mnt/nfs/hoangduy/cache/huggingface
export HF_DATASETS_OFFLINE=1 HF_HUB_OFFLINE=1

# DeepGEMM compat (proven) + cross-node fabric pinning (proven on M3 TP16).
export FLASHINFER_USE_CUDA_NORM=1
export SGLANG_ENABLE_JIT_DEEPGEMM=0
export DG_JIT_NVCC_COMPILER=/mnt/nfs/hoangduy/cuda-12.9/bin/nvcc
export DG_JIT_USE_NVRTC=0
export SGLANG_DG_USE_NVRTC=0
export NCCL_SOCKET_IFNAME=${NCCL_SOCKET_IFNAME:-intranet}
export GLOO_SOCKET_IFNAME=${GLOO_SOCKET_IFNAME:-intranet}
export NCCL_DEBUG=${NCCL_DEBUG:-WARN}
# Node-local triton cache: the shared NFS cache races across nodes/jobs
# (observed: rank1 FileNotFoundError on fused_moe_kernel.json, job 13129).
export TRITON_CACHE_DIR=/tmp/triton-cache-${SLURM_JOB_ID:-nojob}-r${SLURM_PROCID:-0}
# Inductor's DEFAULT cache (/tmp/torchinductor_$USER) is shared by every job on
# the node — stale artifacts from other jobs' triton versions poison it
# (KeyError: 'cubin' in inductor compile workers, job 13141 rank0 on h104).
export TORCHINDUCTOR_CACHE_DIR=/tmp/inductor-cache-${SLURM_JOB_ID:-nojob}-r${SLURM_PROCID:-0}
mkdir -p "$TRITON_CACHE_DIR" "$TORCHINDUCTOR_CACHE_DIR"

# Rank 0 publishes its intranet IP; other ranks wait for it.
HEAD_FILE=$CLIENT/head-ip-${SLURM_JOB_ID:-nojob}
if [ "$rank" = 0 ]; then
  HEAD_IP=$(ip -4 addr show intranet | awk '/inet /{print $2}' | cut -d/ -f1 | head -1)
  [ -n "$HEAD_IP" ] || { note "FATAL: no intranet IPv4 on head"; exit 1; }
  echo "$HEAD_IP" > "$HEAD_FILE"
else
  for _ in $(seq 1 60); do [ -s "$HEAD_FILE" ] && break; sleep 2; done
  HEAD_IP=$(cat "$HEAD_FILE" 2>/dev/null)
  [ -n "$HEAD_IP" ] || { note "FATAL: head IP never appeared"; exit 1; }
fi
note "host=$(hostname) rank=$rank/$NNODES head=$HEAD_IP tp=$TP ctx=$CTX port=$PORT smoke_only=$SMOKE_ONLY"

source /mnt/nfs/hoangduy/venvs/sglang-eval/bin/activate
python -m sglang.launch_server \
  --model-path "$MODEL_PATH" \
  $QUANT_ARGS \
  --disable-shared-experts-fusion \
  --tp "$TP" \
  --nnodes "$NNODES" --node-rank "$rank" --dist-init-addr "$HEAD_IP:$DIST_PORT" \
  --kv-cache-dtype fp8_e4m3 \
  --reasoning-parser glm45 \
  --tool-call-parser glm47 \
  --context-length "$CTX" \
  --mem-fraction-static "$MEM_FRAC" \
  --chunked-prefill-size "$CHUNKED_PREFILL" \
  --trust-remote-code \
  --host 0.0.0.0 --port "$PORT" \
  > "$CLIENT/serve-rank$rank.log" 2>&1 &
SERVER_PID=$!
echo "$SERVER_PID" > "$CLIENT/serve-rank$rank.pid"

DONE_FILE=$CLIENT/client-done-${SLURM_JOB_ID:-nojob}
if [ "$rank" != 0 ]; then
  # Worker rank: hold until the client finishes or the server dies.
  while :; do
    [ -f "$DONE_FILE" ] && { note "client done; worker exiting"; break; }
    kill -0 "$SERVER_PID" 2>/dev/null || { note "server rank $rank died"; break; }
    sleep 10
  done
  kill "$SERVER_PID" 2>/dev/null; sleep 5; kill -9 "$SERVER_PID" 2>/dev/null
  exit 0
fi

# ---- rank 0 only below ----
finish() { touch "$DONE_FILE"; kill "$SERVER_PID" 2>/dev/null; sleep 10; kill -9 "$SERVER_PID" 2>/dev/null; }

healthy=1
for i in $(seq 1 540); do   # up to 90 min: BF16 loads 1.4 TB from NFS
  kill -0 "$SERVER_PID" 2>/dev/null || { note "server died during startup"; break; }
  curl -sf "http://localhost:$PORT/health_generate" >/dev/null 2>&1 && { healthy=0; break; }
  sleep 10
done
gate serve_healthy "$healthy"
[ "$healthy" = 0 ] || { tail -40 "$CLIENT/serve-rank0.log"; finish; exit 1; }
note "server healthy after ~$((i*10))s"

note "smoke: one greedy test completion"
curl -s "http://localhost:$PORT/v1/chat/completions" -H 'Content-Type: application/json' \
  -d '{"model":"glm-5.2","messages":[{"role":"user","content":"A farmer has 17 sheep. All but 9 run away. How many are left? Explain briefly."}],"max_tokens":4096,"temperature":0.0}' \
  > "$CLIENT/smoke-response.json"
"$BVENV/bin/python" - "$CLIENT/smoke-response.json" <<'EOF'
import json, sys
d = json.load(open(sys.argv[1]))
ch = d["choices"][0]; msg = ch["message"]
content = (msg.get("content") or "").strip()
reasoning = (msg.get("reasoning_content") or "").strip()
print(f"[smoke] finish_reason={ch.get('finish_reason')} usage={d.get('usage')}")
print(f"[smoke] reasoning={len(reasoning)}ch content={content[:200]!r}")
assert ch.get("finish_reason") in ("stop", "length") and content and reasoning
print("[smoke] PASS")
EOF
rc=$?; gate smoke_completion "$rc"
[ "$rc" = 0 ] || { finish; exit 1; }

if [ "$SMOKE_ONLY" = 1 ]; then
  note "SMOKE_ONLY=1 -> done"
  finish; exit 0
fi

note "step 2: loglikelihood-shape gate (the EXACT lm-eval payload: max_tokens=1 echo logprobs)"
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
  cat "$CLIENT/ll-gate.json"; finish; exit 1
fi
( cd "$BENCH" && BASE_URL="http://localhost:$PORT" "$BVENV/bin/python" -m quality.probe_endpoint \
    --profile "configs/glm/$PROFILE" --capabilities ) >"$CLIENT/capabilities.txt" 2>&1 || true

note "step 3: general suite (standalone orchestrator, ${RUN_ID})"
( cd "$BENCH" && BASE_URL="http://localhost:$PORT" BASELINE_REF="" PATH="$BVENV/bin:$PATH" \
    "$BVENV/bin/python" -m quality.orchestrator \
    --profile "configs/glm/$PROFILE" --out-root "$ROOT/results" --run-id "$RUN_ID" \
    --execute ) >"$CLIENT/general.log" 2>&1
rc=$?; gate general_suite "$rc"
tail -20 "$CLIENT/general.log" | tee -a "$CLIENT/client.log"

note "arm done rc=$rc"
finish
exit $rc
