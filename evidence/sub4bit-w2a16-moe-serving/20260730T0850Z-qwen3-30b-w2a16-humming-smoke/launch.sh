#!/usr/bin/env bash
# Smoke test: serve a 2-bit (W2A16 g128 sym) compressed-tensors Qwen3-30B-A3B MoE
# checkpoint on vLLM 0.26.0 + PR #48918 (ported) + humming main.
# Venv: serve-sub4 (clone of serve-026 with the PR applied to site-packages).
# Runs entirely on one node: starts the server, gates on /health, sends probe
# prompts, records evidence, then shuts down. Exit 0 only if generation worked.
set -euo pipefail

DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV=/mnt/nfs/hoangduy/venvs/serve-sub4
export PYTHONPATH=/mnt/nfs/hoangduy/venvs/humming-main-site
# humming JIT resolves libnvrtc-builtins.so.13.0 via the loader search path
export LD_LIBRARY_PATH="$VENV/lib/python3.12/site-packages/nvidia/cu13/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
# humming's torch cpp_extension launcher build needs ninja (in venv bin);
# pin the cache dir to the NFS location pre-warmed from the login box
export PATH="$VENV/bin:$PATH"
export HUMMING_CACHE_DIR=/mnt/nfs/hoangduy/claude/home/.humming/cache/
MODEL=/mnt/nfs/hoangduy/hf_assets/Yi30/Qwen3-30B-A3B-Instruct-2507-W2A16-G128-AutoRound-LLMC
PORT=8321
LOG="$DIR/serve.log"

echo "=== node: $(hostname), gpu: $(nvidia-smi -L | head -1)"
echo "=== vllm: $("$VENV/bin/python" -c 'import vllm; print(vllm.__version__)')"
echo "=== humming: $("$VENV/bin/python" -c 'import humming; print(humming.__version__)')"

"$VENV/bin/vllm" serve "$MODEL" \
  --served-model-name qwen3-30b-w2a16 \
  --port "$PORT" \
  --max-model-len 8192 \
  --gpu-memory-utilization 0.45 \
  >"$LOG" 2>&1 &
SERVER_PID=$!
trap 'kill "$SERVER_PID" 2>/dev/null || true' EXIT

# Fail-closed readiness gate: up to 20 min (humming JIT-compiles kernels on
# first use), but bail immediately if the server process dies.
for i in $(seq 1 240); do
  if ! kill -0 "$SERVER_PID" 2>/dev/null; then
    echo "FAIL: server process exited early"; tail -50 "$LOG"; exit 1
  fi
  if curl -sf "http://127.0.0.1:$PORT/health" >/dev/null 2>&1; then
    echo "=== server healthy after ${i}x5s"; break
  fi
  if [ "$i" -eq 240 ]; then
    echo "FAIL: health timeout"; tail -50 "$LOG"; exit 1
  fi
  sleep 5
done

echo "=== backend selection evidence:"
grep -E "WNA16 MoE backend|MoEMethod|Humming|humming" "$LOG" | head -20 || true

run_probe() {
  local prompt="$1"
  curl -sf "http://127.0.0.1:$PORT/v1/chat/completions" \
    -H 'Content-Type: application/json' \
    -d "{\"model\":\"qwen3-30b-w2a16\",\"messages\":[{\"role\":\"user\",\"content\":\"$prompt\"}],\"max_tokens\":200,\"temperature\":0}"
}

echo "=== probe 1: arithmetic"
P1=$(run_probe "What is 17 + 25? Answer with just the number.")
echo "$P1" | "$VENV/bin/python" -c 'import json,sys; r=json.load(sys.stdin); print(r["choices"][0]["message"]["content"]); print("usage:", r["usage"])'

echo "=== probe 2: factual"
P2=$(run_probe "Name the capital of France in one word.")
echo "$P2" | "$VENV/bin/python" -c 'import json,sys; r=json.load(sys.stdin); print(r["choices"][0]["message"]["content"])'

echo "=== probe 3: fluency"
P3=$(run_probe "Write one sentence about mixture-of-experts models.")
echo "$P3" | "$VENV/bin/python" -c 'import json,sys; r=json.load(sys.stdin); print(r["choices"][0]["message"]["content"])'

# Machine-checkable gate: probe 1 must contain 42.
if echo "$P1" | grep -q "42"; then
  echo "SMOKE_RESULT: PASS"
else
  echo "SMOKE_RESULT: FAIL (probe 1 did not contain 42)"
  exit 1
fi
