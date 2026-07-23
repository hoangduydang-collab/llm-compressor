#!/usr/bin/env bash
# Smoke test: load a GLM-5.2 checkpoint on SGLang and run one chat completion.
# Fail-closed: exits nonzero unless the server becomes healthy AND the test
# request returns non-empty content with a sane finish_reason.
#
# Env overrides:
#   MODEL_PATH  (default: PhalaCloud W4AFP8)
#   QUANT_ARGS  (default: "--quantization w4afp8" for the W4AFP8 layout; set
#                to "" for FP8/BF16 official checkpoints)
#   TP          (default: 8)
#   CTX         (default: 131072)
#   PORT        (default: 30001)
#   OUT_DIR     (default: evals/glm52-smoke-<ts>)
#
# GLM-5.2 recommended serving flags per PhalaCloud/GLM-5.2-W4AFP8 README and
# zai-org GLM-5.2 SGLang cookbook: reasoning-parser glm45, tool-call-parser
# glm47, --disable-shared-experts-fusion, kv fp8_e4m3, trust-remote-code.
set -uo pipefail

MODEL_PATH="${MODEL_PATH:-/mnt/nfs/hoangduy/hf_assets/PhalaCloud/GLM-5.2-W4AFP8}"
QUANT_ARGS="${QUANT_ARGS:---quantization w4afp8}"
TP="${TP:-8}"
CTX="${CTX:-131072}"
PORT="${PORT:-30001}"
OUT_DIR="${OUT_DIR:-evals/glm52-smoke-$(date -u +%Y%m%dT%H%M%SZ)}"

source /mnt/nfs/hoangduy/env.sh
source /mnt/nfs/hoangduy/venvs/sglang-eval/bin/activate
export HOME="${WORK_ROOT:-/mnt/nfs/hoangduy}"

# DeepGEMM compat (proven combo from evals/glm52-w4afp8-phala-8k): system nvcc
# 12.4 cannot compile DeepGEMM fp8 kernels (needs >= 12.9) -> disable JIT
# DeepGEMM and point DG at the NFS cuda-12.9 nvcc for anything that still asks.
export FLASHINFER_USE_CUDA_NORM=1
export SGLANG_ENABLE_JIT_DEEPGEMM=0
export DG_JIT_NVCC_COMPILER=/mnt/nfs/hoangduy/cuda-12.9/bin/nvcc
export DG_JIT_USE_NVRTC=0
export SGLANG_DG_USE_NVRTC=0
cd /mnt/nfs/hoangduy/projects/llm-compressor
mkdir -p "$OUT_DIR"

echo "[smoke] host=$(hostname) model=$MODEL_PATH tp=$TP ctx=$CTX port=$PORT"
echo "[smoke] sglang=$(python -c 'import sglang;print(sglang.__version__)')"
nvidia-smi --query-gpu=index,name,memory.used,memory.total --format=csv,noheader | head -8

python -m sglang.launch_server \
  --model-path "$MODEL_PATH" \
  $QUANT_ARGS \
  --disable-shared-experts-fusion \
  --tp "$TP" \
  --kv-cache-dtype fp8_e4m3 \
  --reasoning-parser glm45 \
  --tool-call-parser glm47 \
  --context-length "$CTX" \
  --mem-fraction-static 0.85 \
  --trust-remote-code \
  --host 127.0.0.1 --port "$PORT" \
  > "$OUT_DIR/serve.log" 2>&1 &
SERVER_PID=$!
echo "$SERVER_PID" > "$OUT_DIR/serve.pid"
echo "[smoke] server pid=$SERVER_PID; waiting for health (up to 45 min)"

healthy=0
for i in $(seq 1 270); do
  if ! kill -0 "$SERVER_PID" 2>/dev/null; then
    echo "[smoke] FAIL: server process died during startup"
    tail -40 "$OUT_DIR/serve.log"
    exit 2
  fi
  if curl -sf "http://127.0.0.1:$PORT/health_generate" >/dev/null 2>&1; then
    healthy=1; break
  fi
  sleep 10
done
if [[ "$healthy" != 1 ]]; then
  echo "[smoke] FAIL: server never became healthy"
  tail -40 "$OUT_DIR/serve.log"
  kill -9 "$SERVER_PID" 2>/dev/null
  exit 3
fi
echo "[smoke] server healthy after ~$((i*10))s"
curl -s "http://127.0.0.1:$PORT/get_model_info" | tee "$OUT_DIR/model_info.json"; echo

echo "[smoke] sending test chat completion (temp 1.0, top_p 0.95, reasoning on)"
curl -s "http://127.0.0.1:$PORT/v1/chat/completions" \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "glm-5.2",
    "messages": [{"role": "user", "content": "A farmer has 17 sheep. All but 9 run away. How many are left? Explain briefly."}],
    "max_tokens": 4096,
    "temperature": 1.0,
    "top_p": 0.95
  }' > "$OUT_DIR/response.json" 2>>"$OUT_DIR/client.err"

python - "$OUT_DIR/response.json" <<'EOF'
import json, sys
d = json.load(open(sys.argv[1]))
ch = d["choices"][0]
msg = ch["message"]
content = (msg.get("content") or "").strip()
reasoning = (msg.get("reasoning_content") or "").strip()
fr = ch.get("finish_reason")
usage = d.get("usage", {})
print(f"[smoke] finish_reason={fr} completion_tokens={usage.get('completion_tokens')}")
print(f"[smoke] reasoning_content: {len(reasoning)} chars | content: {len(content)} chars")
print(f"[smoke] content preview: {content[:300]!r}")
assert fr in ("stop", "length"), f"bad finish_reason: {fr}"
assert content, "empty content"
assert reasoning, "reasoning_content missing - reasoning parser not active?"
print("[smoke] PASS")
EOF
rc=$?

kill "$SERVER_PID" 2>/dev/null
sleep 5
kill -9 "$SERVER_PID" 2>/dev/null
if [[ $rc -eq 0 ]]; then echo "[smoke] SMOKE OK"; else echo "[smoke] SMOKE FAILED rc=$rc"; fi
exit $rc
