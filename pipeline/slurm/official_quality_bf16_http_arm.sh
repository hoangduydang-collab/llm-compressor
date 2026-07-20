#!/usr/bin/env bash
# BF16 MiniMax-M3 HTTP arm for the official-pipeline migration (phase 2).
# Runs as 2 ranks on 2x 8xH100 (srun --nodes=2 --ntasks=2): boots the ray
# cluster via test_m3_ray_topology.sh (same proven path as the BF16 TP16
# qualification arms), then rank 0 runs `vllm serve` TP16/ray as an OpenAI
# endpoint on $PORT and holds until $ROOT/client-done appears.
#
# Markers written under $ROOT/bf16/:
#   endpoint-ip   the head node's routable IP (client builds http://IP:PORT)
#   ready         endpoint answered /v1/models
#   serve.log     vllm serve output
set -uo pipefail

ROOT=${1:?usage: official_quality_bf16_http_arm.sh <phase2 root>}
PORT=${PORT:-8001}
MODEL=${MODEL:-/mnt/nfs/hoangduy/hf_assets/MiniMaxAI/MiniMax-M3}
SERVED_NAME=${SERVED_NAME:-MiniMaxAI/MiniMax-M3}
MAX_MODEL_LEN=${MAX_MODEL_LEN:-65536}
HOLD_MAX=${HOLD_MAX:-2160}          # 10s ticks -> 6h
READY_MAX=${READY_MAX:-540}         # 10s ticks -> 90 min load budget

REPO=/mnt/nfs/hoangduy/projects/llm-compressor
ARM=$ROOT/bf16
mkdir -p "$ARM"
rank=${SLURM_PROCID:-0}
note() { echo "[bf16-arm r$rank $(date -u +%H:%M:%S)] $1" | tee -a "$ARM/arm-rank-$rank.log"; }

# Same node env as run_vllm_http_serve_smoke.sh (proven for M3 vLLM serve).
source /mnt/nfs/hoangduy/env.sh
source /mnt/nfs/hoangduy/venvs/quant/bin/activate
cd "$REPO"
export HOME=/mnt/nfs/hoangduy WORK_ROOT=/mnt/nfs/hoangduy
export PYTHONPATH="${PYTHONPATH:+$PYTHONPATH:}$REPO"
export PYTHONUNBUFFERED=1 TOKENIZERS_PARALLELISM=false
export FLASHINFER_USE_CUDA_NORM=1
export FLASHINFER_WORKSPACE_DIR=${FLASHINFER_WORKSPACE_DIR:-$HOME/cache/flashinfer}
export VLLM_DISABLE_SHARED_EXPERTS_STREAM=${VLLM_DISABLE_SHARED_EXPERTS_STREAM:-1}
export PYTORCH_CUDA_ALLOC_CONF=${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}
mkdir -p "$FLASHINFER_WORKSPACE_DIR" 2>/dev/null || true

finish() {
  touch "$ARM/ray_runtime/driver-done" 2>/dev/null || true
  ray stop --force >/dev/null 2>&1 || true
}
trap finish EXIT

# Boot / join the ray cluster (sets local_ip, VLLM_HOST_IP; gates readiness).
note "booting ray cluster"
source pipeline/slurm/test_m3_ray_topology.sh --out "$ARM/ray_runtime" --keep-alive
set +e   # topology script enables -e; everything below handles rc explicitly

if ((rank != 0)); then
  note "worker joined; waiting for driver-done"
  for _ in $(seq 1 86400); do
    [[ -f "$ARM/ray_runtime/driver-done" ]] && break
    sleep 1
  done
  exit 0
fi

note "rank 0: ensuring M3 vLLM patches (NFS venv, applies to both nodes)"
python pipeline/slurm/patch_vllm_m3_serve.py --check >>"$ARM/arm-rank-0.log" 2>&1 || {
  python pipeline/slurm/patch_vllm_m3_serve.py >>"$ARM/arm-rank-0.log" 2>&1 || { note "patch apply FAILED"; exit 1; }
  python pipeline/slurm/patch_vllm_m3_serve.py --check >>"$ARM/arm-rank-0.log" 2>&1 || { note "patch check FAILED"; exit 1; }
}

note "rank 0: starting vllm serve TP16/ray on port $PORT (bf16, ~920 GB)"
vllm serve "$MODEL" \
  --served-model-name "$SERVED_NAME" \
  --host 0.0.0.0 --port "$PORT" \
  --trust-remote-code \
  --tensor-parallel-size 16 \
  --distributed-executor-backend ray \
  --enable-expert-parallel \
  --disable-custom-all-reduce \
  --language-model-only \
  --kv-cache-dtype fp8 --block-size 128 \
  --max-model-len "$MAX_MODEL_LEN" \
  --gpu-memory-utilization 0.9 \
  --tool-call-parser minimax_m3 --reasoning-parser minimax_m3 \
  --enable-auto-tool-choice \
  >"$ARM/serve.log" 2>&1 &
SERVE_PID=$!

note "readiness poll (max $((READY_MAX / 6)) min)"
ready=1
for _ in $(seq 1 "$READY_MAX"); do
  if curl -sf "http://localhost:$PORT/v1/models" -o "$ARM/models.json" 2>/dev/null; then ready=0; break; fi
  kill -0 "$SERVE_PID" 2>/dev/null || { note "vllm serve died during load"; break; }
  sleep 10
done
if [[ "$ready" != 0 ]]; then
  tail -60 "$ARM/serve.log" >>"$ARM/arm-rank-0.log" 2>&1
  note "BF16 endpoint FAILED to become ready"
  exit 1
fi
echo "$local_ip" >"$ARM/endpoint-ip"
touch "$ARM/ready"
note "BF16 endpoint READY at http://$local_ip:$PORT"

note "holding until $ROOT/client-done (max $((HOLD_MAX / 360))h)"
for _ in $(seq 1 "$HOLD_MAX"); do
  [[ -f "$ROOT/client-done" ]] && break
  kill -0 "$SERVE_PID" 2>/dev/null || { note "vllm serve died while holding"; exit 1; }
  sleep 10
done

note "teardown"
kill "$SERVE_PID" 2>/dev/null || true
sleep 20
kill -9 "$SERVE_PID" 2>/dev/null || true
exit 0
