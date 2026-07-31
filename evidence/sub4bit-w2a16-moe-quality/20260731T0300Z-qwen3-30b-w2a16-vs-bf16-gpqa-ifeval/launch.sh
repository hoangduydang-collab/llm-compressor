#!/usr/bin/env bash
# Paired quality eval: in-house W2A16 vs BF16 baseline, gpqa_diamond_cot_zeroshot
# + ifeval, 100 samples each, greedy, via benchmarks repo quality/run_ab.
# One node, 3 GPUs: BF16 TP2 on 0,1 + W2A16 (humming) on 2. Fail-closed:
# GPU-residue gate, health gates, harness check BEFORE any eval request.
set -euo pipefail

DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SERVE_VENV=/mnt/nfs/hoangduy/venvs/serve-sub4
BENCH_VENV=/mnt/nfs/hoangduy/venvs/benchmarks
BENCH_REPO=/mnt/nfs/hoangduy/projects/benchmarks
BF16=/mnt/nfs/hoangduy/hf_assets/Qwen/Qwen3-30B-A3B-Instruct-2507
W2A16=/mnt/nfs/hoangduy/projects/llm-compressor/artifacts/Qwen3-30B-A3B-Instruct-2507-autoround-W2A16-g128-ddp8
RUN_ID="${RUN_ID:-qwen3-30b-w2a16-vs-bf16-g100}"

# serve-sub4 env (humming JIT: nvrtc libs, ninja, NFS kernel cache)
export PYTHONPATH=/mnt/nfs/hoangduy/venvs/humming-main-site
export LD_LIBRARY_PATH="$SERVE_VENV/lib/python3.12/site-packages/nvidia/cu13/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export PATH="$SERVE_VENV/bin:$PATH"
export HUMMING_CACHE_DIR=/mnt/nfs/hoangduy/claude/home/.humming/cache/
# offline HF for eval-side dataset loads (same cache the M3 evals used)
export HF_HOME=/mnt/nfs/hoangduy/cache/huggingface
export HF_HUB_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false

echo "=== node: $(hostname)"
echo "=== GPU state before launch:"
nvidia-smi --query-gpu=index,memory.used --format=csv,noheader
busy=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | awk 'NR<=3 && $1 > 2000' | wc -l)
if [ "$busy" -gt 0 ]; then
  echo "EVAL_RESULT: FAIL (residue on $busy of the first 3 GPUs)"
  exit 1
fi

CUDA_VISIBLE_DEVICES=0,1 "$SERVE_VENV/bin/vllm" serve "$BF16" \
  --served-model-name qwen3-30b-bf16 --port 8410 --tensor-parallel-size 2 \
  --max-model-len 8192 --max-logprobs 20 --gpu-memory-utilization 0.92 \
  >"$DIR/serve-bf16.txt" 2>&1 &
BF16_PID=$!

CUDA_VISIBLE_DEVICES=2 "$SERVE_VENV/bin/vllm" serve "$W2A16" \
  --served-model-name qwen3-30b-w2a16 --port 8411 \
  --max-model-len 8192 --max-logprobs 20 --gpu-memory-utilization 0.85 \
  >"$DIR/serve-w2a16.txt" 2>&1 &
W2A16_PID=$!

trap 'kill "$BF16_PID" "$W2A16_PID" 2>/dev/null || true' EXIT

wait_healthy() {  # port pid label
  for i in $(seq 1 240); do
    if ! kill -0 "$2" 2>/dev/null; then
      echo "EVAL_RESULT: FAIL ($3 server died)"; tail -30 "$DIR/serve-$3.txt"; exit 1
    fi
    if curl -sf "http://127.0.0.1:$1/health" >/dev/null 2>&1; then
      echo "=== $3 healthy after ${i}x5s"; return 0
    fi
    if [ "$i" -eq 240 ]; then
      echo "EVAL_RESULT: FAIL ($3 health timeout)"; tail -30 "$DIR/serve-$3.txt"; exit 1
    fi
    sleep 5
  done
}
wait_healthy 8410 "$BF16_PID" bf16
wait_healthy 8411 "$W2A16_PID" w2a16

echo "=== harness check (fail-closed):"
"$BENCH_VENV/bin/python" "$DIR/harness_check.py" "$DIR/harness-check.json"

echo "=== run_ab:"
cd "$BENCH_REPO"
# general suite shells out to the lm_eval CLI — it lives in the benchmarks venv.
# Prepended AFTER the servers spawn so their env (serve-sub4 bin first) is untouched.
export PATH="$BENCH_VENV/bin:$PATH"
command -v lm_eval >/dev/null || { echo "EVAL_RESULT: FAIL (lm_eval not on PATH)"; exit 1; }
set +e
"$BENCH_VENV/bin/python" -m quality.run_ab \
  --profile /mnt/nfs/hoangduy/projects/llm-compressor/pipeline/configs/benchmarks/qwen3-30b-a3b-2507-w2a16.sh \
  --run-id "$RUN_ID" \
  --general-limit 100 \
  --out-root "$BENCH_REPO/results" \
  --report 2>&1 | tee "$DIR/run_ab-output.txt"
rc=${PIPESTATUS[0]}
set -e

if [ "$rc" -eq 0 ]; then
  echo "EVAL_RESULT: PASS"
elif [ "$rc" -eq 3 ]; then
  echo "EVAL_RESULT: PARTIAL (some legs failed; results on disk)"
else
  echo "EVAL_RESULT: FAIL (run_ab rc=$rc)"
fi
exit "$rc"
