#!/usr/bin/env bash
# Start MiniMax-M3 vLLM serve-verify detached from the current SSH session.
# Use when sbatch fails and you are already on an idle 8-GPU node (e.g. h118).
#
#   bash pipeline/slurm/run_serve_minimax_m3_detached.sh
#
# Options: CONFIG, OUT_DIR, CHECKPOINT (same as submit_serve_minimax_m3.sh)
#
# Monitor:
#   tail -f serves/m3-awq-w4afp8/run.log
#   pgrep -af 'pipeline.run.*--stage serve'
# Stop:
#   kill "$(cat serves/m3-awq-w4afp8/serve.pid)"
#   kill -9 -"$(cat serves/m3-awq-w4afp8/serve.pid)"

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
cd "$REPO_ROOT"

CONFIG="${CONFIG:-pipeline/configs/minimax_m3.yaml}"
OUT_DIR="${OUT_DIR:-serves/m3-awq-w4afp8}"
CHECKPOINT="${CHECKPOINT:-artifacts/MiniMax-M3-awq-W4AFP8/20260707-082218/checkpoint}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-8192}"
GPU_UTIL="${GPU_UTIL:-0.9}"

mkdir -p "$OUT_DIR" /mnt/nfs/hoangduy/logs
PID_FILE="$OUT_DIR/serve.pid"

if [[ ! -d "$CHECKPOINT" || ! -f "$CHECKPOINT/config.json" ]]; then
  echo "ERROR: checkpoint not found at $CHECKPOINT"
  exit 1
fi

if [[ -f "$PID_FILE" ]]; then
  old_pid="$(cat "$PID_FILE" 2>/dev/null || true)"
  if [[ -n "$old_pid" ]] && kill -0 "$old_pid" 2>/dev/null; then
    echo "Serve already running (pid=$old_pid). tail -f $OUT_DIR/run.log"
    exit 0
  fi
fi

RUN_SCRIPT="$(mktemp /tmp/m3_serve_run_XXXXXX.sh)"
cat > "$RUN_SCRIPT" <<EOF
#!/usr/bin/env bash
set -uo pipefail
source /mnt/nfs/hoangduy/env.sh
source /mnt/nfs/hoangduy/venvs/quant/bin/activate
cd /mnt/nfs/hoangduy/projects/llm-compressor
# Record the REAL worker pid: this bash exec's into python below (same pid), so \$\$
# is the python process. \`setsid\` may fork, making the launcher's \$! stale/parent;
# writing here means \`kill \$(cat PID_FILE)\` always targets the actual worker.
echo \$\$ > $(printf '%q' "$PID_FILE")
export HOME=\${WORK_ROOT:-/mnt/nfs/hoangduy}
export PYTHONPATH=/mnt/nfs/hoangduy/projects/llm-compressor
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export PYTHONUNBUFFERED=1
export FLASHINFER_WORKSPACE_DIR=\${FLASHINFER_WORKSPACE_DIR:-\$HOME/cache/flashinfer}
export TOKENIZERS_PARALLELISM=false
mkdir -p "\$FLASHINFER_WORKSPACE_DIR" 2>/dev/null || true
export CONFIG=$(printf '%q' "$CONFIG")
export OUT_DIR=$(printf '%q' "$OUT_DIR")
export CHECKPOINT=$(printf '%q' "$CHECKPOINT")
export MAX_MODEL_LEN=$(printf '%q' "$MAX_MODEL_LEN")
export GPU_UTIL=$(printf '%q' "$GPU_UTIL")

echo "host=\$(hostname) serve-verify started=\$(date -Is)"
echo "checkpoint=\$CHECKPOINT"
python -c "import vllm; print('vllm', vllm.__version__)" 2>/dev/null || echo "vllm: not importable"
nvidia-smi --query-gpu=index,name,memory.total,memory.free --format=csv 2>/dev/null || true

exec python -m pipeline.run --config "\$CONFIG" --stage serve \\
  --checkpoint "\$CHECKPOINT" \\
  --set serve.tensor_parallel_size=8 \\
  --set serve.enable_expert_parallel=true \\
  --set serve.block_size=128 \\
  --set serve.kv_cache_dtype=fp8 \\
  --set serve.max_model_len="\$MAX_MODEL_LEN" \\
  --set serve.gpu_memory_utilization="\$GPU_UTIL" \\
  --set eval.enabled=false
EOF
chmod +x "$RUN_SCRIPT"

echo "host=$(hostname) starting detached MiniMax-M3 vLLM serve-verify"
echo "  config:     $CONFIG"
echo "  checkpoint: $CHECKPOINT"
echo "  out:        $OUT_DIR"
echo "  max_model_len: $MAX_MODEL_LEN"
nvidia-smi --query-gpu=index,name,memory.total --format=csv 2>/dev/null || true

# setsid + nohup: survive SSH disconnect (unlike foreground tmux attach).
nohup setsid bash "$RUN_SCRIPT" >> "$OUT_DIR/run.log" 2>&1 &
echo $! > "$PID_FILE"
echo "started launcher; worker pid written by run script -> $PID_FILE"
echo "  tail -f $OUT_DIR/run.log"
echo "  kill \$(cat $PID_FILE)        # graceful stop"
echo "  kill -9 -\$(cat $PID_FILE)    # hard stop (whole process group)"
echo ""
echo "serve_report.json -> $(dirname "$CHECKPOINT")/serve_report.json"
