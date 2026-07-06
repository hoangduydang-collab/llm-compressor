#!/usr/bin/env bash
# Start MiniMax-M3 quantize detached from the current SSH session.
# Use when sbatch fails and tmux does not survive disconnect.
#
# Each method needs its own idle GPU node (~428B BF16 on CPU RAM).
#
#   Node A: METHOD=gptq bash pipeline/slurm/run_quantize_minimax_m3_detached.sh
#   Node B: METHOD=awq  bash pipeline/slurm/run_quantize_minimax_m3_detached.sh
#
# Env:
#   METHOD     gptq | awq (required)
#   MODEL_ID   default: /mnt/nfs/hoangduy/hf_assets/MiniMaxAI/MiniMax-M3
#   CONFIG     default: pipeline/configs/minimax_m3.yaml
#   SCHEME     default: W4AFP8
#   LOG        default: /mnt/nfs/hoangduy/logs/quantize-m3-<method>.log
#   PID_FILE   default: /mnt/nfs/hoangduy/logs/quantize-m3-<method>.pid
#   EXTRA      optional extra --set flags for pipeline.run
#
# Monitor:
#   tail -f /mnt/nfs/hoangduy/logs/quantize-m3-gptq.log
#   pgrep -af 'pipeline.run.*quantize'
# Stop:
#   kill "$(cat /mnt/nfs/hoangduy/logs/quantize-m3-gptq.pid)"

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
cd "$REPO_ROOT"

METHOD="${METHOD:?set METHOD=gptq or awq}"
CONFIG="${CONFIG:-pipeline/configs/minimax_m3.yaml}"
SCHEME="${SCHEME:-W4AFP8}"
MODEL_ID="${MODEL_ID:-/mnt/nfs/hoangduy/hf_assets/MiniMaxAI/MiniMax-M3}"
EXTRA="${EXTRA:-}"

mkdir -p /mnt/nfs/hoangduy/logs
LOG="${LOG:-/mnt/nfs/hoangduy/logs/quantize-m3-${METHOD}.log}"
PID_FILE="${PID_FILE:-/mnt/nfs/hoangduy/logs/quantize-m3-${METHOD}.pid}"

if [[ ! -f "$MODEL_ID/config.json" ]]; then
  echo "ERROR: model not found at $MODEL_ID"
  exit 1
fi

if [[ -f "$PID_FILE" ]]; then
  old_pid="$(cat "$PID_FILE" 2>/dev/null || true)"
  if [[ -n "$old_pid" ]] && kill -0 "$old_pid" 2>/dev/null; then
    echo "Quantize already running (pid=$old_pid, method=$METHOD)"
    echo "  tail -f $LOG"
    exit 0
  fi
fi

RUN_SCRIPT="$(mktemp /tmp/quantize_m3_${METHOD}_XXXXXX.sh)"
cat > "$RUN_SCRIPT" <<EOF
#!/usr/bin/env bash
set -uo pipefail
source /mnt/nfs/hoangduy/env.sh
source /mnt/nfs/hoangduy/venvs/quant/bin/activate
cd /mnt/nfs/hoangduy/projects/llm-compressor
export HOME=\${WORK_ROOT:-/mnt/nfs/hoangduy}
export FLASHINFER_WORKSPACE_DIR=\${FLASHINFER_WORKSPACE_DIR:-\$HOME/cache/flashinfer}
export TOKENIZERS_PARALLELISM=false
mkdir -p "\$FLASHINFER_WORKSPACE_DIR" 2>/dev/null || true
export CONFIG=$(printf '%q' "$CONFIG")
export METHOD=$(printf '%q' "$METHOD")
export SCHEME=$(printf '%q' "$SCHEME")
export MODEL_ID=$(printf '%q' "$MODEL_ID")

echo "host=\$(hostname) method=\$METHOD scheme=\$SCHEME started=\$(date -Is)"
nvidia-smi --query-gpu=index,name,memory.total --format=csv 2>/dev/null || true
free -g 2>/dev/null || true

exec python -m pipeline.run --config "\$CONFIG" --stage quantize \\
  --set quantization.method="\$METHOD" \\
  --set quantization.scheme="\$SCHEME" \\
  --set model.id="\$MODEL_ID" ${EXTRA}
EOF
chmod +x "$RUN_SCRIPT"

echo "host=$(hostname) starting detached MiniMax-M3 quantize"
echo "  method: $METHOD"
echo "  scheme: $SCHEME"
echo "  model:  $MODEL_ID"
echo "  config: $CONFIG"
nvidia-smi --query-gpu=index,name,memory.total --format=csv 2>/dev/null || true

# setsid + nohup: survive SSH disconnect (unlike foreground tmux attach).
: > "$LOG"
nohup setsid bash "$RUN_SCRIPT" >> "$LOG" 2>&1 &
echo $! > "$PID_FILE"
echo "started pid=$(cat "$PID_FILE")"
echo "  tail -f $LOG"
echo "  kill \$(cat $PID_FILE)   # to stop"
