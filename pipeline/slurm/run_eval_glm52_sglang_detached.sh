#!/usr/bin/env bash
# Start GLM-5.2 SGLang eval detached from the current SSH session.
# Use when sbatch fails with "I/O error writing script/environment to file"
# and you are already on an idle 8-GPU node (e.g. h119).
#
#   bash pipeline/slurm/run_eval_glm52_sglang_detached.sh
#
# Options: CONFIG, OUT_DIR, MODEL_PATH (same as submit_eval_glm52_sglang.sh)
#
# Monitor:
#   tail -f evals/glm52-w4afp8-phala/run.log
#   pgrep -af pipeline.evalsuite
# Stop:
#   kill "$(cat evals/glm52-w4afp8-phala/eval.pid)"

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
cd "$REPO_ROOT"

CONFIG="${CONFIG:-pipeline/configs/eval_glm52_w4afp8_sglang_h100.yaml}"
OUT_DIR="${OUT_DIR:-evals/glm52-w4afp8-phala}"
MODEL_PATH="${MODEL_PATH:-/mnt/nfs/hoangduy/hf_assets/PhalaCloud/GLM-5.2-W4AFP8}"

mkdir -p "$OUT_DIR" /mnt/nfs/hoangduy/logs
PID_FILE="$OUT_DIR/eval.pid"

if [[ -f "$PID_FILE" ]]; then
  old_pid="$(cat "$PID_FILE" 2>/dev/null || true)"
  if [[ -n "$old_pid" ]] && kill -0 "$old_pid" 2>/dev/null; then
    echo "Eval already running (pid=$old_pid). tail -f $OUT_DIR/run.log"
    exit 0
  fi
fi

RUN_SCRIPT="$(mktemp /tmp/glm52_eval_run_XXXXXX.sh)"
cat > "$RUN_SCRIPT" <<EOF
#!/usr/bin/env bash
set -uo pipefail
source /mnt/nfs/hoangduy/env.sh
source /mnt/nfs/hoangduy/venvs/sglang-eval/bin/activate
cd /mnt/nfs/hoangduy/projects/llm-compressor
export HOME=\${WORK_ROOT:-/mnt/nfs/hoangduy}
export PYTHONPATH=/mnt/nfs/hoangduy/projects/llm-compressor
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export CONFIG=$(printf '%q' "$CONFIG")
export OUT_DIR=$(printf '%q' "$OUT_DIR")
export MODEL_PATH=$(printf '%q' "$MODEL_PATH")
exec python -m pipeline.evalsuite.cli run \\
  --config "\$CONFIG" \\
  --model "\$MODEL_PATH" \\
  --out "\$OUT_DIR"
EOF
chmod +x "$RUN_SCRIPT"

echo "host=$(hostname) starting detached GLM-5.2 eval"
echo "  config: $CONFIG"
echo "  out:    $OUT_DIR"
nvidia-smi --query-gpu=index,name,memory.total --format=csv 2>/dev/null || true

# setsid + nohup: survive SSH disconnect (unlike foreground tmux attach).
nohup setsid bash "$RUN_SCRIPT" >> "$OUT_DIR/run.log" 2>&1 &
echo $! > "$PID_FILE"
echo "started pid=$(cat "$PID_FILE")"
echo "  tail -f $OUT_DIR/run.log"
echo "  kill \$(cat $PID_FILE)   # to stop"
