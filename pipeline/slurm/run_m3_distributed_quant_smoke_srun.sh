#!/usr/bin/env bash
# Native llm-compressor DDP smoke: GPTQ then AWQ, one 8xH100 node at a time.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
CONFIG="${CONFIG:-$REPO_ROOT/pipeline/configs/minimax_m3_distributed_smoke.yaml}"
RUN_ID="${RUN_ID:-$(date -u +%Y%m%dT%H%M%SZ)-m3-ddp-quant-smoke}"
RESULT_ROOT="${RESULT_ROOT:-/mnt/nfs/hoangduy/results/m3-distributed-quant-smoke/$RUN_ID}"
LOG_ROOT="${LOG_ROOT:-/mnt/nfs/hoangduy/logs/m3-distributed-quant-smoke/$RUN_ID}"
OFFLOAD_ROOT="${OFFLOAD_ROOT:-/mnt/nfs/hoangduy/offload/m3-distributed-quant-smoke/$RUN_ID}"
MODEL_ID="${MODEL_ID:-/mnt/nfs/hoangduy/hf_assets/MiniMaxAI/MiniMax-M3}"
ENV_FILE="${ENV_FILE:-/mnt/nfs/hoangduy/env.sh}"
VENV_ACTIVATE="${VENV_ACTIVATE:-/mnt/nfs/hoangduy/venvs/quant/bin/activate}"
TIME_LIMIT="${TIME_LIMIT:-24:00:00}"
SAMPLE_INTERVAL="${SAMPLE_INTERVAL:-60}"
DRY_RUN="${DRY_RUN:-0}"

worker_main() {
  local method="${1:?worker requires gptq or awq}"
  local method_root="$RESULT_ROOT/$method"
  local method_logs="$LOG_ROOT/$method"
  local offload_dir="$OFFLOAD_ROOT/$method"
  [[ "$method" == gptq || "$method" == awq ]] || {
    echo "ERROR: unsupported method=$method" >&2
    return 2
  }

  source "$ENV_FILE"
  source "$VENV_ACTIVATE"
  cd "$REPO_ROOT"
  export PYTHONPATH="$REPO_ROOT/src:$REPO_ROOT"
  export PYTHONUNBUFFERED=1 TOKENIZERS_PARALLELISM=false
  export HF_ENABLE_PARALLEL_LOADING="${HF_ENABLE_PARALLEL_LOADING:-true}"
  export HF_PARALLEL_LOADING_WORKERS="${HF_PARALLEL_LOADING_WORKERS:-16}"
  mkdir -p "$method_root" "$method_logs" "$offload_dir"

  {
    grep -E 'MemTotal|MemAvailable|Shmem|SwapTotal|SwapFree' /proc/meminfo
    df -B1 /dev/shm
    nvidia-smi --query-gpu=index,uuid,memory.used,memory.total \
      --format=csv,noheader,nounits
    python - <<'PY'
import torch

count = torch.cuda.device_count()
print(f"torch_cuda_device_count={count}")
if count != 8:
    raise SystemExit(f"expected exactly 8 visible GPUs, found {count}")
PY
  } >"$method_logs/node_preflight.txt" 2>&1

  local mem_available_kb shm_available
  mem_available_kb="$(awk '/^MemAvailable:/ {print $2}' /proc/meminfo)"
  shm_available="$(df -B1 --output=avail /dev/shm | tail -1 | tr -d ' ')"
  if (( mem_available_kb * 1024 < 1200000000000 )); then
    echo "ERROR: MemAvailable below 1.2 TB: ${mem_available_kb} KiB" >&2
    return 3
  fi
  if (( shm_available < 900000000000 )); then
    echo "ERROR: /dev/shm available space below 900 GB: ${shm_available} bytes" >&2
    return 3
  fi

  {
    echo "run_id=$RUN_ID"
    echo "method=$method"
    echo "host=$(hostname)"
    echo "slurm_job_id=${SLURM_JOB_ID:-}"
    echo "slurm_step_id=${SLURM_STEP_ID:-}"
    git rev-parse HEAD
    python --version
    python -m pip show llmcompressor compressed-tensors torch transformers
  } >"$method_logs/environment.txt" 2>&1

  local command=(
    torchrun --nproc_per_node=8 -m pipeline.run
    --config "$CONFIG" --stage quantize --evidence-only
    --set "quantization.method=$method"
    --set "model.id=$MODEL_ID"
    --set "model.offload_folder=$offload_dir"
    --set "output_dir=$method_root"
  )
  printf '%q ' "${command[@]}" >"$method_logs/command.txt"
  printf '\n' >>"$method_logs/command.txt"

  set +e
  /usr/bin/time -v "${command[@]}" \
    >"$method_logs/torchrun.out" 2>"$method_logs/torchrun.err" &
  local quant_pid=$!
  (
    while kill -0 "$quant_pid" 2>/dev/null; do
      date -u +%Y-%m-%dT%H:%M:%SZ
      grep -E 'MemTotal|MemAvailable|Shmem|SwapTotal|SwapFree' /proc/meminfo
      nvidia-smi --query-gpu=index,uuid,memory.used,memory.total,utilization.gpu \
        --format=csv,noheader,nounits
      sleep "$SAMPLE_INTERVAL"
    done
  ) >"$method_logs/resources.log" 2>&1 &
  local sampler_pid=$!

  wait "$quant_pid"
  local rc=$?
  kill "$sampler_pid" 2>/dev/null || true
  wait "$sampler_pid" 2>/dev/null || true
  set -e
  printf '%s\n' "$rc" >"$method_logs/rc"
  return "$rc"
}

if [[ "${1:-}" == --worker ]]; then
  shift
  worker_main "$@"
  exit $?
fi

if [[ -n "${SLURM_JOB_ID:-}" ]]; then
  echo "ERROR: run this controller outside an allocation; it owns top-level srun" >&2
  exit 2
fi

overall=0
for method in gptq awq; do
  command=(
    srun --exclusive --nodes=1 --ntasks=1 --gres=gpu:8
    --time="$TIME_LIMIT" --kill-on-bad-exit=1
    env RUN_ID="$RUN_ID" RESULT_ROOT="$RESULT_ROOT" LOG_ROOT="$LOG_ROOT"
    OFFLOAD_ROOT="$OFFLOAD_ROOT" MODEL_ID="$MODEL_ID" CONFIG="$CONFIG"
    ENV_FILE="$ENV_FILE" VENV_ACTIVATE="$VENV_ACTIVATE"
    SAMPLE_INTERVAL="$SAMPLE_INTERVAL"
    bash "$SCRIPT_DIR/run_m3_distributed_quant_smoke_srun.sh" --worker "$method"
  )
  if [[ "$DRY_RUN" == 1 || "$DRY_RUN" == true ]]; then
    printf '%q ' "${command[@]}"
    printf '\n'
    printf 'worker: torchrun --nproc_per_node=8 -m pipeline.run --config %q --stage quantize --evidence-only --set quantization.method=%q\n' "$CONFIG" "$method"
    continue
  fi

  rc=0
  "${command[@]}" || rc=$?
  echo "method=$method rc=$rc logs=$LOG_ROOT/$method results=$RESULT_ROOT/$method"
  [[ "$rc" -eq 0 ]] || overall=1
done

exit "$overall"
