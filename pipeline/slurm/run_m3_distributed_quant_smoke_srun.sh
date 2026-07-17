#!/usr/bin/env bash
# Native llm-compressor DDP smoke: GPTQ and AWQ, one 8xH100 node per method
# (both methods launch in parallel on separate nodes by default;
# PARALLEL_METHODS=0 restores the sequential single-node-at-a-time behavior).

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
MIN_MEM_AVAILABLE_BYTES="${MIN_MEM_AVAILABLE_BYTES:-1200000000000}"
# r6 lesson: the distributed CPU offload (`DistributedCPUCache`) keeps ONE full
# shared model copy in /dev/shm (`_share_filename_cpu_`), so the real /dev/shm
# requirement is the whole checkpoint (~869 GB for MiniMax-M3), not an IPC
# floor. "auto" sizes the gate from the checkpoint's safetensors index
# (total_size + 5% headroom); the r6 128 GB floor let AWQ launch on a node with
# only 213 GB free and die mid-load.
MIN_SHM_AVAILABLE_BYTES="${MIN_SHM_AVAILABLE_BYTES:-auto}"
# r7 lesson: without an explicit --cpus-per-task, Slurm 21.08 binds the whole
# step task to ONE physical core (Cpus_allowed_list "0,96") even though the
# exclusive job owns all 192 CPUs -- the 8-rank torchrun worker (dataloading,
# shm dispatch memcpy, NCCL progress threads) then serializes onto 2 hardware
# threads. r7's model-dispatch phase alone projected ~1h45m under that binding.
CPUS_PER_TASK="${CPUS_PER_TASK:-192}"
# Launch GPTQ and AWQ concurrently on separate exclusive nodes (each method's
# results/logs/offload trees are already method-scoped, so the arms share
# nothing but the read-only checkpoint and calibration dataset cache).
PARALLEL_METHODS="${PARALLEL_METHODS:-1}"
# 1 (default): metrics/evidence only, no checkpoint. 0: save the quantized
# checkpoint so pipeline/m3_checkpoint_scale_audit.py and
# pipeline/verify_quant_checkpoint.py can run against it.
EVIDENCE_ONLY="${EVIDENCE_ONLY:-1}"

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

  # Reclaim orphaned torch shared-memory segments before the capacity gate.
  # Ranks hard-killed mid-run leak their /dev/shm/torch_* files (torch's shm
  # manager cannot clean up after SIGKILL); the r5 GPTQ crash left 852 GB on
  # gpu-h101, which starved the next arm's model load. We hold this node
  # exclusively, so any $USER-owned torch_* file not mapped by a live process
  # is such leakage.
  local mapped_shm stale_removed=0 stale_file
  mapped_shm="$(awk '$6 ~ /^\/dev\/shm\/torch_/ {print $6}' /proc/[0-9]*/maps 2>/dev/null | sort -u || true)"
  for stale_file in /dev/shm/torch_*; do
    [[ -e "$stale_file" && -O "$stale_file" ]] || continue
    grep -qxF "$stale_file" <<<"$mapped_shm" && continue
    rm -f -- "$stale_file" 2>/dev/null && stale_removed=$((stale_removed + 1)) || true
  done

  # Resolve the /dev/shm requirement: the checkpoint's exact byte size + 5%.
  local min_shm_bytes="$MIN_SHM_AVAILABLE_BYTES"
  if [[ "$min_shm_bytes" == auto ]]; then
    local index_json="$MODEL_ID/model.safetensors.index.json"
    if [[ ! -f "$index_json" ]]; then
      echo "ERROR: MIN_SHM_AVAILABLE_BYTES=auto requires $index_json; set an explicit byte value" >&2
      return 3
    fi
    min_shm_bytes="$(python - "$index_json" <<'PY'
import json
import sys

total = json.load(open(sys.argv[1]))["metadata"]["total_size"]
print(total * 105 // 100)
PY
)"
  fi

  {
    echo "stale_shm_files_removed=$stale_removed"
    echo "min_shm_available_bytes_required=$min_shm_bytes"
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
  if (( mem_available_kb * 1024 < MIN_MEM_AVAILABLE_BYTES )); then
    echo "ERROR: MemAvailable below ${MIN_MEM_AVAILABLE_BYTES} bytes: ${mem_available_kb} KiB" >&2
    return 3
  fi
  if (( shm_available < min_shm_bytes )); then
    echo "ERROR: /dev/shm available space below ${min_shm_bytes} bytes: ${shm_available} bytes" >&2
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
    python - <<'PY'
import torch

print(f"torch_cuda_build={torch.version.cuda}")
print(f"torch_cuda_device_count={torch.cuda.device_count()}")
PY
    nvidia-smi --query-gpu=index,name,driver_version \
      --format=csv,noheader,nounits
  } >"$method_logs/environment.txt" 2>&1

  local command=(
    torchrun --nproc_per_node=8 -m pipeline.run
    --config "$CONFIG" --stage quantize
    --set "quantization.method=$method"
    --set "model.id=$MODEL_ID"
    --set "model.offload_folder=$offload_dir"
    --set "output_dir=$method_root"
  )
  # EVIDENCE_ONLY=1 (default) skips the checkpoint save; set 0 to save the
  # quantized checkpoint (needed by the scale audit / verify tooling).
  if [[ "${EVIDENCE_ONLY:-1}" == 1 || "${EVIDENCE_ONLY:-1}" == true ]]; then
    command+=(--evidence-only)
  fi
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
declare -A method_pids=()
for method in gptq awq; do
  command=(
    srun --exclusive --nodes=1 --ntasks=1 --gres=gpu:8
    --cpus-per-task="$CPUS_PER_TASK"
    --time="$TIME_LIMIT" --kill-on-bad-exit=1
    env RUN_ID="$RUN_ID" RESULT_ROOT="$RESULT_ROOT" LOG_ROOT="$LOG_ROOT"
    OFFLOAD_ROOT="$OFFLOAD_ROOT" MODEL_ID="$MODEL_ID" CONFIG="$CONFIG"
    ENV_FILE="$ENV_FILE" VENV_ACTIVATE="$VENV_ACTIVATE"
    SAMPLE_INTERVAL="$SAMPLE_INTERVAL"
    MIN_MEM_AVAILABLE_BYTES="$MIN_MEM_AVAILABLE_BYTES"
    MIN_SHM_AVAILABLE_BYTES="$MIN_SHM_AVAILABLE_BYTES"
    EVIDENCE_ONLY="$EVIDENCE_ONLY"
    bash "$SCRIPT_DIR/run_m3_distributed_quant_smoke_srun.sh" --worker "$method"
  )
  if [[ "$DRY_RUN" == 1 || "$DRY_RUN" == true ]]; then
    printf '%q ' "${command[@]}"
    printf '\n'
    evidence_flag=""
    [[ "$EVIDENCE_ONLY" == 1 || "$EVIDENCE_ONLY" == true ]] && evidence_flag=" --evidence-only"
    printf 'worker: torchrun --nproc_per_node=8 -m pipeline.run --config %q --stage quantize%s --set quantization.method=%q\n' "$CONFIG" "$evidence_flag" "$method"
    continue
  fi

  if [[ "$PARALLEL_METHODS" == 1 || "$PARALLEL_METHODS" == true ]]; then
    mkdir -p "$LOG_ROOT/$method"
    "${command[@]}" >"$LOG_ROOT/$method/controller-launch.log" 2>&1 &
    method_pids[$method]=$!
    echo "method=$method launched in parallel pid=${method_pids[$method]}"
  else
    rc=0
    "${command[@]}" || rc=$?
    echo "method=$method rc=$rc logs=$LOG_ROOT/$method results=$RESULT_ROOT/$method"
    [[ "$rc" -eq 0 ]] || overall=1
  fi
done

for method in gptq awq; do
  [[ -n "${method_pids[$method]:-}" ]] || continue
  rc=0
  wait "${method_pids[$method]}" || rc=$?
  echo "method=$method rc=$rc logs=$LOG_ROOT/$method results=$RESULT_ROOT/$method"
  [[ "$rc" -eq 0 ]] || overall=1
done

exit "$overall"
