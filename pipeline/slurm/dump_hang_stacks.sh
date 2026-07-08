#!/usr/bin/env bash
# Capture live Python stacks of hung vLLM workers to pinpoint a decode-time
# deadlock (e.g. the "No available shared memory broadcast block found in 60s"
# -> "RPC call to sample_tokens timed out" hang after CUDA-graph capture).
#
# A shm_broadcast timeout means one/more Worker_TP* entered a forward/collective
# and never returned. This dumps every GPU-resident process' Python stack with
# py-spy (non-destructive) TWICE a few seconds apart, so a stuck frame (identical
# across snapshots) is distinguishable from slow-but-progressing work.
#
# Usage (run in a SECOND shell while the serve is hung, before the RPC timeout):
#   bash pipeline/slurm/dump_hang_stacks.sh
#   OUT=/mnt/nfs/hoangduy/logs/m3-hang-stacks.txt bash pipeline/slurm/dump_hang_stacks.sh
#
# Reads the faulting frame from the output: look for the DEEPEST app frame, e.g.
#   fused_allreduce_gemma_rms_norm / all_reduce / NCCL  -> fused-AR / collective deadlock
#   _select_experts / nan_to_num / _compute_routing     -> router path
#   cutlass_moe / grouped_gemm / finalize               -> MoE kernel
#   shm_broadcast acquire_read/dequeue                  -> that rank is WAITING (victim, not culprit)

set -uo pipefail

OUT="${OUT:-/mnt/nfs/hoangduy/logs/m3-hang-stacks.txt}"
SNAPSHOTS="${SNAPSHOTS:-2}"
GAP_SECS="${GAP_SECS:-8}"
FRAMES="${FRAMES:-60}"

# Prefer the cluster uv wrapper ($UV from env.sh); fall back to python -m if needed.
if command -v py-spy >/dev/null 2>&1; then
  PYSPY=(py-spy)
elif [ -n "${UV:-}" ] && "$UV" pip show py-spy >/dev/null 2>&1; then
  PYSPY=(py-spy)
else
  echo "[dump_hang_stacks] py-spy not found; installing into the active venv..."
  if [ -n "${UV:-}" ]; then
    "$UV" pip install py-spy || { echo "[dump_hang_stacks] FAILED to install py-spy"; exit 1; }
  else
    pip install py-spy || { echo "[dump_hang_stacks] FAILED to install py-spy (no \$UV, no pip)"; exit 1; }
  fi
  PYSPY=(py-spy)
fi

_worker_pids() {
  nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null \
    | tr -d ' ' | grep -E '^[0-9]+$' | sort -u
}

pids="$(_worker_pids)"
if [ -z "${pids// }" ]; then
  echo "[dump_hang_stacks] no GPU-resident processes found (already crashed?)."
  exit 1
fi

: > "$OUT"
{
  echo "# hang stacks captured $(date -Is) on $(hostname)"
  echo "# GPU-resident pids: $(echo "$pids" | tr '\n' ' ')"
} | tee -a "$OUT"

for snap in $(seq 1 "$SNAPSHOTS"); do
  echo "" | tee -a "$OUT"
  echo "############ SNAPSHOT $snap/$SNAPSHOTS  $(date -Is) ############" | tee -a "$OUT"
  for p in $pids; do
    name="$(ps -o comm= -p "$p" 2>/dev/null | tr -d ' ')"
    echo "" | tee -a "$OUT"
    echo "===== pid $p ($name) =====" | tee -a "$OUT"
    "${PYSPY[@]}" dump --pid "$p" 2>&1 | head -n "$FRAMES" | tee -a "$OUT"
  done
  [ "$snap" -lt "$SNAPSHOTS" ] && sleep "$GAP_SECS"
done

echo "" | tee -a "$OUT"
echo "[dump_hang_stacks] wrote $OUT" | tee -a "$OUT"
