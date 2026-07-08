#!/usr/bin/env bash
# Capture Python stacks of hung vLLM workers to pinpoint a decode-time deadlock
# (the "No available shared memory broadcast block found in 60s" -> "RPC call to
# sample_tokens timed out" hang after CUDA-graph capture).
#
# A shm_broadcast timeout means one/more Worker_TP* entered a forward/collective
# and never returned. Two capture backends:
#
#   pyspy  (non-destructive): py-spy dump of every GPU-resident process, twice.
#          Requires ptrace permission (Yama kernel.yama.ptrace_scope=0 or
#          CAP_SYS_PTRACE). On locked-down nodes this is "Permission Denied".
#
#   abrt   (destructive):   send SIGABRT to a few workers. With PYTHONFAULTHANDLER=1
#          (set by debug_cudagraph_ima.sh / the serve launchers) Python dumps ALL
#          thread stacks to stderr -> the run's LOG. This KILLS those workers, so
#          only use it once, when already hung. No ptrace needed.
#
# MODE=auto (default) tries py-spy first and falls back to abrt on Permission Denied.
#
# Usage (while the serve is hung, before the RPC timeout):
#   LOG=/mnt/nfs/hoangduy/logs/m3-cudagraph-debug.log bash pipeline/slurm/dump_hang_stacks.sh
#   MODE=abrt LOG=... bash pipeline/slurm/dump_hang_stacks.sh   # force SIGABRT route
#
# Env:
#   OUT         stack output file (default /mnt/nfs/hoangduy/logs/m3-hang-stacks.txt)
#   LOG         run log where worker stderr/faulthandler lands (REQUIRED for abrt)
#   MODE        auto | pyspy | abrt   (default auto)
#   SNAPSHOTS   py-spy snapshots (default 2)
#   GAP_SECS    seconds between py-spy snapshots (default 8)
#   FRAMES      max lines per py-spy dump (default 60)
#   ABRT_COUNT  workers to SIGABRT in abrt mode (default 2)
#
# Reading the result: find the DEEPEST application frame, e.g.
#   fused_allreduce_gemma_rms_norm / all_reduce / NCCL  -> collective deadlock
#   _select_experts / _compute_routing / nan_to_num     -> router path
#   cutlass_moe / grouped_gemm / finalize               -> MoE kernel
#   shm_broadcast acquire_read / dequeue                 -> that rank is WAITING (victim)

set -uo pipefail

OUT="${OUT:-/mnt/nfs/hoangduy/logs/m3-hang-stacks.txt}"
LOG="${LOG:-/mnt/nfs/hoangduy/logs/m3-cudagraph-debug.log}"
MODE="${MODE:-auto}"
SNAPSHOTS="${SNAPSHOTS:-2}"
GAP_SECS="${GAP_SECS:-8}"
FRAMES="${FRAMES:-60}"
ABRT_COUNT="${ABRT_COUNT:-2}"

_worker_pids() {
  nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null \
    | tr -d ' ' | grep -E '^[0-9]+$' | sort -u
}

pids="$(_worker_pids)"
if [ -z "${pids// }" ]; then
  echo "[dump_hang_stacks] no GPU-resident processes found (already crashed?)."
  exit 1
fi

_have_pyspy() {
  if command -v py-spy >/dev/null 2>&1; then return 0; fi
  if [ -n "${UV:-}" ]; then
    "$UV" pip show py-spy >/dev/null 2>&1 && return 0
    echo "[dump_hang_stacks] installing py-spy via \$UV..." >&2
    "$UV" pip install py-spy >/dev/null 2>&1 && return 0
  fi
  return 1
}

_pyspy_denied() { grep -qi 'permission denied\|operation not permitted' "$1"; }

run_pyspy() {
  : > "$OUT"
  { echo "# hang stacks (py-spy) $(date -Is) on $(hostname)";
    echo "# pids: $(echo "$pids" | tr '\n' ' ')"; } | tee -a "$OUT"
  local first_dump; first_dump="$(mktemp)"
  local snap p name
  for snap in $(seq 1 "$SNAPSHOTS"); do
    echo "" | tee -a "$OUT"
    echo "############ SNAPSHOT $snap/$SNAPSHOTS $(date -Is) ############" | tee -a "$OUT"
    for p in $pids; do
      name="$(ps -o comm= -p "$p" 2>/dev/null | tr -d ' ')"
      echo "" | tee -a "$OUT"; echo "===== pid $p ($name) =====" | tee -a "$OUT"
      py-spy dump --pid "$p" 2>&1 | head -n "$FRAMES" | tee -a "$OUT" | tee "$first_dump" >/dev/null
    done
    # bail early if the very first dump was a permission error
    if [ "$snap" -eq 1 ] && _pyspy_denied "$first_dump"; then
      rm -f "$first_dump"; return 3
    fi
    [ "$snap" -lt "$SNAPSHOTS" ] && sleep "$GAP_SECS"
  done
  rm -f "$first_dump"
  echo "" | tee -a "$OUT"; echo "[dump_hang_stacks] wrote $OUT (py-spy)" | tee -a "$OUT"
  return 0
}

run_abrt() {
  echo "[dump_hang_stacks] SIGABRT route (needs PYTHONFAULTHANDLER=1; dumps to LOG=$LOG)."
  echo "[dump_hang_stacks] WARNING: this kills the signalled workers."
  local victims; victims="$(echo "$pids" | head -n "$ABRT_COUNT" | tr '\n' ' ')"
  echo "[dump_hang_stacks] sending SIGABRT to: $victims"
  local mark="=== FAULTHANDLER DUMP $(date +%s) ==="
  # marker so we can slice the fresh traceback out of the (possibly long) LOG
  echo "$mark" >> "$LOG" 2>/dev/null || true
  # shellcheck disable=SC2086
  kill -ABRT $victims 2>/dev/null || true
  sleep 5
  {
    echo "# hang stacks (SIGABRT/faulthandler) $(date -Is) on $(hostname)";
    echo "# signalled pids: $victims";
    echo "# --- traceback slice from $LOG ---";
    if [ -f "$LOG" ]; then
      awk -v m="$mark" 'index($0,m){f=1} f' "$LOG" | head -n 400
    else
      echo "(LOG not found: $LOG)";
    fi
  } | tee "$OUT"
  echo "[dump_hang_stacks] wrote $OUT (SIGABRT). If empty, workers may lack"
  echo "  PYTHONFAULTHANDLER=1 or stderr wasn't redirected to \$LOG; re-run the"
  echo "  serve with debug_cudagraph_ima.sh (it sets both)."
}

case "$MODE" in
  pyspy)
    if _have_pyspy; then run_pyspy || echo "[dump_hang_stacks] py-spy failed (rc=$?)."; \
    else echo "[dump_hang_stacks] py-spy unavailable."; fi ;;
  abrt)
    run_abrt ;;
  auto|*)
    if _have_pyspy; then
      run_pyspy; rc=$?
      if [ "$rc" -eq 3 ]; then
        echo "[dump_hang_stacks] py-spy Permission Denied (ptrace locked) -> SIGABRT fallback."
        run_abrt
      fi
    else
      echo "[dump_hang_stacks] py-spy unavailable -> SIGABRT fallback."
      run_abrt
    fi ;;
esac
