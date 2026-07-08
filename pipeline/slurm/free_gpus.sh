#!/usr/bin/env bash
# Preflight GPU guard: ensure the node's GPUs are actually free BEFORE launching a
# vLLM serve/quantize run. Recurring failure mode: a previous crashed run leaves
# EngineCore/VllmWorker/pt_main_thread processes holding ~70 GiB/GPU, so the next
# serve dies at startup with:
#   ValueError: Free memory on device cuda:X (~6 GiB) ... less than gpu_memory_utilization
#
# This script:
#   1. lists current GPU compute apps and their owners,
#   2. SIGTERMs, then SIGKILLs, THIS USER's leftover vLLM/pipeline processes,
#   3. polls until they are actually gone (not just signalled),
#   4. verifies every GPU has >= MIN_FREE_GIB free,
#   5. refuses to proceed (exit 1) if GPUs are still occupied — by us OR a teammate.
#
# It NEVER kills processes owned by other users (shared node; see cluster-access rule).
#
# Usage:
#   bash pipeline/slurm/free_gpus.sh            # kill own leftovers + verify
#   source pipeline/slurm/free_gpus.sh          # same, but keep shell (for chaining)
#   FORCE=0 bash pipeline/slurm/free_gpus.sh    # verify only, do NOT kill anything
#   MIN_FREE_GIB=70 bash pipeline/slurm/free_gpus.sh
#
# Env:
#   FORCE         (default 1)  1=kill own leftovers; 0=verify only
#   MIN_FREE_GIB  (default 70) per-GPU free memory required to declare "free"
#   KILL_PATTERN  process cmdline regex to match own leftovers
#   WAIT_SECS     (default 45) max seconds to wait for procs to die / mem to free

set -uo pipefail

FORCE="${FORCE:-1}"
MIN_FREE_GIB="${MIN_FREE_GIB:-70}"
WAIT_SECS="${WAIT_SECS:-45}"
KILL_PATTERN="${KILL_PATTERN:-pipeline\.run|EngineCore|VllmWorker|pt_main_thread|from multiprocessing|vllm}"

_min_free_mib=$(( MIN_FREE_GIB * 1024 ))

_have_nvidia_smi() { command -v nvidia-smi >/dev/null 2>&1; }

# PIDs of compute apps currently resident on the GPUs.
_gpu_pids() {
  nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null \
    | tr -d ' ' | grep -E '^[0-9]+$' || true
}

# Subset of _gpu_pids owned by the current user.
_gpu_pids_mine() {
  local p
  for p in $(_gpu_pids); do
    if [ "$(ps -o user= -p "$p" 2>/dev/null | tr -d ' ')" = "$USER" ]; then
      echo "$p"
    fi
  done
}

_print_gpu_apps() {
  echo "--- GPU compute apps (pid / user / etime / cmd) ---"
  local p
  local any=0
  for p in $(_gpu_pids); do
    any=1
    ps -o pid=,user=,etime=,cmd= -p "$p" 2>/dev/null | sed 's/^/  /'
  done
  [ "$any" -eq 0 ] && echo "  (none)"
  echo "--- per-GPU free memory ---"
  nvidia-smi --query-gpu=index,memory.free --format=csv,noheader 2>/dev/null | sed 's/^/  /'
}

# Returns 0 if EVERY GPU has >= MIN_FREE_GIB free, else 1.
_all_gpus_free() {
  local busy
  busy=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits 2>/dev/null \
    | tr -d ' ' | awk -v min="$_min_free_mib" 'NF && $1 < min {n++} END{print n+0}')
  [ "${busy:-1}" -eq 0 ]
}

free_gpus_main() {
  if ! _have_nvidia_smi; then
    echo "[free_gpus] WARNING: nvidia-smi not found; skipping GPU preflight."
    return 0
  fi

  echo "[free_gpus] preflight: require >= ${MIN_FREE_GIB} GiB free on every GPU (FORCE=$FORCE)"
  _print_gpu_apps

  if _all_gpus_free; then
    echo "[free_gpus] OK: all GPUs already have >= ${MIN_FREE_GIB} GiB free."
    return 0
  fi

  local mine other
  mine="$(_gpu_pids_mine | tr '\n' ' ')"
  # other = gpu pids that are NOT mine
  other="$(comm -23 <(_gpu_pids | sort -u) <(_gpu_pids_mine | sort -u) 2>/dev/null | tr '\n' ' ')"

  if [ "$FORCE" != "1" ]; then
    echo "[free_gpus] FORCE=0: not killing anything. GPUs are NOT free."
    return 1
  fi

  if [ -z "${mine// }" ]; then
    echo "[free_gpus] No leftover GPU processes owned by '$USER'."
    if [ -n "${other// }" ]; then
      echo "[free_gpus] FAIL: GPUs occupied by OTHER users (pids: $other)."
      echo "[free_gpus] Refusing to kill other users' jobs. Pick a free node or coordinate."
    else
      echo "[free_gpus] FAIL: memory held but no compute apps listed (driver/ECC state?)."
    fi
    return 1
  fi

  echo "[free_gpus] SIGTERM own leftovers: $mine"
  # shellcheck disable=SC2086
  kill -TERM $mine 2>/dev/null || true

  local waited=0
  while [ "$waited" -lt "$WAIT_SECS" ]; do
    sleep 3; waited=$((waited + 3))
    mine="$(_gpu_pids_mine | tr '\n' ' ')"
    if [ -z "${mine// }" ]; then break; fi
    echo "[free_gpus]   still alive after ${waited}s: $mine"
  done

  # Escalate to SIGKILL for anything that ignored SIGTERM.
  mine="$(_gpu_pids_mine | tr '\n' ' ')"
  if [ -n "${mine// }" ]; then
    echo "[free_gpus] SIGKILL stubborn own procs: $mine"
    # shellcheck disable=SC2086
    kill -KILL $mine 2>/dev/null || true
  fi

  # Also sweep by pattern in case some workers detached from the GPU app list.
  pkill -KILL -u "$USER" -f "$KILL_PATTERN" 2>/dev/null || true

  # Wait for the driver to actually reclaim the memory (freeing lags process exit).
  waited=0
  while [ "$waited" -lt "$WAIT_SECS" ]; do
    if _all_gpus_free && [ -z "$(_gpu_pids_mine | tr '\n' ' ' | tr -d ' ')" ]; then
      echo "[free_gpus] OK: GPUs reclaimed after ${waited}s."
      _print_gpu_apps
      return 0
    fi
    sleep 3; waited=$((waited + 3))
  done

  echo "[free_gpus] FAIL: GPUs still not free after ${WAIT_SECS}s."
  _print_gpu_apps
  other="$(comm -23 <(_gpu_pids | sort -u) <(_gpu_pids_mine | sort -u) 2>/dev/null | tr '\n' ' ')"
  [ -n "${other// }" ] && echo "[free_gpus] Remaining GPU procs owned by others: $other (not killed)."
  return 1
}

free_gpus_main
_free_gpus_rc=$?
# When executed (not sourced), propagate the exit code so callers can `|| exit`.
if [ "${BASH_SOURCE[0]}" = "${0}" ]; then
  exit "$_free_gpus_rc"
fi
