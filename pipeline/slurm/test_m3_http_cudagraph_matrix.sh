#!/usr/bin/env bash
# Controlled MiniMax-M3 HTTP CUDA-graph RCA matrix (no vLLM site-packages edits).
#
# Isolates graph / breakable-cudagraph / sync-CUDA behavior for
# cyankiwi/MiniMax-M3-AWQ-INT4 and classifies each trial via
# pipeline/m3_cudagraph_evidence.py.
#
# Usage (cluster, free 8-GPU node):
#   bash pipeline/slurm/test_m3_http_cudagraph_matrix.sh
#   MATRIX_CASES=async_baseline_1,graphs_off bash pipeline/slurm/test_m3_http_cudagraph_matrix.sh
#
# Local dry-run (no GPUs, no nohup/kill/free_gpus):
#   DRY_RUN=1 bash pipeline/slurm/test_m3_http_cudagraph_matrix.sh
#
# Scope: evidence only. CUDA_LAUNCH_BLOCKING / DEBUG_CUDAGRAPH=1 is labelled
# masked_pass, never "fixed". Do not treat a masked pass as root-cause closure.
#
# See BUGS_AND_FIXES.md "HTTP async cudagraph race" / RCA matrix protocol.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
cd "$REPO_ROOT"

DRY_RUN="${DRY_RUN:-0}"
MATRIX_CASES="${MATRIX_CASES:-}"
READY_TIMEOUT_SECS="${READY_TIMEOUT_SECS:-1800}"
POLL_SECS="${POLL_SECS:-10}"
BASE_PORT="${BASE_PORT:-8100}"
RESULTS_ROOT="${RESULTS_ROOT:-/mnt/nfs/hoangduy/logs/m3-cudagraph-rca}"
if [[ "$DRY_RUN" == "1" || "$DRY_RUN" == "true" ]]; then
  RESULTS_ROOT="${RESULTS_ROOT_LOCAL:-$REPO_ROOT/artifacts/m3-cudagraph-rca-dryrun}"
fi

RUN_ID="${RUN_ID:-$(date +%Y%m%d-%H%M%S)}"
RUN_DIR="${RUN_DIR:-$RESULTS_ROOT/$RUN_ID}"
mkdir -p "$RUN_DIR"

LAUNCHER="$SCRIPT_DIR/run_vllm_http_serve_smoke.sh"
CHAT_SMOKE="$SCRIPT_DIR/smoke_chat_completions.sh"
CLASSIFIER=(python -m pipeline.m3_cudagraph_evidence)

# Baseline contract (plan): cyankiwi, TP8, EP, 8192/0.9, language-model-only.
export CKPT="${CKPT:-/mnt/nfs/hoangduy/hf_assets/cyankiwi/MiniMax-M3-AWQ-INT4}"
export SERVED_NAME="${SERVED_NAME:-cyankiwi/MiniMax-M3-AWQ-INT4}"
export TP="${TP:-8}"
export MAX_MODEL_LEN="${MAX_MODEL_LEN:-8192}"
export GPU_UTIL="${GPU_UTIL:-0.9}"
export LANGUAGE_MODEL_ONLY="${LANGUAGE_MODEL_ONLY:-1}"
export ENABLE_EP="${ENABLE_EP:-1}"
export DISABLE_CUSTOM_AR="${DISABLE_CUSTOM_AR:-1}"
export KV_CACHE_DTYPE="${KV_CACHE_DTYPE:-fp8}"
export BLOCK_SIZE="${BLOCK_SIZE:-128}"

# Case definitions: name|ENFORCE_EAGER|DEBUG_CUDAGRAPH|BREAKABLE|COREDUMP|graphs_on|debug_flag
# BREAKABLE: "" = leave unset; "0" = force VLLM_USE_BREAKABLE_CUDAGRAPH=0
# COREDUMP: 0/1
ALL_CASES=(
  "async_baseline_1|0|0||0|1|0"
  "async_baseline_2|0|0||0|1|0"
  "async_baseline_3|0|0||0|1|0"
  "graphs_off|1|0||0|0|0"
  "blocking_mask|0|1||0|1|1"
  "breakable_off|0|0|0|0|1|0"
  "async_coredump|0|0||1|1|0"
)

_select_cases() {
  local want="$1"
  local c name
  if [[ -z "$want" ]]; then
    printf '%s\n' "${ALL_CASES[@]}"
    return
  fi
  local IFS=','
  # shellcheck disable=SC2206
  local names=($want)
  for c in "${ALL_CASES[@]}"; do
    name="${c%%|*}"
    for w in "${names[@]}"; do
      w="$(echo "$w" | tr -d '[:space:]')"
      if [[ "$name" == "$w" ]]; then
        echo "$c"
      fi
    done
  done
}

CASES=()
while IFS= read -r _line; do
  [[ -n "$_line" ]] && CASES+=("$_line")
done < <(_select_cases "$MATRIX_CASES")
if [[ ${#CASES[@]} -eq 0 ]]; then
  echo "ERROR: no cases selected (MATRIX_CASES=$MATRIX_CASES)"
  exit 2
fi

# Manifest
{
  echo "{"
  echo "  \"run_id\": \"$RUN_ID\","
  echo "  \"host\": \"$(hostname 2>/dev/null || echo unknown)\","
  echo "  \"dry_run\": $([[ "$DRY_RUN" == "1" || "$DRY_RUN" == "true" ]] && echo true || echo false),"
  echo "  \"ckpt\": \"$CKPT\","
  echo "  \"served_name\": \"$SERVED_NAME\","
  echo "  \"max_model_len\": $MAX_MODEL_LEN,"
  echo "  \"gpu_util\": $GPU_UTIL,"
  echo "  \"cases\": ["
  _mi=0
  for c in "${CASES[@]}"; do
    name="${c%%|*}"
    [[ $_mi -gt 0 ]] && echo ","
    printf '    \"%s\"' "$name"
    _mi=$((_mi + 1))
  done
  echo ""
  echo "  ]"
  echo "}"
} >"$RUN_DIR/run_manifest.json"

echo "[matrix] run_dir=$RUN_DIR"
echo "[matrix] cases=${CASES[*]%%|*}"
echo "[matrix] dry_run=$DRY_RUN"

_stop_trial() {
  local pid_file="$1"
  [[ "$DRY_RUN" == "1" || "$DRY_RUN" == "true" ]] && return 0
  if [[ -f "$pid_file" ]]; then
    local pid
    pid="$(cat "$pid_file" 2>/dev/null || true)"
    if [[ -n "${pid:-}" ]]; then
      # Kill process group started by setsid
      kill -TERM -"$pid" 2>/dev/null || kill -TERM "$pid" 2>/dev/null || true
      sleep 2
      kill -KILL -"$pid" 2>/dev/null || kill -KILL "$pid" 2>/dev/null || true
    fi
    rm -f "$pid_file"
  fi
  # Sweep own leftovers only (free_gpus never kills other users).
  MIN_FREE_GIB="${MIN_FREE_GIB:-70}" bash "$SCRIPT_DIR/free_gpus.sh" || true
}

_wait_ready_or_fail() {
  local log="$1"
  local port="$2"
  local deadline=$((SECONDS + READY_TIMEOUT_SECS))
  while (( SECONDS < deadline )); do
    if [[ -f "$log" ]]; then
      if grep -q "Application startup complete" "$log" 2>/dev/null; then
        return 0
      fi
      if grep -qiE 'illegal memory access|cudaErrorIllegalAddress|Engine core initialization failed|Worker failed' "$log" 2>/dev/null; then
        return 1
      fi
    fi
    # Also probe health if process might be up
    if curl -sf "http://127.0.0.1:${port}/health" >/dev/null 2>&1; then
      return 0
    fi
    sleep "$POLL_SECS"
  done
  echo "[matrix] timeout waiting for ready/fail (${READY_TIMEOUT_SECS}s)"
  return 1
}

_run_case() {
  local spec="$1"
  local idx="$2"
  IFS='|' read -r name enforce_eager debug_cg breakable coredump graphs_on debug_flag <<<"$spec"

  local trial_dir="$RUN_DIR/$name"
  mkdir -p "$trial_dir"
  local port=$((BASE_PORT + idx))
  local log="$trial_dir/serve.log"
  local pid_file="$trial_dir/serve.pid"
  local chat_json="$trial_dir/chat.json"
  local meta_json="$trial_dir/meta.json"
  local result_json="$trial_dir/result.json"
  local config_out="$trial_dir/effective_config.txt"
  local dump_pattern="$trial_dir/cuda_coredump_%h.%p.%t"

  echo ""
  echo "========== CASE $name (port=$port) =========="

  # Write meta before launch
  cat >"$meta_json" <<EOF
{
  "case_name": "$name",
  "graphs_on": $([[ "$graphs_on" == "1" ]] && echo true || echo false),
  "debug_cudagraph": $([[ "$debug_flag" == "1" ]] && echo true || echo false),
  "enforce_eager": "$enforce_eager",
  "debug_cudagraph_env": "$debug_cg",
  "vllm_use_breakable_cudagraph": "${breakable:-<unset>}",
  "coredump": $([[ "$coredump" == "1" ]] && echo true || echo false),
  "core_dump_path": $([[ "$coredump" == "1" ]] && echo "\"$dump_pattern\"" || echo null),
  "port": $port,
  "log": "$log"
}
EOF

  if [[ "$DRY_RUN" == "1" || "$DRY_RUN" == "true" ]]; then
    echo "[matrix] DRY_RUN: printing effective config for $name"
    # Must not invoke free_gpus / nohup / kill
    (
      export PRINT_EFFECTIVE_CONFIG=1
      export SKIP_GPU_PREFLIGHT=1
      export APPLY_M3_PATCHES=0
      export PATCH_CKPT_CONFIG=0
      export ENFORCE_EAGER="$enforce_eager"
      export DEBUG_CUDAGRAPH="$debug_cg"
      export PORT="$port"
      export LOG="$log"
      export PID_FILE="$pid_file"
      unset VLLM_USE_BREAKABLE_CUDAGRAPH 2>/dev/null || true
      if [[ "$breakable" == "0" ]]; then
        export VLLM_USE_BREAKABLE_CUDAGRAPH=0
      fi
      bash "$LAUNCHER"
    ) >"$config_out" 2>&1 || true
    # Synthesize a placeholder result for dry-run
    cat >"$result_json" <<EOF
{
  "verdict": "inconclusive",
  "case_name": "$name",
  "notes": ["DRY_RUN — no GPU launch"],
  "dry_run": true,
  "port": $port,
  "log": "$log"
}
EOF
    echo "[matrix] DRY_RUN config written: $config_out"
    # Assert dry-run config has no launch side effects in the printed plan
    if grep -qE 'nohup|setsid vllm|free_gpus' "$config_out" 2>/dev/null; then
      echo "ERROR: dry-run config output mentions nohup/setsid/free_gpus"
      exit 3
    fi
    return 0
  fi

  # Live cluster path: free GPUs between trials
  MIN_FREE_GIB="${MIN_FREE_GIB:-70}" bash "$SCRIPT_DIR/free_gpus.sh" || {
    echo "[matrix] FAIL: GPUs not free before $name"
    echo '{"verdict":"inconclusive","notes":["GPUs not free"]}' >"$result_json"
    return 1
  }

  # Launch
  set +e
  (
    export ENFORCE_EAGER="$enforce_eager"
    export DEBUG_CUDAGRAPH="$debug_cg"
    export PORT="$port"
    export LOG="$log"
    export PID_FILE="$pid_file"
    unset VLLM_USE_BREAKABLE_CUDAGRAPH 2>/dev/null || true
    if [[ "$breakable" == "0" ]]; then
      export VLLM_USE_BREAKABLE_CUDAGRAPH=0
    fi
    if [[ "$coredump" == "1" ]]; then
      export CUDA_ENABLE_COREDUMP_ON_EXCEPTION=1
      export CUDA_COREDUMP_SHOW_PROGRESS=1
      export CUDA_COREDUMP_GENERATION_FLAGS='skip_nonrelocated_elf_images,skip_global_memory,skip_shared_memory,skip_local_memory,skip_constbank_memory'
      export CUDA_COREDUMP_FILE="$dump_pattern"
    else
      unset CUDA_ENABLE_COREDUMP_ON_EXCEPTION CUDA_COREDUMP_SHOW_PROGRESS \
        CUDA_COREDUMP_GENERATION_FLAGS CUDA_COREDUMP_FILE 2>/dev/null || true
    fi
    bash "$LAUNCHER"
  )
  local launch_rc=$?
  set -e
  if [[ $launch_rc -ne 0 ]]; then
    echo "[matrix] launcher exited $launch_rc for $name"
  fi

  local ready=0
  if _wait_ready_or_fail "$log" "$port"; then
    ready=1
  fi

  local chat_ok=false
  if [[ $ready -eq 1 ]]; then
    set +e
    MODEL="$SERVED_NAME" PORT="$port" HOST=127.0.0.1 \
      bash "$CHAT_SMOKE" >"$chat_json" 2>&1
    local chat_rc=$?
    set -e
    if [[ $chat_rc -eq 0 ]] && grep -q '"choices"' "$chat_json" 2>/dev/null; then
      chat_ok=true
    fi
  else
    echo '{"ok": false, "error": "server not ready"}' >"$chat_json"
  fi

  # Update meta with chat_ok
  python - "$meta_json" "$chat_ok" <<'PY'
import json, sys
p = sys.argv[1]
ok = sys.argv[2].lower() == "true"
meta = json.loads(open(p, encoding="utf-8").read())
meta["chat_ok"] = ok
open(p, "w", encoding="utf-8").write(json.dumps(meta, indent=2) + "\n")
PY

  # Classify
  set +e
  "${CLASSIFIER[@]}" "$log" --chat "$chat_json" --meta "$meta_json" -o "$result_json"
  set -e

  _stop_trial "$pid_file"
  echo "[matrix] case $name done → $result_json"
}

idx=0
for spec in "${CASES[@]}"; do
  _run_case "$spec" "$idx"
  idx=$((idx + 1))
done

# Summary
python - "$RUN_DIR" <<'PY'
import json, sys
from pathlib import Path
run_dir = Path(sys.argv[1])
rows = []
for d in sorted(run_dir.iterdir()):
    if not d.is_dir():
        continue
    rj = d / "result.json"
    if not rj.exists():
        continue
    rec = json.loads(rj.read_text(encoding="utf-8"))
    rows.append({
        "case": rec.get("case_name") or d.name,
        "verdict": rec.get("verdict"),
        "server_ready": rec.get("server_ready"),
        "chat_ok": rec.get("chat_ok"),
        "ima": rec.get("ima"),
        "notes": rec.get("notes"),
    })
summary = {"run_dir": str(run_dir), "trials": rows}
out = run_dir / "summary.json"
out.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
print(json.dumps(summary, indent=2))
PY

echo "[matrix] summary: $RUN_DIR/summary.json"

# Dry-run self-check: unique ports/paths, five+ distinct configs, no side-effect cmds
if [[ "$DRY_RUN" == "1" || "$DRY_RUN" == "true" ]]; then
  python - "$RUN_DIR" <<'PY'
import sys
from pathlib import Path
run_dir = Path(sys.argv[1])
ports = set()
configs = []
for d in sorted(p for p in run_dir.iterdir() if p.is_dir()):
    cfg = d / "effective_config.txt"
    text = cfg.read_text(encoding="utf-8", errors="replace") if cfg.exists() else ""
    for bad in ("nohup", "free_gpus", "setsid vllm"):
        if bad in text:
            raise SystemExit(f"DRY_RUN FAIL: {cfg} contains {bad!r}")
    # extract PORT=
    for line in text.splitlines():
        if line.strip().startswith("PORT="):
            ports.add(line.split("=", 1)[1].strip())
        if "ENFORCE_EAGER=" in line or "DEBUG_CUDAGRAPH=" in line or "VLLM_USE_BREAKABLE" in line:
            pass
    # fingerprint
    eager = dbg = brk = None
    for line in text.splitlines():
        s = line.strip()
        if s.startswith("ENFORCE_EAGER="):
            eager = s
        if s.startswith("DEBUG_CUDAGRAPH="):
            dbg = s
        if s.startswith("VLLM_USE_BREAKABLE_CUDAGRAPH="):
            brk = s
    configs.append((d.name, eager, dbg, brk))
if len(ports) != len(configs):
    raise SystemExit(f"DRY_RUN FAIL: expected unique ports, got {ports} for {len(configs)} cases")
# Distinct (eager, dbg, breakable) fingerprints among non-baseline-repeat cases
fps = {(e, d, b) for _, e, d, b in configs}
print(f"DRY_RUN OK: {len(configs)} cases, {len(ports)} unique ports, {len(fps)} distinct env fingerprints")
print("fingerprints:", sorted(fps))
PY
fi

echo "[matrix] DONE"
