#!/usr/bin/env bash
# Periodic checkpoint of DERIVED state for a live unified spec-dec window.
#
# The raw evidence is already durable without help: the arm writes each serve's
# serve.log, cell-config.txt, gate outputs, metrics/*-{pre,post}.txt and aiperf JSON
# to NFS before the next serve begins, and appends one line per finished serve to
# progress.txt. A died allocation therefore loses at most the serve in flight, and
# ONLY=<labels> resumes the rest.
#
# What is NOT durable without this loop is the ANALYSIS. If the operating session
# disappears mid-window, the aggregate has to be re-derived by whoever picks it up.
# So this re-runs the aggregator every INTERVAL seconds and leaves
# aggregate.{json,txt} beside the raw evidence, plus a timestamped history so a
# mid-window trend (drift, a degrading node) stays visible after the fact rather than
# being averaged away at the end.
#
# Deliberately does NOT git-commit: it runs unattended and would race the operator's
# own commits. Committing stays a human/agent decision at milestones.
#
# Usage (detached, alongside the controller):
#   tmux new-session -d -s m3-specdec-unified-ckpt \
#     "ROOT=<window> bash pipeline/slurm/checkpoint_specdec_unified.sh"
set -uo pipefail

ROOT=${ROOT:?set ROOT to the window directory}
ARM=${ARM:-unified}
INTERVAL=${INTERVAL:-600}
PY=${PY:-/mnt/nfs/hoangduy/venvs/quant/bin/python}
REPO=${REPO:-/mnt/nfs/hoangduy/projects/llm-compressor}

C=$ROOT/arm-$ARM
HIST=$ROOT/aggregate-history
mkdir -p "$HIST"
log() { echo "[ckpt $(date -u +%H:%M:%S)] $1" | tee -a "$ROOT/checkpoint.log"; }

log "start root=$ROOT interval=${INTERVAL}s"

snapshot() {
  local ts
  ts=$(date -u +%Y%m%dT%H%M%SZ)
  "$PY" "$REPO/pipeline/specdec_unified_aggregate.py" \
      --root "$ROOT" --arm "$ARM" --out-json "$ROOT/aggregate.json" \
      > "$ROOT/aggregate.txt" 2>&1
  local rc=$?
  cp -f "$ROOT/aggregate.txt" "$HIST/aggregate-$ts.txt" 2>/dev/null || true
  local n
  n=$(grep -cE '^[A-Za-z0-9-]+ done=' "$C/progress.txt" 2>/dev/null || echo 0)
  log "snapshot $ts rc=$rc serves_done=$n"
}

while [ ! -f "$C/arm-done.txt" ]; do
  snapshot
  for _ in $(seq 1 "$((INTERVAL / 10))"); do
    [ -f "$C/arm-done.txt" ] && break
    sleep 10
  done
done

log "arm finished -- final snapshot"
snapshot
log "done"
