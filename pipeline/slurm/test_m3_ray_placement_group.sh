#!/usr/bin/env bash
set -euo pipefail

OUT=""
EXPECTED_BUNDLES=16
TIMEOUT_SECONDS=120
while (($#)); do
  case "$1" in
    --out) OUT=$2; shift 2 ;;
    --expected-bundles) EXPECTED_BUNDLES=$2; shift 2 ;;
    --timeout-seconds) TIMEOUT_SECONDS=$2; shift 2 ;;
    *) echo "unknown argument: $1" >&2; exit 2 ;;
  esac
done
[[ -n "$OUT" ]] || { echo "--out is required" >&2; exit 2; }
mkdir -p "$OUT"

rank=${SLURM_PROCID:-0}
cleanup() {
  if [[ -d /tmp/ray/session_latest/logs ]]; then
    tar -czf "$OUT/ray-logs-rank-$rank.tar.gz" \
      -C /tmp/ray/session_latest logs 2>"$OUT/ray-logs-rank-$rank.err" || true
  fi
  ray stop --force >"$OUT/ray-stop-rank-$rank.txt" 2>&1 || true
}
trap cleanup EXIT
source pipeline/slurm/test_m3_ray_topology.sh --out "$OUT/topology" --keep-alive

if ((rank != 0)); then
  for _ in $(seq 1 600); do
    [[ -f "$OUT/driver-done" ]] && exit 0
    sleep 1
  done
  echo "timed out waiting for placement-group driver" >&2
  exit 1
fi

rc=0
ray status >"$OUT/ray-status-before.txt" 2>&1 || rc=$?
ray list placement-groups --detail \
  >"$OUT/placement-groups-before.txt" 2>&1 || true
if ((rc == 0)); then
  if python - "$OUT/placement-group.json" "$EXPECTED_BUNDLES" "$TIMEOUT_SECONDS" <<'PY'
import json
import sys
import time

import ray
from ray.util.placement_group import placement_group, placement_group_table

output, expected, timeout = sys.argv[1], int(sys.argv[2]), float(sys.argv[3])
started = time.monotonic()
report = {
    "schema_version": 1,
    "expected_bundles": expected,
    "timeout_seconds": timeout,
    "ready": False,
}
group = None
try:
    ray.init(address="auto")
    group = placement_group([{"GPU": 1}] * expected, strategy="PACK")
    ray.get(group.ready(), timeout=timeout)
    report["ready"] = True
    report["table"] = placement_group_table(group)
except Exception as error:
    report["error"] = f"{type(error).__name__}: {error}"
finally:
    report["elapsed_seconds"] = time.monotonic() - started
    with open(output, "w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2, default=str)
    if group is not None:
        ray.util.remove_placement_group(group)
    ray.shutdown()
raise SystemExit(0 if report["ready"] else 1)
PY
  then
    rc=0
  else
    rc=$?
  fi
fi
ray list placement-groups --detail \
  >"$OUT/placement-groups-after.txt" 2>&1 || true
ray status >"$OUT/ray-status-after.txt" 2>&1 || true
printf '%s\n' "$rc" >"$OUT/return_code.txt"
touch "$OUT/driver-done"
exit "$rc"
