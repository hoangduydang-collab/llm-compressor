#!/usr/bin/env bash
set -euo pipefail

PLAN=""; RUN_ROOT=""; MATRIX=""
while (($#)); do
  case "$1" in
    --plan) PLAN=$2; shift 2 ;;
    --run-root) RUN_ROOT=$2; shift 2 ;;
    --matrix) MATRIX=$2; shift 2 ;;
    *) echo "unknown argument: $1" >&2; exit 2 ;;
  esac
done
[[ -n "$PLAN" && -n "$RUN_ROOT" && -n "$MATRIX" ]] || {
  echo "--plan, --run-root, and --matrix are required" >&2; exit 2;
}
INDEX="${SLURM_ARRAY_TASK_ID:-}"
[[ "$INDEX" =~ ^[0-9]+$ ]] || {
  echo "SLURM_ARRAY_TASK_ID must be a non-negative integer" >&2; exit 2;
}

row=$(python - "$PLAN" "$RUN_ROOT/preflight/resolved_tasks.json" "$INDEX" <<'PY'
import json, sys
from pathlib import Path

plan_path, resolved_path, raw_index = sys.argv[1:]
arms = json.load(open(plan_path, encoding="utf-8"))["arms"]
index = int(raw_index)
if index >= len(arms):
    raise SystemExit(f"array index {index} out of range for {len(arms)} arms")
arm = arms[index]
if int(arm["nodes"]) != 1 or int(arm["gpus_per_node"]) != 8:
    raise SystemExit("array launcher requires one-node, eight-GPU arms")
resolved_file = Path(resolved_path)
resolved = (
    json.load(open(resolved_file, encoding="utf-8")).get("aliases", {})
    if resolved_file.is_file()
    else {}
)
fields = (
    arm["model_label"],
    arm["model_path"],
    arm["shard"],
    str(arm["tensor_parallel_size"]),
    str(arm.get("pipeline_parallel_size", 1)),
    arm["distributed_executor_backend"],
    ",".join(resolved.get(name, name) for name in arm["tasks"]),
    "1" if arm["distributional_probe"] else "0",
    str(arm["probe_tokens"]),
)
print("\n".join(fields))
PY
)
mapfile -t fields <<<"$row"
(( ${#fields[@]} == 9 )) || { echo "invalid launch-plan arm record" >&2; exit 2; }

runner="${M3_QUALITY_ARM_RUNNER:-pipeline/slurm/test_m3_quality_eval_arm.sh}"
exec "$runner" \
  --profile production --run-root "$RUN_ROOT" --matrix "$MATRIX" \
  --model-label "${fields[0]}" --model "${fields[1]}" \
  --shard "${fields[2]}" --tensor-parallel-size "${fields[3]}" \
  --pipeline-parallel-size "${fields[4]}" \
  --distributed-executor-backend "${fields[5]}" --tasks "${fields[6]}" \
  --run-probe "${fields[7]}" --probe-tokens "${fields[8]}"
