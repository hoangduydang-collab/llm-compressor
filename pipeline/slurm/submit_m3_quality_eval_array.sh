#!/usr/bin/env bash
set -euo pipefail

MATRIX=""; RUN_ROOT=""; SMOKE_GATE=""; DRY_RUN=0
while (($#)); do
  case "$1" in
    --matrix) MATRIX=$2; shift 2 ;;
    --run-root) RUN_ROOT=$2; shift 2 ;;
    --smoke-gate) SMOKE_GATE=$2; shift 2 ;;
    --dry-run) DRY_RUN=1; shift ;;
    *) echo "unknown argument: $1" >&2; exit 2 ;;
  esac
done
[[ -n "$MATRIX" && -n "$RUN_ROOT" && -n "$SMOKE_GATE" ]] || {
  echo "--matrix, --run-root, and --smoke-gate are required" >&2; exit 2;
}
[[ -d "$RUN_ROOT/preflight" ]] || {
  echo "preflight directory is required before array submission" >&2; exit 2;
}
PLAN="$RUN_ROOT/production_launch_plan.json"
[[ ! -e "$PLAN" ]] || { echo "refusing existing launch plan: $PLAN" >&2; exit 2; }
[[ ! -e "$RUN_ROOT/models" ]] || {
  echo "refusing existing arm output tree: $RUN_ROOT/models" >&2; exit 2;
}
mkdir -p "$RUN_ROOT/logs"

python -m pipeline.m3_quality_eval launch-plan \
  --matrix "$MATRIX" --profile production --smoke-gate "$SMOKE_GATE" \
  --out "$PLAN"

python - "$PLAN" <<'PY'
import json, sys

plan = json.load(open(sys.argv[1], encoding="utf-8"))
for index, arm in enumerate(plan["arms"]):
    print(
        f'index={index} arm={arm["model_label"]}/{arm["shard"]} '
        f'nodes={arm["nodes"]} gpus={arm["gpus_per_node"]}'
    )
PY

metadata=$(python - "$PLAN" <<'PY'
import json, sys

plan = json.load(open(sys.argv[1], encoding="utf-8"))
arms = plan["arms"]
if not arms:
    raise SystemExit("production launch plan has no arms")
if any(int(arm["nodes"]) != 1 or int(arm["gpus_per_node"]) != 8 for arm in arms):
    raise SystemExit("array launcher requires one-node, eight-GPU arms")
limit = plan.get("arm_time_limit")
if not limit:
    raise SystemExit("production launch plan requires arm_time_limit")
print(len(arms), int(plan["max_parallel_arms"]), limit)
PY
)
read -r arm_count max_parallel arm_time_limit <<<"$metadata"

submit=(sbatch --parsable
  "--array=0-$((arm_count - 1))%${max_parallel}"
  --nodes=1 --ntasks=1 --gpus-per-node=8 --exclusive
  "--time=${arm_time_limit}"
  "--output=$RUN_ROOT/logs/production-%A_%a.out"
  "--error=$RUN_ROOT/logs/production-%A_%a.err"
  pipeline/slurm/run_m3_quality_eval_array_arm.sh
  --plan "$PLAN" --run-root "$RUN_ROOT" --matrix "$MATRIX")
printf '%q ' "${submit[@]}"; printf '\n'
if ((DRY_RUN)); then exit 0; fi

{
  printf '%q ' "${submit[@]}"
  printf '\n'
} >"$RUN_ROOT/submission_command.txt"
job_id=$("${submit[@]}")
printf '%s\n' "$job_id" >"$RUN_ROOT/array_job_id.txt"
printf 'submitted_array_job=%s\n' "$job_id"
