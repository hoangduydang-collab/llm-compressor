#!/usr/bin/env bash
set -euo pipefail

PROFILE=""; MATRIX=""; RUN_ROOT=""; SMOKE_GATE=""; DRY_RUN=0
while (($#)); do
  case "$1" in
    --profile) PROFILE=$2; shift 2 ;;
    --matrix) MATRIX=$2; shift 2 ;;
    --run-root) RUN_ROOT=$2; shift 2 ;;
    --smoke-gate) SMOKE_GATE=$2; shift 2 ;;
    --dry-run) DRY_RUN=1; shift ;;
    *) echo "unknown argument: $1" >&2; exit 2 ;;
  esac
done
[[ -n "$PROFILE" && -n "$MATRIX" && -n "$RUN_ROOT" ]] || { echo "--profile, --matrix, --run-root are required" >&2; exit 2; }
mkdir -p "$RUN_ROOT/logs"
PLAN="$RUN_ROOT/${PROFILE}_launch_plan.json"
cmd=(python -m pipeline.m3_quality_eval launch-plan --matrix "$MATRIX" --profile "$PROFILE" --out "$PLAN")
[[ -z "$SMOKE_GATE" ]] || cmd+=(--smoke-gate "$SMOKE_GATE")
"${cmd[@]}"

mapfile -t ARMS < <(python - "$PLAN" "$RUN_ROOT/preflight/resolved_tasks.json" <<'PYPLAN'
import json, sys
from pathlib import Path
resolved=json.load(open(sys.argv[2]))["aliases"] if Path(sys.argv[2]).is_file() else {}
for arm in json.load(open(sys.argv[1]))["arms"]:
    fields = [arm["model_label"], arm["model_path"], arm["shard"], str(arm["nodes"]),
              str(arm["tensor_parallel_size"]), str(arm.get("pipeline_parallel_size", 1)),
              arm["distributed_executor_backend"],
              ",".join(resolved.get(name,name) for name in arm["tasks"]) or "-", "1" if arm["distributional_probe"] else "0",
              str(arm["probe_tokens"])]
    print("\t".join(fields))
PYPLAN
)
TOTAL_NODES=$(python -c 'import json,sys; print(json.load(open(sys.argv[1]))["total_nodes"])' "$PLAN")
PLAN_TIME_LIMIT=$(python -c 'import json,sys; print(json.load(open(sys.argv[1])).get("arm_time_limit") or "")' "$PLAN")
echo "profile=$PROFILE arms=${#ARMS[@]} total_nodes=$TOTAL_NODES"

# Optional wall-clock bound applied to every arm allocation. Production runs the
# full task suite (thousands of samples, 16k-token generations) and must not be
# SIGTERM'd mid-eval, so an explicit generous TIME_LIMIT is expected there.
if [[ -n "$PLAN_TIME_LIMIT" ]]; then
  if [[ -n "${TIME_LIMIT:-}" && "$TIME_LIMIT" != "$PLAN_TIME_LIMIT" ]]; then
    echo "TIME_LIMIT conflicts with matrix arm_time_limit: $TIME_LIMIT != $PLAN_TIME_LIMIT" >&2
    exit 2
  fi
  TIME_LIMIT="$PLAN_TIME_LIMIT"
else
  TIME_LIMIT="${TIME_LIMIT:-}"
fi
time_args=()
[[ -z "$TIME_LIMIT" ]] || time_args=(--time "$TIME_LIMIT")

# Any multi-node arm (BF16 TP8xPP2/Ray) needs the two-node topology gate written
# to $RUN_ROOT/ray_preflight/gate.json before its arm self-checks it. This is
# required for both smoke and production, not smoke alone.
MAX_ARM_NODES=$(python -c 'import json,sys; print(max((a["nodes"] for a in json.load(open(sys.argv[1]))["arms"]), default=1))' "$PLAN")
if ((MAX_ARM_NODES > 1)); then
  ray_check=(srun --exclusive --nodes=2 --ntasks=2 --gpus-per-node=8 --kill-on-bad-exit=1 "${time_args[@]}"
    pipeline/slurm/test_m3_ray_topology.sh --out "$RUN_ROOT/ray_preflight" --stop-after-check)
  printf '%q ' "${ray_check[@]}"; printf '\n'
  if ((DRY_RUN == 0)); then
    "${ray_check[@]}" >"$RUN_ROOT/logs/ray-preflight.out" 2>"$RUN_ROOT/logs/ray-preflight.err"
  fi
fi

pids=()
for row in "${ARMS[@]}"; do
  IFS=$'\t' read -r label model shard nodes tp pp backend tasks probe probe_tokens <<<"$row"
  [[ "$tasks" != "-" ]] || tasks=""
  probe_tokens=${probe_tokens%$'\r'}
  arm=(pipeline/slurm/test_m3_quality_eval_arm.sh --profile "$PROFILE" --run-root "$RUN_ROOT" --matrix "$MATRIX" --model-label "$label" --model "$model" --shard "$shard" --tasks "$tasks" --tensor-parallel-size "$tp" --pipeline-parallel-size "$pp" --distributed-executor-backend "$backend" --run-probe "$probe" --probe-tokens "$probe_tokens")
  launch=(srun --exclusive --nodes="$nodes" --ntasks="$nodes" --gpus-per-node=8 --kill-on-bad-exit=1 "${time_args[@]}" "${arm[@]}")
  printf '%q ' "${launch[@]}"; printf '\n'
  if ((DRY_RUN == 0)); then
    "${launch[@]}" >"$RUN_ROOT/logs/${PROFILE}-${label}-${shard}.out" 2>"$RUN_ROOT/logs/${PROFILE}-${label}-${shard}.err" &
    pids+=("$!")
  fi
done
if ((DRY_RUN)); then exit 0; fi
status=0
for pid in "${pids[@]}"; do wait "$pid" || status=1; done
if [[ "$PROFILE" == smoke ]]; then
  python - "$RUN_ROOT" "$RUN_ROOT/preflight/smoke_sample_manifest.json" <<'PYSMOKE'
import hashlib,json,sys
from pathlib import Path
root=Path(sys.argv[1]); sample=Path(sys.argv[2]); models={}
for path in root.glob('models/*/shards/smoke/smoke_evidence.json'):
    models[path.parts[-4]]=json.load(open(path))
report={'schema_version':1,'profile':'smoke','sample_manifest_sha256':hashlib.sha256(sample.read_bytes()).hexdigest(),'models':models}
json.dump(report,open(root/'smoke_report.json','w'),indent=2)
PYSMOKE
  python -m pipeline.m3_quality_eval smoke-gate --matrix "$MATRIX" --report "$RUN_ROOT/smoke_report.json" --out "$RUN_ROOT/smoke_gate.json" || status=1
fi
exit "$status"
