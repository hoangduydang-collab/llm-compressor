#!/usr/bin/env bash
set -euo pipefail
PROFILE=""; RUN_ROOT=""; MATRIX=""; LABEL=""; MODEL=""; SHARD=""; TASKS=""; TP=""; PP=1; BACKEND=""; RUN_PROBE=0; PROBE_TOKENS=0
while (($#)); do
  case "$1" in
    --profile) PROFILE=$2; shift 2;; --run-root) RUN_ROOT=$2; shift 2;; --matrix) MATRIX=$2; shift 2;;
    --model-label) LABEL=$2; shift 2;; --model) MODEL=$2; shift 2;; --shard) SHARD=$2; shift 2;;
    --tasks) TASKS=$2; shift 2;; --tensor-parallel-size) TP=$2; shift 2;;
    --pipeline-parallel-size) PP=$2; shift 2;;
    --distributed-executor-backend) BACKEND=$2; shift 2;; --run-probe) RUN_PROBE=$2; shift 2;;
    --probe-tokens) PROBE_TOKENS=$2; shift 2;; *) echo "unknown argument: $1" >&2; exit 2;;
  esac
done
: "${EVAL_CONFIG:=$RUN_ROOT/preflight/resolved_eval_config.yaml}"
: "${SAMPLES_MANIFEST:=$RUN_ROOT/preflight/${PROFILE}_sample_manifest.json}"
: "${PROBE_CORPUS:=$RUN_ROOT/preflight/${PROFILE}_probe_corpus.json}"
: "${MODEL_SOURCE:=/mnt/nfs/hoangduy/hf_assets/MiniMaxAI/MiniMax-M3}"
ARM="$RUN_ROOT/models/$LABEL/shards/$SHARD"; mkdir -p "$ARM"
rank=${SLURM_PROCID:-0}; nodes=${SLURM_NTASKS:-1}
if ((nodes > 1)); then
  placement_monitor_pid=""
  python -c 'import json,sys; raise SystemExit(0 if json.load(open(sys.argv[1]))["ready"] else 1)' \
    "$RUN_ROOT/ray_preflight/gate.json"
  source pipeline/slurm/test_m3_ray_topology.sh \
    --out "$ARM/ray_runtime" --keep-alive
  cleanup_ray() {
    if [[ -n "$placement_monitor_pid" ]]; then
      kill "$placement_monitor_pid" 2>/dev/null || true
      wait "$placement_monitor_pid" 2>/dev/null || true
    fi
    touch "$ARM/ray_runtime/driver-done"
    ray stop --force >/dev/null 2>&1 || true
  }
  if ((rank != 0)); then
    for _ in $(seq 1 86400); do
      [[ -f "$ARM/ray_runtime/driver-done" ]] && break
      sleep 1
    done
    cleanup_ray
    exit 0
  fi
  trap cleanup_ray EXIT
  (
    while [[ ! -f "$ARM/ray_runtime/driver-done" ]]; do
      echo "timestamp=$(date -Is)"
      timeout 10s ray list placement-groups --detail 2>&1 || true
      sleep 5
    done
  ) >"$ARM/ray_runtime/placement-monitor.log" 2>&1 &
  placement_monitor_pid=$!
fi
if ((rank != 0)); then exit 0; fi
python - "$ARM/arm_manifest.json" "$RUN_ROOT" "$LABEL" "$SHARD" "$SAMPLES_MANIFEST" "$EVAL_CONFIG" <<'PYMAN'
import hashlib,json,os,subprocess,sys
out,root,label,shard,samples,config=sys.argv[1:]
def sha(p): return hashlib.sha256(open(p,'rb').read()).hexdigest()
run=json.load(open(root+'/run_manifest.json'))
data={k:run[k] for k in ('run_id','git_commit','tokenizer_sha256','chat_template_sha256') if k in run}
data.update(
    schema_version=1,
    model_label=label,
    shard=shard,
    sample_manifest_sha256=sha(samples),
    eval_config_sha256=sha(config),
    slurm_job_id=os.environ.get("SLURM_JOB_ID"),
    slurm_step_id=os.environ.get("SLURM_STEP_ID"),
    slurm_node_name=os.environ.get("SLURMD_NODENAME"),
)
json.dump(data,open(out,'w'),indent=2)
PYMAN
set +e
eval_cmd=(python -m pipeline.evalsuite.cli run --config "$EVAL_CONFIG" --model "$MODEL" --out "$ARM" --tasks "$TASKS" --samples-manifest "$SAMPLES_MANIFEST"
  --set "model.id=$MODEL_SOURCE" --set "serve.tensor_parallel_size=$TP"
  --set "serve.vllm_kwargs.pipeline_parallel_size=$PP"
  --set "serve.vllm_kwargs.distributed_executor_backend=$BACKEND"
  --set serve.vllm_kwargs.enable_expert_parallel=true --set serve.vllm_kwargs.block_size=128 --set serve.vllm_kwargs.kv_cache_dtype=fp8)
if [[ "$PROFILE" == smoke ]]; then eval_cmd+=(--set eval.gen_kwargs.max_gen_toks=256); fi
run_probe() {
  python -m pipeline.m3_distributional_probe run --model "$MODEL" --model-source "$MODEL_SOURCE" --corpus "$PROBE_CORPUS" --out "$ARM/distributional_probe.jsonl" --tensor-parallel-size "$TP" --pipeline-parallel-size "$PP" --distributed-executor-backend "$BACKEND" --top-k 20
}
rc=0
probe_ran=0
if [[ "$PROFILE" == smoke && "$RUN_PROBE" == 1 ]]; then
  run_probe
  rc=$?
  probe_ran=1
fi
if ((rc == 0)); then
  if [[ -n "$TASKS" ]]; then
    "${eval_cmd[@]}"
    rc=$?
  else
    printf '{}\n' >"$ARM/aggregate.json"
  fi
fi
if ((rc == 0 && RUN_PROBE == 1 && probe_ran == 0)); then
  run_probe
  rc=$?
fi
set -e
printf '%s\n' "$rc" >"$ARM/return_code.txt"
if [[ "$PROFILE" == smoke ]]; then
python - "$ARM" "$LABEL" "$SAMPLES_MANIFEST" "$TP" "$PP" "$rc" <<'PYSMOKE'
import hashlib,json,sys
from pathlib import Path
arm,label,samples,tp,pp,rc=Path(sys.argv[1]),sys.argv[2],Path(sys.argv[3]),int(sys.argv[4]),int(sys.argv[5]),int(sys.argv[6])
def load(path, default):
    try: return json.load(open(path))
    except Exception: return default
aggregate=load(arm/'aggregate.json',{})
empty=loops=0
for path in (arm/'generation_health').glob('*.json') if (arm/'generation_health').is_dir() else []:
    health=load(path,{})
    empty += int(health.get('empty_count',0)); loops += int(health.get('periodic_loop_count',0))
probe=load(arm/'distributional_probe.summary.json',{})
evidence={'infrastructure_ok':rc==0,'artifacts_valid':rc==0 and bool(aggregate),
 'tasks_scored':len(aggregate),'sample_manifest_sha256':hashlib.sha256(samples.read_bytes()).hexdigest(),
 'empty_count':empty,'periodic_loop_count':loops,'distributed_world_size':tp * pp,
 'probe':{'tokens':probe.get('tokens',0),'elapsed_seconds':probe.get('elapsed_seconds',0)}}
json.dump(evidence,open(arm/'smoke_evidence.json','w'),indent=2)
PYSMOKE
fi
python - "$ARM/arm_complete.json" "$rc" <<'PYDONE'
import json,sys; json.dump({'complete':int(sys.argv[2])==0},open(sys.argv[1],'w'),indent=2)
PYDONE
exit "$rc"
