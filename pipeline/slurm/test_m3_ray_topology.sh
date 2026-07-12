#!/usr/bin/env bash
set -euo pipefail

OUT=""; MODE=stop
while (($#)); do
  case "$1" in
    --out) OUT=$2; shift 2 ;;
    --keep-alive) MODE=keep; shift ;;
    --stop-after-check) MODE=stop; shift ;;
    *) echo "unknown argument: $1" >&2; return 2 2>/dev/null || exit 2 ;;
  esac
done
[[ -n "$OUT" ]] || { echo "--out is required" >&2; return 2 2>/dev/null || exit 2; }
mkdir -p "$OUT"
rank=${SLURM_PROCID:-0}; world=${SLURM_NTASKS:-1}
attempt=${SLURM_JOB_ID:-local}-${SLURM_STEP_ID:-0}
mapfile -t hosts < <(scontrol show hostnames "${SLURM_JOB_NODELIST:?}")
head_node=${hosts[0]}; local_node=$(hostname -s)
addresses=$(getent ahostsv4 "$local_node"); read -r local_ip _ <<<"$addresses"
head_addresses=$(getent ahostsv4 "$head_node"); read -r head_ip _ <<<"$head_addresses"
export VLLM_HOST_IP=$local_ip
export RAY_ADDRESS=auto
rank_log="$OUT/rank-$rank.log"
{
  echo "rank=$rank world=$world host=$local_node local_ip=$local_ip head=$head_node head_ip=$head_ip"
  python --version
  ray --version
  env | sort | grep -E '^(SLURM_|CUDA_VISIBLE_DEVICES=|NCCL_|VLLM_HOST_IP=|RAY_ADDRESS=)' || true
} >"$rank_log" 2>&1
ray stop --force >>"$rank_log" 2>&1 || true
if ((rank == 0)); then
  ray start --head --node-ip-address="$local_ip" --port=6379 >>"$rank_log" 2>&1
  touch "$OUT/head-ready-$attempt"
else
  for _ in $(seq 1 120); do [[ -f "$OUT/head-ready-$attempt" ]] && break; sleep 1; done
  [[ -f "$OUT/head-ready-$attempt" ]]
  ray start --address="$head_ip:6379" --node-ip-address="$local_ip" >>"$rank_log" 2>&1
fi
python - "$OUT/rank-$rank.json" <<'PYRANK'
import json, os, platform, socket, subprocess, sys
json.dump({"rank":int(os.environ.get("SLURM_PROCID",0)),"world_size":int(os.environ.get("SLURM_NTASKS",1)),"hostname":socket.gethostname(),"vllm_host_ip":os.environ.get("VLLM_HOST_IP"),"cuda_visible_devices":os.environ.get("CUDA_VISIBLE_DEVICES"),"python":platform.python_version(),"ray":subprocess.check_output(["ray","--version"],text=True).strip()},open(sys.argv[1],"w"),indent=2)
PYRANK
touch "$OUT/rank-$rank-ready-$attempt"
if ((rank == 0)); then
  for _ in $(seq 1 180); do
    count=$(find "$OUT" -maxdepth 1 -name "rank-*-ready-$attempt" -type f | wc -l)
    ((count >= world)) && break
    sleep 1
  done
  ray status >"$OUT/ray_status.txt" 2>&1
  python - "$OUT/ray_nodes.json" "$OUT/gate-$attempt.json" "$world" <<'PYGATE'
import json, sys, ray
ray.init(address="auto",ignore_reinit_error=True)
nodes=ray.nodes(); alive=[n for n in nodes if n.get("Alive")]
gpus=sum(float(n.get("Resources",{}).get("GPU",0)) for n in alive)
json.dump(nodes,open(sys.argv[1],"w"),indent=2,default=str)
expected=int(sys.argv[3]); gate={"ready":len(alive)==expected and gpus>=expected*8,"expected_nodes":expected,"alive_nodes":len(alive),"visible_gpus":gpus}
json.dump(gate,open(sys.argv[2],"w"),indent=2); ray.shutdown()
raise SystemExit(0 if gate["ready"] else 1)
PYGATE
fi
for _ in $(seq 1 180); do [[ -f "$OUT/gate-$attempt.json" ]] && break; sleep 1; done
[[ -f "$OUT/gate-$attempt.json" ]]
cp "$OUT/gate-$attempt.json" "$OUT/gate.json"
python -c 'import json,sys; raise SystemExit(0 if json.load(open(sys.argv[1]))["ready"] else 1)' "$OUT/gate-$attempt.json"
if [[ "$MODE" == stop ]]; then ray stop --force >>"$rank_log" 2>&1 || true; fi
