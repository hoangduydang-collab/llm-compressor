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
local_ip=$(ip -4 route get 1.1.1.1 2>/dev/null | awk '
  { for (i = 1; i <= NF; i++) if ($i == "src") { print $(i + 1); exit } }')
if [[ -z "$local_ip" ]]; then
  local_ip=$(hostname -I | awk '{ for (i = 1; i <= NF; i++) if ($i !~ /^127\./) { print $i; exit } }')
fi
[[ -n "$local_ip" ]] || { echo "unable to determine routable IP for $local_node" >&2; exit 2; }
head_ip=""
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
wait_for_ray_nodes() {
  local address=$1 expected=$2 status_file="$OUT/status-rank-$rank-$attempt.txt"
  local active=0
  for _ in $(seq 1 120); do
    if ray status --address="$address" >"$status_file" 2>&1; then
      active=$(awk '
        /^Active:$/ { in_active=1; next }
        /^Pending:$/ { in_active=0 }
        in_active && /^[[:space:]]+1 node_/ { count++ }
        END { print count + 0 }
      ' "$status_file")
      ((active >= expected)) && break
    fi
    sleep 1
  done
  cat "$status_file" >>"$rank_log" 2>&1 || true
  [[ "$active" -ge "$expected" ]]
}
if ((rank == 0)); then
  ray start --head --node-ip-address="$local_ip" --port=6379 >>"$rank_log" 2>&1 &
  ray_cli_pid=$!
  wait_for_ray_nodes "$local_ip:6379" 1
  kill "$ray_cli_pid" 2>/dev/null || true
  wait "$ray_cli_pid" 2>/dev/null || true
  printf '%s\n' "$local_ip" >"$OUT/head-ip-$attempt"
  touch "$OUT/head-ready-$attempt"
else
  for _ in $(seq 1 120); do [[ -f "$OUT/head-ready-$attempt" && -f "$OUT/head-ip-$attempt" ]] && break; sleep 1; done
  [[ -f "$OUT/head-ready-$attempt" ]]
  head_ip=$(<"$OUT/head-ip-$attempt")
  ray start --address="$head_ip:6379" --node-ip-address="$local_ip" >>"$rank_log" 2>&1 &
  ray_cli_pid=$!
  wait_for_ray_nodes "$head_ip:6379" "$world"
  kill "$ray_cli_pid" 2>/dev/null || true
  wait "$ray_cli_pid" 2>/dev/null || true
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
  [[ "$count" -ge "$world" ]] || { echo "only $count/$world ranks joined Ray" >&2; exit 1; }
  ray status --address="$local_ip:6379" >"$OUT/ray_status.txt" 2>&1
  python - "$OUT/ray_status.txt" "$OUT/ray_nodes.json" "$OUT/gate-$attempt.json" "$world" <<'PYGATE'
import json, re, sys
status, nodes_out, gate_out, expected = sys.argv[1:]
lines = open(status).read().splitlines()
in_active = False
nodes = []
for line in lines:
    if line == "Active:":
        in_active = True
    elif line == "Pending:":
        in_active = False
    elif in_active and re.match(r"\s*1 node_", line):
        nodes.append(line.strip())
gpu_match = re.search(r"/([0-9.]+) GPU", open(status).read())
gpus = float(gpu_match.group(1)) if gpu_match else 0.0
expected = int(expected)
json.dump({"active_nodes": nodes, "visible_gpus": gpus}, open(nodes_out, "w"), indent=2)
gate = {"ready": len(nodes) == expected and gpus >= expected * 8,
        "expected_nodes": expected, "alive_nodes": len(nodes), "visible_gpus": gpus}
json.dump(gate, open(gate_out, "w"), indent=2)
raise SystemExit(0 if gate["ready"] else 1)
PYGATE
fi
for _ in $(seq 1 180); do [[ -f "$OUT/gate-$attempt.json" ]] && break; sleep 1; done
[[ -f "$OUT/gate-$attempt.json" ]]
cp "$OUT/gate-$attempt.json" "$OUT/gate.json"
python -c 'import json,sys; raise SystemExit(0 if json.load(open(sys.argv[1]))["ready"] else 1)' "$OUT/gate-$attempt.json"
if [[ "$MODE" == stop ]]; then ray stop --force >>"$rank_log" 2>&1 || true; fi
