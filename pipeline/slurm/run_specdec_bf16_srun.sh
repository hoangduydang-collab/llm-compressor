#!/usr/bin/env bash
# Controller (tmux) for the EAGLE3 BF16 absolute-reference test -- phase F.
#
# Does retraining / finetuning the draft model pay off? The drafter was trained
# against the ORIGINAL BF16 MiniMax-M3, so BF16 acceptance is the ceiling. Phase E
# compared two quantized targets (our 4-bit W4AFP8 vs the vendor's 8-bit MXFP8) and
# found only a ~1.35% acceptance gap -- but that bounds the penalty against a mild
# quant, not against the drafter's actual training distribution. This phase measures
# the ceiling:
#
#   BF16 accepted length - W4AFP8 accepted length = the headroom drafter finetuning
#   on our quantized target could plausibly recover.
#
# 2 arms x 2 nodes = 4 nodes (BF16 M3 is 796 GiB of safetensors; one 8x80 GiB node
# cannot hold it, so TP16 over ray is required):
#
#   bf16-k0  port 8060   nodes h113,h114   unquantized MiniMax-M3, TP16/ray (control)
#   bf16-k3  port 8061   nodes h107,h123   same + eagle3 k=3
#
# NODE SELECTION IS DELIBERATE, NOT ARBITRARY. slurm reports h97, h98 and h101 as
# `idle` while a foreign job holds ~350-390 GiB on each (probed repeatedly through
# 2026-07-27: 252 / 280 / 246 GiB free of 633). slurm is not accounting that memory.
# The same failure mode killed arms in wave 1, so the pre-launch probe below is a
# hard gate, not a formality -- on the first attempt it caught h97 going from
# 633 GiB free to 252 GiB in the minutes between an earlier probe and the launch,
# and refused rather than starting a doomed 2-node arm. The four nodes used here
# were each probed at 633 GiB free with every GPU >=70 GiB immediately before launch.
#
# Comparability: BF16 runs a different parallel topology (TP16 cross-node) than the
# TP8 single-node quantized arms, so its ABSOLUTE speed is not comparable to theirs.
# Each arm carries its own k=0 control and only within-format ratios are quoted.
# Accepted length -- the metric this phase exists for -- is model-intrinsic and
# unaffected by topology. See specdec_bf16_arm.sh for the full divergence list.
#
# A separate file from run_specdec_format_srun.sh on purpose: bash reads a script
# incrementally, so editing a launcher whose controller is still running can
# corrupt the running shell.
set -uo pipefail
REPO=/mnt/nfs/hoangduy/projects/llm-compressor
BF16_CKPT=/mnt/nfs/hoangduy/hf_assets/MiniMaxAI/MiniMax-M3
DRAFTER=/mnt/nfs/hoangduy/hf_assets/Inferact/MiniMax-M3-EAGLE3
SB_DIR="$REPO/artifacts/aiperf-datasets/speedbench"
PY=/mnt/nfs/hoangduy/venvs/quant/bin/python
# Verified-clean pairs (probed 2026-07-27; see header). Override only after probing.
NODES_K0=${NODES_K0:-gpu-h113,gpu-h114}
NODES_K3=${NODES_K3:-gpu-h107,gpu-h123}
# Proven BF16 value: 796 GiB of weights leaves ~22 GiB/GPU for KV at gpu_util 0.9.
MAX_MODEL_LEN=${MAX_MODEL_LEN:-65536}
# Windows whose 8k cells are the comparison arms (recorded for provenance).
W4A8_REF=${W4A8_REF:-/mnt/nfs/hoangduy/results/m3-specdec-eagle3/20260727T073533Z-phaseD}
MXFP8_REF=${MXFP8_REF:-/mnt/nfs/hoangduy/results/m3-specdec-eagle3/20260727T084526Z-format}

TS=$(date -u +%Y%m%dT%H%M%SZ)
ROOT=${ROOT_OVERRIDE:-/mnt/nfs/hoangduy/results/m3-specdec-eagle3/$TS-bf16ref}
mkdir -p "$ROOT" /mnt/nfs/hoangduy/logs/m3-specdec-eagle3
echo "$ROOT" > /mnt/nfs/hoangduy/logs/m3-specdec-eagle3/latest_bf16ref_root
echo "[controller] root=$ROOT"

export RUN_TS=$TS
export LLMC_M3_CAPTURE_SYNC=sync
echo "[controller] run_ts=$RUN_TS mml=$MAX_MODEL_LEN k0_nodes=$NODES_K0 k3_nodes=$NODES_K3"

fail() { echo "[controller] ABORT: $1" >&2; echo "$1" >"$ROOT/abort.txt"; exit 1; }

for d in "$BF16_CKPT" "$DRAFTER" "$SB_DIR" "$W4A8_REF" "$MXFP8_REF"; do
  test -d "$d" || fail "missing dir: $d"
done
printf 'w4afp8=%s\nmxfp8=%s\n' "$W4A8_REF" "$MXFP8_REF" > "$ROOT/reference-windows.txt"

# --- BF16 identity gate --------------------------------------------------------
# The whole phase is meaningless if this checkpoint turns out to be quantized.
bf16_state=$("$PY" - <<PY
import json
c = json.load(open("$BF16_CKPT/config.json"))
t = c.get("text_config") or c
q = c.get("quantization_config") or t.get("quantization_config")
dt = c.get("torch_dtype") or t.get("torch_dtype")
print("QUANTIZED" if q else f"bf16_ok:{dt}:{(c.get('architectures') or ['?'])[0]}")
PY
) || fail "cannot read BF16 config"
case "$bf16_state" in
  bf16_ok:bfloat16:*) ;;
  *) fail "BF16 reference checkpoint failed identity gate: $bf16_state" ;;
esac

# Weights must actually need 2 nodes; if this shrank, the topology rationale is stale.
bf16_gib=$(du -sBG --apparent-size "$BF16_CKPT" 2>/dev/null | awk '{gsub(/G/,"",$1); print $1}')
[ -n "$bf16_gib" ] && [ "$bf16_gib" -gt 640 ] \
  || fail "BF16 weights measured ${bf16_gib:-?} GiB; expected >640 (TP16 rationale)"
echo "[controller] BF16 gate: $bf16_state, ${bf16_gib} GiB (needs TP16)"

# --- SPEED-Bench staging gate (fail closed, same bytes as phase D/E) -----------
test -s "$SB_DIR/manifest.json" || fail "SPEED-Bench not staged: run pipeline/stage_speedbench.py"
for cell in 8k-low 8k-high; do
  f="$SB_DIR/$cell.jsonl"
  test -s "$f" || fail "missing staged cell: $f"
  n=$(wc -l < "$f")
  [ "$n" -ge 100 ] || fail "$cell has only $n entries (need >=100)"
  if grep -q "FULL BENCHMARK DATA SHOULD BE FETCHED" "$f"; then
    fail "$cell still contains masked placeholder rows"
  fi
done
"$PY" - <<PY >"$ROOT/speedbench-manifest.txt" || fail "SPEED-Bench hash gate failed"
import hashlib, json
m = json.load(open("$SB_DIR/manifest.json"))
for name in ["8k-low.jsonl", "8k-high.jsonl"]:
    v = m["files"][name]
    h = hashlib.sha256(open("$SB_DIR/" + name, "rb").read()).hexdigest()
    assert h == v["sha256"], "sha256 drift for " + name
    print(f"{name:16s} n={v['entries']:4d} tok_mean={v['tokens_mean']:9.1f} sha={h[:12]}")
PY
echo "[controller] SPEED-Bench 8k cells hash-verified (prompt-identical to phases D/E)"

drafter_arch=$("$PY" - <<PY
import json
c = json.load(open("$DRAFTER/config.json"))
print(f"{c['architectures'][0]}:{c['hidden_size']}:{c['num_attention_heads']}")
PY
) || fail "cannot read drafter config"
# TP16 splits the drafter's attention heads too: 64/16 = 4 per rank.
[ "$drafter_arch" = "LlamaForCausalLMEagle3:6144:64" ] \
  || fail "unexpected drafter config: $drafter_arch (TP16 head-divisibility unverified)"

aiperf_version=$(/mnt/nfs/hoangduy/venvs/perf/bin/python -c \
  "import importlib.metadata as md; print(md.version('aiperf'))" 2>/dev/null)
case "$aiperf_version" in 0.8.*) ;; *) fail "perf venv aiperf must be 0.8.x, got: ${aiperf_version:-none}";; esac
echo "[controller] aiperf=$aiperf_version drafter=$drafter_arch"

# --- node-health gate (fail closed) -------------------------------------------
# slurm's `idle` is not evidence of free GPUs (h98/h101 reported idle at 246 GiB
# free). Re-probe every node we are about to take, right before taking it.
ALL_NODES="$NODES_K0,$NODES_K3"
for n in ${ALL_NODES//,/ }; do
  free=$(timeout 120 srun --nodes=1 --ntasks=1 --nodelist="$n" --time=00:01:00 \
           --partition=compute --job-name="pF-probe" \
           nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits 2>/dev/null \
         | awk '{s+=$1; if($1<70000) bad++} END {print int(s/1024)":"(bad?"BAD":"ok")}')
  case "$free" in
    *:ok) echo "[controller] node $n probe: ${free%%:*} GiB free, all GPUs >=70 GiB" ;;
    *)    fail "node $n failed GPU-free probe: ${free:-no response} (need all GPUs >=70 GiB)" ;;
  esac
done

( cd "$REPO" && git rev-parse HEAD ) >"$ROOT/actual-commit.txt"

declare -A ARM_PID=()

# $1 arm  $2 spec_k  $3 port  $4 nodelist
launch_arm() {
  local arm=$1 k=$2 port=$3 nodes=$4
  local -a arm_env=(
    "ROOT=$ROOT" "ARM=$arm" "RUN_TS=$RUN_TS" "SPEC_K=$k"
    "CKPT=$BF16_CKPT" "PORT=$port" "DRAFTER=$DRAFTER" "SB_DIR=$SB_DIR"
    "MAX_MODEL_LEN=$MAX_MODEL_LEN" "PYTHONPATH=$REPO"
  )
  env "${arm_env[@]}" \
  srun --exclusive --nodes=2 --ntasks=2 --ntasks-per-node=1 --gpus-per-node=8 \
       --cpus-per-task=192 --time=08:00:00 --kill-on-bad-exit=1 \
       --partition=compute --nodelist="$nodes" \
       --job-name="m3-bf16ref-$arm" --export=ALL \
       bash "$REPO/pipeline/slurm/specdec_bf16_arm.sh" \
       > "$ROOT/$arm-srun.log" 2>&1 &
  ARM_PID[$arm]=$!
  printf 'arm=%s spec_k=%s port=%s nodes=%s pid=%s launched=%s\n' \
    "$arm" "$k" "$port" "$nodes" "${ARM_PID[$arm]}" \
    "$(date -u +%Y-%m-%dT%H:%M:%SZ)" >>"$ROOT/arm-provenance.txt"
  echo "[controller] launched $arm k=$k port=$port nodes=$nodes pid=${ARM_PID[$arm]}"
}

launch_arm bf16-k0 0 8060 "$NODES_K0"
launch_arm bf16-k3 3 8061 "$NODES_K3"

echo "[controller] 2 arms x 2 nodes launched; waiting (796 GiB load over NFS is slow)"
rc_all=0
for arm in "${!ARM_PID[@]}"; do
  if wait "${ARM_PID[$arm]}"; then
    echo "[controller] $arm rc=0"
  else
    rc=$?
    echo "[controller] $arm rc=$rc"
    echo "$arm rc=$rc" >>"$ROOT/arm-failures.txt"
    rc_all=1
  fi
done

echo "[controller] done rc_all=$rc_all root=$ROOT"
date -u +%Y-%m-%dT%H:%M:%SZ > "$ROOT/controller-done.txt"
exit "$rc_all"
