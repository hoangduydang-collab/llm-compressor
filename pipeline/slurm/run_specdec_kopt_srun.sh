#!/usr/bin/env bash
# Controller (tmux) for EAGLE3 spec-dec phase H -- optimal draft depth per entropy tier.
#
# Phase G established that draft depth is workload-dependent, and that k in {3,4,5}
# brackets neither optimum:
#
#   8k-low  conc 1 : ITL 3.280 -> 3.120 -> 2.965 ms for k=3,4,5, marginal gain flat
#                    (-4.9%, -5.0%) => the knee is ABOVE k=5. Still true at conc 10
#                    (7.079 -> 6.897 -> 6.614), just compressed.
#   8k-high conc 1 : k=3 wins at 1.31x; k=4 and k=5 lose (+1.9%, +6.8% ITL) because
#                    position 3 accepts 5.4% and position 4 only 2.0%, against a
#                    marginal step cost of ~7.6% per position => the knee is at or
#                    BELOW k=3.
#
# So: 2 arms x 1 node, one entropy tier each, sweeping outward.
#
#   kopt-low   gpu-h113  ports 8080+  cell 8k-low   k = 0, 5, 6, 7  (+ k=5 repeat)
#   kopt-high  gpu-h114  ports 8090+  cell 8k-high  k = 0, 1, 2, 3  (+ k=1 repeat)
#
# ALL k VALUES FOR A TIER LIVE ON ONE NODE. Phase G spread k=3/4/5 across
# gpu-h107/h123/h108, so its k-trend has node variance folded into it -- fine for the
# INT4-vs-bf16 A/B it was built for (that comparison was same-node), not fine for
# choosing k. Here the k axis never crosses a node; the cost is a serial sweep, whose
# drift is measured by re-serving the first spec k at the end (see the arm script).
#
# Each arm measures its own k=0 control first, on a fresh engine, so every speedup in
# this window is within-window. k=6 and k=7 are unconstrained: vLLM's num_speculative
# _tokens/n_predict divisibility check only fires when the draft config carries
# `n_predict` (an MTP field), and this EAGLE3 config does not.
#
# Drafter is the derived INT4 artifact -- phase G measured no acceptance cost from it
# (3.134 vs bf16's 3.106 at k=3, higher at every position) at lower ITL.
#
# DATASET IDENTITY: same staged SPEED-Bench bytes as phases D-G (sha256-gated below),
# --random-seed 42, same request counts (40 at conc 1, 100 at conc 10). Only k varies.
#
# A separate file from run_specdec_int4drafter_srun.sh on purpose: bash reads a script
# incrementally, so editing a launcher whose controller is still running can corrupt
# the running shell.
set -uo pipefail
REPO=/mnt/nfs/hoangduy/projects/llm-compressor
GPTQ_CKPT="$REPO/artifacts/m3-awq-gptq-prepared/gptq-checkpoint-vllm-w123-abi-overlay"
DRAFTER=${DRAFTER:-/mnt/nfs/hoangduy/hf_assets/derived/MiniMax-M3-EAGLE3-INT4-bf16embed}
SB_DIR="$REPO/artifacts/aiperf-datasets/speedbench"
SITE_0110=/mnt/nfs/hoangduy/venvs/humming-0.1.10-site
PY=/mnt/nfs/hoangduy/venvs/quant/bin/python
# Probed 8 GPUs x >=70 GiB free immediately before launch; re-probed by the gate below.
NODE_LOW=${NODE_LOW:-gpu-h113}
NODE_HIGH=${NODE_HIGH:-gpu-h114}
KS_LOW=${KS_LOW:-"0 5 6 7"}
KS_HIGH=${KS_HIGH:-"0 1 2 3"}
PHASEG_REF=${PHASEG_REF:-/mnt/nfs/hoangduy/results/m3-specdec-eagle3/20260727T102751Z-int4drafter}

TS=$(date -u +%Y%m%dT%H%M%SZ)
ROOT=${ROOT_OVERRIDE:-/mnt/nfs/hoangduy/results/m3-specdec-eagle3/$TS-kopt}
mkdir -p "$ROOT" /mnt/nfs/hoangduy/logs/m3-specdec-eagle3
echo "$ROOT" > /mnt/nfs/hoangduy/logs/m3-specdec-eagle3/latest_kopt_root
echo "[controller] root=$ROOT"

export RUN_TS=$TS
export LLMC_M3_CAPTURE_SYNC=sync
echo "[controller] run_ts=$RUN_TS low=$NODE_LOW ($KS_LOW) high=$NODE_HIGH ($KS_HIGH)"

fail() { echo "[controller] ABORT: $1" >&2; echo "$1" >"$ROOT/abort.txt"; exit 1; }

for d in "$GPTQ_CKPT" "$SITE_0110" "$DRAFTER" "$SB_DIR"; do
  test -d "$d" || fail "missing dir: $d"
done
printf 'phaseG=%s\n' "$PHASEG_REF" > "$ROOT/reference-windows.txt"

# --- drafter identity gate: must be the derived INT4 artifact -------------------
"$PY" - <<PY >"$ROOT/drafter-identity.txt" || fail "drafter identity gate failed"
import hashlib, json
from safetensors import safe_open

def sha(p):
    h = hashlib.sha256()
    with open(p, "rb") as fh:
        while b := fh.read(1 << 22):
            h.update(b)
    return h.hexdigest()

d = "$DRAFTER"
c = json.load(open(f"{d}/config.json"))
q = c["quantization_config"]
assert q["quant_method"] == "compressed-tensors", q["quant_method"]
assert q["format"] == "pack-quantized", q["format"]
g = q["config_groups"]
assert "group_embed" not in g, "group_embed present -- this is not the derived artifact"
for name in ("group_0", "group_lmhead"):
    assert g[name]["weights"]["num_bits"] == 4 and g[name]["weights"]["type"] == "int", name
    assert g[name].get("input_activations") is None, f"{name} declares activation quant (expected W4A16)"
with safe_open(f"{d}/model.safetensors", framework="pt") as f:
    keys = set(f.keys())
assert "embed_tokens.weight" in keys, "missing bf16 embed_tokens"
assert not any(k.startswith("embed_tokens.weight_p") for k in keys), "embed is packed"
assert "lm_head.weight_packed" in keys, "lm_head is not INT4"
man = json.load(open(f"{d}/derivation-manifest.json"))
assert man["out_safetensors_sha256"] == sha(f"{d}/model.safetensors"), "derived artifact modified"
print("drafter:", d)
print("scheme : W4A16 (int4 weights, group 128, no activation quant)")
print("sha256 :", man["out_safetensors_sha256"][:16])
PY
echo "[controller] drafter verified: derived INT4 W4A16, unmodified since derivation"

# --- SPEED-Bench staging gate: byte-identical prompts to phases D-G -------------
test -s "$SB_DIR/manifest.json" || fail "SPEED-Bench not staged"
for cell in 8k-low 8k-high; do
  f="$SB_DIR/$cell.jsonl"
  test -s "$f" || fail "missing staged cell: $f"
  [ "$(wc -l < "$f")" -ge 100 ] || fail "$cell has too few entries"
  grep -q "FULL BENCHMARK DATA SHOULD BE FETCHED" "$f" && fail "$cell has masked rows"
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
echo "[controller] SPEED-Bench 8k cells hash-verified (prompt-identical to phases D-G)"

aiperf_version=$(/mnt/nfs/hoangduy/venvs/perf/bin/python -c \
  "import importlib.metadata as md; print(md.version('aiperf'))" 2>/dev/null)
case "$aiperf_version" in 0.8.*) ;; *) fail "perf venv aiperf must be 0.8.x, got: ${aiperf_version:-none}";; esac
for patch in ct_input_format grouped_expert_bounds tma_store_fence tma_store_commit; do
  ( cd "$REPO" && PYTHONPATH="$REPO" "$PY" \
      "pipeline/slurm/patch_humming_$patch.py" --site "$SITE_0110" --check ) \
      >>"$ROOT/patch-checks.log" 2>&1 \
    || fail "humming $patch patch missing on $SITE_0110"
done
echo "[controller] aiperf=$aiperf_version humming 0.1.10 patches present"

# --- node-health gate: 8 GPUs visible AND each >=70 GiB free -------------------
for n in "$NODE_LOW" "$NODE_HIGH"; do
  r=$(timeout 120 srun --nodes=1 --ntasks=1 --nodelist="$n" --time=00:01:00 \
        --partition=compute --job-name="pH-probe" \
        nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits 2>/dev/null \
      | awk '{c++; s+=$1; if ($1 < 70000) bad++}
             END {printf "%d:%d:%s", c, int(s/1024), (bad || c != 8 ? "BAD" : "ok")}')
  case "$r" in
    *:ok) echo "[controller] node $n: ${r%%:*} GPUs, $(echo "$r" | cut -d: -f2) GiB free" ;;
    *)    fail "node $n failed GPU probe: ${r:-no response}" ;;
  esac
done

( cd "$REPO" && git rev-parse HEAD ) >"$ROOT/actual-commit.txt"

declare -A ARM_PID=()

# $1 arm  $2 cell  $3 ks  $4 repeat_k  $5 port_base  $6 node
launch_arm() {
  local arm=$1 cell=$2 ks=$3 rep=$4 pb=$5 node=$6
  local -a arm_env=(
    "ROOT=$ROOT" "ARM=$arm" "RUN_TS=$RUN_TS" "CELL=$cell" "KS=$ks" "REPEAT_K=$rep"
    "CKPT=$GPTQ_CKPT" "PORT_BASE=$pb" "SB_DIR=$SB_DIR" "DRAFTER=$DRAFTER"
    "HUMMING_M3_W4A8_CACHE_ROOT=/mnt/nfs/hoangduy/.humming"
    "M3_W4A8_BACKEND=humming" "VLLM_HUMMING_MOE_GEMM_TYPE=indexed"
    "VLLM_HUMMING_USE_F16_ACCUM=0" "PYTHONPATH=$SITE_0110:$REPO"
  )
  env "${arm_env[@]}" \
  srun --exclusive --nodes=1 --ntasks=1 --gres=gpu:8 --cpus-per-task=192 \
       --time=06:00:00 --kill-on-bad-exit=1 --partition=compute \
       --nodelist="$node" --job-name="m3-pH-$arm" --export=ALL \
       bash "$REPO/pipeline/slurm/specdec_kopt_arm.sh" \
       > "$ROOT/$arm-srun.log" 2>&1 &
  ARM_PID[$arm]=$!
  printf 'arm=%s cell=%s ks="%s" repeat_k=%s port_base=%s node=%s pid=%s launched=%s\n' \
    "$arm" "$cell" "$ks" "$rep" "$pb" "$node" "${ARM_PID[$arm]}" \
    "$(date -u +%Y-%m-%dT%H:%M:%SZ)" >>"$ROOT/arm-provenance.txt"
  echo "[controller] launched $arm cell=$cell ks='$ks' node=$node pid=${ARM_PID[$arm]}"
}

launch_arm kopt-low  8k-low  "$KS_LOW"  5 8080 "$NODE_LOW"
launch_arm kopt-high 8k-high "$KS_HIGH" 1 8090 "$NODE_HIGH"

echo "[controller] 2 arms launched (5 serves each); waiting"
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
