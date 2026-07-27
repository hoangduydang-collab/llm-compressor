#!/usr/bin/env bash
# Controller (tmux) for EAGLE3 spec-dec phase G -- quantized DRAFTER vs bf16 drafter.
#
# Phases D-F all varied the TARGET's weight format and found the drafter's acceptance
# is indifferent to it (4-bit / 8-bit / 16-bit within ~1%). This phase varies the
# DRAFTER instead, and asks a different question: not "does acceptance survive?" but
# "can we make drafting cheaper?" Phase D pins the cost we are attacking -- at 8k-low
# conc 1 on the W4AFP8 target, a k=0 step is 7.313 ms and a k=3 step is 10.49 ms, so
# drafting costs 3.18 ms/step (~1.06 ms per draft token) and caps the speedup at 2.17x.
#
# 3 arms x 1 node, one per draft depth. Each arm serves the INT4 drafter, runs the
# grid, tears down, serves the bf16 drafter, and runs the identical grid -- so every
# INT4-vs-bf16 comparison is same node, same allocation, same target checkpoint, same
# prompts. See specdec_int4drafter_arm.sh for why the A/B must not cross nodes.
#
#   int4-k3  ports 8070/8170   k=3   (directly comparable to phase D's k=3 arm)
#   int4-k4  ports 8071/8171   k=4
#   int4-k5  ports 8072/8172   k=5
#   probe    port  8079        serve-only test of the AS-PUBLISHED checkpoint
#
# The measured drafter is DERIVED, not the published artifact:
# pipeline/prepare_int4_drafter.py restores the bf16 embed_tokens and drops the
# group_embed quantization group, carrying all 34 other published tensors
# byte-for-byte. Reason: vLLM's _maybe_share_embeddings reads
# `self.model.model.embed_tokens.weight` with no hasattr guard, and a
# compressed-tensors quantized embedding has no `weight` -- only weight_packed. The
# access raises AttributeError during drafter load. It is also pointless to quantize:
# when the draft embedding matches the target's, vLLM deletes it and shares the
# target's table (our own phase D serve log shows exactly that), so an INT4 embedding
# could only have added error, never speed. The `probe` arm records what the
# published artifact actually does, since that is a fact about vLLM support worth
# having on the record.
#
# DATASET IDENTITY: the arms read the same staged SPEED-Bench files as phases D/E,
# gated below on sha256 equality with the manifest, and use --random-seed 42 with the
# same request counts and sampling. Neither half of an arm, nor any arm, sees a
# different prompt stream.
#
# A separate file from run_specdec_bf16_srun.sh on purpose: bash reads a script
# incrementally, so editing a launcher whose controller is still running can corrupt
# the running shell.
set -uo pipefail
REPO=/mnt/nfs/hoangduy/projects/llm-compressor
GPTQ_CKPT="$REPO/artifacts/m3-awq-gptq-prepared/gptq-checkpoint-vllm-w123-abi-overlay"
DRAFTER_FP=/mnt/nfs/hoangduy/hf_assets/Inferact/MiniMax-M3-EAGLE3
DRAFTER_INT4=${DRAFTER_INT4:-/mnt/nfs/hoangduy/hf_assets/derived/MiniMax-M3-EAGLE3-INT4-bf16embed}
DRAFTER_PUB=${DRAFTER_PUB:-/mnt/nfs/hoangduy/hf_assets/Sebesky/MiniMax-M3-EAGLE3-RTN-INT4}
SB_DIR="$REPO/artifacts/aiperf-datasets/speedbench"
SITE_0110=/mnt/nfs/hoangduy/venvs/humming-0.1.10-site
PY=/mnt/nfs/hoangduy/venvs/quant/bin/python
# Verified-clean at launch (see probe gate). h125/h126 report slurm-idle but return
# no GPUs to an srun step; h97/h98/h101 were held by an unaccounted foreign job
# earlier today, so the throwaway probe arm gets h97 and the measurement arms do not.
NODE_K3=${NODE_K3:-gpu-h107}
NODE_K4=${NODE_K4:-gpu-h123}
NODE_K5=${NODE_K5:-gpu-h108}
NODE_PROBE=${NODE_PROBE:-gpu-h97}
# Phase D window whose k=0 / k=3 cells this phase is anchored to (provenance only;
# the in-arm bf16 half is the comparison).
PHASED_REF=${PHASED_REF:-/mnt/nfs/hoangduy/results/m3-specdec-eagle3/20260727T073533Z-phaseD}

TS=$(date -u +%Y%m%dT%H%M%SZ)
ROOT=${ROOT_OVERRIDE:-/mnt/nfs/hoangduy/results/m3-specdec-eagle3/$TS-int4drafter}
mkdir -p "$ROOT" /mnt/nfs/hoangduy/logs/m3-specdec-eagle3
echo "$ROOT" > /mnt/nfs/hoangduy/logs/m3-specdec-eagle3/latest_int4drafter_root
echo "[controller] root=$ROOT"

export RUN_TS=$TS
export LLMC_M3_CAPTURE_SYNC=sync
echo "[controller] run_ts=$RUN_TS nodes=$NODE_K3,$NODE_K4,$NODE_K5 probe=$NODE_PROBE"

fail() { echo "[controller] ABORT: $1" >&2; echo "$1" >"$ROOT/abort.txt"; exit 1; }

for d in "$GPTQ_CKPT" "$SITE_0110" "$DRAFTER_FP" "$DRAFTER_INT4" "$DRAFTER_PUB" "$SB_DIR" "$PHASED_REF"; do
  test -d "$d" || fail "missing dir: $d"
done
printf 'phaseD=%s\n' "$PHASED_REF" > "$ROOT/reference-windows.txt"

# --- drafter identity gates (fail closed) --------------------------------------
# The derived INT4 drafter must be INT4 where it computes and bf16 where it does not.
"$PY" - <<PY >"$ROOT/drafter-identity.txt" || fail "drafter identity gate failed"
import hashlib, json, sys
from safetensors import safe_open

def sha(p):
    h = hashlib.sha256()
    with open(p, "rb") as fh:
        while b := fh.read(1 << 22):
            h.update(b)
    return h.hexdigest()

fp, i4, pub = "$DRAFTER_FP", "$DRAFTER_INT4", "$DRAFTER_PUB"

# fp drafter: no quantization at all.
c_fp = json.load(open(f"{fp}/config.json"))
assert c_fp.get("quantization_config") is None, "fp drafter is quantized"
assert c_fp["architectures"][0] == "LlamaForCausalLMEagle3", c_fp["architectures"]

# derived INT4 drafter.
c4 = json.load(open(f"{i4}/config.json"))
q = c4["quantization_config"]
assert q["quant_method"] == "compressed-tensors", q["quant_method"]
assert q["format"] == "pack-quantized", q["format"]
groups = q["config_groups"]
assert "group_embed" not in groups, "group_embed survived the derivation"
for g in ("group_0", "group_lmhead"):
    assert groups[g]["weights"]["num_bits"] == 4, g
    assert groups[g]["weights"]["type"] == "int", g
assert c4["architectures"] == c_fp["architectures"], "arch drift"
for k in ("hidden_size", "num_attention_heads", "vocab_size", "intermediate_size"):
    assert c4[k] == c_fp[k], f"{k} drift: {c4[k]} vs {c_fp[k]}"

with safe_open(f"{i4}/model.safetensors", framework="pt") as f:
    keys = set(f.keys())
    # embedding restored to bf16, lm_head + linears still packed INT4.
    assert "embed_tokens.weight" in keys, "derived drafter lacks bf16 embed_tokens"
    assert not any(k.startswith("embed_tokens.weight_p") for k in keys), "embed still packed"
    assert "lm_head.weight_packed" in keys, "lm_head is not INT4"
    packed = sorted(k for k in keys if k.endswith(".weight_packed"))
    assert len(packed) == 9, f"expected 9 packed tensors, got {len(packed)}: {packed}"

# The derivation manifest must match the artifacts actually on disk.
man = json.load(open(f"{i4}/derivation-manifest.json"))
assert man["derived_from"]["int4"]["safetensors_sha256"] == sha(f"{pub}/model.safetensors"), \
    "published checkpoint changed since derivation"
assert man["derived_from"]["fp"]["safetensors_sha256"] == sha(f"{fp}/model.safetensors"), \
    "fp drafter changed since derivation"
assert man["out_safetensors_sha256"] == sha(f"{i4}/model.safetensors"), "derived artifact modified"

print("fp        :", f"{c_fp['architectures'][0]} bf16 sha={man['derived_from']['fp']['safetensors_sha256'][:12]}")
print("published :", f"int4+int4embed sha={man['derived_from']['int4']['safetensors_sha256'][:12]}")
print("derived   :", f"int4 compute + bf16 embed sha={man['out_safetensors_sha256'][:12]}")
print("packed    :", ", ".join(packed))
PY
echo "[controller] drafter identity verified (derived INT4 = published bytes + bf16 embed)"

# --- SPEED-Bench staging gate: byte-identical prompts to phases D/E -------------
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

# --- node-health gate (fail closed) -------------------------------------------
# slurm's `idle` is not evidence of usable GPUs, in BOTH directions: h97/h98/h101 have
# reported idle while a foreign job held ~350-390 GiB, and h125/h126 report idle but
# hand an srun step zero GPUs. So require exactly 8 GPUs visible AND every one of them
# >=70 GiB free -- an empty nvidia-smi must fail, not pass.
probe_node() {
  local n=$1
  timeout 120 srun --nodes=1 --ntasks=1 --nodelist="$n" --time=00:01:00 \
    --partition=compute --job-name="pG-probe" \
    nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits 2>/dev/null \
  | awk '{n++; s+=$1; if ($1 < 70000) bad++}
         END {printf "%d:%d:%s", n, int(s/1024), (bad || n != 8 ? "BAD" : "ok")}'
}
for n in "$NODE_K3" "$NODE_K4" "$NODE_K5" "$NODE_PROBE"; do
  r=$(probe_node "$n")
  case "$r" in
    *:ok) echo "[controller] node $n: ${r%%:*} GPUs, $(echo "$r" | cut -d: -f2) GiB free" ;;
    *)    fail "node $n failed GPU probe: ${r:-no response} (need 8 GPUs, each >=70 GiB free)" ;;
  esac
done

( cd "$REPO" && git rev-parse HEAD ) >"$ROOT/actual-commit.txt"

declare -A ARM_PID=()

# $1 arm  $2 spec_k  $3 port  $4 node  $5.. extra env
launch_arm() {
  local arm=$1 k=$2 port=$3 node=$4; shift 4
  local -a arm_env=(
    "ROOT=$ROOT" "ARM=$arm" "RUN_TS=$RUN_TS" "SPEC_K=$k"
    "CKPT=$GPTQ_CKPT" "PORT=$port" "SB_DIR=$SB_DIR"
    "DRAFTER_INT4=$DRAFTER_INT4" "DRAFTER_FP=$DRAFTER_FP"
    "HUMMING_M3_W4A8_CACHE_ROOT=/mnt/nfs/hoangduy/.humming"
    "M3_W4A8_BACKEND=humming" "VLLM_HUMMING_MOE_GEMM_TYPE=indexed"
    "VLLM_HUMMING_USE_F16_ACCUM=0" "PYTHONPATH=$SITE_0110:$REPO"
    "$@"
  )
  env "${arm_env[@]}" \
  srun --exclusive --nodes=1 --ntasks=1 --gres=gpu:8 --cpus-per-task=192 \
       --time=06:00:00 --kill-on-bad-exit=1 --partition=compute \
       --nodelist="$node" --job-name="m3-pG-$arm" --export=ALL \
       bash "$REPO/pipeline/slurm/specdec_int4drafter_arm.sh" \
       > "$ROOT/$arm-srun.log" 2>&1 &
  ARM_PID[$arm]=$!
  printf 'arm=%s spec_k=%s port=%s node=%s pid=%s launched=%s\n' \
    "$arm" "$k" "$port" "$node" "${ARM_PID[$arm]}" \
    "$(date -u +%Y-%m-%dT%H:%M:%SZ)" >>"$ROOT/arm-provenance.txt"
  echo "[controller] launched $arm k=$k port=$port node=$node pid=${ARM_PID[$arm]}"
}

# Serve-only test of the published artifact. Cheap, throwaway node, cannot fail the run.
launch_arm probe-published 3 8079 "$NODE_PROBE" \
  "PROBE_ONLY=1" "PROBE_DRAFTER=$DRAFTER_PUB"

launch_arm int4-k3 3 8070 "$NODE_K3"
launch_arm int4-k4 4 8071 "$NODE_K4"
launch_arm int4-k5 5 8072 "$NODE_K5"

echo "[controller] 3 A/B arms + 1 probe launched; waiting"
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
