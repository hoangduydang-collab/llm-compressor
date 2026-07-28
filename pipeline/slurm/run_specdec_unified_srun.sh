#!/usr/bin/env bash
# Controller (tmux) for the UNIFIED one-node EAGLE3 spec-dec window.
#
# OBJECTIVE
# ---------
# Re-measure the whole optimization story on ONE node in ONE allocation, so that every
# number in the collaborator report is a same-node comparison rather than a join
# across five nodes and six windows:
#
#   rung 1  W4AFP8 on CUTLASS      (barebone, no spec-dec)
#   rung 2  W4AFP8 on Humming      (our kernel work)          <- axis 0
#   rung 3  + spec-dec at tuned k                              <- axis 1
#   rung 4  + drafter kernel choice / drafter precision         <- axes 2, 3
#
# Two design decisions that matter, both agreed with the user up front:
#
# 1. THE BAREBONE LEG IS RE-MEASURED ON SPEED-BENCH, not quoted. The published
#    "95 -> 137 tok/s" CUTLASS->Humming figure is from the PINNED reasoning shape
#    (1k in / 8k forced out via ignore_eos), while every spec-dec number is
#    SPEED-Bench with natural stopping. Wave 2 measured that ignore_eos inflates
#    acceptance by +33%, so chaining the two would multiply a pinned-shape kernel gain
#    by a natural-shape spec-dec gain -- a real defect a reviewer would find. Both legs
#    now share one workload, one node, one window.
#
# 2. THE STREAM ON/OFF SUB-RUNG IS DROPPED. The 95-vs-103 pair is the shared-stream
#    cudagraph fix, whose "off" condition lives in a 2026-07-24 IMA-debug env matrix
#    built around a different checkpoint (cyankiwi AWQ). Reviving it is
#    disproportionate to one bar, so the two CUTLASS serves here are REPLICATES of one
#    CUTLASS config, not a stream A/B. The stream fix stays owned by the perf report.
#
# STATISTICS ARE IN THE ORDERING, so the serve list below is deliberate, not arbitrary:
#   * A/B pairs INTERLEAVE (Humming/CUTLASS at k=0; INT4/bf16 at k=5 and k=2). A
#     6-hour serial window can drift monotonically; blocking the arms would alias that
#     drift straight into the effect, interleaving cannot.
#   * Replicate counts come from the measured cross-engine floors, not taste:
#     conc-1 sd 1.02%, conc-10 sd 0.16% (four replicates, phases H/I/I.2), and
#     se_diff = sd*sqrt(2/n). n=3 -> 0.83% at conc 1, which resolves the ~2.2-2.5%
#     axis-1 and axis-3 effects; conc-10 effects are already 6-12 sd at n=1.
#   * n=3 goes on the k=0 Humming control (denominator of every ratio in the report),
#     on k=5/k=6 (the low-tier contenders, 2.2% apart), on k=2 (high-tier optimum),
#     and on the bf16 drafter at k=5 (axis 3's only magnitude claim).
#
# WHAT IS NEW HERE versus phases D-I.2:
#   * the CUTLASS rung on SPEED-Bench at all;
#   * axis 2 measured at the HIGH-entropy deployment k (k=2) for the first time;
#   * axis 3 measured at the high-entropy deployment k (k=2) for the first time --
#     phase G ran k=3/4/5 only, so this closes the hole it left;
#   * axis 3 replicated well enough to state a magnitude instead of a sign.
#
# COST: 26 serves, ~6.3 h measured from phase H's real per-serve timings
# (spec serve boot 4m15s, k=0 boot ~5-8 min, cell time = requests * 2048 / tok/s).
# Request counts are 30/80 rather than 40/100 -- within-cell request scatter is not the
# limiting noise (cross-engine is), and the cut is what fits 26 serves in budget.
#
# A separate file from run_specdec_kernel_srun.sh on purpose: bash reads a script
# incrementally, so editing a launcher whose controller is still running can corrupt
# the running shell.
set -uo pipefail
REPO=/mnt/nfs/hoangduy/projects/llm-compressor
GPTQ_CKPT="$REPO/artifacts/m3-awq-gptq-prepared/gptq-checkpoint-vllm-w123-abi-overlay"
DRAFTER_INT4=${DRAFTER_INT4:-/mnt/nfs/hoangduy/hf_assets/derived/MiniMax-M3-EAGLE3-INT4-bf16embed}
DRAFTER_BF16=${DRAFTER_BF16:-/mnt/nfs/hoangduy/hf_assets/Inferact/MiniMax-M3-EAGLE3}
SB_DIR="$REPO/artifacts/aiperf-datasets/speedbench"
SITE_0110=/mnt/nfs/hoangduy/venvs/humming-0.1.10-site
PY=/mnt/nfs/hoangduy/venvs/quant/bin/python
NODE=${NODE:?set NODE to a node probed free immediately before launch}
PORT_BASE=${PORT_BASE:-8080}
ONLY=${ONLY:-}                  # label subset for resume after a died allocation
N_C1=${N_C1:-30}
N_C10=${N_C10:-80}
# Known-good staged prompt digests (phases D-I.2 all gated on these).
SHA_LOW=${SHA_LOW:-bfcf60739f43}
SHA_HIGH=${SHA_HIGH:-325e2a9dc34f}
# Cross-window anchors: the k=0 Humming control and k=5 cell should land near these.
REF_KOPT=/mnt/nfs/hoangduy/results/m3-specdec-eagle3/20260727T105919Z-kopt
REF_KERNEL=/mnt/nfs/hoangduy/results/m3-specdec-eagle3/20260727T154725Z-kernel

TS=$(date -u +%Y%m%dT%H%M%SZ)
ROOT=${ROOT_OVERRIDE:-/mnt/nfs/hoangduy/results/m3-specdec-eagle3/$TS-unified}
mkdir -p "$ROOT" /mnt/nfs/hoangduy/logs/m3-specdec-eagle3
echo "$ROOT" > /mnt/nfs/hoangduy/logs/m3-specdec-eagle3/latest_unified_root
echo "[controller] root=$ROOT"

export RUN_TS=$TS
export LLMC_M3_CAPTURE_SYNC=sync
echo "[controller] run_ts=$RUN_TS node=$NODE only='${ONLY:-<all>}'"

fail() { echo "[controller] ABORT: $1" >&2; echo "$1" >"$ROOT/abort.txt"; exit 1; }

# ============================ THE SERVE LIST ==================================
# label;k;backend;drafter;kernel;cells
SERVES=$(cat <<'LIST'
# --- axis 0: barebone kernel leg, interleaved, both tiers (k=0 = the denominator) ---
L0-hum-k0-r1;0;humming;none;default;8k-low 8k-high
L0-cut-k0-r1;0;cutlass;none;default;8k-low 8k-high
L0-hum-k0-r2;0;humming;none;default;8k-low 8k-high
L0-cut-k0-r2;0;cutlass;none;default;8k-low 8k-high
L0-hum-k0-r3;0;humming;none;default;8k-low 8k-high
# --- axis 1 (low tier) interleaved with axis 3 (bf16 vs INT4 at k=5) ---
A1-k5-int4-r1;5;humming;int4;default;8k-low
A3-k5-bf16-r1;5;humming;bf16;default;8k-low
A1-k6-r1;6;humming;int4;default;8k-low
A1-k5-int4-r2;5;humming;int4;default;8k-low
A3-k5-bf16-r2;5;humming;bf16;default;8k-low
A1-k6-r2;6;humming;int4;default;8k-low
A1-k5-int4-r3;5;humming;int4;default;8k-low
A3-k5-bf16-r3;5;humming;bf16;default;8k-low
A1-k6-r3;6;humming;int4;default;8k-low
A1-k7-r1;7;humming;int4;default;8k-low
# --- axis 1 (high tier) interleaved with axis 3 at the high-entropy k (NEW) ---
A1-k2-int4-r1;2;humming;int4;default;8k-high
A3-k2-bf16-r1;2;humming;bf16;default;8k-high
A1-k2-int4-r2;2;humming;int4;default;8k-high
A1-k1-r1;1;humming;int4;default;8k-high
A1-k3-r1;3;humming;int4;default;8k-high
A1-k2-int4-r3;2;humming;int4;default;8k-high
# --- axis 2: drafter kernel, at BOTH deployment k (k=2 cells are NEW) ---
A2-k5-humlm;5;humming;int4;hum-lmhead;8k-low
A2-k5-humall;5;humming;int4;hum-all;8k-low
A2-k5-machall;5;humming;int4;machete-all;8k-low
A2-k2-humlm;2;humming;int4;hum-lmhead;8k-high
A2-k2-machall;2;humming;int4;machete-all;8k-high
LIST
)
n_serves=$(printf '%s\n' "$SERVES" | grep -cvE '^\s*(#|$)')
echo "[controller] serve list: $n_serves serves"
printf '%s\n' "$SERVES" > "$ROOT/serve-list.txt"
[ "$n_serves" = 26 ] || fail "serve list is $n_serves entries, expected 26 (edit was unintended?)"

# ============================ PRE-FLIGHT GATES ================================
for d in "$GPTQ_CKPT" "$SITE_0110" "$DRAFTER_INT4" "$DRAFTER_BF16" "$SB_DIR"; do
  test -d "$d" || fail "missing dir: $d"
done
printf 'kopt=%s\nkernel=%s\n' "$REF_KOPT" "$REF_KERNEL" > "$ROOT/reference-windows.txt"

# --- staged prompt gate: the exact bytes phases D-I.2 measured -----------------
"$PY" - "$SB_DIR" "$SHA_LOW" "$SHA_HIGH" <<'PY' >"$ROOT/speedbench-manifest.txt" \
  || fail "staged prompt gate failed"
import hashlib, json, sys
d, want = sys.argv[1], {"8k-low": sys.argv[2], "8k-high": sys.argv[3]}
for cell, pre in want.items():
    p = f"{d}/{cell}.jsonl"
    h = hashlib.sha256(open(p, "rb").read()).hexdigest()
    n = sum(1 for _ in open(p))
    assert h.startswith(pre), f"{cell}: sha {h[:12]} != expected {pre} -- prompts changed"
    print(f"{cell+'.jsonl':16s} n={n:4d} sha={h[:12]}")
print("prompt bytes identical to phases D-I.2")
PY
echo "[controller] staged prompts verified"

# --- drafter identity gates: INT4 is W4A16-derived, bf16 is genuinely unquantized ---
"$PY" - "$DRAFTER_INT4" "$DRAFTER_BF16" <<'PY' >"$ROOT/drafter-identity.txt" \
  || fail "drafter identity gate failed"
import hashlib, json, sys
from safetensors import safe_open

def sha(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()

int4, bf16 = sys.argv[1], sys.argv[2]

# ---- INT4 half: the derived artifact, W4A16, lm_head premise intact ----
cfg = json.load(open(f"{int4}/config.json"))
g = cfg["quantization_config"]["config_groups"]
assert "group_embed" not in g, "group_embed present -- not the derived artifact"
for name in ("group_0", "group_lmhead"):
    w = g[name]["weights"]
    assert w["num_bits"] == 4 and w["type"] == "int", name
    assert w["group_size"] == 128 and w["symmetric"] is True, name
    assert g[name].get("input_activations") is None, f"{name} declares activation quant"
with safe_open(f"{int4}/model.safetensors", framework="pt") as f:
    keys = set(f.keys())
    out_features, _ = f.get_slice("lm_head.weight_packed").get_shape()
assert "embed_tokens.weight" in keys and not any(
    k.startswith("embed_tokens.weight_p") for k in keys), "embed not bf16"
assert out_features == 200064, f"lm_head out={out_features}"
per_rank = out_features // 8
assert per_rank == 25008 and per_rank % 128 == 48, f"per-rank {per_rank} -- premise changed"
man = json.load(open(f"{int4}/derivation-manifest.json"))
assert man["out_safetensors_sha256"] == sha(f"{int4}/model.safetensors"), "derived artifact modified"
print("INT4 drafter :", int4)
print("  scheme     : W4A16 (int4, group 128, symmetric, no activation quant)")
print(f"  lm_head    : {out_features} -> {per_rank}/rank at TP8, % 128 == {per_rank % 128}")
print("  sha256     :", man["out_safetensors_sha256"][:16])

# ---- bf16 half: axis 3's A/B is only meaningful if this one is NOT quantized ----
bcfg = json.load(open(f"{bf16}/config.json"))
assert "quantization_config" not in bcfg, "bf16 drafter declares a quantization_config"
dt = bcfg.get("torch_dtype") or bcfg.get("dtype")
assert dt in ("bfloat16", "float16"), f"bf16 drafter dtype={dt}"
assert bcfg.get("architectures", [None])[0] == "LlamaForCausalLMEagle3", bcfg.get("architectures")
print("bf16 drafter :", bf16)
print(f"  dtype      : {dt}, no quantization_config -- the A/B counterpart is genuine")
PY
echo "[controller] both drafters verified (INT4 W4A16 + genuine bf16)"

# --- vLLM patch gates ----------------------------------------------------------
# hum-lmhead / hum-all cells cannot run without the ParallelLMHead fix, and
# machete-all cannot run without the vocab-pad lever. Apply then re-check, so a
# fresh venv is handled but a failed apply still fails closed.
for patch in patch_vllm_humming_lmhead patch_vllm_eagle3_lmhead_pad; do
  ( cd "$REPO" && "$PY" "pipeline/slurm/$patch.py" ) >>"$ROOT/patch-checks.log" 2>&1 \
    || fail "$patch apply failed"
  ( cd "$REPO" && "$PY" "pipeline/slurm/$patch.py" --check ) >>"$ROOT/patch-checks.log" 2>&1 \
    || fail "$patch still not applied after apply"
done
# Source-level gate for the humming lm_head fix: prove the guarded fallbacks are the
# ones actually loaded, not merely that a marker string is in the file.
"$PY" - <<'PY' >>"$ROOT/patch-checks.log" 2>&1 || fail "humming lm_head source gate failed"
import inspect
from vllm.model_executor.layers.quantization.utils import humming_utils
src = inspect.getsource(humming_utils.prepare_humming_layer)
for need in ("layer.embedding_dim", "[layer.num_embeddings_per_partition]",
             "sum(output_partition_sizes)"):
    assert need in src, f"missing fallback: {need}"
assert "has_bias=layer.has_bias" not in src, "unguarded has_bias read still present"
from vllm.model_executor.layers.vocab_parallel_embedding import VocabParallelEmbedding
isrc = inspect.getsource(VocabParallelEmbedding.__init__)
for need in ("self.embedding_dim", "self.num_embeddings_per_partition", "self.params_dtype"):
    assert need in isrc, f"VocabParallelEmbedding no longer sets {need}"
print("humming lm_head fallbacks present and VocabParallelEmbedding still supplies them")
PY
for patch in ct_input_format grouped_expert_bounds tma_store_fence tma_store_commit; do
  ( cd "$REPO" && PYTHONPATH="$REPO" "$PY" \
      "pipeline/slurm/patch_humming_$patch.py" --site "$SITE_0110" --check ) \
      >>"$ROOT/patch-checks.log" 2>&1 \
    || fail "humming $patch patch missing on $SITE_0110"
done
echo "[controller] vLLM + humming patches verified"

# --- kernel registry gate ------------------------------------------------------
"$PY" - <<'PY' >"$ROOT/kernel-registry.txt" || fail "kernel registry gate failed"
from vllm.model_executor.kernels.linear import _POSSIBLE_KERNELS
from vllm.platforms import PlatformEnum
names = [k.__name__ for k in _POSSIBLE_KERNELS[PlatformEnum.CUDA]]
print("CUDA MP priority:", " > ".join(names))
for need in ("MacheteLinearKernel", "MarlinLinearKernel", "HummingLinearKernel"):
    assert need in names, f"{need} absent from this vLLM build"
assert names.index("MacheteLinearKernel") < names.index("MarlinLinearKernel") \
       < names.index("HummingLinearKernel"), "kernel priority changed -- cells mislabelled"
print("priority intact: Machete > Marlin > Humming (default cell = Machete x8 + Marlin)")
PY

aiperf_version=$(/mnt/nfs/hoangduy/venvs/perf/bin/aiperf --version 2>&1 | head -1)
printf 'aiperf=%s\n' "$aiperf_version" > "$ROOT/harness-version.txt"

# --- node-health gate: 8 GPUs visible AND each >= 70 GiB free -------------------
# slurm `idle` is unreliable in both directions: nodes have reported idle while a
# foreign job held ~350 GiB, and others hand an srun step zero GPUs.
r=$(timeout 120 srun --nodes=1 --ntasks=1 --nodelist="$NODE" --time=00:01:00 \
      --partition=compute --job-name="uni-probe" \
      nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits 2>/dev/null \
    | awk '{c++; s+=$1; if ($1 < 70000) bad++}
           END {printf "%d:%d:%s", c, int(s/1024), (bad || c != 8 ? "BAD" : "ok")}')
case "$r" in
  *:ok) echo "[controller] node $NODE: ${r%%:*} GPUs, $(echo "$r" | cut -d: -f2) GiB free" ;;
  *)    fail "node $NODE failed GPU probe: ${r:-no response}" ;;
esac

( cd "$REPO" && git rev-parse HEAD ) >"$ROOT/actual-commit.txt"

# ============================ LAUNCH ==========================================
ARM=unified
# M3_W4A8_BACKEND is deliberately NOT exported here -- the arm sets it per serve, and
# a window-level default would silently win for one of the two barebone rungs.
arm_env=(
  "ROOT=$ROOT" "ARM=$ARM" "RUN_TS=$RUN_TS"
  "CKPT=$GPTQ_CKPT" "PORT_BASE=$PORT_BASE" "SB_DIR=$SB_DIR"
  "DRAFTER_INT4=$DRAFTER_INT4" "DRAFTER_BF16=$DRAFTER_BF16"
  "SERVES=$SERVES" "ONLY=$ONLY" "N_C1=$N_C1" "N_C10=$N_C10"
  "HUMMING_M3_W4A8_CACHE_ROOT=/mnt/nfs/hoangduy/.humming"
  "VLLM_HUMMING_MOE_GEMM_TYPE=indexed" "VLLM_HUMMING_USE_F16_ACCUM=0"
  "PYTHONPATH=$SITE_0110:$REPO"
)
env "${arm_env[@]}" \
srun --exclusive --nodes=1 --ntasks=1 --gres=gpu:8 --cpus-per-task=192 \
     --time=10:00:00 --kill-on-bad-exit=1 --partition=compute \
     --nodelist="$NODE" --job-name="m3-uni-$ARM" --export=ALL \
     bash "$REPO/pipeline/slurm/specdec_unified_arm.sh" \
     > "$ROOT/$ARM-srun.log" 2>&1 &
ARM_PID=$!
printf 'arm=%s serves=%s port_base=%s node=%s pid=%s only=%s launched=%s\n' \
  "$ARM" "$n_serves" "$PORT_BASE" "$NODE" "$ARM_PID" "${ONLY:-<all>}" \
  "$(date -u +%Y-%m-%dT%H:%M:%SZ)" >>"$ROOT/arm-provenance.txt"
echo "[controller] launched $ARM on $NODE pid=$ARM_PID"

wait "$ARM_PID"; rc=$?
echo "$rc" > "$ROOT/arm-$ARM.rc"
echo "[controller] arm rc=$rc"

{
  echo "unified one-node spec-dec window"
  echo "root=$ROOT node=$NODE serves=$n_serves rc=$rc only=${ONLY:-<all>}"
  echo "requests: conc1=$N_C1 conc10=$N_C10"
  cat "$ROOT/arm-$ARM/arm-done.txt" 2>/dev/null
} > "$ROOT/controller-done.txt"
echo "[controller] done -> $ROOT/controller-done.txt"
