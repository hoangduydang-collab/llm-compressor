#!/usr/bin/env bash
# Controller (tmux) for EAGLE3 spec-dec phase I -- drafter W4A16 kernel selection.
#
# QUESTION: 60% of the drafter's per-rank weight traffic (lm_head, 153.6 M of
# 254.3 M params) runs on MarlinLinearKernel rather than Machete, because
# check_machete_supports_shape needs out_features % 128 == 0 and TP8 gives
# 200064/8 = 25008 (% 128 == 48). vLLM itself warns the layer is padded/sliced every
# forward. Phase G's measured INT4 saving was ~half the bandwidth prediction; this is
# the leading candidate for the missing half. Is a different kernel faster?
#
# ONE ARM, ONE NODE, FIVE SERVES, all at k=5 / 8k-low / conc 1 and 10:
#   A-baseline      no env                                  -> {Machete, Marlin}
#   B-hum-lmhead    VLLM_DISABLED_KERNELS=Marlin*           -> {Machete, Humming}
#   C-hum-all       VLLM_DISABLED_KERNELS=Machete*,Marlin*   -> {Humming}
#   D-machete-all   LLMC_EAGLE3_LMHEAD_PAD=1024             -> {Machete}
#   A-repeat        no env, re-served last                  -> drift control
#
# k=5 / 8k-low is chosen deliberately: it is the low-entropy deployment optimum
# (phase H) AND the cell where drafting is the largest share of step cost, so it is
# the most sensitive place to measure a drafting-side kernel change. 8k-high is
# omitted on purpose -- the drafter reads identical bytes there, so that cell would
# only dilute the effect with a larger base step.
#
# ARM SCRIPT holds the full kernel-eligibility analysis (every can_implement read, not
# assumed) and the argument for why VLLM_DISABLED_KERNELS is drafter-scoped here.
#
# HONEST PRIOR: Marlin was designed for exactly this memory-bound batch-1 regime and
# Machete's advantage is at larger batch, so the current assignment may already be
# near-optimal and the whole lever may be worth <1%. The per-forward pad/slice that
# vLLM warns about is the reason it is worth five serves rather than zero. Cell C is
# also the first time HummingLinearKernel has ever run for W4A16 on CUDA -- it sits
# below Marlin in the priority list and Marlin always succeeds, so upstream's own
# path is untested. Treat a Humming win or loss as new information either way.
#
# Humming stays on 0.1.10 + the four declared patches, exactly as phases G/H, so the
# only thing that varies in this window is kernel assignment. If Humming wins, the
# 0.1.11 packed-K dequant path becomes the natural follow-up -- NOT bundled here,
# because changing backend version and kernel together would confound both.
#
# A separate file from run_specdec_kopt_srun.sh on purpose: bash reads a script
# incrementally, so editing a launcher whose controller is still running can corrupt
# the running shell.
set -uo pipefail
REPO=/mnt/nfs/hoangduy/projects/llm-compressor
GPTQ_CKPT="$REPO/artifacts/m3-awq-gptq-prepared/gptq-checkpoint-vllm-w123-abi-overlay"
DRAFTER=${DRAFTER:-/mnt/nfs/hoangduy/hf_assets/derived/MiniMax-M3-EAGLE3-INT4-bf16embed}
SB_DIR="$REPO/artifacts/aiperf-datasets/speedbench"
SITE_0110=/mnt/nfs/hoangduy/venvs/humming-0.1.10-site
PY=/mnt/nfs/hoangduy/venvs/quant/bin/python
NODE=${NODE:?set NODE to a node probed free immediately before launch}
CELL=${CELL:-8k-low}
SPEC_K=${SPEC_K:-5}
PORT_BASE=${PORT_BASE:-8080}
PHASEG_REF=${PHASEG_REF:-/mnt/nfs/hoangduy/results/m3-specdec-eagle3/20260727T102751Z-int4drafter}
PHASEH_REF=${PHASEH_REF:-/mnt/nfs/hoangduy/results/m3-specdec-eagle3/20260727T105919Z-kopt}

TS=$(date -u +%Y%m%dT%H%M%SZ)
ROOT=${ROOT_OVERRIDE:-/mnt/nfs/hoangduy/results/m3-specdec-eagle3/$TS-kernel}
mkdir -p "$ROOT" /mnt/nfs/hoangduy/logs/m3-specdec-eagle3
echo "$ROOT" > /mnt/nfs/hoangduy/logs/m3-specdec-eagle3/latest_kernel_root
echo "[controller] root=$ROOT"

export RUN_TS=$TS
export LLMC_M3_CAPTURE_SYNC=sync
echo "[controller] run_ts=$RUN_TS node=$NODE cell=$CELL k=$SPEC_K"

fail() { echo "[controller] ABORT: $1" >&2; echo "$1" >"$ROOT/abort.txt"; exit 1; }

for d in "$GPTQ_CKPT" "$SITE_0110" "$DRAFTER" "$SB_DIR"; do
  test -d "$d" || fail "missing dir: $d"
done
printf 'phaseG=%s\nphaseH=%s\n' "$PHASEG_REF" "$PHASEH_REF" > "$ROOT/reference-windows.txt"

# --- drafter identity gate: must be the derived INT4 W4A16 artifact -------------
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
    w = g[name]["weights"]
    assert w["num_bits"] == 4 and w["type"] == "int", name
    assert w["group_size"] == 128, f"{name} group_size={w['group_size']}, expected 128"
    assert w["symmetric"] is True, f"{name} is not symmetric"
    assert g[name].get("input_activations") is None, f"{name} declares activation quant (expected W4A16)"
# The kernel-eligibility analysis depends on these exact shapes: Machete needs
# in%64 and out%128, and lm_head's per-rank out is the whole point of this phase.
with safe_open(f"{d}/model.safetensors", framework="pt") as f:
    keys = set(f.keys())
    lm = f.get_slice("lm_head.weight_packed").get_shape()
assert "embed_tokens.weight" in keys, "missing bf16 embed_tokens"
assert not any(k.startswith("embed_tokens.weight_p") for k in keys), "embed is packed"
assert "lm_head.weight_packed" in keys, "lm_head is not INT4"
out_features, packed_in = lm
assert out_features == 200064, f"lm_head out_features={out_features}, expected 200064"
per_rank = out_features // 8
assert per_rank == 25008 and per_rank % 128 == 48, f"per-rank out {per_rank} -- premise changed"
man = json.load(open(f"{d}/derivation-manifest.json"))
assert man["out_safetensors_sha256"] == sha(f"{d}/model.safetensors"), "derived artifact modified"
print("drafter    :", d)
print("scheme     : W4A16 (int4, group 128, symmetric, no activation quant)")
print(f"lm_head    : out={out_features} -> {per_rank}/rank at TP8, % 128 == {per_rank % 128}")
print("            => Machete ineligible for lm_head (needs % 128 == 0); premise holds")
print("sha256     :", man["out_safetensors_sha256"][:16])
PY
echo "[controller] drafter verified: derived INT4 W4A16, lm_head premise holds"

# --- kernel-registry gate: the candidates must exist, in the expected order ------
# The whole experiment design rests on this priority list and on Humming being
# eligible-but-unreachable. Assert it against the vLLM we will actually serve with
# rather than the one the analysis was read from.
"$PY" - <<'PY' >"$ROOT/kernel-registry.txt" || fail "kernel registry gate failed"
from vllm.model_executor.kernels.linear import _POSSIBLE_KERNELS, _LINEAR_BACKEND_KERNEL_MAP
from vllm.platforms import PlatformEnum
names = [k.__name__ for k in _POSSIBLE_KERNELS[PlatformEnum.CUDA]]
print("CUDA MP priority:", " > ".join(names))
for need in ("MacheteLinearKernel", "MarlinLinearKernel", "HummingLinearKernel"):
    assert need in names, f"{need} absent from this vLLM build"
assert names.index("MarlinLinearKernel") < names.index("HummingLinearKernel"), \
    "Marlin no longer precedes Humming -- the baseline cell would not be the baseline"
assert names.index("MacheteLinearKernel") < names.index("MarlinLinearKernel"), \
    "Machete no longer precedes Marlin -- the 8/1 split premise changed"
# Documented in the arm: --linear-backend cannot reach Humming, which is why the
# lever is VLLM_DISABLED_KERNELS instead.
assert "humming" not in _LINEAR_BACKEND_KERNEL_MAP, \
    "a 'humming' --linear-backend now exists; prefer it over VLLM_DISABLED_KERNELS"
from vllm.model_executor.layers.quantization.utils.machete_utils import (
    MACHETE_PREPACKED_BLOCK_SHAPE as B)
assert list(B) == [64, 128], f"Machete block shape changed to {B}"
print("machete block shape:", list(B), "(in % 64, out % 128)")
print("registry OK: Machete > Marlin > Humming, Humming unreachable by default")
PY
echo "[controller] kernel registry verified"

# --- Humming importability gate: cells B and C are meaningless without it --------
PYTHONPATH="$SITE_0110:$REPO" "$PY" - <<'PY' >"$ROOT/humming-availability.txt" || fail "humming not importable"
from vllm.utils.import_utils import has_humming
import humming
assert has_humming(), "has_humming() is False -- HummingLinearKernel would self-reject"
print("humming import OK, version:", getattr(humming, "__version__", "unknown"))
PY
echo "[controller] humming importable under the arm's PYTHONPATH"

# --- eagle3 lm_head padding patch: applied, and inert unless opted in -----------
( cd "$REPO" && "$PY" pipeline/slurm/patch_vllm_eagle3_lmhead_pad.py ) \
  >>"$ROOT/patch-checks.log" 2>&1 || fail "could not apply eagle3 lmhead-pad patch"
( cd "$REPO" && "$PY" pipeline/slurm/patch_vllm_eagle3_lmhead_pad.py --check ) \
  >>"$ROOT/patch-checks.log" 2>&1 || fail "eagle3 lmhead-pad patch not verifiable after apply"
"$PY" - <<'PY' >>"$ROOT/patch-checks.log" || fail "lmhead-pad patch is not inert by default"
import os
from vllm.model_executor.layers.vocab_parallel_embedding import (
    DEFAULT_VOCAB_PADDING_SIZE, pad_vocab_size)
assert "LLMC_EAGLE3_LMHEAD_PAD" not in os.environ, "controller must not preset the pad var"
assert DEFAULT_VOCAB_PADDING_SIZE == 64, DEFAULT_VOCAB_PADDING_SIZE
assert pad_vocab_size(200064, 64) == 200064, "default padding is not a no-op"
assert pad_vocab_size(200064, 1024) == 200704, pad_vocab_size(200064, 1024)
assert (200704 // 8) % 128 == 0, "padded per-rank out is still not Machete-eligible"
print("pad patch inert at default 64; 1024 -> 200704 (25088/rank, % 128 == 0)")
PY
echo "[controller] lmhead-pad patch applied and confirmed inert by default"

# --- SPEED-Bench staging gate: byte-identical prompts to phases D-H -------------
test -s "$SB_DIR/manifest.json" || fail "SPEED-Bench not staged"
f="$SB_DIR/$CELL.jsonl"
test -s "$f" || fail "missing staged cell: $f"
[ "$(wc -l < "$f")" -ge 100 ] || fail "$CELL has too few entries"
grep -q "FULL BENCHMARK DATA SHOULD BE FETCHED" "$f" && fail "$CELL has masked rows"
"$PY" - <<PY >"$ROOT/speedbench-manifest.txt" || fail "SPEED-Bench hash gate failed"
import hashlib, json
m = json.load(open("$SB_DIR/manifest.json"))
name = "$CELL.jsonl"
v = m["files"][name]
h = hashlib.sha256(open("$SB_DIR/" + name, "rb").read()).hexdigest()
assert h == v["sha256"], "sha256 drift for " + name
print(f"{name:16s} n={v['entries']:4d} tok_mean={v['tokens_mean']:9.1f} sha={h[:12]}")
PY
echo "[controller] SPEED-Bench $CELL hash-verified (prompt-identical to phases D-H)"

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
# slurm `idle` is unreliable in both directions here: nodes have reported idle while
# a foreign job held ~350 GiB, and others hand an srun step zero GPUs. Require
# exactly 8 GPU lines, not just a healthy sum.
r=$(timeout 120 srun --nodes=1 --ntasks=1 --nodelist="$NODE" --time=00:01:00 \
      --partition=compute --job-name="pI-probe" \
      nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits 2>/dev/null \
    | awk '{c++; s+=$1; if ($1 < 70000) bad++}
           END {printf "%d:%d:%s", c, int(s/1024), (bad || c != 8 ? "BAD" : "ok")}')
case "$r" in
  *:ok) echo "[controller] node $NODE: ${r%%:*} GPUs, $(echo "$r" | cut -d: -f2) GiB free" ;;
  *)    fail "node $NODE failed GPU probe: ${r:-no response}" ;;
esac

( cd "$REPO" && git rev-parse HEAD ) >"$ROOT/actual-commit.txt"

ARM=kernel
arm_env=(
  "ROOT=$ROOT" "ARM=$ARM" "RUN_TS=$RUN_TS" "CELL=$CELL" "SPEC_K=$SPEC_K"
  "CKPT=$GPTQ_CKPT" "PORT_BASE=$PORT_BASE" "SB_DIR=$SB_DIR" "DRAFTER=$DRAFTER"
  "HUMMING_M3_W4A8_CACHE_ROOT=/mnt/nfs/hoangduy/.humming"
  "M3_W4A8_BACKEND=humming" "VLLM_HUMMING_MOE_GEMM_TYPE=indexed"
  "VLLM_HUMMING_USE_F16_ACCUM=0" "PYTHONPATH=$SITE_0110:$REPO"
)
env "${arm_env[@]}" \
srun --exclusive --nodes=1 --ntasks=1 --gres=gpu:8 --cpus-per-task=192 \
     --time=08:00:00 --kill-on-bad-exit=1 --partition=compute \
     --nodelist="$NODE" --job-name="m3-pI-$ARM" --export=ALL \
     bash "$REPO/pipeline/slurm/specdec_kernel_arm.sh" \
     > "$ROOT/$ARM-srun.log" 2>&1 &
ARM_PID=$!
printf 'arm=%s cell=%s k=%s port_base=%s node=%s pid=%s launched=%s\n' \
  "$ARM" "$CELL" "$SPEC_K" "$PORT_BASE" "$NODE" "$ARM_PID" \
  "$(date -u +%Y-%m-%dT%H:%M:%SZ)" >>"$ROOT/arm-provenance.txt"
echo "[controller] launched $ARM on $NODE pid=$ARM_PID"

wait "$ARM_PID"; rc=$?
echo "$rc" > "$ROOT/arm-$ARM.rc"
echo "[controller] arm rc=$rc"

{
  echo "phase I -- drafter W4A16 kernel selection"
  echo "root=$ROOT node=$NODE cell=$CELL k=$SPEC_K rc=$rc"
  for cfg in A-baseline B-hum-lmhead C-hum-all D-machete-all A-repeat; do
    d="$ROOT/arm-$ARM/$cfg"
    printf '%-15s kernels=%-18s pad_warn=%-4s acc_c1=%s\n' "$cfg" \
      "$(cat "$d/wna16-kernels.txt" 2>/dev/null || echo -)" \
      "$(cat "$d/marlin-pad-warning.txt" 2>/dev/null || echo -)" \
      "$(cat "$d/accepted-$CELL-c1.txt" 2>/dev/null || cat "$d/accepted-$CELL-repeat-c1.txt" 2>/dev/null || echo -)"
  done
} | tee "$ROOT/summary.txt"
echo "[controller] done -- $ROOT"
