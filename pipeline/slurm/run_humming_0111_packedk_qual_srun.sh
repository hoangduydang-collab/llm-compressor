#!/usr/bin/env bash
# Qualify humming 0.1.11 (packed-K dequant layout) for MiniMax-M3 W4A8.
#
# Why. Upstream 0.1.11 auto-enables a new packed-K weight layout
# (LayerMeta.use_packed_k_layout) for exactly our config -- WGMMA, 8-bit
# activations, no fused e8m0 scale, weight-scale group size 128 -- changing
# the B loaders, mainloop dequant, weight repack, and the sm90 tuning path
# (max_block_m=128). It also still ships all four defects we patch: the
# pack-quantized schema gap, the grouped last-expert bound, and the two TMA
# store-synchronization bugs (fence + commit-group; tma_commit_store_group is
# still defined and never called in 0.1.11). This run qualifies the patched
# 0.1.11 side-install with the same correctness probes that closed the
# store-sync investigation (m3-arm3-commit-verify/20260725T162957Z: 0/96 bad,
# sweep clean), before any perf comparison against the 0.1.10 arms.
#
# Gates, all fail-closed before GPU work:
#   - humming resolves from the 0.1.11 side-install at version 0.1.11
#   - all four declared patches present (--check)
#   - use_packed_k_layout actually engages for the probe layer config
#     (the whole point of the qualification)
#
# Uses a dedicated JIT cache dir so 0.1.10 cubins are never mixed in.
set -uo pipefail
REPO=/mnt/nfs/hoangduy/projects/llm-compressor
SITE=/mnt/nfs/hoangduy/venvs/humming-0.1.11-site
VENV=/mnt/nfs/hoangduy/venvs/quant

TS=$(date -u +%Y%m%dT%H%M%SZ)
ROOT=${ROOT_OVERRIDE:-/mnt/nfs/hoangduy/results/m3-humming-0111-packedk-qual/$TS}
mkdir -p "$ROOT"

( cd "$REPO" && git rev-parse HEAD ) >"$ROOT/commit.txt"

cat >"$ROOT/run.sh" <<EOF
set -uo pipefail
source $VENV/bin/activate
export LD_LIBRARY_PATH="$VENV/lib/python3.12/site-packages/nvidia/cu13/lib\${LD_LIBRARY_PATH:+:\$LD_LIBRARY_PATH}"
export PYTHONPATH="$SITE:$REPO"
export HOME=/mnt/nfs/hoangduy
export HUMMING_CACHE_DIR=/mnt/nfs/hoangduy/.humming/cache-m3-gptq-w4a8-0111-v1
cd $REPO
python - <<'PY' || exit 9
import importlib.metadata as md
import humming
assert humming.__file__.startswith("$SITE/"), humming.__file__
version = md.version("humming-kernels")
assert version == "0.1.11", version
from humming.layer import HummingLayerMeta
from pipeline.m3_humming_grouped_tile_forensics import build_layer_config
meta = HummingLayerMeta(**build_layer_config())
print({"humming": version, "use_packed_k_layout": meta.use_packed_k_layout})
assert meta.use_packed_k_layout, "packed-K layout did not engage; nothing to qualify"
PY
python pipeline/slurm/patch_humming_ct_input_format.py --site $SITE --check || exit 10
python pipeline/slurm/patch_humming_grouped_expert_bounds.py --site $SITE --check || exit 11
python pipeline/slurm/patch_humming_tma_store_fence.py --site $SITE --check || exit 12
python pipeline/slurm/patch_humming_tma_store_commit.py --site $SITE --check || exit 13
python -m pipeline.m3_humming_grouped_tile_forensics --out "\$1/forensics" --repeats 96 || exit 1
python -m pipeline.m3_humming_grouped_scale_probe --out "\$1/sweep"
EOF

srun --nodes=1 --ntasks=1 --gres=gpu:1 --cpus-per-task=16 \
     --time=02:00:00 --kill-on-bad-exit=1 --job-name=m3-humming-0111-qual \
     bash "$ROOT/run.sh" "$ROOT" >"$ROOT/srun.log" 2>&1
rc=$?
echo "rc=$rc" >"$ROOT/probe.rc"
echo "QUAL_RC=$rc root=$ROOT"
exit "$rc"
