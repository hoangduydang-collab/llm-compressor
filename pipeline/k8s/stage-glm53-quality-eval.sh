#!/usr/bin/env bash
# Prepare the GLM-5.3 paired quality evaluation. CPU ONLY — this script must
# never need a GPU, so that everything which can fail cheaply does fail cheaply,
# before an 8xH100 arm is scheduled.
#
# It does four things, in this order, each fail-closed:
#   1. datasets  — pull every corpus the general suite needs into the SHARED HF
#                  cache, so both arms can run with HF_HUB_OFFLINE=1 and cannot
#                  score different revisions of the same benchmark.
#   2. parity    — render both arm profiles through `quality.orchestrator
#                  --dry-run` and refuse if any PROTOCOL field differs. Two arms
#                  that disagree on decoding, budgets, tasks, few-shot counts or
#                  concurrency measure the harness, not the quantizer.
#   3. templates — diff the tokenizer and chat-template digests ACROSS the two
#                  checkpoints. Arm A is our conversion of GLM-5.3; arm B is
#                  PhalaCloud's. If those templates differ, the arms are answering
#                  differently-worded prompts and no delta between them means
#                  anything. This is the precondition for reading any result.
#   4. inventory — record both checkpoints' shard/tensor inventories, which is
#                  also where the unexplained 21.7 MB size difference between our
#                  artifact and PhalaCloud's gets pinned down.
#
# PREREQUISITE, done from the operator's laptop, NOT here: the benchmarks repo is
# on SSH-only internal GitLab and no pod holds a credential for it. Push the
# working tree in from a local clone instead of cloning on the pod:
#
#   tar -C "$(dirname "$BENCH_LOCAL")" -czf /tmp/bench.tgz \
#       --exclude=.git --exclude='*.pyc' --exclude=__pycache__ "$(basename "$BENCH_LOCAL")"
#   kubectl -n evaluation cp /tmp/bench.tgz <pod>:/tmp/bench.tgz
#   kubectl -n evaluation exec <pod> -- \
#       tar -C /mnt/cephfs/hoangduy/projects -xzf /tmp/bench.tgz
#
# 76 MB / 576 files, so this costs seconds. Staging into
# /mnt/cephfs/hoangduy/projects keeps it inside our own area — it adds a
# directory to the shared PVC and touches nothing a collaborator owns.
set -uo pipefail

BENCH="${BENCH:-/mnt/cephfs/hoangduy/projects/benchmarks}"
BVENV="${BVENV:-/work/bvenv}"
OURS="${OURS:-/mnt/cephfs/hoangduy/results/glm53-w4afp8-mtp/checkpoint}"
PHALA="${PHALA:-/mnt/cephfs/.hf-cache/models--PhalaCloud--GLM-5.3-W4AFP8/snapshots/7e77d7b5592d748778459a0dac802e7fd407e593}"
OUT="${OUT:-/mnt/cephfs/hoangduy/results/glm53-quality-paired}"
# The default task set is the GLM-5.2 official seven MINUS the two that cannot
# run unattended here: gpqa_diamond_zeroshot (gated dataset, needs an accepted
# licence on the token's account) and ifeval (fail-closed on served
# chat-template provenance, which is an operator observation of the running
# engine). Both stay addable for the formal run.
TASKS="${TASKS:-gsm8k mmlu arc_challenge hellaswag truthfulqa_mc2}"

export HF_HOME=/mnt/cephfs/.hf-cache
export HF_HUB_CACHE=/mnt/cephfs/.hf-cache
export HF_DATASETS_CACHE=/mnt/cephfs/.hf-cache/datasets

mkdir -p "$OUT"
fail=0
note() { echo "[stage $(date -u +%H:%M:%S)] $1"; }
gate() { echo "$1=$2" >> "$OUT/stage-gates.txt"; note "gate $1=$2"
         [ "$2" = 0 ] || fail=1; }

note "bench=$BENCH ours=$OURS phala=$PHALA"
[ -d "$BENCH/quality" ] || { note "FATAL: benchmarks not staged at $BENCH (see header)"; exit 1; }

# ---- client venv ------------------------------------------------------------
if [ ! -x "$BVENV/bin/python" ]; then
  note "building client venv at $BVENV"
  python -m venv "$BVENV" || exit 1
  "$BVENV/bin/pip" install -q --upgrade pip
  "$BVENV/bin/pip" install -q "lm-eval[api,ifeval]==0.4.10" "openai>=1.40" jsonschema 2>&1 | tail -5
fi
PY="$BVENV/bin/python"

# ---- 1. datasets -----------------------------------------------------------
# Downloaded ONCE here, with the network available, then frozen. The arms run
# offline against exactly these bytes. A missing corpus must break staging, which
# costs a CPU pod, rather than an arm, which costs 8 H100s.
note "step 1: staging datasets for [$TASKS]"
HF_HUB_OFFLINE=0 HF_DATASETS_OFFLINE=0 "$PY" - "$TASKS" <<'PY' 2>&1 | tee "$OUT/datasets.log"
import json, sys, traceback
from pathlib import Path
tasks = sys.argv[1].split()
# lm-eval owns the task->dataset mapping; asking IT to build the task is the only
# way to be sure the staged corpus is the one the eval will look for. Guessing HF
# repo ids here would stage the wrong thing and still "succeed".
from lm_eval.tasks import TaskManager
tm = TaskManager()
staged, failed = {}, {}
for t in tasks:
    try:
        d = tm.load_task_or_group([t])
        n = 0
        for name, obj in d.items():
            task = getattr(obj, "task", obj)
            for split in ("test_docs", "validation_docs"):
                fn = getattr(task, split, None)
                if fn is None:
                    continue
                try:
                    docs = fn()
                except Exception:
                    continue
                if docs is not None:
                    n = max(n, len(list(docs)))
                    break
        staged[t] = {"subtasks": len(d), "docs": n}
        print(f"[stage] {t}: {len(d)} subtask(s), {n} docs")
    except Exception as e:
        failed[t] = f"{type(e).__name__}: {e}"
        print(f"[stage] {t}: FAILED {type(e).__name__}: {e}")
        traceback.print_exc()
Path("/mnt/cephfs/hoangduy/results/glm53-quality-paired/datasets.json").write_text(
    json.dumps({"staged": staged, "failed": failed}, indent=2), encoding="utf-8")
raise SystemExit(1 if failed else 0)
PY
gate datasets $?

# Prove the freeze actually holds: re-resolve every task with the network shut
# off. If this fails, the arms would have failed the same way after burning the
# startup cost of a 394 GB load.
note "step 1b: re-resolving the same tasks OFFLINE (the freeze must hold)"
HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 "$PY" - "$TASKS" <<'PY' 2>&1 | tail -20
import sys
from lm_eval.tasks import TaskManager
bad = []
tm = TaskManager()
for t in sys.argv[1].split():
    try:
        tm.load_task_or_group([t]); print(f"[offline] {t}: ok")
    except Exception as e:
        bad.append(t); print(f"[offline] {t}: FAILED {type(e).__name__}: {e}")
raise SystemExit(1 if bad else 0)
PY
gate datasets_offline $?

# ---- 2. arm parity ---------------------------------------------------------
# The dry run is the same code path as --execute for everything except sending
# requests, so a field that renders identically here renders identically there.
note "step 2: arm parity (protocol fields must be identical)"
for arm in ours phala; do
  ( cd "$BENCH" && "$PY" -m quality.orchestrator \
      --profile "configs/glm/glm-5.3-w4afp8-$arm.sh" \
      --dry-run --suite general ) > "$OUT/dryrun-$arm.txt" 2>&1
  note "rendered $arm (rc=$?) -> $OUT/dryrun-$arm.txt"
done
"$PY" - <<PY
import re, sys
from pathlib import Path
out = Path("$OUT")
# Compare the rendered plans with the two arms' IDENTITIES normalised away.
# Whatever differs after that is a protocol difference, which is exactly what
# must not exist.
def norm(p):
    t = p.read_text(encoding="utf-8", errors="replace")
    for ident in ("glm-5.3-w4afp8-ours", "glm-5.3-w4afp8-phala",
                  "$OURS", "$PHALA"):
        t = t.replace(ident, "<ARM>")
    return [l.rstrip() for l in t.splitlines()
            # drop lines that only carry identity/paths
            if "<ARM>" not in l or "--tasks" in l or "model_args" in l]
a, b = norm(out / "dryrun-ours.txt"), norm(out / "dryrun-phala.txt")
if a == b:
    print("[parity] identical after identity normalisation"); sys.exit(0)
import difflib
d = list(difflib.unified_diff(a, b, "ours", "phala", lineterm="", n=1))
print("[parity] PROTOCOL DIVERGENCE — the arms are not comparable:")
print("\n".join(d[:60]))
sys.exit(1)
PY
gate arm_parity $?

# ---- 3. tokenizer / chat template across the two checkpoints ---------------
note "step 3: tokenizer + chat-template identity across checkpoints"
"$PY" - <<PY | tee "$OUT/template-parity.json"
import hashlib, json, sys
from pathlib import Path
FILES = ("tokenizer.json", "tokenizer_config.json", "chat_template.jinja",
         "special_tokens_map.json", "generation_config.json")
def digests(root):
    r = Path(root)
    out = {}
    for f in FILES:
        p = r / f
        out[f] = hashlib.sha256(p.read_bytes()).hexdigest() if p.is_file() else None
    return out
a, b = digests("$OURS"), digests("$PHALA")
# tokenizer.json / tokenizer_config.json / chat_template.jinja decide what text
# the model is shown. generation_config.json does NOT (the server's flags and the
# task's pinned kwargs govern decoding here), so a difference there is recorded
# but does not block.
BLOCKING = ("tokenizer.json", "tokenizer_config.json", "chat_template.jinja")
diff = {f: {"ours": a[f], "phala": b[f]} for f in FILES if a[f] != b[f]}
res = {"ours": a, "phala": b, "differs": sorted(diff),
       "blocking_differences": sorted(f for f in diff if f in BLOCKING)}
print(json.dumps(res, indent=2))
sys.exit(1 if res["blocking_differences"] else 0)
PY
gate template_parity $?

# ---- 4. inventory ----------------------------------------------------------
# Non-gating: a size difference is a question, not a defect. Recorded because
# PhalaCloud's GLM-5.3 is 21.7 MB larger than our artifact and nothing has yet
# explained which tensors account for it.
note "step 4: checkpoint inventory (non-gating)"
"$PY" - <<PY > "$OUT/inventory.json"
import json
from pathlib import Path
def inv(root):
    idx = json.loads((Path(root) / "model.safetensors.index.json").read_text())
    wm = idx["weight_map"]
    shards = sorted(set(wm.values()))
    total = sum((Path(root) / s).stat().st_size for s in shards)
    return {"tensors": len(wm), "shards": len(shards), "bytes": total,
            "metadata_total_size": idx.get("metadata", {}).get("total_size")}
a, b = inv("$OURS"), inv("$PHALA")
names_a = set(json.loads((Path("$OURS") / "model.safetensors.index.json").read_text())["weight_map"])
names_b = set(json.loads((Path("$PHALA") / "model.safetensors.index.json").read_text())["weight_map"])
print(json.dumps({
    "ours": a, "phala": b,
    "byte_delta_phala_minus_ours": b["bytes"] - a["bytes"],
    "tensor_delta": b["tensors"] - a["tensors"],
    "only_in_ours": sorted(names_a - names_b)[:40],
    "only_in_phala": sorted(names_b - names_a)[:40],
    "only_in_ours_count": len(names_a - names_b),
    "only_in_phala_count": len(names_b - names_a),
}, indent=2))
PY
note "inventory -> $OUT/inventory.json"
head -30 "$OUT/inventory.json"

note "==== staging summary ===="
cat "$OUT/stage-gates.txt"
[ "$fail" = 0 ] && note "ALL GATES PASS — arms may be launched" \
                || note "GATES FAILED — do not launch an arm"
exit "$fail"
