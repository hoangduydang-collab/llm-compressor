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
# Default = the M3 "full4" seven, which the GLM-5.2 run also used. Both of the
# tasks that used to be excluded are now runnable:
#   gpqa_diamond_cot_zeroshot - gated dataset; needs HF_TOKEN in the env AND the
#     licence accepted on that account. Note the task id: the CoT variant
#     (generate_until), not gpqa_diamond_zeroshot, which is multiple_choice and a
#     different benchmark despite the similar name.
#   ifeval - fail-closed on served chat-template provenance, which is an
#     observation of the RUNNING engine. Satisfied by capturing the engine's own
#     args (chat_template=None etc. => no override), recorded in the profiles.
TASKS="${TASKS:-gsm8k ifeval gpqa_diamond_cot_zeroshot mmlu arc_challenge hellaswag truthfulqa_mc2}"

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

# gpqa_diamond_zeroshot needs Idavidrein/gpqa, the general suite's one gated
# dataset. Say so up front rather than letting the download fail 40 lines later.
# Never echo the token itself: presence and length only.
if echo "$TASKS" | grep -q gpqa; then
  if [ -n "${HF_TOKEN:-}" ]; then
    note "gpqa requested and HF_TOKEN is present (${#HF_TOKEN} chars)"
    note "  a token alone is not enough: the licence must be ACCEPTED on that"
    note "  account at https://huggingface.co/datasets/Idavidrein/gpqa"
  else
    note "FATAL: gpqa requested but HF_TOKEN is unset. Create the secret and"
    note "  mount it (see glm53-quality-arm.yaml.tmpl), or drop gpqa from TASKS."
    exit 1
  fi
fi

# ---- client venv ------------------------------------------------------------
if [ ! -x "$BVENV/bin/python" ]; then
  note "building client venv at $BVENV"
  # --system-site-packages: reuse the image's CUDA-matched torch instead of
  # letting pip pull ~2.5 GB of its own. See the same note in
  # glm53_quality_arm.sh.
  python -m venv --system-site-packages "$BVENV" || exit 1
  "$BVENV/bin/pip" install -q --upgrade pip
  "$BVENV/bin/pip" install -q "lm-eval[api,ifeval]==0.4.10" "openai>=1.40" jsonschema 2>&1 | tail -5
fi
PY="$BVENV/bin/python"

# ---- 0b. tokenizer-only dirs (the served-templating identity) ---------------
# The arm profiles point SERVED_TOKENIZER at these, NOT at the checkpoints.
# evidence.py treats an existing local directory as a `local_content` identity and
# pins it with _sha256_dir, which hashes every byte of every file it finds. Aimed
# at a checkpoint that is 394 GB of shared-cephfs reads per arm (measured: 22
# minutes) for an identity that is supposed to describe the tokenizer and template,
# not the weights. Copying those few files out makes the digest instant AND
# correct. Rebuilt every run so it can never drift from the checkpoints.
note "step 0b: building tokenizer-only identity dirs"
TOKDIR="$OUT/tok"
for arm in ours phala; do
  case "$arm" in ours) SRC="$OURS";; phala) SRC="$PHALA";; esac
  rm -rf "$TOKDIR/$arm"; mkdir -p "$TOKDIR/$arm"
  for f in tokenizer.json tokenizer_config.json special_tokens_map.json \
           chat_template.jinja vocab.json merges.txt added_tokens.json; do
    [ -f "$SRC/$f" ] && cp -f "$SRC/$f" "$TOKDIR/$arm/$f"
  done
  n=$(ls -1 "$TOKDIR/$arm" | wc -l)
  [ "$n" -gt 0 ] || { note "FATAL: no tokenizer files found under $SRC"; exit 1; }
  note "  $arm: $n file(s)"
done
# Print the digests the profiles must declare. A mismatch here is a fail-closed
# refusal inside evidence.py, so surface the right values rather than making the
# operator reverse-engineer them from a traceback.
"$PY" - "$TOKDIR" <<'PY' | tee "$OUT/tok-digests.txt"
import sys
sys.path.insert(0, "/mnt/cephfs/hoangduy/projects/benchmarks")
from quality.general.evidence import _sha256_dir
for arm in ("ours", "phala"):
    dg, n = _sha256_dir(f"{sys.argv[1]}/{arm}")
    print(f"SERVED_TOKENIZER_REVISION[{arm}] = {dg}  ({n} files)")
PY

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
        # A GROUP task (mmlu = 57 subtasks) comes back as a nested dict keyed by
        # a ConfigurableGroup, so the flat `d.items()` walk reported "1 subtask,
        # 0 docs" for mmlu -- which looked like a staging failure and was only a
        # counting failure. Flatten first, and SUM leaf docs rather than taking a
        # max, because for a group the total is what the run will actually score.
        def leaves(node):
            if isinstance(node, dict):
                for v in node.values():
                    yield from leaves(v)
            else:
                yield node

        n, k = 0, 0
        for obj in leaves(d):
            task = getattr(obj, "task", obj)
            k += 1
            for split in ("test_docs", "validation_docs"):
                fn = getattr(task, split, None)
                if fn is None:
                    continue
                try:
                    docs = fn()
                except Exception:
                    continue
                if docs is not None:
                    n += len(list(docs))
                    break
        if not n:
            # Zero docs after a successful load is not a warning, it is a broken
            # corpus: the task would silently score nothing.
            raise RuntimeError(f"{t} resolved {k} leaf task(s) but 0 documents")
        staged[t] = {"leaf_tasks": k, "docs": n}
        print(f"[stage] {t}: {k} leaf task(s), {n} docs")
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
# CRLF GUARD, FIRST. These profiles are authored on Windows and sourced by bash on
# Linux, and a CR survives into the VALUE: BASE_URL became
# 'http://127.0.0.1:30000\r/v1', which command.py then refused as a character its
# --model_args parser would split on. That cost a full staging round, and it
# presented as "PROTOCOL DIVERGENCE" rather than "this file has CRLF" -- so check
# it explicitly and say so.
for arm in ours phala; do
  f="$BENCH/configs/glm/glm-5.3-w4afp8-$arm.sh"
  if grep -qU $'\r' "$f" 2>/dev/null; then
    note "FATAL: $f has CRLF line endings; bash keeps the CR inside every value"
    note "  fix: python -c \"import pathlib,sys;p=pathlib.Path(sys.argv[1]);p.write_bytes(p.read_bytes().replace(b'\\\\r\\\\n',b'\\\\n'))\" $f"
    gate profile_line_endings 1
  fi
done
[ "$fail" = 0 ] || { note "GATES FAILED — do not launch an arm"; exit 1; }
gate profile_line_endings 0

# GENERAL_TASKS is exported so the parity check renders THE SAME task set the arms
# will actually run. Without it the profile default was rendered instead, which
# still contains ifeval, and ifeval is fail-closed on served chat-template
# provenance -- so parity failed on a task that was never going to run. A parity
# check that validates a different plan than the one executed is worse than none.
#
# Each render's rc is captured and gated SEPARATELY from the diff. A crashed
# render also produces two unequal files, so without this a crash is reported as
# a protocol divergence and the operator goes looking for a config difference
# that does not exist.
render_rc=0
for arm in ours phala; do
  ( cd "$BENCH" && GENERAL_TASKS="$TASKS" "$PY" -m quality.orchestrator \
      --profile "configs/glm/glm-5.3-w4afp8-$arm.sh" \
      --dry-run --suite general ) > "$OUT/dryrun-$arm.txt" 2>&1
  rc=$?
  note "rendered $arm (rc=$rc) -> $OUT/dryrun-$arm.txt"
  if [ "$rc" != 0 ]; then
    render_rc=1
    note "  render FAILED; tail:"
    tail -6 "$OUT/dryrun-$arm.txt" | sed 's/^/    /'
  fi
done
gate profile_renders "$render_rc"
if [ "$render_rc" != 0 ]; then
  note "a profile did not render, so the diff below would be meaningless"
  note "GATES FAILED — do not launch an arm"
  exit 1
fi
# QUOTED heredoc + argv. An unquoted <<PY makes bash expand the body, and a
# backtick in a COMMENT is then run as a command -- `is_local` in step 3 below
# did exactly that. Nothing here needs shell expansion, so nothing gets it.
"$PY" - "$OUT" "$OURS" "$PHALA" <<'PY'
import difflib, re, sys
from pathlib import Path
out, ours, phala = Path(sys.argv[1]), sys.argv[2], sys.argv[3]

# Normalise away everything that is ARM IDENTITY, so that whatever remains is a
# PROTOCOL difference -- the only kind that invalidates the comparison.
#
# The list is explicit rather than clever, and it grew: the first version stripped
# only the profile/model names, so the per-arm tokenizer-identity fields I added
# later (tok/<arm> and their content digests) read as a protocol divergence and
# failed the gate on a comparison that was in fact sound. An identity field that
# is SUPPOSED to differ must be named here.
_SHA64 = re.compile(r"\b[0-9a-f]{64}\b")


def norm(p):
    t = p.read_text(encoding="utf-8", errors="replace")
    for ident in ("glm-5.3-w4afp8-ours", "glm-5.3-w4afp8-phala", ours, phala):
        t = t.replace(ident, "<ARM>")
    lines = []
    for line in t.splitlines():
        line = line.rstrip()
        # Per-arm served-tokenizer identity: the directory and its content
        # digest are meant to differ, and the digest is a bare 64-hex token.
        if "served tokenizer:" in line or "served revision:" in line:
            line = _SHA64.sub("<DIGEST>", line).replace("/tok/ours", "/tok/<ARM>") \
                         .replace("/tok/phala", "/tok/<ARM>")
        # NOT normalised, deliberately: `served template sha256` must be EQUAL,
        # because that is what decides both arms see the same prompt text.
        if "<ARM>" in line and not ("--tasks" in line or "model_args" in line):
            continue
        lines.append(line)
    return lines


a, b = norm(out / "dryrun-ours.txt"), norm(out / "dryrun-phala.txt")
if a == b:
    print("[parity] identical after identity normalisation")
    sys.exit(0)
d = list(difflib.unified_diff(a, b, "ours", "phala", lineterm="", n=1))
print("[parity] PROTOCOL DIVERGENCE - the arms are not comparable:")
print("\n".join(d[:60]))
sys.exit(1)
PY
gate arm_parity $?

# ---- 3. tokenizer / chat template across the two checkpoints ---------------
note "step 3: tokenizer + chat-template identity across checkpoints"
"$PY" - "$OURS" "$PHALA" <<'PY' | tee "$OUT/template-parity.json"
import hashlib, json, sys
from pathlib import Path
OURS, PHALA = sys.argv[1], sys.argv[2]
FILES = ("tokenizer.json", "tokenizer_config.json", "chat_template.jinja",
         "special_tokens_map.json", "generation_config.json")
def digests(root):
    r = Path(root)
    out = {}
    for f in FILES:
        p = r / f
        out[f] = hashlib.sha256(p.read_bytes()).hexdigest() if p.is_file() else None
    return out
a, b = digests(OURS), digests(PHALA)
# tokenizer.json and chat_template.jinja decide what text the model is shown, so a
# byte difference there blocks. generation_config.json does NOT (the server's flags
# and the task's pinned kwargs govern decoding here), so it is recorded only.
BLOCKING = ("tokenizer.json", "chat_template.jinja")
diff = {f: {"ours": a[f], "phala": b[f]} for f in FILES if a[f] != b[f]}
blocking = sorted(f for f in diff if f in BLOCKING)

# tokenizer_config.json is compared SEMANTICALLY rather than by digest. Measured
# on these two checkpoints, it differs only in is_local / local_files_only --
# HF loader bookkeeping recording how each snapshot happened to be written, which
# changes neither tokenization nor templating. Blocking on the raw digest would
# fail a comparison that is in fact sound. Any difference OUTSIDE this inert set
# does block, because then the two arms really are configured differently.
INERT = {"is_local", "local_files_only", "_name_or_path", "name_or_path",
         "tokenizer_file", "auto_map"}
cfg_note = None
try:
    ca = json.loads((Path(OURS) / "tokenizer_config.json").read_text(encoding="utf-8"))
    cb = json.loads((Path(PHALA) / "tokenizer_config.json").read_text(encoding="utf-8"))
    semantic = sorted(k for k in set(ca) | set(cb)
                      if k not in INERT and ca.get(k) != cb.get(k))
    inert_diff = sorted(k for k in set(ca) | set(cb)
                        if k in INERT and ca.get(k) != cb.get(k))
    cfg_note = {"semantic_differences": semantic, "inert_differences": inert_diff,
                "keys_compared": len(set(ca) | set(cb))}
    if semantic:
        blocking.append("tokenizer_config.json:" + ",".join(semantic))
except Exception as e:
    cfg_note = {"error": "%s: %s" % (type(e).__name__, e)}
    blocking.append("tokenizer_config.json:unreadable")

res = {"ours": a, "phala": b, "differs": sorted(diff),
       "tokenizer_config": cfg_note, "blocking_differences": blocking}
print(json.dumps(res, indent=2))
sys.exit(1 if blocking else 0)
PY
gate template_parity $?

# ---- 4. inventory ----------------------------------------------------------
# Non-gating: a size difference is a question, not a defect. Recorded because
# PhalaCloud's GLM-5.3 is 21.7 MB larger than our artifact and nothing has yet
# explained which tensors account for it.
note "step 4: checkpoint inventory (non-gating)"
"$PY" - "$OURS" "$PHALA" <<'PY' > "$OUT/inventory.json"
import json, sys
from pathlib import Path
OURS, PHALA = sys.argv[1], sys.argv[2]
def inv(root):
    idx = json.loads((Path(root) / "model.safetensors.index.json").read_text())
    wm = idx["weight_map"]
    shards = sorted(set(wm.values()))
    total = sum((Path(root) / s).stat().st_size for s in shards)
    return {"tensors": len(wm), "shards": len(shards), "bytes": total,
            "metadata_total_size": idx.get("metadata", {}).get("total_size")}
a, b = inv(OURS), inv(PHALA)
names_a = set(json.loads((Path(OURS) / "model.safetensors.index.json").read_text())["weight_map"])
names_b = set(json.loads((Path(PHALA) / "model.safetensors.index.json").read_text())["weight_map"])
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
