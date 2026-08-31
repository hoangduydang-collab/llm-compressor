#!/usr/bin/env bash
# One GLM-5.3 quality arm on the Rancher cluster: serve on SGLang (single node)
# -> capability gates (fail-closed) -> standalone general-suite orchestrator ->
# shutdown. Sequential-arm design: arms never overlap; A/B deltas vs the peer
# arm are rebuilt offline with `quality.rebuild_delta`.
#
# THIS IS A PORT, NOT A NEW DESIGN. Every serve flag below is copied from
# pipeline/slurm/glm52_quality_arm.sh, the arm runner that produced
# GLM52_OFFICIAL_EVAL_RESULTS.html ("Serve env/flags proven by
# evals/glm52-w4afp8-smoke-20260722T0958Z (PASS)"). Deviating from it silently
# would make the GLM-5.3 numbers non-comparable with the GLM-5.2 run that is our
# only reference point for what a healthy W4AFP8 arm looks like. Every deviation
# that IS present is called out in the "CLUSTER-2 DEVIATIONS" block and has a
# mechanical reason.
#
# Env (required): ARM PROFILE PORT ROOT
# Env (optional): MODEL_PATH TP CTX RUN_ID MEM_FRAC CHUNKED_PREFILL LIMIT
#                 REASONING_PARSER TOOL_PARSER SUITE
set -uo pipefail

ARM=${ARM:?}; PROFILE=${PROFILE:?}; PORT=${PORT:?}; ROOT=${ROOT:?}
MODEL_PATH="${MODEL_PATH:?MODEL_PATH is required (no cluster-2 default exists)}"
QUANT_ARGS="${QUANT_ARGS:---quantization w4afp8}"
TP="${TP:-8}"
# 65536, not the GLM-5.2 arm's 131072. Nothing in this suite needs more than the
# 32768-token generation budget plus a 25-shot prompt, and a lower ceiling bounds
# the worst-case KV a single runaway sequence can reserve. The KV POOL size is set
# by memory, not by this (measured: max_total_num_tokens=101120 with a BF16 KV
# cache, ~2x that with fp8_e4m3), so this is a per-sequence cap, not a capacity
# knob.
CTX="${CTX:-65536}"
RUN_ID="${RUN_ID:-glm53full7}"
# 0.75 + chunked prefill 2048: echo/loglikelihood requests compute logits for
# ALL prompt tokens (vocab x batch tokens, TP all-gather) — at 0.85 the GLM-5.2
# scheduler OOM'd and died (job 13136). 2048-token chunks bound that buffer.
# Our checkpoint is 394 GB against PhalaCloud GLM-5.2's 407 GB, so this budget
# is if anything slightly looser here than where it was proven.
MEM_FRAC="${MEM_FRAC:-0.75}"
CHUNKED_PREFILL="${CHUNKED_PREFILL:-2048}"
# GLM-5.2 served with the glm45 reasoning parser and the glm47 tool-call parser.
# These are OVERRIDABLE rather than hardcoded because GLM-5.3's template family
# is not something this script can verify offline; step 2b probes the served
# behaviour and records it, so a wrong parser shows up as evidence instead of as
# a silently mis-scored suite.
REASONING_PARSER="${REASONING_PARSER:-glm45}"
TOOL_PARSER="${TOOL_PARSER:-glm47}"

# ---- CLUSTER-2 DEVIATIONS from pipeline/slurm/glm52_quality_arm.sh ----------
# 1. No `source /mnt/nfs/hoangduy/env.sh` and no venv activation. That cluster's
#    NFS root and its sglang-eval/benchmarks venvs do not exist here; this arm
#    runs inside the lmsysorg/sglang image and builds its client venv locally.
# 2. DG_JIT_NVCC_COMPILER is dropped: it pointed at /mnt/nfs/hoangduy/cuda-12.9,
#    a cluster-1 path that does not exist here. DeepGEMM's JIT is now left
#    ENABLED (see the DeepGEMM note below for why the GLM-5.2 disables were
#    removed), so if a compile needs a host compiler this is the knob to revisit.
# 3. TRITON_CACHE_DIR / TORCHINDUCTOR_CACHE_DIR still go to /tmp, which the pod
#    spec backs with a per-pod emptyDir. On cluster 1 these had to be set to
#    escape a NODE-SHARED default that other jobs poisoned; here the pod
#    boundary already gives that isolation. They stay set anyway so the cache
#    can never be pointed at cephfs, where it WOULD be shared with collaborators.
# 4. BENCH is a staged copy under our own cephfs area, not a clone: the
#    benchmarks repo is on SSH-only internal GitLab and this pod holds no
#    credential for it. pipeline/k8s/stage-glm53-quality-eval.sh pushes the
#    working tree in from the operator's laptop instead.
# ----------------------------------------------------------------------------

REPO="${REPO:-/work/repo}"
BENCH="${BENCH:-/mnt/cephfs/hoangduy/projects/benchmarks}"
# A PERSISTENT client venv on the PVC, built once and reused by every arm --
# the same pattern cluster 1 used (`source /mnt/nfs/hoangduy/venvs/quant/bin/
# activate`), which is why the M3 runs never had to think about scoring pins.
#
# The point is not saving the ~40 s of pip: it is that the pinned closure
# (lm_eval 0.4.10 / datasets 5.0.0 / nltk 3.10.0) is resolved and VERIFIED once
# instead of re-resolved per pod, where a resolver change could silently hand one
# arm a different scorer than the other. Two arms reading it is read-only.
#
# The path names the image tag on purpose: the venv is built with
# --system-site-packages against THIS image's torch, so it is only valid for
# pods running the same image. A different sglang image needs its own venv.
BVENV="${BVENV:-/mnt/cephfs/hoangduy/venvs/eval-sglang-0.5.17}"

CLIENT=$ROOT/client-$ARM
mkdir -p "$CLIENT"
note() { echo "[$ARM $(date -u +%H:%M:%S)] $1" | tee -a "$CLIENT/client.log"; }
gate() { echo "$1=$2" >> "$CLIENT/gates.txt"; note "gate $1=$2"; }

# A PRIVATE HF cache, deliberately not the shared /mnt/cephfs/.hf-cache.
#
# Two reasons, and the first is about other people's work:
#
# 1. /mnt/cephfs/.hf-cache is SHARED -- it holds other collaborators' models
#    (DeepSeek-V4-Flash, Qwen3.6/3.8, GLM-4.7-FP8) and datasets (SPEED-Bench,
#    ultrachat_200k). ADDING an entry there is additive and harmless; MUTATING
#    one is not, and the documented fix for a stale Arrow schema requires
#    removing the dataset's caches outright. Doing that in the shared tree could
#    break a collaborator's offline run, so our runs get their own tree instead.
#
# 2. It is also what makes the ifeval population digest correct.
#    OFFICIAL_POPULATION_DOC_DIGEST is a hash over what `datasets` hands lm-eval
#    per row, and a cache built by an older `datasets` keeps its Arrow schema:
#    the IFEval `kwargs` column comes back as a null-padded fixed struct instead
#    of List(Json), so the same 541 items hash differently. The provenance record
#    is explicit that force_redownload alone is NOT sufficient -- only a cache
#    with nothing pre-existing re-infers the schema. A private root gives us that
#    without deleting anything anyone else may be using.
#
# MODEL_PATH is an absolute snapshot path, so nothing here needs the shared hub
# cache to load the weights; the shared tree stays read-only for us.
# Overridable through EVAL_HF_ROOT, NOT through HF_HOME -- an ambient
# HF_HOME=/mnt/cephfs/.hf-cache must not be able to silently put us back on the
# shared cache, whose Arrow schema fails the ifeval population digest.
EVAL_HF_ROOT="${EVAL_HF_ROOT:-/mnt/cephfs/hoangduy/hf-private}"
export HF_HOME="$EVAL_HF_ROOT"
export HF_HUB_CACHE="$EVAL_HF_ROOT/hub"
export HF_DATASETS_CACHE="$EVAL_HF_ROOT/datasets"
# Offline is not a performance knob, it is the pairing guarantee: neither arm may
# reach the network mid-eval and silently score a different corpus revision than
# the other. Datasets are pre-staged by stage-glm53-quality-eval.sh; a cache miss
# must fail this arm rather than quietly re-download.
export HF_DATASETS_OFFLINE=1 HF_HUB_OFFLINE=1

# The IFEval scorer's corpora. An operator-staged input, not a pip dependency:
# `ifeval_runtime.require_declared_source_root` refuses an unset NLTK_DATA, and
# the scorer's own downloader is a raising stub, so an unstaged corpus is a
# refusal rather than a quiet download. Staged by stage-glm53-quality-eval.sh.
# Must name exactly ONE absolute, normalised, non-symlink directory -- a
# `:`-separated list is refused precisely because it reintroduces the ambiguity
# the gate exists to remove.
export NLTK_DATA="${NLTK_DATA:-/mnt/cephfs/hoangduy/nltk_data}"

# DeepGEMM: LEFT ENABLED, which is a change from the GLM-5.2 arm this was ported
# from. That arm exported SGLANG_ENABLE_JIT_DEEPGEMM=0 / DG_JIT_USE_NVRTC=0 /
# SGLANG_DG_USE_NVRTC=0 as a "compat" combo, and those three disables were the
# prime suspect for the throughput this run actually measured: 454 tok/s
# aggregate at num_concurrent=16 (~28 tok/s per stream), which is far below what
# 8xH100 should give a W4AFP8 MoE and slow enough that single hard items crossed
# lm-eval's 300 s client timeout.
#
# The vendor's own serving command for this checkpoint sets NONE of them, and
# describes the checkpoint as "served by SGLang's native CUTLASS W4A8 grouped
# GEMM" -- so disabling the JIT path those FP8 layers compile through was ours,
# not the reference configuration.
#
# RISK, stated rather than discovered later: cluster-2 compute nodes have no gcc
# (see the cluster notes), so an NVCC-backed JIT compile can fail. If the server
# dies at startup with a compiler error, restore the three disables -- that
# failure is loud and costs one ~12 min load, whereas the silent 3x slowdown cost
# a 60-minute invocation.
export FLASHINFER_USE_CUDA_NORM=1
export TRITON_CACHE_DIR=/tmp/triton-cache-${ARM}
export TORCHINDUCTOR_CACHE_DIR=/tmp/inductor-cache-${ARM}
mkdir -p "$TRITON_CACHE_DIR" "$TORCHINDUCTOR_CACHE_DIR"

note "host=$(hostname) arm=$ARM model=$MODEL_PATH tp=$TP ctx=$CTX port=$PORT run_id=$RUN_ID"
note "mem_frac=$MEM_FRAC chunked_prefill=$CHUNKED_PREFILL reasoning_parser=$REASONING_PARSER"

# ---- step 0: preflight (fail-closed, before any GPU is touched) -------------
# The repo's evaluation-harness contract (CLAUDE.md) requires the harness
# identity recorded BEFORE launch, so a score can never be reported without the
# conditions that produced it.
note "step 0: preflight"
[ -d "$BENCH/quality" ] || { note "FATAL: benchmarks not staged at $BENCH"; exit 1; }
[ -f "$MODEL_PATH/model.safetensors.index.json" ] || {
  note "FATAL: no weight index at $MODEL_PATH"; exit 1; }

if [ ! -x "$BVENV/bin/python" ]; then
  note "building client venv at $BVENV"
  # --system-site-packages is not cosmetic: lm-eval depends on torch, and the
  # sglang image already ships a CUDA-matched build (2.11.0+cu130). An isolated
  # venv makes pip download ~2.5 GB of torch it does not need, per pod, and then
  # shadows the image's build with a generic one. lm-eval 0.4.10 only requires
  # torch>=1.8, so the resident version satisfies it and pip leaves it alone.
  python -m venv --system-site-packages "$BVENV" >/dev/null 2>&1 \
    || virtualenv --system-site-packages "$BVENV" >/dev/null 2>&1
  # lm-eval 0.4.10 is the version the GLM-5.2 official run used; pinning it is
  # what makes task aliases and metric keys comparable across the two runs.
  # The extras are not optional: quality/general/command.py documents
  # `lm_eval[api,ifeval]==0.4.10` — [api] provides the local-completions /
  # local-chat-completions backends this whole arm depends on, and [ifeval]
  # provides the IFEval scorer's own deps.
  "$BVENV/bin/pip" install -q --upgrade pip
  # Install the repository's PINNED CLOSURE rather than a hand-rolled list.
  # `lm-eval[api,ifeval]==0.4.10` on its own leaves `datasets` and `nltk` to
  # pip's resolver, and evidence-backed IFEval enforces both EXACTLY at runtime:
  # `evidence.require_pinned_datasets_version` (datasets==5.0.0, because
  # OFFICIAL_POPULATION_DOC_DIGEST is a hash over what that library hands
  # lm-eval per row) and `nltk_resources` (nltk==3.10.0, a DIRECT, coherent,
  # SOLE install -- an overlay is prohibited). Getting this wrong is silent
  # until the suite refuses, which on the first full7 attempt was after 45
  # minutes of weight load.
  "$BVENV/bin/pip" install -q -r "$BENCH/quality/general/requirements-ifeval.txt" \
    "openai>=1.40" jsonschema 2>&1 | tail -5
fi
"$BVENV/bin/python" - <<'PY' | tee -a "$CLIENT/client.log"
import json, lm_eval, openai, sys
print(json.dumps({"lm_eval": lm_eval.__version__, "openai": openai.__version__}))
PY

# The scoring closure and the corpora, checked BEFORE the GPU is touched. Each of
# these is a hard refusal inside the general suite; discovering any of them after
# the weight load costs ~45 minutes of 8xH100 time and produces no measurement.
# Everything here is verified by the repository's own code, not by inspecting the
# tree myself -- the point is to run the same checks the suite will run.
if echo "${GENERAL_TASKS:-ifeval}" | grep -q ifeval; then
  "$BVENV/bin/python" - "$BENCH" <<'PY' | tee -a "$CLIENT/client.log"
import os, sys
sys.path.insert(0, sys.argv[1])
import datasets, nltk, lm_eval
from quality.general import nltk_resources as NR
from quality.general import evidence as EV
from quality.general import ifeval_runtime as IR
bad = []
if datasets.__version__ != EV.REQUIRED_DATASETS_VERSION:
    bad.append("datasets %s != %s" % (datasets.__version__, EV.REQUIRED_DATASETS_VERSION))
if nltk.__version__ != NR.REQUIRED_NLTK_VERSION:
    bad.append("nltk %s != %s" % (nltk.__version__, NR.REQUIRED_NLTK_VERSION))
try:
    root = IR.resolve_private_root()
    print("nltk source root OK:", os.path.basename(root))
except Exception as e:
    bad.append("NLTK_DATA: %s" % e)
try:
    ident = NR.preflight()
    print("nltk resources OK:", ", ".join(r["spec"] for r in ident["resources"]))
except Exception as e:
    bad.append("nltk corpora: %s" % e)
# The official-population CONTENT identity, the check that killed run
# full7-20260831t045151z 19 s into the suite. Reuses the SAME two helpers the
# suite's own preflight calls, so a pass here is the identical comparison rather
# than a lookalike. eval_docs is the whole unlimited population, so this is
# limit-independent.
try:
    import lm_eval.tasks as _T
    _td = _T.get_task_dict(["ifeval"], _T.TaskManager())
    def _leaves(node):
        if isinstance(node, dict):
            for v in node.values():
                yield from _leaves(v)
        else:
            yield node
    _obj = next(iter(_leaves(_td)))
    _docs = list(_obj.eval_docs)
    EV.verify_official_population_digest("ifeval", EV.doc_hashes_of(_docs))
    print("ifeval official population OK: %d docs" % len(_docs))
except Exception as e:
    bad.append("ifeval population: %s" % str(e)[:400])
if bad:
    print("IFEVAL PREFLIGHT FAILED:")
    for b in bad:
        print("  -", b)
    sys.exit(1)
print("ifeval scoring preflight OK")
PY
  gate ifeval_preflight "${PIPESTATUS[0]}"
  grep -q '^ifeval_preflight=0$' "$CLIENT/gates.txt" || {
    note "FATAL: refusing to load 394 GB of weights for a suite that would refuse"
    note "  to score ifeval. Fix staging (pipeline/k8s/stage-glm53-quality-eval.sh)"
    note "  or drop ifeval from GENERAL_TASKS."; exit 1; }
fi

"$BVENV/bin/python" - <<PY > "$CLIENT/harness_manifest.json"
import hashlib, json, os, subprocess
from pathlib import Path
ck = Path("$MODEL_PATH")
def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest() if p.is_file() else None
import lm_eval
print(json.dumps({
    "arm": "$ARM", "profile": "$PROFILE", "run_id": "$RUN_ID",
    "checkpoint": str(ck), "node": os.environ.get("NODE_NAME"),
    "serve": {"engine": "sglang", "tp": $TP, "ctx": $CTX,
              "mem_fraction_static": $MEM_FRAC,
              "chunked_prefill_size": $CHUNKED_PREFILL,
              "kv_cache_dtype": "fp8_e4m3",
              "quant_args": "$QUANT_ARGS",
              "disable_shared_experts_fusion": True,
              "reasoning_parser": "$REASONING_PARSER",
              "tool_call_parser": "$TOOL_PARSER"},
    "lm_eval_version": lm_eval.__version__,
    "ported_from": "pipeline/slurm/glm52_quality_arm.sh",
    "digests": {f: sha(ck / f) for f in (
        "config.json", "tokenizer.json", "tokenizer_config.json",
        "chat_template.jinja", "generation_config.json")},
    "score_comparable_to_public_recipe": False,
    "score_comparability_note": (
        "Paired candidate-vs-peer comparison under one harness. Greedy decoding "
        "and an FP8 KV cache on both arms; MC tasks scored as loglikelihood with "
        "a chat template. Not comparable to any public leaderboard number, nor to "
        "PhalaCloud's own model-card table (temperature 1.0 / top_p 0.95)."),
}, indent=2))
PY
rc=$?; gate preflight "$rc"
[ "$rc" = 0 ] || { note "FATAL: preflight failed"; cat "$CLIENT/harness_manifest.json"; exit 1; }
note "manifest: $CLIENT/harness_manifest.json"

# ---- step 1: serve ---------------------------------------------------------
note "step 1: serve on SGLang"
(
  exec python -m sglang.launch_server \
    --model-path "$MODEL_PATH" \
    $QUANT_ARGS \
    --disable-shared-experts-fusion \
    --tp "$TP" \
    `# FP8 KV cache: KEPT. It is absent from the vendor's serving command, and it` \
    `# IS a quality variable (it quantizes attention K/V on top of the INT4 expert` \
    `# weights under test) -- but an FP8 KV cache is standard practice for` \
    `# production serving of this model class, so the measurement that matters is` \
    `# the one taken under it. Operator decision, 2026-08-31. Both arms carry it,` \
    `# so the ours-vs-peer delta is unaffected either way; what it changes is what` \
    `# the ABSOLUTE numbers describe, and they should describe the deployed` \
    `# configuration.` \
    `#` \
    `# It also doubles the KV pool: measured without it, SGLang computed` \
    `# max_total_num_tokens=101120 (16 concurrent reasoning requests sat at` \
    `# token_usage 0.42, no queue). With it, roughly 200k tokens -- headroom that` \
    `# matters when several hard items generate 20-30k tokens at once, which is` \
    `# the regime GPQA-CoT and ifeval enter at GENERAL_MAX_GEN_TOKS=32768.` \
    --kv-cache-dtype fp8_e4m3 \
    `#` \
    `# NO --speculative-algorithm: the MTP/nextn draft layer ships in BOTH` \
    `# checkpoints and the vendor command does enable EAGLE (accept length ~2.9),` \
    `# but this run measures the quantized model's own decode path, not a serving` \
    `# optimisation. Speculative decoding stays OFF by decision.` \
    --reasoning-parser "$REASONING_PARSER" \
    --tool-call-parser "$TOOL_PARSER" \
    --context-length "$CTX" \
    --mem-fraction-static "$MEM_FRAC" \
    `# KEPT, against the vendor command, on evidence: echo/loglikelihood requests` \
    `# compute logits for every prompt token and GLM-5.2's scheduler OOM'd at 0.85` \
    `# (job 13136). The vendor's command serves chat only and never exercises that` \
    `# path, so it cannot speak to this bound.` \
    --chunked-prefill-size "$CHUNKED_PREFILL" \
    `# --enable-metrics is a DEVIATION from the vendor command, added on purpose:` \
    `# without it SGLang serves no /metrics endpoint, scrape_metrics records` \
    `# "(scrape failed)", and the run has no token-spend column -- one of the` \
    `# three axes the M3 report uses. It only starts a Prometheus exporter, does` \
    `# not touch the inference path, and both arms carry it.` \
    --enable-metrics \
    --trust-remote-code \
    --host 0.0.0.0 --port "$PORT"
) > "$CLIENT/serve.log" 2>&1 &
SERVER_PID=$!
echo "$SERVER_PID" > "$CLIENT/serve.pid"

# 270*10s = 45 min. cephfs pages in a 394 GB checkpoint at roughly 20-30 MB/s
# per stream and SGLang mmaps the shards, so "loading shards: 100%" is followed
# by a long silent page-in with no log line at all (see the cluster notes in
# BUGS_AND_FIXES.md). Do not shorten this on the assumption the server hung.
healthy=1
for i in $(seq 1 270); do
  kill -0 "$SERVER_PID" 2>/dev/null || { note "server died during startup"; break; }
  curl -sf "http://localhost:$PORT/health_generate" >/dev/null 2>&1 && { healthy=0; break; }
  sleep 10
done
gate serve_healthy "$healthy"
[ "$healthy" = 0 ] || { tail -60 "$CLIENT/serve.log"; exit 1; }
note "server healthy after ~$((i*10))s"

# ---- step 1b: decode throughput floor --------------------------------------
# Measured BEFORE committing to a multi-hour suite, because throughput is not a
# cosmetic property here: lm-eval's api backend applies a 300 s TOTAL per-request
# deadline (`timeout=300`, aiohttp ClientTimeout) and, in 0.4.10, a timeout is
# NOT survivable -- the handler that should retry reads an unbound `outputs` and
# raises UnboundLocalError, killing the invocation. Run
# full7-20260831t054731z died that way at ifeval 519/541 after 60 minutes of
# completed inference, at ~28 output tok/s per stream.
#
# So the arm now measures aggregate decode throughput with a short burst and
# refuses to start the suite below a floor. One minute here replaces an hour of
# inference that cannot be scored.
THROUGHPUT_FLOOR="${THROUGHPUT_FLOOR:-250}"
THROUGHPUT_PROBE_N="${THROUGHPUT_PROBE_N:-8}"
note "step 1b: decode throughput probe (${THROUGHPUT_PROBE_N} concurrent x 256 tokens)"
tp_before=$(curl -sf "http://localhost:$PORT/metrics" 2>/dev/null \
  | awk '/^sglang:generation_tokens_total/ {print $2; exit}')
tp_t0=$(date +%s)
tp_pids=""
for _ in $(seq 1 "$THROUGHPUT_PROBE_N"); do
  # --max-time is mandatory: an unbounded curl here would hang the arm forever.
  curl -sf --max-time 180 "http://localhost:$PORT/v1/chat/completions" \
    -H 'Content-Type: application/json' \
    -d '{"model":"m","messages":[{"role":"user","content":"Count from 1 to 300, one number per line."}],"max_tokens":256,"temperature":0}' \
    >/dev/null 2>&1 &
  tp_pids="$tp_pids $!"
done
# Wait for THESE pids only, never a bare `wait`. The SGLang server is itself a
# background child of this script, so bare `wait` blocks on a process that never
# exits -- it hung an arm for 64 minutes with the server idle and healthy.
# shellcheck disable=SC2086
wait $tp_pids 2>/dev/null || true
tp_t1=$(date +%s)
tp_after=$(curl -sf "http://localhost:$PORT/metrics" 2>/dev/null \
  | awk '/^sglang:generation_tokens_total/ {print $2; exit}')
tp_rate=$("$BVENV/bin/python" - "$tp_before" "$tp_after" "$tp_t0" "$tp_t1" <<'PY'
import sys
try:
    b, a, t0, t1 = float(sys.argv[1]), float(sys.argv[2]), float(sys.argv[3]), float(sys.argv[4])
    dt = max(t1 - t0, 1.0)
    print("%.0f" % ((a - b) / dt))
except Exception:
    print("0")
PY
)
note "decode throughput: ${tp_rate} tok/s aggregate over $((tp_t1 - tp_t0))s (floor ${THROUGHPUT_FLOOR})"
if [ "${tp_rate:-0}" -lt "$THROUGHPUT_FLOOR" ] 2>/dev/null; then
  gate throughput 1
  note "FATAL: ${tp_rate} tok/s is below the ${THROUGHPUT_FLOOR} tok/s floor. At this"
  note "  rate a 32768-token item needs $((32768 / (${tp_rate:-1} > 0 ? ${tp_rate:-1} : 1) )) s and will trip lm-eval's 300 s"
  note "  client deadline, which 0.4.10 turns into a fatal UnboundLocalError."
  note "  Investigate the serve config (DeepGEMM, mem-fraction, batching) rather"
  note "  than paying for a suite that cannot finish. Override with THROUGHPUT_FLOOR=0."
  grep -aiE "deepgemm|fallback|not compiled|nvcc|compiler" "$CLIENT/serve.log" | tail -10
  exit 1
fi
gate throughput 0

# ---- step 2: loglikelihood-shape gate --------------------------------------
note "step 2: loglikelihood-shape gate (the EXACT lm-eval payload: max_tokens=1 echo logprobs)"
# NOT probe_endpoint's text_offset probe: that sends max_tokens=0, which SGLang
# 400s while vLLM accepts — a probe artifact, not the eval path (job 13128).
curl -s "http://localhost:$PORT/v1/completions" -H 'Content-Type: application/json' \
  -d '{"model":"glm","prompt":"The capital of France is Paris. The capital of Germany is","max_tokens":1,"echo":true,"logprobs":5,"temperature":0.0}' \
  > "$CLIENT/ll-gate.json"
"$BVENV/bin/python" -c "
import json
d=json.load(open('$CLIENT/ll-gate.json'))
lp=d['choices'][0]['logprobs']
assert lp['tokens'] and lp['token_logprobs'], 'echo logprobs missing'
print('[ll-gate] ok:', len(lp['tokens']), 'prompt tokens with logprobs')
"
cap=$?; gate ll_shape "$cap"
if [ "$cap" != 0 ]; then
  note "FATAL: /completions echo+logprobs (lm-eval loglikelihood shape) unsupported"
  cat "$CLIENT/ll-gate.json"
  kill "$SERVER_PID" 2>/dev/null; exit 1
fi

# ---- step 2b: reasoning-separation probe (non-gating provenance) -----------
# The reasoning parser decides what the SERVER strips before the scorer sees a
# response; the GLM-5.2 report notes that a budget-exhausted, answer-less
# response scores zero precisely because glm45 strips the CoT. Whether glm45 is
# right for GLM-5.3 is not knowable offline, so record what the server actually
# did. Recorded, never inferred — and non-gating, because an unparsed CoT
# affects both arms identically as long as they serve with the same parser.
curl -s "http://localhost:$PORT/v1/chat/completions" -H 'Content-Type: application/json' \
  -d '{"model":"glm","messages":[{"role":"user","content":"What is 17*23? Answer with the number only."}],"max_tokens":512,"temperature":0.0}' \
  > "$CLIENT/reasoning-probe.json"
# Quoted heredoc + argv, so neither the JSON nor the probe code is subject to
# shell expansion (the '</think>' literal and the f-strings below would otherwise
# be at the mercy of it).
"$BVENV/bin/python" - "$CLIENT/reasoning-probe.json" <<'PY' 2>&1 | tee -a "$CLIENT/client.log"
import json, sys
d = json.load(open(sys.argv[1], encoding="utf-8"))
m = d["choices"][0]["message"]
print("[reasoning-probe] message keys:", sorted(m))
for k in ("reasoning", "reasoning_content"):
    v = m.get(k)
    print("[reasoning-probe] %s: %s" % (
        k, "absent" if v is None else "%d chars" % len(v)))
c = m.get("content") or ""
print("[reasoning-probe] content[:200]:", repr(c[:200]))
# A leaked marker means the server did NOT strip the CoT, so the scorer would see
# reasoning text as the answer. Same on both arms, hence recorded not gated.
print("[reasoning-probe] think_marker_leaked:", "</think>" in c or "<think>" in c)
print("[reasoning-probe] finish_reason:", d["choices"][0].get("finish_reason"))
print("[reasoning-probe] usage:", json.dumps(d.get("usage", {})))
PY

# Full capability probe for provenance (non-gating).
( cd "$BENCH" && BASE_URL="http://localhost:$PORT" "$BVENV/bin/python" -m quality.probe_endpoint \
    --profile "$PROFILE" --capabilities ) >"$CLIENT/capabilities.txt" 2>&1 || true

# ---- step 3: general suite -------------------------------------------------
note "step 3: general suite (standalone orchestrator, ${RUN_ID})"
# BASELINE_REF="" -> standalone mode (no live baseline in the sequential design;
# deltas/report rebuilt offline via quality.rebuild_delta after both arms finish).
LIMIT_ARGS=""
if [ -n "${LIMIT:-}" ]; then
  LIMIT_ARGS="--limit $LIMIT"
  note "LIMIT=$LIMIT -> DIAGNOSTIC/NON-FORMAL run (the orchestrator labels it so)"
fi
SUITE_ARGS=""
[ -n "${SUITE:-}" ] && SUITE_ARGS="--suite $SUITE"
# Thinking ON is the M3/GLM-5.2 protocol and it is load-bearing, not cosmetic:
# --reasoning-mode threads the profile's THINK_ON_EXTRA_BODY
# ({"chat_template_kwargs": {"enable_thinking": true}}) into every generative
# request. It also means an answer-less, budget-exhausted response scores zero,
# which is exactly how M3's AWQ non-termination pathology became visible.
REASONING_MODE_ARGS=""
[ -n "${REASONING_MODE:-}" ] && REASONING_MODE_ARGS="--reasoning-mode $REASONING_MODE"

# TOKEN SPEND. The benchmarks repo at this ref has no usage-proxy capture -- the
# per-request accounting behind M3's "token spend 2.19x BF16" and the GLM-5.2
# report's exhaustion rates is not in the pushed tree. SGLang's Prometheus
# endpoint is a coarse substitute that needs no benchmarks change: scraping it
# either side of the suite gives per-arm TOTALS (generated tokens, request
# counts), which is the axis that mattered most. It is aggregate, not
# per-request, so it cannot attribute spend to a task or an item -- do not
# report it as if it could.
scrape_metrics() {
  curl -sf "http://localhost:$PORT/metrics" 2>/dev/null \
    | grep -aE "^sglang:(prompt_tokens_total|generation_tokens_total|num_requests_total|num_aborted_requests_total|cached_tokens_total)" \
    > "$CLIENT/metrics-$1.txt" || echo "(scrape failed)" > "$CLIENT/metrics-$1.txt"
  note "metrics[$1]: $(tr '\n' ' ' < "$CLIENT/metrics-$1.txt" | cut -c1-200)"
}
scrape_metrics before

( cd "$BENCH" && BASE_URL="http://localhost:$PORT" BASELINE_REF="" PATH="$BVENV/bin:$PATH" \
    "$BVENV/bin/python" -m quality.orchestrator \
    --profile "$PROFILE" --out-root "$ROOT/results" --run-id "$RUN_ID" \
    $SUITE_ARGS $LIMIT_ARGS $REASONING_MODE_ARGS --execute ) >"$CLIENT/general.log" 2>&1
rc=$?; gate general_suite "$rc"
scrape_metrics after
tail -30 "$CLIENT/general.log" | tee -a "$CLIENT/client.log"

# ---- step 4: shutdown ------------------------------------------------------
note "step 4: shutdown"
kill "$SERVER_PID" 2>/dev/null; sleep 10; kill -9 "$SERVER_PID" 2>/dev/null
note "arm done rc=$rc"
exit $rc
