# GLM-5.3 AA quality rerun — handoff

- Protocol version: 1
- Task: rerun AA GPQA Diamond on the two-node GLM-5.3 serve, and make the
  truncation hypothesis testable
- Packet revision: 2026-09-11-r1
- Planner owner: planner session 2026-09-11 (local, CPU-only, no cluster access)
- Base Git commit: `2e144c1843414dfc44c51594d048cb6c4aa52ed5` (branch `duy-branch`)
- Active packet: **Packet A** below (state `READY_FOR_EXECUTOR`, CPU-only, no GPU)
- Packet B (the GPQA rerun) is **drafted but NOT authorized** — it is blocked on
  Packet A's answer.

Read `CLAUDE.md`, `PLANNER_EXECUTOR_PROTOCOL.md` and
`docs/status/2026-09-11-glm53-aa-gpqa-and-three-objectives.md` first. This
handoff supersedes the serving assumptions in
`docs/glm53-aa-v42-public-evaluation-plan.md` (see "Stale claims" at the end);
it does not supersede that document's benchmark definitions.

---

## 1. What this session established (all measured, not inferred)

### 1.1 The two serving nodes are ONE endpoint, not two

`ca-gpu02` + `ca-gpu03`, namespace `evaluation`, pods
`glm-5-3-w4afp8-sglang-gpu02` (label `role: head`) and `…-gpu03`
(`role: worker`), 8 GPUs each, up 23 h as of 2026-09-11.

They are a single SGLang server: `--tp 16 --nnodes 2`, gpu02 is
`--node-rank 0` at `10.132.254.16:25000`, gpu03 is rank 1. Service
`glm-5-3-w4afp8-sglang` selects `role: head` only, so there is exactly one
HTTP endpoint (`:30000`), and its `Endpoints` object lists only gpu02.

**Consequence:** the status doc's plan that "two endpoints means both arms run
at once instead of back to back, so the 83 h becomes roughly 46 h"
(`docs/status/2026-09-11-…md` lines 165–167) does not hold. There is one server
serving one checkpoint. The wall-clock win available here comes from tp=16 +
speculative decoding + higher concurrency, not from running paired arms
concurrently.

### 1.2 It serves OUR checkpoint; there is no Phala arm anywhere

```
--model-path /mnt/cephfs/hoangduy/results/glm53-w4afp8-mtp/checkpoint
--served-model-name glm-5.3-w4afp8
```

Same path as the Sep-11 in-house arm
(`docs/superpowers/specs/2026-09-02-glm53-aa-gpqa-v3-design.md` line 20).
`/v1/models` returns `glm-5.3-w4afp8`, `max_model_len` 1048576. No PhalaCloud
checkpoint is served on any node. A paired rerun would require standing up a
second 8- or 16-GPU serve, and **no capacity exists** (§1.6).

### 1.3 Live serve config, from `/get_server_info`

| Field | Value |
|---|---|
| `tp_size` | 16 |
| `max_total_num_tokens` | **598848** |
| `context_length` | 1048576 |
| `max_prefill_tokens` | 16384 |
| `chunked_prefill_size` | 2048 |
| `mem_fraction_static` | 0.80 |
| `kv_cache_dtype` | fp8_e4m3 |
| `quantization` | w4afp8 |
| `speculative_algorithm` | EAGLE (`num_steps` 3, `eagle_topk` 1, `num_draft_tokens` 4) |
| `max_running_requests` | **16** |
| `reasoning_parser` | glm45 |
| SGLang version | 0.5.17 |
| `disable_radix_cache` | False |

Also: `hostNetwork: true`, `NCCL_IB_DISABLE=0`,
`NCCL_IB_HCA=^mlx5_4,mlx5_8`, `NCCL_SOCKET_IFNAME=en0`. No
`rdma/hca_shared_ib` resource request and no `IPC_LOCK` capability — IB is
reached via the host network stack instead of the device plugin.

**`context_length` 1048576 is NOT backed by 1M of KV.** The args request
`--max-total-tokens 1048576`; the engine returned 598848. 598,848 tokens is the
real ceiling and it cannot be raised without restarting the collaborator's
server at a higher `mem_fraction_static` (0.85–0.90 would reach only ~750–830K).

### 1.4 Measured throughput and EAGLE accept length

Endpoint is healthy and was **completely idle** — 23 h of logs contain only
`GET /health 200`. Verified with a live generation (HTTP 200, 1.15 s,
`finish_reason: stop`, `reasoning_content` populated, glm45 parser working).

Concurrency sweep via `/generate`, `ignore_eos: true`, 600 tokens/request,
30-token prompts:

| Concurrency | Aggregate | Per-stream | vs conc 1 |
|---:|---:|---:|---:|
| 1 | 90.8 tok/s | 90.9 | 1.00× |
| 2 | 160.2 tok/s | 81.0 | 1.76× |
| 4 | 275.6 tok/s | 72.2 | 3.04× |
| 8 | 421.2 tok/s | 54.2 | 4.64× |

EAGLE accept length from server logs, by batch size:

```
running-req=1   accept_len 2.38–2.70
running-req=2   accept_len 2.23–2.75
running-req=4   accept_len 2.16–2.76
running-req=8   accept_len 2.31–2.80
```

**Accept length does not degrade from batch 1 to 8.** EAGLE accepts ~2.5 of 4
draft tokens throughout. There is therefore *no speculative-decoding argument
for low concurrency* in this range — the usual verification-goes-compute-bound
effect has not kicked in at this model size and W4AFP8 weight footprint.

90.8 tok/s single-stream also settles the IB question: cross-node TP-16 on a
TCP fallback could not reach that. The `hostNetwork` + `NCCL_IB_HCA` setup
works. Treat the IB path as validated for this deployment (this supersedes the
"unvalidated" note in `../RANCHER_GPU_ACCESS.md` for these two pods only).

Caveat: the sweep used 30-token prompts and 600-token generations. At 26K–131K
context the attention term grows and decode slows materially. Treat these as
upper bounds on throughput.

### 1.5 Concurrency decision: **4** at the AA-standard 131,072 cap

The binding constraint is the 598,848-token KV pool, not spec decoding.

| `max_new_tokens` | Per full-cap req | Full-cap traces the pool holds |
|---:|---:|---:|
| 131,072 (AA std) | ~133K | **4.5** |
| 262,144 | ~264K | 2.3 |
| 524,288 | ~526K | **1.14** |

At concurrency 4, each slot gets 149,712 tokens — enough for a complete 131,072
trace plus prompt, so **retraction is structurally impossible**, not merely
unlikely.

Concurrency 8 over-commits. Long requests accumulate in the batch because short
ones retire fast: from the Sep-11 distribution, 120 × 131,072 = 15.7M of ~26.2M
total token-steps, so near-cap requests hold **~60% of all slot-time**, not
12%. At conc 8 that is ~4.8 of 8 slots near-cap → ~629K > 598,848. Retraction
would hit precisely in the long-trace regime that dominates wall clock.
Concurrency 6 fits (~501K, ~16% headroom) if someone wants to push.

Concurrency 2 is a bad trade: 12% better per-request latency for 72% worse
aggregate throughput — roughly 19 extra hours on a 990-completion run.

Rough wall clock for ~26.2M generated tokens (optimistic, see §1.4 caveat):
conc 2 ≈ 45 h, **conc 4 ≈ 26 h**, conc 6 ≈ 21 h. Sep-11 baseline was 37.1 h at
conc 8 / tp=8 / no spec dec.

### 1.6 Cluster capacity as of 2026-09-11

`scripts/gpu-free.sh --verify` (both witnesses agree; pool has grown to **9**
nodes, 72 GPUs):

```
gpu01–gpu08: 8/8 used     gpu09: 6/8 used
TOTAL FREE: 2 of 72       fully-free nodes: 0
```

`kernels` holds 5 whole nodes (`gpu07-hold7`, `gpu08-hold7`, `spec-dev-0/1/2`).
No 8-GPU pod can schedule. Also: **the EP GPTQ gate pod is gone** —
`glm53-ep-gptq-20260910t120700z` no longer exists and there are **zero Pending
pods** in `evaluation`. It never ran (it was Pending 17 h), so no work was lost,
but it will not run unless resubmitted. The gpu03 hold is also gone; gpu03 is
now the serving worker.

### 1.7 Where the Sep-11 numbers actually came from — and the real data gap

The status doc says the harness "kept only aggregates, no per-item scores"
(line 46) yet reports "120/990 hit the 131,072-token cap" (line 38). Both are
true, because they come from different streams.

`scrape_metrics` (`pipeline/k8s/glm53_quality_arm.sh` lines 478–483) snapshots
SGLang's Prometheus endpoint before/after the AA run, capturing exactly:

```
prompt_tokens_total  generation_tokens_total  num_requests_total
num_aborted_requests_total  cached_tokens_total
```

- **Avg completion tokens 26,463** = `generation_tokens_total` delta ÷ 990.
- **990/990 successful, all HTTP 200** = `num_requests_total` and
  `num_aborted_requests_total` deltas.
- **The cap count is NOT derivable from those five**, and SGLang exposes no
  finish-reason counter at all (verified against the live `/metrics`: no such
  series exists). The only available source is
  `sglang:generation_tokens_histogram_bucket`, whose edges include
  `le=100000` and `le=200000`. With a 131,072 cap,
  `bucket(200000) − bucket(100000)` counts completions in 100,001–131,072.
  That is the 120.

Two consequences:

1. **"120" is a slight over-count.** A request that generated 105K tokens and
   stopped *naturally* falls in the same bucket. So 12.1% is an upper bound and
   the 87.9% ceiling is correspondingly a little pessimistic.
2. **Confirmation that per-item scores were genuinely absent.** 87.9% is exactly
   870/990 = (990 − 120)/990 — "assume all 120 cap-hits wrong, all 870 others
   right." That bound needs only the *count*. Anyone holding per-item scores
   would have reported the actual score among truncated items instead.

**The precise gap is not missing data, it is a missing join key:**

| Stream | Has | Lacks |
|---|---|---|
| SGLang metrics | request-level completion lengths | any item identity |
| AA harness output | aggregate pass@1 | per-item correctness |

The histogram knows 120 requests ran near the cap but not *which* of the 198
questions or 5 repeats. The score knows 81.82% overall. Neither joins to the
other. **That, and only that, is what makes the truncation hypothesis
untestable.**

Implementation note for any rerun: `scrape_metrics` must be extended to
snapshot the histogram *buckets*, not just the five scalar counters. The
histogram is cumulative since server boot (the live one already holds 7,662
requests), so it must be diffed before/after.

### 1.8 The 512K cap was considered and is not recommended

The user proposed raising `max_new_tokens` to 512K. Analysis:

- At a 512K cap the pool holds **1.14** full-cap traces. One near-cap request
  alone occupies 87.5% of the pool, leaving ~74,560 tokens for everything else.
  No concurrency makes retraction structurally impossible; the hard-guarantee
  answer is concurrency **1** (90.8 tok/s).
- Three effects compound: occupancy squeeze, the long request also decoding
  slowest (attention grows with context), and convoy collapse as short requests
  retire and the batch drains to the straggler.
- Total work grows sharply. From the Sep-11 distribution (870 short averaging
  ~12.0K, 120 at the cap): ~26.2M tokens at the 131K cap → ~46.5M if the 120
  average 300K → ~73.4M if they mostly run to 512K. At a 512K cap **~80% of all
  generated tokens live in ~12% of requests**, and those are exactly the
  requests the pool can run about one at a time.
- Untested prior worth checking first: traces that exceed 131K on GPQA are
  frequently **degenerate repetition loops**, not productive reasoning. Those
  consume whatever cap they are given. If most of the 120 are loops, a larger
  cap buys zero score at very high cost.
- Raising the cap also forfeits comparability: 131,072 is part of the
  `gpqa_diamond_aa_v3` recipe, and the whole point of the packaged harness is
  that we wrote no prompt, no extraction, and no budget of our own.

If a raised cap is wanted for the full set anyway, **256K is the defensible
number** — the pool holds 2.3 full-cap traces so concurrency 2 is guaranteed,
and `docs/glm53-aa-v42-public-evaluation-plan.md` line 159 already names 256K
as what AA-LCR output-policy parity requires. It is a number with independent
justification rather than one derived from the advertised context length.

### 1.9 A plan this session proposed and then retracted

An earlier draft proposed "Phase 2: re-run only the ~120 truncating items at a
raised cap." **That is not executable** — per §1.7 the existing artifacts do not
say which items truncated. Per-item capture is a hard prerequisite for any
targeted probe, not a parallel nice-to-have. This is why Packet A comes first.

---

## 2. Packet A (ACTIVE) — artifact inventory, CPU-only

# Execution packet: Sep-11 AA GPQA artifact inventory

- Protocol version: 1
- State: READY_FOR_EXECUTOR
- Packet revision: 2026-09-11-r1
- Planner owner: planner session 2026-09-11
- Intended executor: any executor with `evaluation` namespace access
- Base Git commit: `2e144c1843414dfc44c51594d048cb6c4aa52ed5`
- Decision question: **Can per-item correctness be joined to per-request
  completion length for the Sep-11 AA GPQA run — either from artifacts already
  retained, or from a rerun requiring only configuration changes?**

### Objective and hypothesis

The truncation hypothesis ("truncation is probably most of the gap to AA's
91.7%") has been unverified since Sep-11 solely because per-item correctness
and per-request completion length cannot be joined (§1.7).

Hypothesis to test: **upstream OpenAI `simple-evals` normally writes per-example
records plus an HTML report, and `nvidia-simple-evals` wraps that task — so the
"only aggregates" claim may itself be an untested assumption about output that
was written and never inspected.**

If per-sample records are present, the truncation question is answerable with
**zero GPU time** and Packet B may not be needed at all. This is the cheapest
possible next action and it strictly gates everything downstream. Per the repo
prime directive it runs *before* any capture code is designed.

### Scope and non-goals

- In scope: read-only inventory of the Sep-11 AA artifact tree; read-only
  inspection of the installed `nvidia-simple-evals==26.3` package to determine
  what per-sample output the task is *capable* of writing and under which
  config key.
- Not authorized: any GPU allocation; any write to the results tree; launching
  any evaluation; touching the `glm-5-3-w4afp8-sglang-gpu02/03` pods or the
  collaborator's serve in any way; modifying `scrape_metrics` or the arm script.

### Preconditions and exact environment

- Cluster: Rancher `c-bk8md` (`infermesh-test`), namespace `evaluation`.
  See `../RANCHER_GPU_ACCESS.md`.
- No GPU request. Tolerates the GPU taint because `model-cache-shared` is
  region-pinned to ca-van3 where every node is GPU-tainted (same rationale as
  `pipeline/k8s/stage-glm53-aa-v3.yaml`).
- Required environment variables: none.
- Required check before launch: `kubectl -n evaluation get pods` must succeed
  (a `401 Unauthorized` means the kubeconfig token expired — regenerate per
  `../RANCHER_GPU_ACCESS.md` and redo the two setup fixes).

### Required inputs

| Input | Exact path or identifier | Required validation |
| --- | --- | --- |
| AA artifact root | `/mnt/cephfs/hoangduy/results/glm53-quality-paired/20260902t0431z` | `test -d` → exit 0 |
| AA venv | `/mnt/cephfs/hoangduy/venvs/nvidia-simple-evals-26.3` | `test -x <venv>/bin/python` → exit 0 |
| PVC | `model-cache-shared` mounted at `/mnt/cephfs` | mount present in pod |

Do **not** assume the leaf directory name `results-formal-198-c8`. The arm
script sets `--output-dir "$AA_OUT/results"` while the status doc records
`results-formal-198-c8`, so the leaf was overridden. Inventory the whole
`aa-gpqa-v3/` subtree.

### Workspace policy

- Protected paths: the entire
  `/mnt/cephfs/hoangduy/results/glm53-quality-paired/` tree is **read-only** for
  this packet.
- Permitted untracked roots: `/mnt/cephfs/hoangduy/runlogs/` (log output only).
- Stop: if the artifact root does not exist, or if the venv is missing — return
  that fact as the answer rather than reconstructing anything.

### Resource contract

- Nodes: 1 (any; CPU-only pod, no GPU request)
- GPUs per node: **0**
- Exclusivity: shared
- Time limit: 00:30:00 (`activeDeadlineSeconds: 1800`)
- Expected runtime: 2–10 minutes

### Commands

#### Setup and revision verification

```bash
cd /path/to/llm-compressor
git fetch --all --quiet
git rev-parse HEAD
# expect: 2e144c1843414dfc44c51594d048cb6c4aa52ed5 (or record the actual SHA)
kubectl -n evaluation get pods >/dev/null && echo "kubeconfig OK"
```

#### Launch

Write this manifest to `pipeline/k8s/hd-aa-artifact-inventory.yaml` and apply
it. It is modelled on `pipeline/k8s/stage-glm53-aa-v3.yaml` (same image,
tolerations, PVC and log convention) and requests no GPU.

```yaml
---
apiVersion: batch/v1
kind: Job
metadata:
  name: hd-aa-artifact-inventory
  namespace: evaluation
  labels: {owner: hoangduy.dang, purpose: artifact-inventory}
spec:
  ttlSecondsAfterFinished: 86400
  backoffLimit: 0
  activeDeadlineSeconds: 1800
  template:
    metadata:
      labels: {owner: hoangduy.dang, purpose: artifact-inventory}
    spec:
      restartPolicy: Never
      tolerations:
        - {key: nvidia.com/gpu, operator: Exists, effect: NoSchedule}
      containers:
        - name: inventory
          image: lmsysorg/sglang:v0.5.17
          imagePullPolicy: IfNotPresent
          env:
            - {name: PYTHONUNBUFFERED, value: "1"}
            - name: NODE_NAME
              valueFrom: {fieldRef: {fieldPath: spec.nodeName}}
          resources:
            requests: {cpu: "2", memory: "8Gi"}
            limits: {cpu: "4", memory: "16Gi"}
          command: ["bash", "-lc"]
          args:
            - |
              set -uo pipefail
              ROOT=/mnt/cephfs/hoangduy/results/glm53-quality-paired/20260902t0431z
              VENV=/mnt/cephfs/hoangduy/venvs/nvidia-simple-evals-26.3
              mkdir -p /mnt/cephfs/hoangduy/runlogs
              LOG=/mnt/cephfs/hoangduy/runlogs/hd-aa-artifact-inventory.log
              exec > >(tee -a "$LOG") 2>&1
              echo "==> durable log: $LOG"
              echo "==> node=${NODE_NAME:-unknown} date=$(date -u +%FT%TZ)"

              echo "=== [1] artifact root ==="
              test -d "$ROOT" || { echo "FATAL: missing $ROOT"; exit 10; }
              for ARM in ours phala; do
                D="$ROOT/client-$ARM/aa-gpqa-v3"
                echo "--- arm=$ARM dir=$D ---"
                test -d "$D" || { echo "  ABSENT"; continue; }
                find "$D" -type f -printf '%10s  %p\n' | sort -k2
              done

              echo "=== [2] per-sample candidates ==="
              find "$ROOT" -type f \( -name '*.jsonl' -o -name '*sample*' \
                -o -name '*allresults*' -o -name '*.html' -o -name '*result*' \) \
                -printf '%10s  %p\n' | sort -k2

              echo "=== [3] structure of every json/yml/yaml (keys only) ==="
              find "$ROOT" -type f \( -name '*.json' -o -name '*.yml' -o -name '*.yaml' \) \
                -print0 | while IFS= read -r -d '' F; do
                echo "--- $F ($(stat -c%s "$F") bytes) ---"
                python3 - "$F" <<'PY'
              import json, sys
              p = sys.argv[1]
              try:
                  if p.endswith(('.yml', '.yaml')):
                      import yaml; d = yaml.safe_load(open(p))
                  else:
                      d = json.load(open(p))
              except Exception as e:
                  print("  UNPARSEABLE:", type(e).__name__, e); sys.exit(0)
              def shape(o, depth=0, path=""):
                  pad = "  " * (depth + 1)
                  if isinstance(o, dict):
                      for k in list(o)[:40]:
                          v = o[k]
                          t = type(v).__name__
                          n = f" len={len(v)}" if isinstance(v, (list, dict, str)) else ""
                          print(f"{pad}{k}: {t}{n}")
                          if depth < 2 and isinstance(v, (dict, list)):
                              shape(v, depth + 1, f"{path}.{k}")
                  elif isinstance(o, list) and o:
                      print(f"{pad}[0] -> {type(o[0]).__name__}")
                      if depth < 2:
                          shape(o[0], depth + 1, path + "[0]")
              shape(d)
              PY
              done

              echo "=== [4] first record of every jsonl ==="
              find "$ROOT" -type f -name '*.jsonl' -print0 | while IFS= read -r -d '' F; do
                echo "--- $F lines=$(wc -l < "$F") ---"
                head -c 4000 "$F"; echo
              done

              echo "=== [5] JOIN TEST: per-item correctness + length present? ==="
              python3 - "$ROOT" <<'PY'
              import json, os, sys
              root = sys.argv[1]
              CORRECT = {"correct","score","is_correct","passed","acc","exact_match","grade"}
              IDENT   = {"question_id","item_id","id","index","doc_id","question","idx"}
              LENGTH  = {"finish_reason","completion_tokens","stop_reason",
                         "num_generated_tokens","usage","output_tokens"}
              hits = []
              for dp, _, fns in os.walk(root):
                  for fn in fns:
                      if not fn.endswith((".json", ".jsonl")):
                          continue
                      p = os.path.join(dp, fn)
                      try:
                          if fn.endswith(".jsonl"):
                              with open(p) as fh:
                                  recs = [json.loads(l) for l in fh if l.strip()][:5]
                          else:
                              d = json.load(open(p))
                              recs = d if isinstance(d, list) else [d]
                              recs = [r for r in recs if isinstance(r, dict)][:5]
                      except Exception:
                          continue
                      keys = set()
                      for r in recs:
                          keys |= set(r)
                          for v in r.values():
                              if isinstance(v, dict):
                                  keys |= set(v)
                      c, i, l = keys & CORRECT, keys & IDENT, keys & LENGTH
                      if c or l:
                          hits.append((p, sorted(c), sorted(i), sorted(l)))
              if not hits:
                  print("VERDICT: NO per-item correctness or length fields found.")
                  print("  -> Packet B (rerun with per-item capture) IS required.")
              else:
                  print("VERDICT: candidate per-item records FOUND:")
                  for p, c, i, l in hits:
                      print(f"  {p}\n    correctness={c} identity={i} length={l}")
                  print("  -> If one file carries correctness AND identity, the")
                  print("     hypothesis may be answerable with ZERO GPU time.")
              PY

              echo "=== [6] SHA-256 + size of every artifact ==="
              find "$ROOT" -type f -print0 | xargs -0 sha256sum

              echo "=== [7] what simple-evals CAN write (rerun feasibility) ==="
              if [ -x "$VENV/bin/python" ]; then
                "$VENV/bin/python" -c "
              import importlib.util as u, os, pathlib
              for m in ('simple_evals','core_evals.simple_evals'):
                  s = u.find_spec(m)
                  print(m, '->', s.origin if s else 'NOT FOUND')
                  if s and s.origin:
                      print('  dir:', os.path.dirname(s.origin))
              "
                echo "--- framework.yml (gpqa_diamond_aa_v3 block) ---"
                find "$VENV" -name framework.yml -print -exec \
                  grep -n -A 40 "gpqa_diamond_aa_v3" {} \;
                echo "--- per-sample / output writers in the package ---"
                SE=$(find "$VENV" -type d -name simple_evals | head -1)
                [ -n "$SE" ] && grep -rn \
                  "allresults\|htmls\|jsonl\|single_eval_result\|convo\|to_json\|output_path\|metadata" \
                  "$SE" --include=*.py | head -60
              else
                echo "WARN: $VENV/bin/python absent — rerun feasibility UNDETERMINED"
              fi
              echo "==> inventory complete rc=0"
```

```bash
kubectl apply -f pipeline/k8s/hd-aa-artifact-inventory.yaml
```

#### Monitoring

```bash
kubectl -n evaluation get job/hd-aa-artifact-inventory -w
kubectl -n evaluation logs -f job/hd-aa-artifact-inventory
```

#### Aggregation and packaging

```bash
kubectl -n evaluation logs job/hd-aa-artifact-inventory \
  > docs/evidence/2026-09-11-aa-artifact-inventory.log
kubectl -n evaluation delete -f pipeline/k8s/hd-aa-artifact-inventory.yaml
```

### Success gates and expected artifacts

- Gate: section `[5] JOIN TEST` prints an explicit `VERDICT:` line.
- Gate: section `[6]` lists a SHA-256 for every file under the artifact root.
- Gate: section `[7]` either names the `simple_evals` module path and its
  per-sample writers, or explicitly reports the venv absent.
- Expected artifact: `docs/evidence/2026-09-11-aa-artifact-inventory.log`
  (committed), plus the durable copy at
  `/mnt/cephfs/hoangduy/runlogs/hd-aa-artifact-inventory.log`.

### Allowed adaptations

- If `python3` is absent from the image, substitute the venv interpreter
  (`/mnt/cephfs/hoangduy/venvs/nvidia-simple-evals-26.3/bin/python`) and record
  the substitution.
- If `yaml` is unavailable for section [3], report the `.yml` files as
  UNPARSEABLE and continue. Do not install packages into the PVC venv.

### Pre-authorized record-and-proceed conditions

- `client-phala/aa-gpqa-v3` absent while `client-ours` is present: record and
  continue with the `ours` arm only.

### Pre-authorized retries

- Trigger: pod evicted or `Pending` > 10 min (the cluster is full; a CPU pod
  should still schedule on a cpuworker).
- Maximum retry count: 2. Fresh run ID required: no. Inputs unchanged: all.

### Stop-and-return conditions

- Artifact root absent (exit 10) — return that as the finding.
- Any write attempt to the results tree, or any pod in `Error` — stop, return
  logs.
- `401 Unauthorized` from kubectl — stop, report token expiry.

### Prohibited actions

- Requesting any GPU.
- Touching, restarting, exec'ing into, or port-forwarding to
  `glm-5-3-w4afp8-sglang-gpu02` / `-gpu03`, or altering the collaborator's
  serve. These are **not ours** — read-only `get`/`logs` only, and only if
  needed. (Repo standing rule: on this shared cluster, delete only pods you
  created; never exec into someone else's without asking.)
- Deleting or modifying anything under `glm53-quality-paired/`.
- Launching any evaluation, canary, or GPQA run. Packet B is not authorized.

### Return contract

- Commit `docs/evidence/2026-09-11-aa-artifact-inventory.log`.
- Report the `[5] VERDICT` line verbatim.
- Report, as a table: every per-sample candidate file with byte size, SHA-256,
  and which of {identity, correctness, length} fields it carries.
- Report whether `simple_evals` can emit per-sample records and the exact config
  key that controls it — this decides whether Packet B needs new code or only a
  config change.
- Record the actual Git SHA used and the pod's node name.

### Final instruction

Commit and push the evidence packet, set the state to `RETURNED_FOR_ANALYSIS`,
and stop. Do not launch Packet B.

---

## 3. Packet B (DRAFTED, NOT AUTHORIZED) — AA GPQA rerun

State: `PLANNER_ANALYSIS`. Blocked on Packet A. Do not run.

Decision question it would answer: *what does GLM-5.3 W4AFP8 score on
`gpqa_diamond_aa_v3` on the tp=16 two-node serve, with per-item records
retained?*

Settled parameters (from §1.5, do not re-derive):

| Knob | Value | Why |
|---|---|---|
| Endpoint | `http://<svc glm-5-3-w4afp8-sglang>:30000/v1/chat/completions` | the one head endpoint (§1.1) |
| Concurrency | **4** | largest value where a full 131K trace fits its pool share (§1.5) |
| `max_new_tokens` | **131072** | AA recipe; raising it forfeits comparability (§1.8) |
| temperature / top_p | 0.6 / 1.0 | unchanged from Sep-11 |
| thinking | ON | unchanged |
| Arms | **ours only** | no Phala checkpoint is served and no capacity exists (§1.2, §1.6) |
| Expected wall clock | ~26 h, treat as optimistic | §1.4 caveat |

Mandatory additions before it may be authorized:

1. **Per-item capture** — per-request `finish_reason` + `completion_tokens`
   keyed to item id *and* repeat index, joinable to per-item correctness. Packet
   A determines whether this is a config change or needs glue. **Without this,
   the rerun repeats the Sep-11 dead end and must not be launched.**
2. **Histogram snapshotting** — extend `scrape_metrics` to diff
   `generation_tokens_histogram` buckets, not just the five scalar counters
   (§1.7).
3. **Fail-closed harness gate before GPU spend**, per the `CLAUDE.md`
   evaluation-harness contract: tokenizer/chat-template hashes, reasoning mode,
   task alias and harness version (`nvidia-simple-evals==26.3`,
   `gpqa_diamond_aa_v3`), sampling params, serving backend/topology, and sample
   manifest hash.
4. **Provenance honesty block.** Even at conc 4 and a 131K cap this run differs
   from Sep-11 on: tp (8→16), `mem_fraction_static` (0.75→0.80), context
   (164,800→1,048,576) and speculative decoding (none→EAGLE). EAGLE's rejection
   sampling is distribution-preserving *in theory*, so the score should not
   move, but that is a theoretical claim. **This number is therefore NOT paired
   against Sep-11's Phala 79.39%** and must not be reported as if it were. With
   ±1.27 stderr, only a >2.5-point shift would be visible anyway.
5. A **repetition/n-gram check** on any near-cap traces, to test the
   degenerate-loop prior in §1.8 before anyone spends on a raised cap.

Open risk not yet resolved: the serve is the collaborator's and could be
restarted or reconfigured mid-run. A 26 h run should capture
`/get_server_info` at start and end and compare, so a silent reconfiguration
invalidates the run loudly rather than quietly.

---

## 4. Highest-value use of this endpoint (planner opinion, not authorized)

GPQA can run on any 8-GPU node whenever one frees. **AA-LCR at 256K
output-policy parity can only run here** — 598,848 tokens of pool is the only
thing on the cluster that lifts the 164,800 ceiling named as blocking it
(`docs/glm53-aa-v42-public-evaluation-plan.md` line 159), and it is the first
real test of the FP8 DSA indexer. If the endpoint's lifetime is limited, spend
it on the benchmark that is impossible elsewhere.

---

## 5. Stale claims this handoff corrects

| Source | Stale claim | Correction |
|---|---|---|
| `docs/status/2026-09-11-…md` lines 148–150, 165–167 | "Zhou Yu's two serving nodes"; paired arms run concurrently; 83 h → 46 h | One tp=16 endpoint, one arm. No concurrent pairing (§1.1) |
| `docs/status/2026-09-11-…md` line 152 | "164.8K is the measured FP8 KV pool on 8xH100" | Still true for a tp=8 serve; this tp=16 serve has 598,848 (§1.3) |
| `docs/status/2026-09-11-…md` line 46 | "The harness kept only aggregates, no per-item scores" | Accurate about *scores*; per-request completion **lengths** exist as a Prometheus histogram. The gap is the join key (§1.7). Whether per-sample files exist is Packet A's question |
| `docs/glm53-aa-v42-public-evaluation-plan.md` lines 53, 116, 130, 144, 159 | context budgets computed against a 164,800-token window | Recompute against 598,848 for this deployment (§1.3) |
| `docs/status/2026-09-11-…md` lines 143–145 | EP GPTQ pod "Pending 17 h" | Pod no longer exists; zero Pending pods. Must be resubmitted (§1.6) |
| `../RANCHER_GPU_ACCESS.md` IB note | InfiniBand path "unvalidated" | Validated for these two pods via `hostNetwork` + `NCCL_IB_HCA`; 90.8 tok/s single-stream tp=16 cross-node (§1.4). Unchanged for other workloads |
| `../RANCHER_GPU_ACCESS.md` header | 6 nodes / 48 GPUs | 9 nodes / 72 GPUs (§1.6) |

## 6. Evidence for every claim above

All §1 measurements are reproducible from a machine with the kubeconfig:

```bash
bash scripts/gpu-free.sh --verify                        # §1.6
kubectl -n evaluation get pod glm-5-3-w4afp8-sglang-gpu02 -o jsonpath='{.spec.containers[*].args}'   # §1.1
kubectl -n evaluation get endpoints glm-5-3-w4afp8-sglang                                            # §1.1
kubectl port-forward -n evaluation svc/glm-5-3-w4afp8-sglang 18000:30000
curl -s localhost:18000/get_server_info                  # §1.3
curl -s localhost:18000/metrics | grep generation_tokens_histogram_bucket   # §1.7
kubectl -n evaluation logs glm-5-3-w4afp8-sglang-gpu02 | grep "Decode batch"   # §1.4 accept_len
```

The concurrency sweep (§1.4) used `/generate` with `ignore_eos: true`, 600
tokens, 30-token prompts, at concurrency 1/2/4/8. It was run against the idle
endpoint and cost ~9,000 generated tokens total. Startup logs had already
rotated (39,391 retained lines, all health probes), which is why the NCCL
transport line could not be read directly and throughput was used instead.
