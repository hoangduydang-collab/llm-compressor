# GLM-5.3 W4AFP8 Rancher Evaluation Guide Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Publish a concise task-first runbook that lets an evaluation-pipeline engineer serve either GLM-5.3 W4AFP8 checkpoint on Rancher under the validated quality-evaluation contract.

**Architecture:** One Markdown guide owns the operational handoff: immutable checkpoint identity, a reusable single-pod SGLang deployment, endpoint readiness checks, evaluation-client settings, validated scores, and cleanup. It references existing scripts and artifacts rather than duplicating the full evaluation framework.

**Tech Stack:** Kubernetes/Rancher, CephFS RWX PVC, SGLang 0.5.17, OpenAI-compatible HTTP API, lm-eval 0.4.10, Markdown.

## Global Constraints

- Target cluster/project: `infermesh-test-my`, `c-bk8md:p-dsbn4`; default namespace: `evaluation`.
- Mount the namespace-local `model-cache-shared` PVC at `/mnt/cephfs` with `readOnly: true`.
- In-house checkpoint: `/mnt/cephfs/hoangduy/results/glm53-w4afp8-mtp/checkpoint`.
- PhalaCloud checkpoint: `/mnt/cephfs/.hf-cache/models--PhalaCloud--GLM-5.3-W4AFP8/snapshots/7e77d7b5592d748778459a0dac802e7fd407e593`.
- Match the validated server: `lmsysorg/sglang:v0.5.17`, TP8 on 8×H100-80GB, W4AFP8, 65,536 context, memory fraction 0.75, chunked prefill 2,048, FP8 E4M3 KV cache, shared-expert fusion disabled, `glm45` reasoning parser, and `glm47` tool parser.
- Keep speculative decoding off when reproducing reported numbers.
- Generative evaluation uses thinking enabled, 32,768 maximum generated tokens, seed 0, and 16 concurrent requests.
- Loglikelihood tasks must not apply the GLM-5.3 chat template: `GENERAL_CHAT_TEMPLATE_ON_LOGLIKELIHOOD=0`.
- Report the corrected in-house run `full7-20260901t064327z`; do not present the invalid first in-house loglikelihood run as model quality.
- State that internal paired-harness scores are not public-leaderboard or PhalaCloud-model-card comparable.
- Do not include credentials, token values, or commands that create secrets.
- Do not launch, reserve, or delete GPU resources while authoring or verifying the document.

## File Structure

- Create: `docs/glm53-w4afp8-rancher-evaluation.md` — complete collaborator-facing hosting and evaluation handoff.

---

### Task 1: Write and verify the collaborator runbook

**Files:**
- Create: `docs/glm53-w4afp8-rancher-evaluation.md`

**Interfaces:**
- Consumes: `pipeline/k8s/glm53-quality-arm.yaml.tmpl`, `pipeline/k8s/glm53_quality_arm.sh`, both `benchmarks/configs/glm/glm-5.3-w4afp8-*.sh` profiles, and the selected CephFS result artifacts.
- Produces: a standalone runbook whose pod exposes `http://127.0.0.1:30000/v1` internally and can be port-forwarded to the evaluator.

- [ ] **Step 1: Write the quick-reference and prerequisites**

Put the reader's highest-value facts first: exact cluster/project/namespace, both immutable paths, PVC, 8-GPU requirement, endpoint, and access assumptions. Include:

```bash
kubectl config use-context infermesh-test-my
kubectl config set-context --current --namespace=evaluation
kubectl auth can-i create pods -n evaluation
kubectl auth can-i create pods/exec -n evaluation
kubectl get pvc model-cache-shared -n evaluation
bash scripts/gpu-free.sh
```

Explain that GPU occupancy changes continuously and must be checked immediately before launch.

- [ ] **Step 2: Add one reusable hosting manifest**

Embed a Pod manifest parameterized by `MODEL_PATH` and `SERVED_MODEL_NAME`. It must request eight GPUs, tolerate the GPU taint, allocate 64 GiB `/dev/shm`, mount `model-cache-shared` read-only, use `lmsysorg/sglang:v0.5.17`, and launch:

```bash
python -m sglang.launch_server \
  --model-path "$MODEL_PATH" \
  --served-model-name "$SERVED_MODEL_NAME" \
  --quantization w4afp8 \
  --disable-shared-experts-fusion \
  --tp 8 \
  --kv-cache-dtype fp8_e4m3 \
  --reasoning-parser glm45 \
  --tool-call-parser glm47 \
  --context-length 65536 \
  --mem-fraction-static 0.75 \
  --chunked-prefill-size 2048 \
  --enable-metrics \
  --trust-remote-code \
  --host 0.0.0.0 \
  --port 30000
```

Do not add speculative-decoding flags.

- [ ] **Step 3: Add launch and readiness checks**

Document substituting one of the two model paths, applying the manifest, following logs, allowing up to approximately 45 minutes for CephFS page-in, checking `/health_generate`, and port-forwarding:

```bash
kubectl apply -f glm53-w4afp8-serve.yaml
kubectl logs -n evaluation -f pod/glm53-w4afp8-eval
kubectl exec -n evaluation glm53-w4afp8-eval -- \
  curl -fsS http://127.0.0.1:30000/health_generate
kubectl port-forward -n evaluation pod/glm53-w4afp8-eval 30000:30000
```

Add compact chat/reasoning and `echo=true` logprobs probes so the evaluator verifies both API paths before a long run.

- [ ] **Step 4: Add the evaluation-pipeline contract**

Specify `OPENAI_BASE_URL=http://127.0.0.1:30000/v1`, model name matching `--served-model-name`, empty/dummy API key, thinking request body `{"chat_template_kwargs":{"enable_thinking":true}}`, seed 0, concurrency 16, and maximum generation 32,768. Separate the generative path from loglikelihood scoring and call out:

```bash
GENERAL_CHAT_TEMPLATE_ON_LOGLIKELIHOOD=0
```

Explain that applying the template to multiple-choice/loglikelihood prompts leaves an open `<think>` tag before the scored continuation and invalidates those scores.

- [ ] **Step 5: Add validated throughput and quality results**

Report the pre-suite probe as 512 aggregate output tok/s for both checkpoints with eight concurrent 256-token requests. Include the seven headline results as percentages:

| Task | In-house corrected | PhalaCloud |
|---|---:|---:|
| GSM8K exact match | 97.65 | 97.19 |
| IFEval strict prompt accuracy | 89.65 | 90.76 |
| GPQA Diamond CoT exact match | 61.11 | 55.56 |
| MMLU accuracy | 86.67 | 86.66 |
| ARC Challenge normalized accuracy | 68.77 | 69.80 |
| HellaSwag normalized accuracy | 89.37 | 89.29 |
| TruthfulQA MC2 accuracy | 62.99 | 62.50 |

Attach provenance: in-house `full7-20260901t064327z`; PhalaCloud `full7-20260831t135418z`; commit `7787de48`; SGLang 0.5.17; lm-eval 0.4.10. State all gates passed and scores are internal-harness comparisons only.

- [ ] **Step 6: Add cleanup and failure guidance**

Include:

```bash
kubectl delete pod glm53-w4afp8-eval -n evaluation
```

Cover only the failures most useful to this reader: `Pending` due to unavailable 8-GPU blocks, long but healthy CephFS startup, OOM risk if memory fraction/chunked prefill differ, missing reasoning fields/parser mismatch, and invalid loglikelihood scores from chat-template application.

- [ ] **Step 7: Verify the document mechanically**

Run:

```bash
rg -n "glm53-w4afp8-mtp/checkpoint|7e77d7b5592d748778459a0dac802e7fd407e593|lmsysorg/sglang:v0.5.17|GENERAL_CHAT_TEMPLATE_ON_LOGLIKELIHOOD=0|512|full7-20260901t064327z|full7-20260831t135418z" docs/glm53-w4afp8-rancher-evaluation.md
git diff --check
```

Expected: every required identity/guard appears, and `git diff --check` exits zero.

- [ ] **Step 8: Review against source evidence**

Compare the manifest flags to `pipeline/k8s/glm53_quality_arm.sh`, paths and task behavior to the two benchmark profiles, and scores to the selected general-result artifacts. Confirm the obsolete 33.51% in-house MMLU result is mentioned only as an invalid-run warning, not a model score.

- [ ] **Step 9: Commit**

```bash
git add docs/glm53-w4afp8-rancher-evaluation.md
git commit -m "docs: add GLM-5.3 Rancher evaluation handoff"
```

## Self-review

- Spec coverage: reader, task order, both models, immutable paths, hosting manifest, readiness probes, endpoint contract, seven scores, provenance, comparability, cleanup, and failure guidance are all included.
- Scope remains one standalone Markdown deliverable; no pipeline or cluster mutation is required.
- No placeholders, secrets, mutable HF references, or unverified public-quality claims are present.
