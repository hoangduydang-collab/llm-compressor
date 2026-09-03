# Host and evaluate GLM-5.3 W4AFP8 on Rancher

For evaluation-pipeline engineers who need either checkpoint behind an
OpenAI-compatible SGLang endpoint.

## Quick reference

| Item | Value |
|---|---|
| Cluster / project | `infermesh-test-my` / `c-bk8md:p-dsbn4` |
| Namespace | `evaluation` |
| Shared model PVC | `model-cache-shared` (40 TiB, RWX), mounted at `/mnt/cephfs` |
| GPU topology | One node: 8×H100-80GB, tensor parallelism 8 |
| Server | `lmsysorg/sglang:v0.5.17`, port `30000` |
| In-house model | `/mnt/cephfs/hoangduy/results/glm53-w4afp8-mtp/checkpoint` |
| PhalaCloud model | `/mnt/cephfs/.hf-cache/models--PhalaCloud--GLM-5.3-W4AFP8/snapshots/7e77d7b5592d748778459a0dac802e7fd407e593` |

These are the exact checkpoint paths used by the recent successful quality
runs. Both are readable through the shared PVC. Use the pinned PhalaCloud
snapshot path, not a mutable Hugging Face branch or cache ref.

## 1. Check access and capacity

```bash
kubectl config use-context infermesh-test-my
kubectl config set-context --current --namespace=evaluation

kubectl auth can-i create pods -n evaluation
kubectl auth can-i create pods/exec -n evaluation
kubectl get pvc model-cache-shared -n evaluation
```

You need permission to create/exec pods in an InferMesh project namespace.
Filesystem permissions allow both checkpoints to be read; Kubernetes/Rancher
project membership is the remaining access gate.

This model needs an entire 8-GPU node. Check current occupancy immediately before
launching. From the `AI lab` workspace:

```bash
bash ../scripts/gpu-free.sh
```

The cluster is shared and occupancy changes quickly. A pod that remains
`Pending` with `Insufficient nvidia.com/gpu` has not reserved a usable 8-GPU
block.

## 2. Start SGLang

Save the following as `glm53-w4afp8-serve.yaml`. Set both identity variables:

- In-house: `MODEL_PATH` is the in-house path and `SERVED_MODEL_NAME` is
  `glm-5.3-w4afp8-ours`.
- PhalaCloud: `MODEL_PATH` is the PhalaCloud path and `SERVED_MODEL_NAME` is
  `glm-5.3-w4afp8-phala`.

The PVC is deliberately mounted read-only.

```yaml
apiVersion: v1
kind: Pod
metadata:
  name: glm53-w4afp8-eval
  namespace: evaluation
  labels:
    purpose: quality-eval
    model: glm-5-3-w4afp8
spec:
  restartPolicy: Never
  tolerations:
    - key: nvidia.com/gpu
      operator: Exists
      effect: NoSchedule
  containers:
    - name: sglang
      image: lmsysorg/sglang:v0.5.17
      imagePullPolicy: IfNotPresent
      env:
        - name: MODEL_PATH
          value: /mnt/cephfs/hoangduy/results/glm53-w4afp8-mtp/checkpoint
        - name: SERVED_MODEL_NAME
          value: glm-5.3-w4afp8-ours
      resources:
        requests:
          cpu: "16"
          memory: 200Gi
          nvidia.com/gpu: 8
        limits:
          cpu: "32"
          memory: 600Gi
          nvidia.com/gpu: 8
      command: ["bash", "-lc"]
      args:
        - |
          exec python -m sglang.launch_server \
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
      ports:
        - {name: http, containerPort: 30000}
      volumeMounts:
        - {name: model-cache, mountPath: /mnt/cephfs, readOnly: true}
        - {name: dshm, mountPath: /dev/shm}
        - {name: tmp, mountPath: /tmp}
  volumes:
    - name: model-cache
      persistentVolumeClaim:
        claimName: model-cache-shared
        readOnly: true
    - name: dshm
      emptyDir: {medium: Memory, sizeLimit: 64Gi}
    - name: tmp
      emptyDir: {sizeLimit: 50Gi}
```

Launch and watch it:

```bash
kubectl apply -f glm53-w4afp8-serve.yaml
kubectl describe pod glm53-w4afp8-eval -n evaluation
kubectl logs -f pod/glm53-w4afp8-eval -n evaluation
```

Loading approximately 400 GB from CephFS took 36–39 minutes in the measured
runs. Allow up to 45 minutes before treating a quiet page-in period as a hang.

## 3. Verify the endpoint

Health check:

```bash
kubectl exec -n evaluation glm53-w4afp8-eval -- \
  curl -fsS http://127.0.0.1:30000/health_generate
```

For local pipeline development, keep this running:

```bash
kubectl port-forward -n evaluation pod/glm53-w4afp8-eval 30000:30000
```

The API is then at `http://127.0.0.1:30000/v1`. Use the selected
`SERVED_MODEL_NAME` as the request's `model` value. The measured harness sent
these same logical IDs; explicitly pinning the alias here only stabilizes API
identity and does not change inference.

Verify chat and reasoning parsing:

```bash
MODEL="glm-5.3-w4afp8-ours"

curl -sS http://127.0.0.1:30000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d "{
    \"model\": \"$MODEL\",
    \"messages\": [{\"role\": \"user\", \"content\": \"What is 17 * 23? Return only the answer.\"}],
    \"temperature\": 0,
    \"max_tokens\": 128,
    \"chat_template_kwargs\": {\"enable_thinking\": true}
  }"
```

The response should contain final `content`, parsed `reasoning_content`, no
leaked `<think>` marker, and a normal `stop` finish reason.

Verify the completions/logprobs path used by multiple-choice evaluation:

```bash
curl -sS http://127.0.0.1:30000/v1/completions \
  -H "Content-Type: application/json" \
  -d "{
    \"model\": \"$MODEL\",
    \"prompt\": \"The capital of France is\",
    \"temperature\": 0,
    \"max_tokens\": 1,
    \"echo\": true,
    \"logprobs\": 5
  }"
```

Do not start a long evaluation until health, reasoning parsing, and echo/logprobs
all work.

## 4. Evaluation-pipeline contract

The recent quality evaluation ran the server and client in one pod over
`http://127.0.0.1:30000/v1`. Port-forwarding is convenient for development; for
a formal rerun, use the existing single-pod arm template
`pipeline/k8s/glm53-quality-arm.yaml.tmpl` so network topology and provenance stay
unchanged.

Use these settings when extending the pipeline:

```bash
export OPENAI_BASE_URL=http://127.0.0.1:30000/v1
export BASE_URL=http://127.0.0.1:30000
export OPENAI_API_KEY=EMPTY
export GENERAL_NUM_CONCURRENT=16
export GENERAL_MAX_GEN_TOKS=32768
export GENERAL_CHAT_TEMPLATE_ON_LOGLIKELIHOOD=0
```

- Generative tasks use `/chat/completions`, server-side templating, seed `0`, and
  `{"chat_template_kwargs":{"enable_thinking":true}}`.
- Loglikelihood tasks use `/completions` with echo/logprobs and client-side
  tokenization from the same local checkpoint.
- **Never apply the GLM-5.3 chat template to loglikelihood prompts.** It appends
  an open `<think>` tag before the scored continuation and invalidates MMLU,
  ARC, HellaSwag, and TruthfulQA scores.
- Keep speculative decoding off when comparing with the results below.

## 5. Measured baseline

Both checkpoints passed preflight, server health, throughput, loglikelihood
shape, and the complete seven-task suite.

The short pre-suite probe measured **512 aggregate output tokens/s** for both
models: eight concurrent requests generating 256 tokens over approximately four
seconds. Treat this as a readiness probe, not a full serving benchmark.

| Task | In-house W4AFP8 | PhalaCloud W4AFP8 |
|---|---:|---:|
| GSM8K exact match | 97.65% | 97.19% |
| IFEval strict prompt accuracy | 89.65% | 90.76% |
| GPQA Diamond CoT exact match | 61.11% | 55.56% |
| MMLU accuracy | 86.67% | 86.66% |
| ARC Challenge normalized accuracy | 68.77% | 69.80% |
| HellaSwag normalized accuracy | 89.37% | 89.29% |
| TruthfulQA MC2 accuracy | 62.99% | 62.50% |

Provenance:

- In-house: corrected run `full7-20260901t064327z`
- PhalaCloud: compatible run `full7-20260831t135418z`
- Evaluation code: `llm-compressor` commit `7787de48`
- Server/client: SGLang 0.5.17, lm-eval 0.4.10, seed 0, concurrency 16
- Generative protocol: thinking enabled, maximum 32,768 generated tokens

These are internal candidate-vs-peer results. They are not directly comparable
with public leaderboards or PhalaCloud's model-card table because the prompt,
sampling, and generation-budget protocols differ. GPQA is one greedy sample per
item, not avg@k.

Do not use the first in-house result under `full7-20260831t135418z` as model
quality. Its loglikelihood path incorrectly applied the chat template and
reported invalid low scores, including 33.51% MMLU. The corrected run above set
`GENERAL_CHAT_TEMPLATE_ON_LOGLIKELIHOOD=0`.

Result artifacts:

- In-house:
  `/mnt/cephfs/hoangduy/results/glm53-quality-paired/full7-20260901t064327z/results/glm-5.3-w4afp8-ours/sglang/quality/general.glm53-full7-20260901t064327z.json`
- PhalaCloud:
  `/mnt/cephfs/hoangduy/results/glm53-quality-paired/full7-20260831t135418z/results/glm-5.3-w4afp8-phala/sglang/quality/general.glm53-full7-20260831t135418z.json`

## 6. Cleanup and common failures

Release the eight GPUs as soon as testing finishes:

```bash
kubectl delete pod glm53-w4afp8-eval -n evaluation
```

- **Pending:** no single node currently has eight free GPUs. Check pod events and
  wait or coordinate; do not assume GPUs from different nodes can be pooled.
- **Long startup:** CephFS page-in can be silent for tens of minutes.
- **OOM or scheduler crash:** restore memory fraction `0.75`, context `65536`,
  and chunked prefill `2048` before changing other variables.
- **Missing `reasoning_content`:** confirm `glm45`, thinking request data, and
  SGLang 0.5.17.
- **Implausibly low multiple-choice scores:** confirm the loglikelihood client is
  not applying a chat template.
