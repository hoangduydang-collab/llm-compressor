# Execution packet: GLM-5.3 expert-parallel GPTQ chain

- Protocol version: 1
- State: READY_FOR_EXECUTOR
- Packet revision: 2026-09-10-r1
- Planner owner: Cursor planning session
- Intended executor: Cursor Rancher executor
- Base Git commit: `072a750211ee5477507797204fcd0c4c51bd6f23`
- Decision question: Can the expert-local GPTQ implementation pass tiny and
  real-width EP4/EP8 correctness and memory gates, then produce a complete
  full-model W4A16 EP8 checkpoint?

## Objective and hypothesis

Run one immutable eight-H100 chain. The hypothesis is that expert ownership
reduces routed Hessian residency from approximately 19 GiB per rank at EP4 to
9.5 GiB at EP8, without changing the DDP4-control logits beyond the tolerances
already passed by the tiny lifecycle test. Only a passing representative gate
may start the full run.

## Scope and non-goals

- In scope: tiny NCCL lifecycle, five-layer real-width EP4/EP8, offloaded DDP4
  control, static checkpoint checks, logits comparison, full EP8 quantization,
  durable evidence and failure-only holding.
- Not authorized: serving, quality claims, MTP grafting, retries, node pinning,
  recipe changes, activation quantization, or cleanup of failed-run evidence.
- A successful full run is a quantized base-model artifact. Transformers omits
  the MTP layer, so the result is not yet a production speculative-decoding
  checkpoint.

## Preconditions and exact environment

- Local repository: `C:\Users\hoangduy.dang\AI lab\llm-compressor`
- Branch: `duy-branch`
- Executable revision: `072a750211ee5477507797204fcd0c4c51bd6f23`
- Kubernetes namespace: `evaluation`
- Image:
  `docker.io/lmsysorg/sglang@sha256:16aba8925507e631e1dc1e23d95d026533602591775f6a8db68b74ee99746155`
- Python semantic pins: `transformers==5.14.1`,
  `compressed-tensors==0.17.2a20260707`, `torch==2.11.0`; the Job asserts these
  exact versions before testing.
- The base commit must be visible on a remote branch before launch.
- The authoritative Rancher report's namespace and Rancher accounting must
  agree. `--queue` permits submission when no full node is currently free; the
  atomic eight-GPU request then remains Pending until one node has all eight.

## Required inputs

1. BF16 checkpoint:
   `/mnt/cephfs/.hf-cache/models--zai-org--GLM-5.3-BF16/snapshots/304b8051cfb2b260b61ce0cbe330e02a98e73639`
   - must pass `pipeline.verify_source_snapshot`;
   - architecture `GlmMoeDsaForCausalLM`;
   - 78 decoder layers;
   - no source quantization configuration.
2. Representative calibration fixture:
   `pipeline/fixtures/glm53_ep_gptq_representative.jsonl`
   - SHA-256:
     `b33f7477d98cb06ac24bb1f17689dd339e2dd7666f23a10a7ec88ff2d77a5be9`;
   - eight fixed text rows, maximum 512 tokens.
3. Full calibration data:
   `HuggingFaceH4/ultrachat_200k`, split `train_sft`, first 256 rows before
   seed-42 shuffle, maximum 2048 tokens.

## Quantization contract

The same-family sources are:

- https://huggingface.co/JANGQ-AI/GLM-5.3-W4A16
- https://huggingface.co/canada-quant/glm-5.3-w4a16-mtp

Use symmetric group-128 `W4A16` GPTQ on routed expert gate/up/down projections
only. Keep all attention and DSA indexer projections, router and correction
bias, shared experts, layers 0-2 dense MLPs, embeddings, norms and `lm_head`
high precision. `fp8_dynamic_targets` is empty. The full calibration's
256-by-2048 UltraChat inputs deliberately differ from Canada Quant's
256-by-4096 private chat/code mix; do not call those datasets equivalent.

## Workspace and artifact policy

- Code checkout and virtualenv: node-local `/work`.
- Representative BF16 subset and temporary scratch: node-local `/scratch`.
- Protected input: the entire BF16 snapshot above; never modify it.
- Durable run root:
  `/mnt/cephfs/hoangduy/results/glm53-ep-gptq/20260910t120700z`
- Durable logs:
  `/mnt/cephfs/hoangduy/results/glm53-ep-gptq/20260910t120700z/logs`
- Durable reports:
  `/mnt/cephfs/hoangduy/results/glm53-ep-gptq/20260910t120700z/reports`
- Lane outputs are the four named directories under
  `/mnt/cephfs/hoangduy/results/glm53-ep-gptq/20260910t120700z/lanes`:
  `representative-ep4`, `representative-ep8`, `representative-ddp4`, and
  `full-ep8`.
- Offload roots use the same four lane names under
  `/mnt/cephfs/hoangduy/results/glm53-ep-gptq/20260910t120700z/offload`.
- Status:
  `/mnt/cephfs/hoangduy/results/glm53-ep-gptq/20260910t120700z/chain-status.json`
- Stop on any pre-existing Job with the same name or pre-existing durable run
  root. Do not merge evidence from different packet revisions.

## Resource contract

- Nodes: one
- GPUs: eight H100-80GB requested atomically
- Placement: scheduler-selected fully free node; no hostname selector
- Container request: 16 CPU, 700 GiB memory, 220 GiB ephemeral storage
- Container limit: 48 CPU, 1700 GiB memory, 260 GiB ephemeral storage
- Shared memory: 900 GiB `emptyDir`
- Task layout: tiny test spawns two ranks; representative EP4 and DDP4 use four
  ranks; representative EP8 and full use eight ranks
- Concurrency: strictly sequential
- Kubernetes deadline: seven days
- Expected runtime: representative work several hours; full quantization may
  take one to three days. This is an estimate, not a timeout.
- Retry policy: `backoffLimit: 0`, `restartPolicy: Never`

## Ordered commands

From the workspace root, render and inspect without applying:

```powershell
& 'C:\Program Files\Git\bin\bash.exe' `
  llm-compressor/pipeline/k8s/launch-glm53-ep-gptq-chain.sh `
  --run-tag 20260910t120700z `
  --ref 072a750211ee5477507797204fcd0c4c51bd6f23 `
  --queue `
  --dry-run
```

The launcher must report:

- an authoritative `scripts/gpu-free.sh --verify` agreement;
- the current largest schedulable pod and explicit queue-mode behavior;
- valid rendered YAML and Bash;
- exact image digest, commit and eight-GPU request;
- no `kubectl apply` in dry-run mode.

After the user approves the shown claim, apply with:

```powershell
& 'C:\Program Files\Git\bin\bash.exe' `
  llm-compressor/pipeline/k8s/launch-glm53-ep-gptq-chain.sh `
  --run-tag 20260910t120700z `
  --ref 072a750211ee5477507797204fcd0c4c51bd6f23 `
  --queue
```

The immutable container then runs:

1. exact source and dependency checks;
2. full and representative recipe-scope preflight;
3. `test_expert_parallel_nccl.py` on GPUs 0-1;
4. physical five-layer BF16 subset creation, preserving layers 0-4 and all 256
   experts;
5. representative EP4 save and validation;
6. representative EP8 save and validation;
7. representative offloaded DDP4 save and validation;
8. EP4/EP8 memory gate;
9. DDP4-versus-EP4/EP8 fixed-prompt logits gate;
10. full EP8 save and validation.

## Gates and interpretation

Representative checkpoint gates require:

- all 1,536 expert projections across layers 3-4;
- `weight_packed`, `weight_scale` and `weight_shape` for every projection;
- no packed non-expert weights;
- finite positive scales;
- complete paired phase evidence for every expected rank;
- per-rank CUDA allocator peak below 76 GiB;
- EP8 maximum peak no greater than 80% of EP4 maximum;
- EP4 and EP8 final-token logits close to DDP4 at `rtol=2e-3`,
  `atol=2e-4`.

The DDP4 lane uses CPU Hessian offload and is a numerical control only. Do not
compare its timing to EP timing.

Full checkpoint gates require all 57,600 expert projections across layers 3-77,
the same packed-key and scale invariants, complete eight-rank phase evidence,
and the pipeline's existing save/offline verification.

## Failure, hold and success behavior

On a captured stage failure, the Job writes `state: failure-hold`, the stage,
exit code and UTC timestamp to `chain-status.json`, then sleeps exactly 86,400
seconds while continuing to own all eight GPUs. After that interval it exits
nonzero. Do not delete or signal the Pod during the hold without separate user
approval.

On full success, `chain-status.json` contains `state: success` and the exact
full checkpoint path. The process exits immediately and releases all GPUs.
There is no success hold.

## Monitoring and return evidence

Use read-only commands:

```powershell
kubectl get job,pod -n evaluation -l purpose=glm53-ep-gptq-chain -o wide
kubectl logs -n evaluation -f job/glm53-ep-gptq-20260910t120700z
kubectl exec -n evaluation pod/glm-5-3-w4afp8-sglang-gpu02 -- `
  cat /mnt/cephfs/hoangduy/results/glm53-ep-gptq/20260910t120700z/chain-status.json
```

The executor returns the Job/Pod manifest and scheduler events, status JSON,
all report JSON files, log paths, exact checkpoint path, byte size and small
artifact hashes. No retry or downstream serving is authorized by this packet.
