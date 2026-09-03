# GLM-5.3 W4AFP8 Rancher evaluation guide design

## Reader and outcome

The reader is an engineer extending the evaluation pipeline. They need to get
either GLM-5.3 W4AFP8 checkpoint serving on Rancher with the same contract used
by the validated quality runs, confirm that the endpoint is ready, and connect
their evaluation work without rediscovering cluster or model-specific details.

The guide will be a concise, task-first runbook at:
`docs/glm53-w4afp8-rancher-evaluation.md`.

## Content order

1. A short quick-reference block: cluster/project, namespace, PVC, both immutable
   model paths, engine image, GPU topology, and endpoint.
2. Prerequisite access and GPU-capacity checks.
3. A reusable Kubernetes pod manifest for either checkpoint, mounting
   `model-cache-shared` read-only and exposing SGLang only inside the pod.
4. Launch, logs, health, reasoning, logprobs, and port-forward checks.
5. The evaluation-pipeline contract: OpenAI base URL, model identity, reasoning
   request body, concurrency, generation budget, and the requirement to avoid a
   chat template on loglikelihood tasks.
6. Headline throughput and seven-task quality results.
7. Result provenance, comparability limits, cleanup, and common failure modes.

## Verified serving contract

The guide will reproduce the latest successful harness manifests:

- `lmsysorg/sglang:v0.5.17`
- one 8×H100-80GB node, tensor parallelism 8
- `--quantization w4afp8`
- context length 65,536
- static memory fraction 0.75
- chunked prefill size 2,048
- FP8 E4M3 KV cache
- shared-expert fusion disabled
- reasoning parser `glm45`
- tool parser `glm47`
- no speculative decoding

Both model paths will be copied exactly from the completed manifests. The
PhalaCloud path will use its pinned snapshot revision rather than a mutable HF
reference.

## Results contract

Headline values will come from:

- in-house: corrected full run `full7-20260901t064327z`
- PhalaCloud: compatible full run `full7-20260831t135418z`

Both used commit `7787de48`, SGLang 0.5.17, lm-eval 0.4.10, seed 0,
16 concurrent requests, thinking enabled for generative tasks, and a 32,768-token
generative budget. Both passed preflight, serving, throughput, loglikelihood
shape, and general-suite gates.

The obsolete first in-house run under `full7-20260831t135418z` will not be used
as the in-house baseline. It applied the GLM-5.3 chat template to loglikelihood
tasks, placing continuations after an open thinking tag and invalidating the
multiple-choice scores. The corrected rerun set
`GENERAL_CHAT_TEMPLATE_ON_LOGLIKELIHOOD=0`.

The guide will state that the results are an internal candidate-vs-peer
comparison. They are not directly comparable with public leaderboards or
PhalaCloud's model-card table because the harness and sampling protocol differ.

## Safety and scope

- The guide will not include kubeconfig credentials, HF tokens, or secret values.
- The shared model PVC will be mounted read-only.
- GPU occupancy must be checked immediately before launch.
- Launching the pod and later deleting it remain explicit operator actions.
- The guide covers model hosting and endpoint handoff, not redesigning the
  evaluation framework or quantization recipe.

## Acceptance criteria

- Every path and serving flag matches a successful harness manifest.
- Commands are directly usable after substituting only a pod name and model
  choice.
- Readiness checks exercise health, chat/reasoning output, and echo/logprobs.
- All seven headline scores are traceable to the selected result artifacts.
- The invalid first in-house loglikelihood result cannot be mistaken for the
  corrected result.
- A collaborator can identify how to connect an OpenAI-compatible evaluation
  client and how to release the reserved GPUs.
