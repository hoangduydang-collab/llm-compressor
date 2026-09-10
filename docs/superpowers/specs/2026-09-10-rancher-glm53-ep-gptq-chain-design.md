# Rancher GLM-5.3 EP GPTQ Chain Design

## Decision

Queue one immutable Kubernetes Job that atomically requests all eight GPUs on one
Rancher worker and runs this fail-closed sequence:

1. the existing tiny two-rank H100/NCCL lifecycle suite;
2. a real-width, five-layer GLM-5.3 representative gate at EP4 and EP8;
3. an offloaded DDP4 numerical control for the representative checkpoint;
4. full-model EP8 GPTQ quantization.

Every stage starts only after all earlier stages pass. A stage failure writes its
exit code and evidence to CephFS, then holds the node for 24 hours so the failed
process state can be inspected. Successful full quantization exits immediately
and releases the node; there is no post-success hold and no automatic retry.

## Quantization contract and prior art

The same-family baseline is the public GLM-5.3 W4A16 work from
`JANGQ-AI/GLM-5.3-W4A16`, corroborated by the calibrated
`canada-quant/glm-5.3-w4a16-mtp` GLM-5.3-Flash recipe:

- https://huggingface.co/JANGQ-AI/GLM-5.3-W4A16
- https://huggingface.co/canada-quant/glm-5.3-w4a16-mtp

Both contracts quantize routed-expert gate/up/down projections to symmetric
INT4 and keep attention, the indexer, router, shared experts, dense prefix,
embeddings, norms and `lm_head` at high precision. The calibrated recipe uses
GPTQ, group size 128, 256 samples, and a sequential per-layer walk.

This chain adopts that weight-only contract:

- scheme: `W4A16`, symmetric INT4, group size 128;
- targets: only `model.layers.N.mlp.experts.E.{gate,up,down}_proj`;
- BF16: every attention and indexer projection, layers 0-2 dense MLPs, router,
  shared experts, embeddings, final norm and `lm_head`;
- MTP: outside the Transformers model and therefore not emitted by this run.

The MTP omission remains a release limitation and must be repaired or explicitly
accepted before serving. The full run produces a quantized base-model artifact;
it does not claim a production-ready speculative-decoding checkpoint.

The existing GLM-5.3 pipeline uses 256 UltraChat samples at 2048 tokens. That is
retained for the full run to avoid introducing a new calibration corpus while
testing EP. It differs from Canada Quant's 4096-token in-distribution chat/code
mix and is recorded as a deliberate calibration deviation.

## Representative model and inputs

Build a physical depth-truncated copy of the immutable BF16 snapshot:

`/mnt/cephfs/.hf-cache/models--zai-org--GLM-5.3-BF16/snapshots/304b8051cfb2b260b61ce0cbe330e02a98e73639`

The subset contains layers 0-4:

- layers 0-2: dense prefix;
- layer 3: first MoE layer, covering the dense-to-MoE boundary;
- layer 4: second MoE layer, covering a consecutive-MoE boundary;
- all original dimensions and all 256 routed experts are preserved.

The subset is built once on the Job's node-local `emptyDir` and reused unchanged
by DDP4, EP4 and EP8. A committed eight-row JSONL fixture supplies the same
global calibration inputs to every lane. The inputs have varied lengths, are
tokenized by the pinned BF16 tokenizer, and divide evenly across EP4 and EP8.

Representative lanes use eight samples at a maximum sequence length of 512.
They are correctness and memory gates, not quality measurements.

## Representative gates

Each lane saves a packed checkpoint and complete per-rank phase evidence.
The validator requires:

- exactly 1,536 routed expert projections in layers 3-4
  (`2 layers * 256 experts * 3 projections`);
- no routed expert or required packed qparameter missing;
- all scales finite and strictly positive;
- attention, indexer, router, shared experts, dense prefix, embeddings, norms
  and `lm_head` absent from the INT4 target set;
- complete paired phase evidence for every expected rank;
- successful checkpoint release and Transformers reload;
- fixed-prompt logits matching the offloaded DDP4 control at
  `rtol=2e-3`, `atol=2e-4`, inherited from the passing tiny lifecycle suite;
- maximum rank-local CUDA allocator peak below 76 GiB;
- EP8 maximum peak at most 80% of EP4's maximum peak.

The approximately 19 GiB EP4 and 9.5 GiB EP8 figures are Hessian-only
predictions, not total-memory thresholds. Raw peaks, scratch use, phase times
and packed-word differences remain evidence even when a threshold fails.

The DDP4 control uses the existing CPU Hessian offload path because a real-width
unoffloaded DDP rank requires about 76 GiB for Hessians before weights and would
reproduce the known OOM. This control is numerical only and is not used for
performance comparison.

## Full run

Only a passing representative validator permits the full EP8 command. It loads
the complete immutable BF16 snapshot, quantizes every routed expert in decoder
layers 3-77, and saves to a unique run root under:

`/mnt/cephfs/hoangduy/results/glm53-ep-gptq/<run-tag>/full`

The full config uses:

- eight ranks and `gptq_expert_parallel: true`;
- `W4A16`;
- 256 UltraChat samples, maximum sequence length 2048, seed 42;
- all-expert calibration and sequential `GlmMoeDsaDecoderLayer` processing;
- 32 GB shared CPU weight cache and CephFS offload scratch;
- no activation quantization, sample generation, serving or evaluation.

Full success requires the pipeline's existing save and offline checkpoint gates,
complete phase evidence, and a final chain status document containing commit,
node, image, versions, source revision, paths, sizes and SHA-256 hashes for small
metadata files. Quality and serving validation remain separate downstream work.

## Kubernetes and failure semantics

Use a `batch/v1` Job in namespace `evaluation`:

- request and limit exactly eight `nvidia.com/gpu`;
- `restartPolicy: Never`, `backoffLimit: 0`;
- no node pin by default;
- `lmsysorg/sglang:v0.5.17`;
- `IPC_LOCK`, 900 GiB `/dev/shm`, 220 GiB node-local scratch;
- `model-cache-shared` mounted read-write at `/mnt/cephfs`;
- source checked out at one pushed commit and verified before GPU work;
- semantic dependency pins installed in an isolated node-local virtualenv;
- durable combined logs and status JSON under the run root.

The chain wrapper catches each stage's return code. On failure it records the
failed stage and UTC deadline, sleeps for 86,400 seconds, then exits nonzero.
On complete success it exits zero immediately. `activeDeadlineSeconds` must
cover the expected quantization runtime plus the possible failure hold; it must
not truncate the 24-hour inspection window.

## Launch safety

Before applying the Job:

1. render and parse the manifest locally;
2. extract and `bash -n` the container script;
3. require a pushed commit and no unresolved placeholders;
4. run `scripts/gpu-free.sh --verify`;
5. require a fully free eight-GPU node;
6. show the exact manifest, occupancy and requested resources for user approval.

No Pod is created and no GPU is claimed before that final approval.
