# Active executor handoff: native W4AFP8 representative qualification

Protocol: PLANNER_EXECUTOR_PROTOCOL v1. Packet revision: 1 (2026-09-12).
State: READY_FOR_EXECUTOR — deployment discovery, then gated representative runs.
Branch: `duy-branch`. Implementation/base commit:
`d90ea79c49b4e62d314cc8d9eba190fdfba0ddf4`.
Record the actual handoff commit from `git rev-parse HEAD` after pulling this file.

**Owner instruction:** “push to my branch, for executor access, just write a
handoff with info you have, the executor will figure the rancher stuff”. This
explicitly delegates Rancher context, credentials, namespace, mounts, placement,
container wrapping and launch/monitoring plumbing to the executor. It overrides
requirements in the base protocol that the planner must resolve those deployment
details first. Record the resolved deployment before starting calibration. It
does not authorize changing quantization math, calibration data or pass criteria.

**Decision:** does FP8-before-GPTQ execute at real width with acceptable memory,
preserve its native checkpoint exactly, and load/forward in the serving runtime?
Run the five-layer representative only; return evidence before a full rerun.

## What is implemented and already tested

- Group A: attention, indexer `wk`/`wq_b`, shared experts and dense-prefix MLPs.
  Its block-FP8 working weights and frozen scales are prepared before group B's
  routed INT4 GPTQ Hessians. Existing sequential replay propagates both errors.
- Direct native SGLang W4AFP8 save through the existing collective writer; no
  intermediate CT checkpoint or post-quant conversion is required.
- Native policy: INT4 group 128, E4M3 block 128x128, fixed-unit expert input
  scales, dynamic linear activations. MTP absent; no speculative decoding.
- 138 CPU tests passed. See [implementation](glm53-fp8-before-gptq-implementation.md)
  and [raw numerical evidence](evidence/2026-09-12-fp8-before-gptq-cpu.json).
- Local Slurm job **841530** is queued for the five-case two-H100 NCCL gate.
  [Local controller/evidence](../results/glm53-ep-gptq/20260912-fp8-before-gptq-h100/).
  Last checked: PENDING (Resources); this is not a pass. Do not cancel this or
  unrelated jobs. Run the same gate in your actual quantization environment.

## Known Rancher inputs; executor verifies and resolves them

Historical context/project: `infermesh-test-my` / `c-bk8md:p-dsbn4`; namespace
`evaluation`; PVC `model-cache-shared` mounted at `/mnt/cephfs`.
The planner cannot verify these remotely. Use the real cluster's occupancy and
scheduler tools; the old launcher expected `scripts/gpu-free.sh --verify`.
Do not assume that tool exists in this repo or transplant local Slurm commands.

Historical unquantized source:
`/mnt/cephfs/.hf-cache/models--zai-org--GLM-5.3-BF16/snapshots/304b8051cfb2b260b61ce0cbe330e02a98e73639`.
Preserve that snapshot's bytes and source verification record. Path relocation is
allowed; a different model/source revision is not an equivalent substitution.

Historical quantization image:
`docker.io/lmsysorg/sglang@sha256:16aba8925507e631e1dc1e23d95d026533602591775f6a8db68b74ee99746155`.
Historical serving precedent: SGLang 0.5.17, `--quantization w4afp8`,
`--disable-shared-experts-fusion`, TP8. Record the actual serving image digest,
SGLang version/revision and supported flags; the old mutable tag is not a pin.

Local qualified environment: Python 3.12, Torch 2.11.0+cu128,
compressed-tensors 0.17.2a20260707, Transformers 5.12.1, safetensors 0.8.0.
The older Rancher chain instead asserts Transformers **5.14.1**. Do not silently
claim the environments match. Prefer the qualified versions; if deployment needs
5.14.1 or another existing runtime environment, record all versions and rerun the
CPU exporter/recipe and NCCL gates below there before real-width calibration.
Use an isolated environment; do not upgrade an environment serving another job.

## Scope, resources, adaptations and stops

One node with eight full H100s; run EP4 then EP8 sequentially on the same node.
Use four visible GPUs for EP4 and all eight for EP8. Overall allocation ceiling:
six hours including setup, calibration, export checks and bounded serving smoke.
Record CPU/RAM allocation, shared-memory size, disk capacity and mount type.
Release resources after completion or failure; no 24-hour diagnostic GPU hold.

Use a clean detached checkout of the implementation commit, or verify every
source/config/test hash against the committed CPU evidence before proceeding.
Keep runtime artifacts outside protected `src/`, `pipeline/`, `tests/` and `envs/`.
The planner workspace has unrelated `.gitignore`, `predictive_ptq/` and
`results/predictive-ptq/` changes; do not copy or clean them. Use a fresh run ID;
stop on an existing Job name, output root or subset destination collision.

The executor may resolve Rancher deployment, remap paths preserving inputs, fix
launcher-only errors and adapt an isolated dependency environment with the
mandatory requalification above. Preserve each failed attempt. One fresh retry
is allowed for a launch/placement failure before calibration starts. No automatic
retry after OOM, numerical failure, calibration or export failure. Do not change
thresholds, recipes, solver behavior, sample count, source, or FP8 policy.

**Do not execute the old `launch-glm53-ep-gptq-chain.sh` unchanged.** Its template
uses W4A16/CT validators and unconditionally proceeds to full EP8. Reuse its
source verification, subset builder, log capture and resource plumbing only.
The older W4A16 packet remains a historical contract for its existing artifacts;
this is the active handoff for the new native W4AFP8 qualification task.

## Commands inside the executor's prepared allocation

Set `NATIVE_SOURCE` to the verified source above, `NATIVE_RUN_ROOT` to a fresh
absolute durable result path, and `NATIVE_SUBSET` to a fresh node-local subset
path. Record all three values. Commands assume the isolated quantization Python
is active and the current directory is the implementation checkout.

```bash
export PYTHONPATH="$PWD/src:$PWD"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
python -m pipeline.verify_source_snapshot "$NATIVE_SOURCE" \
  --require-unquantized --expect-layers 78 --expect-arch GlmMoeDsaForCausalLM
sha256sum pipeline/fixtures/glm53_ep_gptq_representative.jsonl
```

Required fixture SHA256:
`14401275814cf0556f20aa686d1e4ddd76e4909e113756d3e22cee2c852f92bd`.
The historical packet has a different hash; do not reuse it. Calibration stays
at eight samples, 512 maximum tokens, seed 42, all experts, sequential replay,
and the existing prefetch settings.

Capture each command's stdout/stderr and exit code; any failed gate stops before
the next stage. Create fresh log/report directories under the fresh run root.

```bash
python -m pytest -q pipeline/tests/test_native_sglang_save.py \
  pipeline/tests/test_glm53_ep_gptq_configs.py \
  tests/llmcompressor/modifiers/quantization/test_weight_preparation.py
CUDA_VISIBLE_DEVICES=0,1 python -m pytest -vv -rA \
  -o tmp_path_retention_policy=all \
  --junitxml="$NATIVE_RUN_ROOT/tiny-nccl.xml" \
  tests/llmcompressor/modifiers/gptq/test_expert_parallel_nccl.py
```

NCCL must report **five passed, zero skipped**. Check the JUnit counts explicitly;
a zero pytest exit code with unavailable GPUs is insufficient. Preserve per-rank
phase, packed-difference, FP8-parity and activation-logits JSON/text artifacts.

```bash
python -m pipeline.build_subset_checkpoint \
  --snapshot "$NATIVE_SOURCE" --out "$NATIVE_SUBSET" --layers 5
```

This keeps physical layers 0–4, including all 256 routed experts in layers 3–4.
Do not reduce hidden width, expert count, or replace the model with a toy fixture.

Run EP4, validate it, then run EP8. Set `NATIVE_RANKS=4` with
`CUDA_VISIBLE_DEVICES=0,1,2,3`; then set `NATIVE_RANKS=8` with
`CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7`. Use distinct offload/output paths per lane.

```bash
python -m torch.distributed.run --standalone \
  --nproc_per_node="$NATIVE_RANKS" -m pipeline.run \
  --config pipeline/configs/glm53_ep_gptq_w4afp8_representative.yaml \
  --stage quantize \
  --set "model.id=$NATIVE_SUBSET" \
  --set "model.offload_folder=$NATIVE_RUN_ROOT/offload/ep$NATIVE_RANKS" \
  --set "output_dir=$NATIVE_RUN_ROOT/lanes/ep$NATIVE_RANKS"
```

Locate exactly one `checkpoint` directory per fresh lane; fail on zero or multiple
matches instead of selecting by modification time. Set `NATIVE_CHECKPOINT` to it:

```bash
python -m pipeline.native_sglang_save "$NATIVE_CHECKPOINT"
```

## Validation and measurements

- Every quantization and native verification command must exit zero. Preserve
  `native_sglang_manifest.json`, config, index, recipe and hashes. The native
  verifier checks all quantized payload hashes and complete save-time inventory.
- Independently require exactly **1,536** routed projection weights:
  layers `{3,4}` × expert IDs `0..255` × `{gate_proj,up_proj,down_proj}`.
  Each has native INT8-packed `.weight` and `.weight_scale_inv`; each expert has
  `w1.input_scale`, `w2.input_scale`, `w3.input_scale`. Group-A weights must be
  E4M3 with FP32 block scales, including the source model's actual DSA indexers.
  Require no `.weight_packed`, `.weight_shape` or `.weight_g_idx` remnants;
  fixed-unit MoE input policy and `num_nextn_predict_layers=0` must be explicit.
- Reuse `pipeline.metrics.summarize_phases(paths)`,
  `pipeline.validate_glm53_ep_gptq.validate_phase_summary(summary, world_size=N)`
  and `peak_cuda_bytes(paths)` on each run directory's `quant_metrics*.jsonl`.
  Require complete rank coverage, finite memory records and successful phases,
  including `fp8_weight_preparation`, calibration/replay, `checkpoint_save` and
  source `offline_verification`. Inspect both start/end records and outcome.
- Adopt the existing representative memory gates for this new run: each lane's
  peak allocated CUDA memory <76 GiB; EP8 peak <=80% of EP4 peak. Missing values
  fail. Report reserved memory too; do not treat the old W4A16 peaks as new results.
- Report elapsed load/dispatch, FP8 preparation, calibration, GPTQ, propagation,
  save and offline verification, plus host memory and available I/O counters.
  Nested timers overlap; do not add them into a fictitious total. Hashing and CT
  cache restoration add work even though checkpoint conversion is eliminated.

Do **not** run the old `validate_glm53_ep_gptq` CLI or
`compare_glm53_ep_gptq` loader on native output: they assume CT tensor names and
Transformers loading. Their pure metric/logit comparison helpers can be reused.

## Serving smoke and activation sensitivity

After serialization passes, use the executor's pinned SGLang W4AFP8 runtime to
load each native representative checkpoint, with TP8 and
`--disable-shared-experts-fusion`. Start from the known flags in
[the serving guide](glm53-w4afp8-rancher-evaluation.md), recording any flags needed
for the deployed version. Use a bounded 2,048-token context, fixed input token
IDs saved to JSON, greedy generation capped at eight new tokens, and no MTP.
Require successful model load, health and forward; preserve outputs and finite
logprob/logit checks. A five-layer model is not expected to answer correctly.
If the runtime cannot load the artifact, preserve the exact failure and return;
do not run a converter to make this direct-native gate pass.

Replay/Hessian differences passed CPU weight-calibration gates, but
activation-enabled DDP/EP logits differ by up to 0.312% relative Frobenius error.
This is **not** proven runtime equivalence. Compare fixed inputs/runtime flags
between EP4 and EP8, retain numerical arrays when supported, and report relative
Frobenius, normalized maximum and absolute maximum differences. Keep activation
policy and tokenization identical. Do not invent or relax a quality threshold.
A controlled same-weight activation comparison and matched baseline quality eval
remain follow-up analysis if these diagnostics cannot isolate the effect.
Do not use a full-model converted/MTP checkpoint as a five-layer quality control.

## Return contract and next decision

Commit and push small evidence to `duy-branch`: actual source/handoff SHA,
container digests and package versions, resolved launch YAML/commands/resources,
source and fixture hashes, subset config/manifest, per-command exit codes,
Job/Pod/node/GPU identity, raw logs, per-rank metrics, native manifests and
verification reports, serving request/response/logit artifacts, deviations and
terminal scheduler state. Record large checkpoint paths, byte sizes and SHA256
on durable storage. Leave completed checkpoints intact and release the allocation.

Classify calibration, native serialization, memory, serving smoke and numerical
comparison separately. On a partial failure, retain the completed lane and return
the first failing operation and raw evidence. Do not label the task fully qualified
from checkpoint hashing alone. Return for analysis before full-model quantization
or meaningful model-quality benchmarking; no full run is in this handoff.
