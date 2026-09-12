# FP8 weights before expert GPTQ, with direct W4AFP8 save

Implemented 2026-09-12 against branch `duy-branch` after pulling `383a4411`.

## Behavior and reuse

Group A is attention, indexer `wk`/`wq_b`, shared experts, and the initial dense
MLPs. Group B is routed INT4 experts. Before collecting Hessians in each resident
subgraph, the new opt-in helper observes A once, freezes its block-FP8 scales,
and replaces its working weights with the FP8 quantize/dequantize result. GPTQ
then sees A's weight error. Existing sequential replay carries A and B weight
error to later subgraphs. Activation QDQ remains disabled during this walk.
This is ordered weight calibration; it does not jointly optimize all weights or
activations and does not establish an accuracy improvement without evaluation.

The [pinned research](research/2026-09-12-glm53-fp8-int4-sequential-reuse.md)
identifies MoE-Quant's FP8-dequantized input weights and expert-only GPTQ as the
closest reuse precedent. Implementation uses existing CT observers, `fake_quantize`,
this fork's distributed parameter publisher, EP ownership and sequential replay.
No new GPTQ solver, FP8 rounding kernel, distributed cache or shard writer is added.

Each modifier now updates only its own targets, including in DDP. Prepared FP8
scales are retained through calibration finalization and export. Collective
preflight checks target ownership, schemes and storage metadata before onloading.
Disabled error propagation, overlapping targets and unsupported pipelines fail.

## Configuration and output

New recipes, separate from the historical W4A16 execution packet:

- `pipeline/configs/glm53_ep_gptq_w4afp8_representative.yaml`
- `pipeline/configs/glm53_ep_gptq_w4afp8_full.yaml`

They set `scheme: W4AFP8`, `fp8_scheme: FP8_BLOCK`,
`fp8_weights_before_gptq: true`, `checkpoint_format: sglang-w4afp8`, and
`gptq_expert_parallel: true`. The library modifier flag is
`quantize_weights_before_calibration: true`. Defaults remain opt-out.

The native-save adapter uses CT's module distribution/compression and the existing
Transformers safetensors writer. It repacks resident compressed INT4 values before
the first shard write. It preserves FP8 payloads, renames native scales, and widens
FP8 scales losslessly to FP32. No intermediate CT checkpoint or post-save converter
is invoked. The in-memory model returns to CT compressed layout after saving. Scale validation
adds a collective pass, hashing adds per-module CPU transfer/work, and restoring
the CT layout rewrites offload caches. End-to-end storage savings need measurement.

The serving contract is GLM native INT4 group 128 plus E4M3 128x128 FP8 blocks.
Expert input scales use the existing **fixed unit** policy; metadata explicitly
sets static MoE inputs and dynamic linear inputs. This is not activation-scale
optimization. Unsupported inventories and group activation ordering are rejected.
MTP is **absent** (`num_nextn_predict_layers: 0`); speculative decoding still needs
separately qualified draft-layer assembly.

`native_sglang_manifest.json` records full tensor inventory and save-time SHA256
hashes of all quantized tensors. The source-only offline gate reads the checkpoint
one tensor at a time and checks exact serialization without onloading distributed
model state. It does not certify SGLang execution or model quality.

## Qualification and execution

CPU tests cover independent FP8 inputs/Hessians, a two-block reference walk,
frozen scales/payload save and reload, DDP/EP numerical parity, CPU/disk collective
publication and saving, unsupported-contract rejection, and native byte conformance.
**138 CPU tests passed** across the targeted regression (100), native exporter
(25 plus 2 entrypoint tests), and real lifecycle (11) suites. Five NCCL cases
were skipped because this session has no two-GPU allocation. Ruff and
`git diff --check` pass. The installed environment is PyTorch 2.11.0+cu128,
compressed-tensors 0.17.2a20260707 and Transformers 5.12.1 on Python 3.12.

[Machine-readable evidence](evidence/2026-09-12-fp8-before-gptq-cpu.json)
records source hashes, per-rank comparisons and test results. The first routed
Hessians pass strict comparison. Across ranks, maximum relative Frobenius errors
are 0.023% for replay/Hessians and 0.086% for weight-only logits. Aggregate
weight-calibration comparisons use a 0.1% relative-Frobenius gate and a 1%
normalized-maximum gate; exact FP8 byte/scale checks remain exact.

Activation-enabled logits are recorded separately: maximum relative Frobenius
error is 0.312%, with maximum absolute difference 0.00518. They do **not** pass
the old pointwise DDP/EP logits tolerance in this enlarged fixture. Activation
rounding can amplify small weight differences, so enabled-activation DDP/EP
runtime equivalence remains unqualified. Each arm's activation-enabled output
still passes its own tight save/reload comparison. No old test tolerance is
relaxed.

The real GLM block-FP8 save/reload fixture uses multiple 128-wide column blocks:
the installed CT decompressor infers a one-column scale grid as per-channel, which
cannot represent multiple FP8 row blocks. Native SGLang uses explicit block
metadata. CPU coverage does not certify the generic CT loader for that ambiguous
geometry.

On a fresh GPU executor, first run the bounded two-GPU gate with the pinned
environment (the new cases are `early-fp8-parity` and `early-fp8-disk`):

```bash
PYTHONPATH=src:. python -m pytest -q tests/llmcompressor/modifiers/gptq/test_expert_parallel_nccl.py
```

Then adapt the representative recipe's source, offload and output paths to the
executor mounts and run a fresh representative EP4/EP8 qualification. Example
entrypoint, under the executor's existing resource wrapper:

```bash
PYTHONPATH=src:. torchrun --standalone --nproc_per_node=4 -m pipeline.run \
  --config pipeline/configs/glm53_ep_gptq_w4afp8_representative.yaml --stage quantize
```

The output can be checked independently with:

```bash
PYTHONPATH=src:. python -m pipeline.native_sglang_save /path/to/checkpoint
```

Require complete preparation/calibration/save phase records, bounded memory,
exact serialization verification, a pinned SGLang load/forward smoke and paired
quality comparison with identical calibration/evaluation data. Only fresh
representative evidence should unlock the new full recipe. Existing W4A16 results
do not qualify this different calibration path. No full-model or GPU job was
launched as part of this implementation.
