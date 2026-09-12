# Mixed calibration CPU smoke — 2026-09-12

Reproducible synthetic local inputs and the real pinned GLM tokenizer; no model weights,
model forward passes, quantization runs or GPU allocations. This verifies data
preparation/consumption, not quantized model quality or production dataset access.

The input records are synthetic (8 generic conversations, 8 agent sessions), with
long predictable prefixes and later reasoning/tool/patch turns. The preparation
command succeeded. Both mixed pipeline configs consumed identical token IDs;
all selected agent windows had nonzero offsets and assistant markup. Runtime
padding/truncation settings did not alter the tokenizer identity hash.
See integration-result.json for raw observed counts, offsets, anchor turns,
preview text and both consumer hashes. prepare.log preserves dataset-loader output.

To reproduce preparation from the repository root, using the quantization Python
environment and a new output directory:

```bash
python results/mixed-calibration/20260912-cpu/generate_smoke_inputs.py \
  --output /tmp/mixed-calibration-inputs
python -m pipeline.prepare_calibration \
  --config /tmp/mixed-calibration-inputs/mix.yaml \
  --tokenizer zai-org/GLM-5.3-BF16 \
  --tokenizer-revision 304b8051cfb2b260b61ce0cbe330e02a98e73639 \
  --output /tmp/mixed-calibration-reproduction
```

Load that directory through `build_calibration_dataset_with_partition` with each
mixed pipeline config's calibration replaced with `prepared_dataset` pointing to
it, `num_samples=8`, `max_seq_length=128`. Verify the returned partition manifests
against integration-result.json. The original smoke used the same tokenizer
revision from its local snapshot path. Names/paths do not enter the behavior hash.
