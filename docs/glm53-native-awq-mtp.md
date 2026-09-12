# Direct GLM W4AFP8 output for AWQ/GPTQ, with integrated MTP

Implemented on `duy-branch`, 2026-09-12. CPU verification is recorded below; full-scale serving remains an executor qualification task.

MTP is the extra draft prediction layer used for speculative decoding. The BF16
release includes it, but the main Transformers model does not load/save it. Adding
it back lets a serving engine propose tokens with the draft and verify them with
the target model; useful proposals can improve generation speed. Ordinary decoding
does not require MTP, and this assembly does not improve main-model quantization
accuracy. The draft receives RTN/block-FP8 quantization without AWQ/GPTQ calibration.

## One pipeline invocation

Both GLM-5.3 full recipes select `checkpoint_format: sglang-w4afp8` and `mtp_policy: source-rtn`:

- `pipeline/configs/glm53_distributed_w4afp8_awq_full.yaml`
- `pipeline/configs/glm53_ep_gptq_w4afp8_full.yaml`

The quantize stage writes native SGLang tensors on the first main-model checkpoint save and assembles MTP automatically. It does not invoke `to_sglang_w4afp8`, `patch_indexer_fp8` or a standalone graft command. Main shards are not rewritten for MTP assembly. Use the final checkpoint directly for the runtime qualification gate.

Main routed experts retain the selected AWQ/GPTQ method. Attention, shared/dense MLPs and indexer `wk`/`wq_b` use block-FP8. GPTQ retains `fp8_weights_before_gptq: true` so its downstream Hessians see those rounded weights. AWQ keeps its existing smoothing/fold lifecycle; this change extends its serialization path.

MTP is loaded from the same source revision as the main model. Its routed experts use the existing group-128 INT4 RTN routine, its designated attention/shared/indexer weights use existing block-FP8 kernels, and its remaining tensors keep their source dtype. In particular, the source router correction bias is FP32. Draft RTN is explicit provenance; it is not calibrated AWQ/GPTQ.

The library default is `mtp_policy: absent`. The five-layer representative GPTQ recipe remains absent-MTP because a truncated main model cannot be paired with the original full-depth draft layer. `source-rtn` requires the matching complete BF16 source, including its MTP tensors; a main-only subset must fail rather than silently omit MTP.

## Verification and recovery

The final native manifest records MTP presence, layer number, source provenance, quantization policy and hashes for every added tensor. The offline native verifier checks the combined artifact. Source metadata/inventory checks run before calibration; new MTP shards are staged and checked before final metadata publication. Interrupted assembly is marked by `.native_mtp_incomplete.json`, and ordinary publication failures restore original metadata and remove only files created by the failed assembly. Preserve an interrupted directory for diagnosis; deleting its marker alone does not establish that publication finished.

```bash
PYTHONPATH=src:. python -m pipeline.native_sglang_save /path/to/final/checkpoint
```

This gate verifies serialization. The actual executor runtime must separately load and forward both the target and draft model and check speculative behavior. Fixed-unit expert activation scales remain the current explicit policy; this work does not optimize them or establish a quality, acceptance-rate or throughput gain.

See [reuse research](research/2026-09-12-native-mtp-reuse.md), [source evidence](evidence/2026-09-12-native-mtp-source-contract.json), and [executor handoff](glm53-native-w4afp8-executor-handoff.md).


## CPU regression finding

The initial combined run exposed an existing test error: after collective saving,
both ranks triggered CT decompression against shared disk-cache paths. The installed
`ModelCompressor.decompress_model` explicitly lacks distributed decompression; its
independent replacements can delete a file before another rank reads it. The
fixture now checks restored compressed values on shared disk on every rank, then
uses CT's existing `remove_dispatch(onload_tensors=True)` to materialize private
CPU state for each rank's forward check. Single-process disk-forward coverage is
retained. This does not add support for distributed CT inference; the production
pipeline already disables distributed post-save sample generation.

The original failures and installed-source fingerprints are preserved in
[CPU evidence](../results/glm53-native-awq-mtp/20260912-cpu/).


## Final validation

The combined targeted suite passed **144 tests with zero failures, errors or
skips** in 107.09 seconds. Ruff and `git diff --check` passed. Coverage includes
real AWQ calibration/native save, a real Transformers GLM native-save-to-MTP
composition, exact preserved tensors, both single-file/indexed artifacts,
source identity checks, publication failures and the repaired collective fixture.
[Machine-readable results and raw logs](../results/glm53-native-awq-mtp/20260912-cpu/validation.json)
record the environment and earlier failures. SGLang runtime and quality checks
remain in the executor handoff; no full quantization or new GPU job was launched.
