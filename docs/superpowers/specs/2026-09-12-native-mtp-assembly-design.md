# Objective-2 continuation: direct AWQ and optional native MTP assembly

Date: 2026-09-12. Status: approved by owner; includes direct AWQ export per explicit follow-up.
Implementation baseline: `c3872fe8`; existing MTP helper baseline: 32 CPU tests
passed in 3.68 seconds. Implementation and CPU validation authorized; GPU qualification remains executor work.

## Problem and scope

Direct native GPTQ W4AFP8 export currently declares MTP absent. Serving with a
speculative draft layer still requires a separate graft operation. Objective 2
calls for a final serving-format artifact without those manual repair steps.
This proposal integrates optional MTP assembly into the export lifecycle using
the same BF16 source snapshot as the main quantization job.

Direct AWQ export is included: use the same native writer as GPTQ, skip transform-only modifiers when resolving schemes, and preserve AWQ calibration/folds. GPTQ retains its early-FP8 requirement. Activation-scale optimization remains separate.
The current representative/full recipes and queued GPU test retain their existing
absent-MTP behavior unless an explicit new option enables assembly.

## Reuse and alternatives

1. **Recommended: integrate the existing MTP quantization/graft primitives as a
   final export-assembly stage.** Emit only new MTP shards and small metadata;
   main-model shards are never converted or rewritten. Stage additions before
   committing the new inventory. This needs source/preflight and manifest work,
   but does not need another distributed model implementation.
2. Add an MTP model to the main calibration and collective writer. This can
   share one shard writer but requires a separate model/forward contract and
   distributed handling for a layer the Transformers main model does not load.
   It is larger than the assembly requirement and changes calibration scope.
3. Keep the standalone post-save graft. Existing code works for historical
   artifacts, but it leaves the manual step that this continuation should remove.

Existing `pipeline/graft_mtp_w4afp8.py` supplies INT4 group-128 RTN, FP8 block
quantization, classification and safetensors writing. Existing
`pipeline/graft_mtp_head.py` supplies source inventory and coverage checks.
Reuse the native exporter/manifest verifier and the established SGLang expert
scale-name mapping. Reconcile the old graft's input-scale names against the
actual native contract; do not blindly append a legacy-format shard.

Primary runtime reference: [SGLang v0.5.17 NextN implementation](https://github.com/sgl-project/sglang/blob/v0.5.17/python/sglang/srt/models/deepseek_nextn.py).
It constructs the draft decoder separately and delegates weight loading with
`is_nextn=True`; its MTP path must be qualified independently of main-model load.
The actual executor runtime revision remains authoritative for serving tests.

## Approved behavior

- Add an opt-in native-export MTP policy, default absent. Source-RTN assembly is
  valid only for native GLM W4AFP8 output with a matching, complete BF16 source. Preserve the source FP32 router correction bias.
- Before expensive calibration, validate source config/index/header metadata,
  source/target depth and architecture, expected expert IDs/projections, required
  draft tensors, supported dtypes/geometries and destination collisions. Use the
  main job's source path; do not select a different vendor checkpoint silently.
- Quantize the draft's routed experts by the existing RTN convention, its defined
  attention/shared/indexer weights by existing block-FP8 kernels, and preserve
  the designated BF16 tensors. Retain explicit fixed-unit expert input scales.
  Record that draft experts are RTN, not calibrated GPTQ.
- Assemble on the source process after collective main-model saving/restoration
  has finished, at a stage where other ranks need not enter another collective.
  Read bounded tensors and use bounded new-shard buffers.
- Stage MTP files separately, verify their full inventory and hashes, then
  publish index/config/native-manifest changes. Ordinary failures restore the
  original metadata and remove only this attempt's new files. An incomplete
  assembly marker makes an interrupted artifact fail verification until repaired.
- Set MTP presence true only after complete assembly. Extend the native manifest
  with layer number, source provenance, RTN policy and hashes for all added
  tensors, including copied BF16/FP32 tensors. The standalone verifier supports both
  absent and assembled policies and rejects partial/mismatched metadata.

## Validation and boundaries

CPU tests must cover full tiny source inventory, exact reused quantizer output,
source mismatch/missing tensors before calibration, correct input-scale names,
main-shard byte preservation, duplicate assembly, partial-write/metadata failure,
manifest corruption, and the pipeline's source-only finalization order.
Re-run existing graft and native exporter suites. Preserve absent-MTP behavior.

No full checkpoint conversion or new GPU allocation is required for this local
implementation. Speculative serving still requires the executor's pinned-runtime
load/forward and acceptance-rate checks; successful assembly is not proof of
speculative correctness, speedup or main-model quality improvement.
