# Final independent review: direct AWQ and native MTP

Reviewed base `c3872fe8` through head `cb9f0b1c` using the supplied complete implementation/design/docs diff, Task 2 brief and report, implementation plan, and committed CPU validation evidence. This review is read-only with respect to the repository; only this requested report was written under `/tmp`. No tests were rerun, no GPU work was launched, and no git state was changed.

**Task 2 spec verdict: PASS.**

**Overall approval verdict: APPROVE.** Task 1 already has an independent PASS/APPROVE; this review also checked its integration with Task 2 and the final documentation.

## Findings

- Critical: none.
- Important: none.
- Minor: none requiring a change within this implementation scope.

## Reviewed contracts

- **Source identity and early rejection:** `pipeline/native_mtp.py:257` resolves Hub IDs through the existing cache helper with the loaded config's `_commit_hash`; it never falls back to a moving revision. Local snapshot paths retain individual cache symlinks. Metadata/header bounds, duplicate JSON keys, source config/architecture/depth, complete expert IDs/projections, required draft copies, dtype and geometry are checked before `oneshot`. Config/index/header fingerprints and file identities are retained and rechecked across calibration and while source tensors are read. Destination preflight runs before calibration when saving is enabled.
- **Closed inventory and native ABI:** source inventory is explicitly constructed from GLM geometry, with the depth-one runtime ambiguity rejected. The recorded full source has 791 entries and native output has 2337. The F32 correction bias is preserved alongside BF16 copies. Expert weights use the existing RTN and nibble packer, with existing block-FP8 conversion for the ten designated modules, including non-block-divisible output dimensions. Expert input scales use `.w1/.w2/.w3.input_scale` and fixed unit BF16 values. No new quantization algorithm was introduced.
- **Payload integrity:** staged tensor names, geometry, dtype and hashes must agree before publication. Every added tensor, including BF16/F32 copies, receives a required final-verifier hash. The verifier independently reconstructs the closed draft inventory and checks presence, layer, RTN/fixed-unit policy, source fingerprint structure, copy metadata and separate MTP shard membership. Existing absent-MTP manifests remain supported.
- **Publication and failure behavior:** `pipeline/native_mtp.py:489` creates an exclusive marker before staging. Shard links cannot overwrite collisions; successfully created filenames and replaced metadata are tracked independently. Ordinary failures restore original metadata and remove this attempt's files. Interruptions and rollback failures retain a marker that both verification and repeat assembly reject. This is guarded transactional publication; it does not claim that several filesystem metadata replacements form a single filesystem transaction. Single-file main artifacts gain an index without replacing their main shard; indexed artifacts retain main shard names and bytes.
- **Storage and lifecycle:** assembly reads only existing main shard headers and metadata. It does not load or hash main tensor payloads during staging. Source payloads are read one tensor at a time with a bounded output buffer. `pipeline/quantize.py:1534` places assembly on the source process after the final save barrier and before the existing whole-artifact verifier. The saving path has no later collective. Both full AWQ/GPTQ recipes enable native output and `source-rtn`; the default and representative policy remain `absent`. Recipe provenance records the selected policy.
- **Collective fixture repair:** `pipeline/tests/test_native_sglang_save.py:347` still verifies the native artifact and exact restored compressed keys, values, dtypes, quantization status, config and hook state on every rank while using shared cache state. CT's existing `remove_dispatch(onload_tensors=True)` then replaces cache mappings with onloaded CPU dictionaries without deleting shared cache entries. A test-only barrier precedes repeated restoration assertions and finite forward on each rank. The single-process disk-forward test remains unchanged. This preserves the intended serialization/restoration checks while avoiding the dependency's explicitly unsupported distributed decompression operation; it does not hide an artifact failure or assert distributed inference support.
- **Documentation and qualification:** the implementation guide, objectives and executor handoff consistently distinguish main AWQ/GPTQ calibration from RTN draft quantization. They describe same-source resolution, automatic assembly, recovery markers, fixed-unit scales and the absent representative policy. Separate AWQ and MTP runtime load/forward requirements remain explicit. CPU serialization results are not presented as SGLang runtime, acceptance-rate, throughput or paired-quality qualification, and the handoff does not authorize a new full run.

## Targeted inspection outside the supplied diff

Only concrete integration risks motivated the following additional reads:

1. `pipeline/quantize.py` model-loading and function-tail code: verified that preflight receives the actual loaded model config and that barriers later in the file belong to the non-saving branch; distributed sample generation remains disabled.
2. `pipeline/serve_ignore.py:65`, `pipeline/graft_mtp_head.py:66` and `:80`: verified that single-file weight-map discovery is header-only and that reused graft helpers reject an existing draft and incorrect target depth.
3. Existing `classify`, `quantize_int4_group_rtn`, and `pipeline/sglang_w4afp8_kernels.py` implementations: verified the reused classification, scale convention, nibble packing and FP8 padding/cropping assumptions.
4. Installed CT `ModelCompressor.decompress_model`, `remove_dispatch`, `remove_module_offload`, and `DistributedDiskCache`: verified the explicit distributed-decompression limitation and that the fixture's materialization step does not delete shared cache files.
5. Existing test setup and unchanged verifier tail: verified CPU onload destinations, remaining restoration assertions, and final per-tensor hash comparisons.

## Validation evidence and limits

`results/glm53-native-awq-mtp/20260912-cpu/validation.json` records **144 passed, zero failures/errors/skips in 107.09 seconds**, with the real AWQ lifecycle, real GLM native-writer/MTP composition, source mutation, malformed inventory, collisions, corruption, rollback and post-barrier ordering coverage. The reports record passing Ruff and diff checks. Original failing collective-forward evidence and the repaired focused run are preserved. This review accepted those completed checks and did not rerun them.

No approval blocker remains in the reviewed implementation. SGLang GPU target/draft load, forward, speculative behavior, performance and quality remain executor qualification work.
