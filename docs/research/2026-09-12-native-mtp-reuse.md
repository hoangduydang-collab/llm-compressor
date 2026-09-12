# Native MTP assembly: source and implementation reuse

The owner approved automatic MTP assembly and direct native W4AFP8 output for both AWQ and GPTQ on 2026-09-12. This extends the existing native exporter, rather than introducing a replacement quantization algorithm.

## Sources inspected

The pinned [GLM-5.3-BF16 config](https://huggingface.co/zai-org/GLM-5.3-BF16/blob/304b8051cfb2b260b61ce0cbe330e02a98e73639/config.json), [weight index](https://huggingface.co/zai-org/GLM-5.3-BF16/blob/304b8051cfb2b260b61ce0cbe330e02a98e73639/model.safetensors.index.json) and bounded safetensors header reads confirm 791 tensors under layer 78: 768 routed expert matrices and 23 other tensors. The router correction bias is FP32; the other 790 tensors are BF16. No weight payload was downloaded for this investigation. Header shapes, dtypes and config/index hashes are recorded in [machine-readable evidence](../evidence/2026-09-12-native-mtp-source-contract.json).

[SGLang's NextN model](https://github.com/sgl-project/sglang/blob/v0.5.17/python/sglang/srt/models/deepseek_nextn.py) constructs its draft decoder separately from the target. Its `eh_proj` is an ordinary Linear for W4AFP8, and its draft norms use RMSNorm. This supports preserving those source tensors rather than giving them FP8 scales.

The [SGLang weight loader](https://github.com/sgl-project/sglang/blob/v0.5.17/python/sglang/srt/models/deepseek_common/deepseek_weight_loader.py) requires one NextN layer. It maps draft-specific norms/projection to the NextN model and decoder weights to the draft decoder; target embeddings and head are shared. W4AFP8 adds the special expert input-scale mapping for `w1/w2/w3.input_scale`. The depth-one legacy special case uses layer 0 rather than layer 1 and must not be silently applied to a normal appended-layer fixture.

## Existing code to reuse

- `pipeline/native_sglang_save.py`: collective CT compression, native INT4 packing, exact FP8 preservation, config and manifest writing, restoration after save, serialization gate.
- `pipeline/graft_mtp_w4afp8.py`: established group-128 INT4 RTN convention (max/8, clamp to [-8, 7]), classification and FP8 quantization primitives.
- `pipeline/sglang_w4afp8_kernels.py`: existing nibble packing and block-FP8 kernels.
- `pipeline/graft_mtp_head.py`: inventory/header checks and shard safety patterns.
- `huggingface_hub` cache resolution: resolve the main model's loaded revision, locally, without downloading another checkpoint.

The missing integration is source validation before calibration, method-independent native preflight, source-only MTP finalization and manifest publication with failure recovery. Calling the old graft unchanged does not supply these guarantees and emits different expert input-scale names. The integrated path uses the existing quantizers and writes only the missing MTP shards.

Draft expert RTN does not use AWQ smoothing or GPTQ Hessians. Serialization checks establish artifact integrity; the executor must separately qualify the actual pinned runtime's draft load/forward and speculative decoding behavior. No inference-speed or acceptance-rate improvement is established by this implementation.
