# MiniMax-M3 AWQ/GPTQ Repair Matrix Design

## Goal

Resolve the remaining MiniMax-M3 quality failure by determining whether the
layer-8 anomaly is specific to AWQ smoothing, while bringing the existing GPTQ
checkpoint through the same confirmed vLLM loader repairs and quality checks.

## Checkpoint preparation

The AWQ portable checkpoint remains the control. The original GPTQ checkpoint
is re-exported once to the same `w1/w2/w3` routed-expert naming convention;
shared experts use the confirmed vLLM-native ignore alias. Re-exported payloads
remain byte-identical and immutable.

## Static audit

A streaming Safetensors audit compares BF16 base, working reference, repaired
AWQ, and repaired GPTQ tensors at every sparse layer 3-59. It records norm and router tensor
statistics, derives the AWQ scale vector from base/candidate normalization
weights, and checks whether router/shared input weights preserve the expected
elementwise compensation. Layer 8 is mandatory and missing tensors fail loudly.

## AWQ repair hypothesis

MiniMax-M3 uses MiniMaxM3VLRMSNorm, whose effective weight is 1 + weight.\nRegister that class with the existing offset-norm calibration context so AWQ\nsmooths the effective weight and restores it as smoothed_weight - 1. This is\nthe primary repair. Also add an explicit MiniMax-M3-only configuration switch\nthat removes only the
`post_attention_layernorm -> MoE input projections` AWQ mapping. Attention
smoothing and expert `up_proj -> down_proj` smoothing remain unchanged. A fresh
checkpoint produced with this switch tests whether pathological MLP-input
smoothing causes the layer-8 failure.

## Parallel matrix

Run working reference, repaired AWQ control, repaired GPTQ W4A8/W4A16/HTTP,
and no-MLP-input-smoothing AWQ W4A8/W4A16/HTTP arms concurrently once the two
prepared checkpoints exist. All offline arms collect the existing layer 3-9
boundary and parameter evidence. CUDA graphs stay disabled.

Classification first decides whether repaired GPTQ passes. If it does, the
remaining defect is AWQ-specific. It then checks whether the no-smoothing AWQ
checkpoint passes or removes the layer-8 explosion. If GPTQ fails at the same
boundary, the verdict moves to shared compression/export logic; a different
boundary is reported separately.

## Handoff

The executor returns checkpoint preparation manifests, the static scale audit,
all arm reports and boundary records, exact job/node/return codes, deviations,
retries, and retained-log hashes. `srun` is the only scheduler interface.
