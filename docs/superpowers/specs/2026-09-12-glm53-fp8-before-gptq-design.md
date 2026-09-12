# GLM-5.3 FP8 weights before EP GPTQ

Approved by the owner in the 2026-09-12 conversation after the [reuse investigation](../../research/2026-09-12-glm53-fp8-int4-sequential-reuse.md).

Group A (attention, shared experts, dense prefix and indexer wk/wq_b) must have block-FP8-rounded weights before group B (INT4 routed experts) collects GPTQ Hessians. Reuse compressed-tensors observation, quantization and offload APIs, following IST-DASLab MoE-Quant's dequantized-working-weight pattern. Process one resident decoder block at a time. Keep the existing GPTQ mathematics, expert ownership, activation policy and sequential error propagation.

Expose an opt-in FP8 preparation setting for GPTQ recipes. Validate disjoint targets, static symmetric block FP8, and sequential execution before calibration. Materialize the working weight using its frozen scales before the block's first statistics forward. Skip later weight observation for prepared modules so export cannot choose new scales. Keep legacy recipes and the running W4A16 experiment reproducible. Add separate W4AFP8 full and representative recipes rather than mutating the active packet.

Direct SGLang save should reuse the current distributed compressed-tensors save path and existing INT4 packing/FP8 rename helpers. Emit the native layout on the first shard write, with explicit runtime activation metadata and honest MTP presence. Never invoke a whole-checkpoint converter as a hidden second pass. Reject unsupported layouts before expensive calibration. The existing SGLang activation policy must be explicit; calibration continues to isolate weight error. Optional MTP assembly must reuse existing source/graft helpers if implemented, and absent MTP must be marked absent.

Tests must compare expert calibration inputs/Hessians against an independently FP8-prepared reference, demonstrate downstream propagation, preserve frozen FP8 scales through save/reload, and exercise two-process EP including disk offload. Native export needs exact integer/FP8 conformance, collective save, and unsupported-contract rejection. Full H100/runtime quality qualification belongs to a fresh executor packet, not this local CPU implementation.
