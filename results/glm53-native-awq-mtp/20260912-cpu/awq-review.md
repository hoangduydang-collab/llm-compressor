# Task 1 independent scoped review

Reviewed immutable diff `/tmp/native-awq-task1-review.diff`, requirements `/tmp/native-awq-mtp-task-1-brief.md`, and validation report `/tmp/native-awq-mtp-task-1-report.md` for base `829ee0f4` to head `8e30719d160b6686476235af31b0001be0e69844`.

Spec-compliance verdict: PASS for Task 1.

Code-quality verdict: APPROVE. No actionable findings in the scoped change.

The config accepts plain AWQ and GPTQ only for W4AFP8 with FP8_BLOCK rest targets and retains GPTQ's mandatory early FP8 preparation. Concrete target overlap and serving-layout checks remain in the existing preflight. Scheme resolution explicitly uses QuantizationMixin, which includes both quantization modifier classes and excludes transform-only AWQModifier. The full AWQ recipe changes only checkpoint format; calibration and folding behavior are preserved. Saving continues through the existing collective native writer with no converter or rewritten main-model shards.

The added coverage exercises entrypoint routing for both methods and an actual sequential GLM AWQ lifecycle. It checks native first writes, verifier acceptance, independently decoded INT4 nibble equality, FP8 byte/scale equality, fixed-unit expert input scale, folded norm preservation, and restored CT compressed state. Existing shared-writer fixtures provide disk-offload and collective coverage; the actual AWQ lifecycle test itself remains CPU/in-memory, as disclosed.

Validation assessment: relied on the supplied exact test results and did not rerun tests. The full native suite was not fully green: 29 passed and 1 failed in an existing two-process disk-cache restored-forward check; the unchanged isolated rerun passed. Its ordering before the new AWQ lifecycle and lack of writer/cache changes provide no concrete causal link to this patch, but do not prove an unrelated root cause. Preserve the reported failure logs and treat collective disk-offload execution as an unresolved runtime qualification concern. This review does not qualify full-scale GPU execution or serving.

Read-only implementation review; no source changes or commits. Only this requested review artifact was written.
