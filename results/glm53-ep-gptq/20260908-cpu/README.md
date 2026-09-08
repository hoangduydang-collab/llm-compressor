# CPU validation

Result: **212 passed, 4 CUDA-dependent skips**, 88.04 seconds, `qualified.log`.
Environment and limitations: [implementation report](../../../docs/glm53-ep-gptq-implementation.md).

Exact final command (existing interpreter; no package upgrades):

```bash
env -u SLURM_STEP_ID OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 .venv-prequant/bin/python -m pytest -q -rs -o tmp_path_retention_policy=all --basetemp=/tmp/pytest-of-e1129930/glm53-ep-qualified-cpu tests/llmcompressor/modeling/moe/test_expert_parallel.py tests/llmcompressor/modeling/moe/test_expert_parallel_equivalence.py tests/llmcompressor/modeling/moe/test_expert_parallel_integration.py tests/llmcompressor/modifiers/gptq/test_gptq_quantize.py tests/llmcompressor/modifiers/gptq/test_hessian_reduce_offload.py tests/llmcompressor/modifiers/gptq/test_expert_parallel_plan.py tests/llmcompressor/modifiers/gptq/test_expert_parallel_distributed.py tests/llmcompressor/modifiers/gptq/test_expert_parallel_oneshot.py tests/llmcompressor/modifiers/gptq/test_expert_parallel_nccl.py tests/llmcompressor/modifiers/quantization/test_qparam_broadcast_offload.py tests/llmcompressor/transformers/compression/test_packed_offload_parameter_types.py tests/llmcompressor/transformers/compression/test_save_coordination.py pipeline/tests/test_metrics.py tests/llmcompressor/utils/test_phase_logging.py pipeline/tests/test_save_prewarm.py pipeline/tests/test_distributed_quantize_contract.py pipeline/tests/test_distributed.py
```

`baseline.log`: pre-change 98-test baseline.
`ambient-slurm-run.log`: initial combined attempt, ambient SLURM snapshot assertion.
`retained-path-run.log`: second attempt, tests passed but teardown counted retained
artifacts outside the fixture's recognized pytest temporary directory as leakage.
`qualified.log`: passing run with the ambient step variable unset and retained
artifacts under the recognized directory. No existing assertion was bypassed.
Packed differences are from the second run's passing tiny pure-GPTQ lifecycle
case: 21 tensors per rank, zero differing words. This does not imply GPU/full-size
bitwise equivalence. The fourth skipped test is a single-GPU accumulator test;
three skips are the two-GPU NCCL lifecycle cases.
