# MiniMax-M3 Static Serving ABI Gate Report

## Result

The new CPU-only serving ABI gate ran successfully and correctly blocked GPU
execution for the in-house GPTQ checkpoint.

Run root:

`results/m3-quality/20260712-140347-m3-static-abi`

Validation before preflight:

```text
38 passed
```

## Failure

Preflight stopped with:

```text
ValueError: static serving ABI validation failed for inhouse_gptq:
language_model.model.layers.10.block_sparse_moe.gate,
language_model.model.layers.10.block_sparse_moe.shared_experts.down_proj,
language_model.model.layers.10.block_sparse_moe.shared_experts.gate_proj
```

The complete report is:

`preflight/serving_abi/inhouse_gptq.json`

It reports:

- `valid: false`
- 21,888 quantized routed-expert modules
- 852 plain quantizable modules
- 57 unignored runtime routers
- 171 unignored runtime shared-expert modules

The representative source/runtime mappings are:

```text
model.language_model.layers.10.mlp.gate
  -> language_model.model.layers.10.block_sparse_moe.gate

model.language_model.layers.10.mlp.shared_experts.down_proj
  -> language_model.model.layers.10.block_sparse_moe.shared_experts.down_proj

model.language_model.layers.10.mlp.shared_experts.gate_proj
  -> language_model.model.layers.10.block_sparse_moe.shared_experts.gate_proj
```

These runtime modules are plain and are not matched by the checkpoint's
compressed-tensors ignore patterns. This is a static metadata/namespace
mismatch, not a vLLM kernel-quality result.

## Interpretation

The gate confirms that the GPTQ checkpoint cannot currently claim a valid
Transformers-to-vLLM serving contract. No GPU serve or quality run was
started from this preflight. The failure is consistent with the earlier
garbage-output investigation: routed experts are quantized, while the
runtime router/shared-expert modules are left plain without corresponding
ignore coverage.

BF16 is exempt from this compressed-tensors ABI contract. The AWQ control
must be evaluated independently after its own ABI report is generated.

Do not bypass this gate with a runtime patch. The next implementation should
either re-export/configure the checkpoint with vLLM-compatible ignore rules
or explicitly prove that the runtime module aliases are covered without
changing tensor data.
