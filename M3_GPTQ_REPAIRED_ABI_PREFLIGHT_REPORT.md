# MiniMax-M3 Repaired GPTQ ABI Preflight

## Result

The direct GPTQ checkpoint failed the expected static ABI check, then an
immutable portable serving overlay was created and passed the full CPU-only
preflight.

Run root:

`results/m3-quality/20260712-142048-m3-gptq-repaired`

Focused verification:

```text
34 passed
```

## Direct checkpoint

The source checkpoint failed with `valid: false` and 228
`plain_runtime_module_not_ignored` errors covering the known
`block_sparse_moe.gate` and `block_sparse_moe.shared_experts.*` namespace
boundary.

The direct report is preserved at:

`static_direct/inhouse_gptq.json`

## Portable overlay

The overlay was generated without changing tensor payloads. Provenance checks
passed:

- source and overlay config hashes are distinct;
- source and overlay Safetensors index hashes are identical;
- vLLM router and shared-expert ignore aliases were added;
- `tensor_payload_unchanged` is `true`.

The overlay checkpoint itself is intentionally excluded from the committed
evidence bundle.

## Repaired aggregate preflight

All three active models were inspected before any dynamic imports or GPU work:

```text
bf16 ABI_VALID
inhouse_gptq ABI_VALID
cyankiwi_awq ABI_VALID
SAMPLE_BOUNDS_VALID
OVERLAY_PROVENANCE_VALID
```

The preflight completed successfully in about 79 seconds. It emitted:

- all three `preflight/serving_abi/*.json` reports;
- checkpoint diagnostics for every model;
- smoke and production probe corpora;
- smoke and production sample manifests;
- resolved task/config metadata and hashes.

No GPU serving or quality arm was launched by this step. The next authorized
step is the parallel smoke run using the repaired GPTQ overlay, cyankiwi AWQ,
and BF16/Ray control.
