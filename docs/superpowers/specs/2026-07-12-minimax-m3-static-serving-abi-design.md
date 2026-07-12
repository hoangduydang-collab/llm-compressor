# MiniMax-M3 Static Serving ABI Gate Design

## Goal

Reject compressed MiniMax-M3 checkpoints before GPU allocation when their
quantization metadata cannot describe the tensors under vLLM's runtime module
namespace. Runtime probes confirm numerical behavior; they do not discover
config/index mismatches.

## Contract

The gate reads only `config.json` and the Safetensors index. It builds the
actual packed/plain module inventory, derives Transformers aliases for vLLM
fused names, and evaluates every compressed-tensors ignore entry against both
inventories. It records per-pattern source/runtime match counts.

Known quantizable modules stored as plain weights—attention projections, dense
MLPs, routers, shared experts, vision linears, MSA indexers, and `lm_head`—must
match an ignore rule in the vLLM namespace. Packed modules must not match an
ignore rule. Every packed module must have a scale. Required categories with
only source-side matches are fatal namespace mismatches, and required rules
matching no runtime modules are fatal.

The preflight writes one ABI report per active quantized checkpoint and raises
before launch-plan or GPU execution when any report fails. BF16 is exempt from
the compressed-tensors contract.

## Related deterministic fixes

Smoke manifests allocate at least one sample to every group leaf because
lm-eval interprets an empty leaf list as unlimited. MiniMax-M3 MMLU-Pro uses
`exact_match,custom-extract` under lm-eval 0.4.12. The distributional probe
accepts and forwards the distributed executor backend so BF16 TP16 uses Ray.

## Verification

CPU tests reproduce the failing checkpoint shape: Transformers shared-expert
ignore names with vLLM `block_sparse_moe` tensor names must fail. Compact
cyankiwi-style vLLM regexes must pass. Tests also cover packed-module conflicts,
MMLU leaf allocation, the metric contract, and BF16 backend propagation.
