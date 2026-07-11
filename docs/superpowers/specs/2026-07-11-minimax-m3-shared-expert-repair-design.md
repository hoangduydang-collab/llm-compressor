# MiniMax-M3 Shared-Expert Repair Design

## Scope

Resolve the MiniMax-M3 quality issue before returning to CUDA-graph diagnosis.
The repair must reuse the existing checkpoint tensors, preserve every source
file, and require only runtime GPU execution from the executor agent.

## Confirmed boundary

Matrix `20260711-144120-routed-diagnostics` proves that both candidate schemes
construct zero-valued packed shared-expert parameters and produce zero shared
output on all 48 probes. The loader sees 171 shared tensors and leaves exactly
171 unmatched. The reference loads BF16 shared weights and has nonzero shared
output on every probe. Candidate W4A8 and W4A16 enter the first routed expert
identically, and their LM-head fingerprints match the reference.

The checkpoint intentionally keeps shared experts in BF16. Its persisted ignore
regex names the Transformers module path, `mlp.shared_experts`, while vLLM
constructs `block_sparse_moe.shared_experts`. Compressed Tensors therefore
mistakenly constructs packed shared parameters that cannot accept the existing
BF16 tensors.

## Repair

Create a metadata-only checkpoint overlay that appends this ignore alias:

```text
re:.*block_sparse_moe[.]shared_experts[.].*
```

Retain the existing Transformers regex. The operation must be idempotent,
reject a missing quantization configuration, symlink all payload files, copy
only `config.json`, and never mutate the source checkpoint. The permanent
pipeline recipe must also persist both naming variants for future checkpoints.

Do not patch vLLM matching logic, repack shared weights, rewrite tensor shards,
or re-quantize the model in this experiment.

## Validation matrix

Launch three concurrent exclusive `srun` allocations, one eight-GPU node each:

1. `repaired_w4a8_offline`: canonical offline quality plus loader,
   fingerprint, and MoE probes.
2. `repaired_w4a16_offline`: the same overlay with routed activation
   quantization disabled, preserving the activation control.
3. `repaired_w4a8_http`: canonical HTTP chat serving with the repaired W4A8
   overlay and the production MiniMax-M3 chat arguments.

All arms retain TP8, expert parallelism, eager execution, block size 128, FP8
KV cache, 2048 context, 0.85 GPU utilization, disabled custom all-reduce,
disabled shared-expert auxiliary stream, deterministic canonical prompts, and
the same pushed code/environment. Scheduler mechanics may change when recorded;
quality variables may not.

## Evidence and verdicts

Each arm records its overlay config hash and alias, source config/index hashes,
job/node, environment, return code, retries/deviations, raw canonical responses,
and retained full-log hashes. Offline arms also return structured fingerprints,
loader audit, and rank-aligned MoE probes.

The aggregate classifier requires all three arms and distinguishes:

- `quality_repair_pass`: W4A8 passes offline and HTTP, all 171 shared tensors
  match, runtime shared parameters are nonzero BF16 weights, and every first
  real-prompt shared output is nonzero;
- `activation_boundary_after_shared_repair`: repaired W4A16 passes while W4A8
  fails after shared loading is proven healthy;
- `shared_ignore_repair_failed`: shared tensors remain unmatched, packed/zero,
  or produce zero output;
- `candidate_interface_disagreement`: repaired W4A8 offline and HTTP differ;
- explicit infrastructure, invalid-control, or missing-evidence verdicts.

If `quality_repair_pass` is returned, the quality issue is resolved and the next
handoff resumes only the documented CUDA-graph issue. Otherwise, analysis stays
on the quality boundary selected by the classifier.

## Testing

CPU tests cover immutable/idempotent overlay creation, permanent recipe aliases,
all classifier branches, dry-run arm envelopes, exactly three exclusive `srun`
commands, and absence of `sbatch`. The executor reruns the focused pytest suite
before GPU execution.
