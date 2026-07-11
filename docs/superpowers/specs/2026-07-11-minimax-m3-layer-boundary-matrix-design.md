# MiniMax-M3 Layer-Boundary Matrix Design

## Scope

Continue only the MiniMax-M3 quality investigation. The shared-expert repair is
retained, CUDA graphs remain disabled, and the cluster is used for maximum
parallelism: independent hypotheses run on separate exclusive eight-H100 nodes.

## Confirmed boundary

Matrix `20260711-152808-shared-repair` proves that all 171 shared-expert tensors
now match and every sampled shared output is finite and nonzero, yet W4A8 and
W4A16 still fail. Both candidates stay bounded through five sampled MoE calls
and enter the sixth at roughly `176,000` norm, while the reference remains near
`177`. The next experiment must identify the exact layer and component that
first creates that jump.

## Instrumentation

Add an environment-gated, eager-only boundary probe to the installed vLLM M3
model module. For the first canonical prefill and selected layers 3 through 9,
emit bounded structured records at:

- decoder input hidden state and residual;
- attention input and output;
- MoE input and output;
- decoder output hidden state and residual.

Every record includes rank, layer, boundary, token count, shape, finite
fraction, L2 norm, absolute maximum, and a bounded sample digest. The hook must
skip CUDA-graph capture, zero/dummy inputs, generation steps, and repeated calls.
It must use the runtime `layer_id` rather than global call order.

Extend parameter/load auditing to include `block_sparse_moe.gate` as
`moe_router`. This establishes whether the router is already the intended FP32
`GateLinear` and whether its checkpoint tensor reaches the runtime parameter.

## Parallel matrix

Launch eleven independent `srun` arms concurrently:

1. Reference W4A16, EP on, FP8 KV control.
2. Repaired candidate W4A8, EP on, FP8 KV control.
3. Repaired candidate W4A16, EP on, FP8 KV control.
4. Candidate W4A8 plus the vLLM-native router ignore alias.
5. Candidate W4A16 plus the same router alias.
6. Reference W4A16 with EP off.
7. Candidate W4A8 with EP off.
8. Candidate W4A16 with EP off.
9. Candidate W4A8 with BF16/auto KV cache.
10. Candidate W4A16 with BF16/auto KV cache.
11. Router-alias W4A8 through canonical HTTP serving.

The router, EP, and KV arms each change one variable relative to their matching
control. The HTTP arm runs concurrently so a successful router repair can close
the interface check without another cluster round. All offline arms collect the
same layer/router evidence. TP8, eager execution, deterministic prompts, EP
unless named otherwise, disabled custom all-reduce, disabled shared-expert
stream, block size 128, context 2048, and all existing serving patches remain
fixed.

## Evidence and decisions

Each arm returns a manifest, normalized quality output, structured boundary
records, router and existing fingerprints, loader summaries, environment, job
and node identity, compact log excerpts, and retained-log hashes. Aggregation
reports quality per arm, router health, the first non-finite or explosive
boundary, and cross-arm boundary comparisons.

Interpretation is evidence-led:

- router alias restores quality or moves the first divergence: confirm router
  construction/loading and retain the alias;
- EP-off restores quality: isolate expert sharding/collective layout;
- auto/BF16 KV restores quality: isolate main or sparse-index cache handling;
- attention output is the first divergent boundary: investigate fused
  QKV/indexer/attention runtime;
- MoE output is first: investigate routed expert payload, routing, and scaling;
- only decoder residual/output diverges: investigate deferred all-reduce and
  residual fusion.

No arm result is treated as a fix unless its matching reference/control is
valid and the changed variable alone explains the boundary movement.

## Testing and handoff

CPU tests cover injection syntax/idempotence, router tracking, structured log
parsing, overlay aliases, all arm envelopes, exactly eleven concurrent exclusive
`srun` commands, classifier branches, and absence of `sbatch`. The executor runs
the focused tests and dry-run before launching all arms, commits the compact
evidence bundle, reports every job/node/deviation/retry, and stops for analysis.
