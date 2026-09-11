# GPTQ reachable-path validator hardening

## Scope

Harden only the path executed by the GLM-5.3 EP GPTQ chain:

`representative checkpoint -> static validator -> phase/peak report -> memory
gate -> logits gate -> full checkpoint validator`.

The pipeline emits two checkpoint layouts:

- representative lanes: one `model.safetensors`;
- full lane: sharded safetensors plus `model.safetensors.index.json`.

Unrelated serving, conversion, malformed third-party checkpoint, stale-run, and
unused helper paths are out of scope.

## Root cause and implementation

`validate_glm53_ep_gptq` duplicates an index-only loader even though
`pipeline.serve_ignore.weight_map_of` is the repository's canonical loader for
both emitted layouts. Replace the private index reader with `weight_map_of` and
pass its filename map into scale loading. Keep exhaustive required-key,
forbidden-packed-weight, positive-finite-scale, phase, and peak checks unchanged.

## Tests

Use real temporary safetensors files rather than mocked key lists:

1. A single-file representative-layout checkpoint with complete packed keys and
   positive scales passes `validate_checkpoint`.
2. A sharded full-layout checkpoint and index with the same contract passes.
3. A real single-file checkpoint containing a non-positive scale fails with the
   existing scale error.

Existing pure tests remain useful for combinatorial key and threshold behavior.
Do not add tests for branches this chain cannot produce.

## Pre-relaunch evidence

Before releasing the current failed holder:

1. Replay the fixed key and scale validator against its saved EP4 checkpoint.
2. Reuse its real four-rank metrics to confirm phase completeness and a non-null
   peak; these already pass and need no additional implementation.
3. With explicit GPU approval, load that checkpoint through the exact
   `_last_token_logits` path and run one forward pass on GPU 0.

Only after all checks pass should the fixed commit be pushed and a replacement
be queued on `gpu04` before deleting the holder.
