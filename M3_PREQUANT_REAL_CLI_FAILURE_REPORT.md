# MiniMax-M3 Pre-Quantization Gate: Real CLI Failure

## Summary

The pre-quantization compatibility gate passes its synthetic and regression
tests, but the first real MiniMax-M3 CLI run fails while constructing the
meta-model. No calibration data, checkpoint weights, forward pass, or GPU was
used.

## Reproduction

Environment:

- `/mnt/nfs/hoangduy/venvs/quant`
- `PYTHONPATH="$PWD/src:$PWD"`
- Git revision: `124a36d2`

Command:

```bash
python -m pipeline.prequant_compatibility \
  --config pipeline/configs/minimax_m3.yaml \
  --output results/prequant-validation/minimax-m3.json
```

Observed result:

```text
NotImplementedError: Offload of type meta and distributed=False has not been implemented
```

The output report was not written and the command returned exit code `1`.

## Failure boundary and likely cause

`pipeline/prequant_compatibility.py::_build_meta_model` creates the model under
`accelerate.init_empty_weights()`, then calls `linearize_moe(model)` so the
planner sees the same per-expert Linear representation used by calibration.

The failure occurs in:

```text
linearize_moe
  -> LinearExperts2D.from_experts_module
  -> compressed_tensors.offload.offload_module
```

The linearized expert modules are created successfully, but
`from_experts_module` unconditionally initializes runtime offload for the
source module's `meta` device. `compressed-tensors` intentionally has no
`meta` offload backend.

The narrow fix candidate is to skip runtime offload initialization when the
source experts are on `meta`, leaving the newly linearized modules as
meta-only modules. The guard must remain limited to `meta`; CPU, CUDA, and
other production offload paths must retain their existing behavior.

## Verification before failure

Passed:

- Pre-quantization focused and regression tests: `50 passed`
- Ruff `0.15.21`: all changed-file checks passed
- Python compilation
- `git diff --check`

The real CLI failure occurs only when exercising MiniMax's actual MoE
linearization path, so the synthetic tests do not currently cover this
meta-offload boundary.

## Handoff request

Add a focused regression test for meta-device MoE linearization, verify it
fails before the fix, implement the narrowly scoped meta-offload guard, then
rerun:

1. the focused pre-quantization and MoE tests;
2. the full related planner/regression suite;
3. Ruff and compilation; and
4. the real MiniMax CLI command above.

Do not allocate GPUs or run calibration as part of this fix.
