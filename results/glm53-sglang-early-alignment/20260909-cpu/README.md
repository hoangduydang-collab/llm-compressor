# Native block FP8 conversion validation

2026-09-09, CPU only. **116 passed in 32.94 s**, pytest exit 0.
`pytest.log` is the captured final command output. `validation.json` records
package versions and the tested file hashes. Fourteen upstream Torch JIT
DeprecationWarnings were emitted. No GPU, model download or full checkpoint used.

From the repository root:

```bash
OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 PYTHONPATH=src:. \
  .venv-prequant/bin/python -m pytest -q \
  pipeline/tests/test_sglang_native_fp8.py \
  pipeline/tests/test_to_sglang_w4afp8.py \
  pipeline/tests/test_verify_sglang_w4afp8.py \
  pipeline/tests/test_patch_indexer_fp8.py \
  pipeline/tests/test_graft_mtp_w4afp8.py \
  pipeline/tests/test_sglang_w4afp8_kernels.py
```

Ruff passes for the changed converter, verifier and new native conformance test.
The historical converter test file retains its pre-existing line-length and
ambiguous-variable findings; added tests introduce none. `git diff --check` passes.

See [implementation scope](../../../docs/glm53-sglang-w4afp8-early-alignment.md).
This proves tiny-checkpoint serialization/conversion, not GPU calibration,
serving, full-model quality, or measured shared-storage speedup.
