# Task 1: direct native AWQ output

Scoped implementation on `duy-branch`:

- Allow plain AWQ W4AFP8 with FP8_BLOCK rest targets to select native SGLang saving; retain mandatory early FP8 weight preparation for GPTQ and reject unsupported methods/layouts.
- Resolve planned schemes only from QuantizationMixin modifiers, skipping transform-only AWQModifier. Existing preflight still rejects target overlaps and incompatible serving inventory before calibration.
- Enable `checkpoint_format: sglang-w4afp8` in the full GLM-5.3 AWQ recipe without changing calibration.
- Cover AWQ and GPTQ native entrypoint selection; add real tiny GLM sequential AWQ/gridsearch lifecycle. Assert native first-write inventory, verifier success, signed INT4 nibble equality against independently compressed CT state, exact FP8 weight-byte/scale preservation, fixed-unit expert input scale, preserved folded norm, and restoration of expert/FP8 COMPRESSED CT state.

Validation performed with CPU settings, no GPU allocation or full checkpoint conversion:

1. `OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 PYTHONPATH=src:. .venv-prequant/bin/python -m pytest pipeline/tests/test_native_sglang_save.py -k 'real_awq_lifecycle or pipeline_preflights' pipeline/tests/test_glm53_ep_gptq_configs.py -q` — 5 passed, 32 deselected, 14 warnings, 35.56s. This command's `-k` deselected config tests; config tests were separately run below. Log: `/tmp/native-awq-task1-focused.log`.
2. `OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 PYTHONPATH=src:. .venv-prequant/bin/python -m pytest pipeline/tests/test_glm53_ep_gptq_configs.py -q` — final expanded cases: 11 passed, 14 warnings, 18.31s. Log: `/tmp/native-awq-task1-config.log`.
3. `OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 PYTHONPATH=src:. .venv-prequant/bin/python -m pytest pipeline/tests/test_native_sglang_save.py -q` — 29 passed, 1 failed, 15 warnings, 121.17s. Failure: existing `test_two_process_collective_native_save[True]`, which executes before the new real AWQ test. The failure occurs in restored-model forward (`assert_restored` -> CT `ct_decompress_hook` -> `DiskCache.onload`) opening `/tmp/pytest-of-e1129930/pytest-17/test_two_process_collective_na0/offload/ct_disk_cache_1_123872038635792.safetensors`. Full exact traceback retained in `/tmp/native-awq-task1-native.log`; failed tmp directory retained. Single-process disk-offload, native CPU collective saving, writer-failure restoration, verifier rejection cases, and new actual AWQ test all passed. Isolated unchanged rerun: `OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 PYTHONPATH=src:. .venv-prequant/bin/python -m pytest 'pipeline/tests/test_native_sglang_save.py::test_two_process_collective_native_save[True]' -q` — 1 passed, 14 warnings, 47.36s; log `/tmp/native-awq-task1-disk-rerun.log`.
4. `.venv-prequant/bin/ruff check pipeline/config.py pipeline/quantize.py pipeline/tests/test_glm53_ep_gptq_configs.py pipeline/tests/test_native_sglang_save.py` — all checks passed.
5. `git diff --check` — passed.

Limits: real AWQ lifecycle runs on CPU in memory; disk-offload and collective saving are exercised by the existing shared native-writer fixtures in the native suite. This does not qualify full-scale GPU execution or serving. No converter is invoked; no main-model checkpoint conversion was run. Warnings observed are existing torch.jit.script_method deprecations and the Transformers offloaded-save memory warning. The collective disk-cache failure is intermittent and occurs before the new AWQ lifecycle test; its exact race/root cause is not proven. No unrelated cache changes or fixture weakening applied; retain this as a runtime qualification concern. Unrelated docs, .gitignore, predictive_ptq and results/predictive-ptq changes are excluded.

Commit: `8e30719d160b6686476235af31b0001be0e69844` (`Support direct native SGLang W4AFP8 export for AWQ`). No push performed.
