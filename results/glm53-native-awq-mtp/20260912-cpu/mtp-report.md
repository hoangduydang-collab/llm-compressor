# Task 2: native MTP assembly implementation

Status: implemented, CPU validation passed, and committed as `f25b179b` on `duy-branch`. No GPU allocation or full checkpoint conversion performed. No push.

## Interfaces and lifecycle

- `pipeline.native_mtp.preflight_native_mtp(source, model_config) -> NativeMTPPlan` validates the complete source config/index/header inventory without loading tensor payloads. Hub sources require the loaded config's `_commit_hash` and cache-only lookup at that revision; local snapshot paths and cache symlinks are preserved.
- `plan.provenance()` returns JSON-safe source ID, snapshot path, revision, source/native tensor counts, config/index SHA256, and per-shard header SHA256. `plan.layer` is the draft layer. Plan retains source file identities and header/metadata fingerprints through calibration.
- `pipeline.native_mtp.assemble_native_mtp(checkpoint, plan, *, max_shard_bytes=1024**3) -> dict` reuses existing INT4 RTN, nibble packing, block-FP8 kernels and classification, plus graft inventory helpers. Reads source tensors on CPU individually; buffers at most the shard cap or one larger tensor. Checks positive finite generated scales, including BF16 underflow.
- Entry point validates source and stale destination before `oneshot`, then assembles only on the source rank after the final existing save barrier, immediately before the existing whole-artifact verifier. No later collective is introduced.
- `quantization.mtp_policy` defaults to `absent`; `source-rtn` requires native W4AFP8. Both full GLM-5.3 AWQ/GPTQ recipes enable it; representative remains absent. Recipe provenance records policy.

## Publication and verification

- Full source geometry is a closed inventory of 791 tensors: 768 expert weights plus 23 other tensors. Native output is exactly 2337 tensors: 768 packed INT4 weights, 768 BF16 weight scales, 768 fixed-unit BF16 `.w1/.w2/.w3.input_scale`, 10 FP8 weights, 10 FP32 block scales, and 13 copies, including the F32 correction bias.
- Supports ordinary single-file and indexed native main checkpoints. Main shard names and bytes remain unchanged; single-file main checkpoints gain an index referencing the original shard and new draft shards.
- Stages additions privately, verifies every new tensor against its expected geometry/dtype/hash, then publishes with collision-safe hard links and metadata replacements. Only main shard headers are checked during staging; main tensor payloads are never reread or hashed by assembly.
- Ordinary failures restore replaced metadata and remove only this attempt's files. An interruption retains `.native_mtp_incomplete.json`; final verifier and repeat assembly reject any marker, including invalid/empty/dangling markers. Manual interrupted-artifact repair is deliberately not provided.
- Final verifier independently reconstructs closed native MTP inventory and checks presence/layer/RTN/fixed-unit policy, source fingerprints, separate MTP shard inventory, copy metadata, and every added hash including BF16/F32 copies. Existing final full-artifact verification runs once.
- Legacy graft changes are documentation/import-order only; no legacy conversion behavior changed.

## Validation evidence so far

1. First focused MTP run: 41 passed, one fixture-only failure from creating two tensor names with shared storage; fixed the fixture by cloning the added extra-expert tensor.
2. Focused MTP + existing cache resolution suite: 60 passed in 34.46 s (`/tmp/native-mtp-focused.log`).
3. Combined requested suite command:

   ```bash
   OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 PYTHONPATH=src:. .venv-prequant/bin/python -m pytest -q pipeline/tests/test_native_mtp.py pipeline/tests/test_native_sglang_save.py pipeline/tests/test_glm53_ep_gptq_configs.py pipeline/tests/test_graft_mtp_w4afp8.py pipeline/tests/test_resolve_weight_index.py tests/pipeline/test_r8_fp8_recipe.py --junitxml=/tmp/native-mtp-final.xml > /tmp/native-mtp-final.log 2>&1
   ```

   Initial combined result: **142 passed, 1 failed in 115.44 s**. Sole failure is the previously observed native disk-cache test `test_two_process_collective_native_save[True]`: worker 1's post-restoration forward reaches CT DiskCache.onload after a referenced shared cache file disappears. Root is investigating the existing test synchronization; no MTP code occurs in that stack. Raw initial log/JUnit retained for attribution.
4. Ruff passed all touched Python files; `git diff --check` passed.
5. Root independently compared the source inventory against all 791 pinned public source headers with exact dtype/shape agreement; expected native count 2337. Evidence: `results/glm53-native-awq-mtp/20260912-cpu/full-source-geometry.json`.

CPU coverage includes exact reused outputs; cropped FP8 kv_a geometry; copied F32 router bias; BF16/FP16/FP32 main dtypes; both single/indexed main artifacts; source config/index/shard replacement across calibration; source change during tensor reads; missing/extra expert and required copy; malformed/oversized header, truncated payload, unsafe path, duplicate JSON; no-main-payload-read guarantee; scale underflow rollback; quantizer/staged corruption/collision/metadata publication rollback; interrupted publication marker; partial or corrupted native manifest; source-only lifecycle ordering and assembly failure after final barrier.

Runtime limitations: CPU serialization/source-contract coverage does not qualify SGLang load/forward, acceptance rate, performance, or main-model calibration quality. Those remain executor work under the pinned serving runtime.

## Additional integration evidence and existing fixture repair

- Direct real-model composition passed: existing real Transformers `GlmMoeDsaForCausalLM` fixture, real installed CT observers and native save context, then complete tiny MTP assembly and full verifier. `1 passed in 30.93 s`; log `/tmp/native-mtp-real-writer.log`. Main shard hashes remained unchanged.
- Root identified the combined-suite disk failure's precise cause: installed `ModelCompressor.decompress_model` explicitly does not support distributed decompression, and its independent per-rank state replacement deletes shared `DistributedDiskCache` files before a peer reads them. This is distinct from serialization and from the new source-only MTP stage. The production pipeline already avoids distributed sanity generation.
- Repaired only the existing test fixture: `assert_restored(..., forward=False)` verifies exact compressed values and restoration on the shared cache, then `remove_dispatch(model, onload_tensors=True)` materializes each rank's CPU parameter dictionaries without deleting shared cache files. A test-only barrier precedes the existing full restoration assertions and finite forward on every rank. Default single-process assertions remain unchanged. No production dependency or distributed inference code was modified.
- Focused fixture command: `OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 PYTHONPATH=src:. .venv-prequant/bin/python -m pytest -q pipeline/tests/test_native_sglang_save.py::test_two_process_collective_native_save --junitxml=/tmp/native-mtp-collective-repair.xml > /tmp/native-mtp-collective-repair.log 2>&1`.

## Final validation

- Repaired collective CPU and disk fixture cases: **2 passed in 69.79 s** (`/tmp/native-mtp-collective-repair.log`, `/tmp/native-mtp-collective-repair.xml`).
- Final combined command (same six suites, including the new real-writer composition case):

  ```bash
  OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 PYTHONPATH=src:. .venv-prequant/bin/python -m pytest -q pipeline/tests/test_native_mtp.py pipeline/tests/test_native_sglang_save.py pipeline/tests/test_glm53_ep_gptq_configs.py pipeline/tests/test_graft_mtp_w4afp8.py pipeline/tests/test_resolve_weight_index.py tests/pipeline/test_r8_fp8_recipe.py --junitxml=/tmp/native-mtp-final-pass.xml > /tmp/native-mtp-final-pass.log 2>&1
  ```

  Result: **144 passed, 15 existing warnings in 107.09 s**. Initial failing run evidence was not overwritten.
- Final lint command: `.venv-prequant/bin/ruff check pipeline/native_mtp.py pipeline/tests/test_native_mtp.py pipeline/native_sglang_save.py pipeline/tests/test_native_sglang_save.py pipeline/config.py pipeline/recipe.py pipeline/quantize.py pipeline/graft_mtp_w4afp8.py` — **All checks passed**.
- `git diff --check` — passed.

Commit: `f25b179b` — `Assemble native GLM MTP from the pinned BF16 source`. Exactly ten task files committed. Root documentation and unrelated `.gitignore`, `predictive_ptq/`, and `results/predictive-ptq/` remain untouched by this commit. No push.
