# Final replay fixes report

## Outcome

Implemented final whole-branch review findings 1–3 only in the empty-output
replay module and its CPU tests. No quality-eval implementation, test, or
documentation files were modified.

## Pinned evidence and SHA

- Exact attempt UID:
  `8e98c89a40db606e115a1d388e89a58518582d44f2f48dafcf389a1e1e146878`
- Trusted rendered-prompt SHA-256:
  `006f5eef8c151c3e7418a047e8a171a33bad2d77fad989a8b7f1552648d60b93`
- The SHA was derived from the matching committed row in
  `results/m3-quality/20260715T075800Z-m3-paired-reasoning-r4/models/inhouse_gptq/shards/smoke/samples/mmlu_pro.jsonl`.
- Evidence file Git blob SHA-1:
  `992195281a94d0a6b212d42af146e13ad4dddc05`.
- Starting commit:
  `017bef1ada278a525e7f51f7bc0be3d693dd8a2b`.

## Implemented behavior

- Rejects any caller-selected UID other than the exact pinned UID before
  reading the source JSONL, and rejects rendered-prompt drift against the
  explicit trusted SHA-256 before runtime/model initialization.
- Retains JSON-safe effective generation evidence per control after
  `modify_gen_kwargs()` and `maybe_truncate()`: normalized sampling kwargs,
  original task stops, effective stops, EOS-only model stops, and effective
  maximum tokens.
- Resolves source/output paths before replay and allows benchmark-tree output
  only below `diagnostics/` or `replays/`, while rejecting samples,
  aggregate, matrix, gates, manifest, and generation-health targets even
  beneath an otherwise permitted diagnostic directory.
- Resolving the destination before classification prevents a symlink alias
  from redirecting a diagnostic-looking name to a protected benchmark
  artifact.

## TDD evidence

Initial focused RED run:

```text
python -m pytest -q pipeline/tests/test_m3_empty_output_replay.py -k "unpinned_requested_uid or rendered_prompt_drift or benchmark_artifact_targets or diagnostic_sidecar_paths or symlink_alias or pinned_adapter"
9 failed, 2 passed, 1 skipped, 37 deselected
```

The failures showed that the wrong UID and drifted prompt were accepted, all
six protected artifact classes reached replay, and effective generation
arguments were missing. The allowed diagnostic/replay paths already passed.

After tightening the protected-name cases to place forbidden artifacts under
`diagnostics/`, the focused test was deliberately RED on all six error-contract
assertions before the minimal production-message update.

## Verification evidence

```text
python -m pytest -q pipeline/tests/test_m3_empty_output_replay.py pipeline/tests/test_m3_quality_eval.py pipeline/tests/test_m3_quality_evidence.py pipeline/tests/test_static_checkpoint.py pipeline/tests/test_lmeval_runner.py
154 passed, 1 skipped in 6.78s

python -m ruff check pipeline/m3_empty_output_replay.py pipeline/tests/test_m3_empty_output_replay.py
All checks passed!

git -c safe.directory=D:/Work/llm-compressor diff --check
clean
```

## Concerns

- Windows on this host does not permit creating a symbolic link without
  additional privilege, so the real symlink-alias regression skipped locally.
  The test remains enabled and exercises the resolved-alias policy on hosts
  where symlink creation is available. The existing resolved source/output
  alias regression and all non-symlink destination-policy cases passed here.
- No GPU or real vLLM runtime was used; the pinned adapter sequence and exact
  report evidence were verified with the existing fake-runtime contract.
