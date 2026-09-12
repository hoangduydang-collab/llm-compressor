# Native AWQ/MTP CPU validation — 2026-09-12

Final combined suite: **144 passed, zero failed/errors/skipped**, 107.09 seconds.
See `validation.json`, `environment.json`, `combined-pass.txt` and JUnit XML.
The 791-source/2337-native geometry check matches the pinned public source headers.

The initial AWQ and combined run logs are deliberately retained. The repeated
collective disk-cache failure was traced to unsupported CT distributed
decompression during the test's post-save forward, and repaired using CT's
existing materialization API while retaining all-rank shared-state and forward
assertions. `disk-forward-limitation.json` records the cause and dependency source
hashes; `collective-repair.txt` records both focused cases passing.

This is CPU source/serialization evidence, not serving, acceptance-rate, speed or
main-model quality qualification. The executor handoff defines those remaining
checks. No model weight payload was downloaded for the public geometry check.
