#!/usr/bin/env python3
"""Repair transformers' sharded offloaded ``save_pretrained``.

    python envs/hotfix-transformers-sharded-save.py [--check]

THE BUG (transformers 5.14.0/5.14.1, and upstream main as of 2026-07-18).
``PreTrainedModel.save_pretrained`` reverts the weight format per shard for
offloaded models, then updates the sharded weight map with::

    weight_map.update({k: os.path.basename(shard_file)} for k in shard_state_dict.keys())

The brace closes too early, so ``dict.update`` is handed a *generator of
one-element dicts* instead of a dict -> ``ValueError``. It is then swallowed by a
broad ``except Exception`` and re-raised as the misleading "we could not revert
some weight conversions ... unlucky sharding" ``RuntimeError``. Every sharded
offloaded original-format save therefore dies at the end of shard 1, and the
error text sends you looking at shard sizes instead of at a typo.

WHY THIS FILE EXISTS. The MiniMax-M3 environment carried this repair as an
in-place edit to the installed ``modeling_utils.py``. A ``pip freeze`` records
versions, not source edits, so ``envs/m3-quant-freeze.txt`` reproduced the
version and silently dropped the patch -- GLM-5.2 then tripped
``assert_transformers_offloaded_save_healthy`` (pipeline/quantize.py) three
minutes into a 4-GPU run. Automating the repair is what makes the environment
actually reproducible; the gate stays as the backstop.

Idempotent, and fail-closed: it refuses to guess. If the marker is absent or
appears more than once, nothing is written and the exit code is non-zero.

Exit codes: 0 patched or already healthy | 1 could not patch | 2 --check found
an unhealthy install.
"""

import argparse
import inspect
import shutil
import sys
from pathlib import Path

# Exactly the substring pipeline.quantize._offloaded_save_health looks for.
BROKEN = "} for k in shard_state_dict.keys())"
FIXED = " for k in shard_state_dict.keys()})"
# Present only once the per-shard revert exists at all (i.e. 5.14+).
REVERT_MARKER = "revert_weight_conversion(model_to_save, shard_state_dict)"


def health(source: str) -> str:
    """Mirror of ``pipeline.quantize._offloaded_save_health``.

    Duplicated deliberately: this script must run before the repo is
    installed, so it cannot import from the pipeline package.
    """
    if REVERT_MARKER not in source:
        return "shimmed"  # pre-5.14; the repo's save shims own this path
    if "} for k in shard_state_dict.keys()" in source:
        return "broken"
    return "healthy"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--check",
        action="store_true",
        help="report status without writing (exit 2 if unhealthy)",
    )
    ap.add_argument(
        "--no-backup",
        action="store_true",
        help="skip the .orig copy (for ephemeral containers)",
    )
    args = ap.parse_args()

    try:
        from transformers import __version__ as tv
        from transformers import modeling_utils as mu
    except Exception as exc:  # noqa: BLE001
        print(f"FAIL: cannot import transformers: {exc}")
        return 1

    path = Path(mu.__file__)
    print(f"transformers {tv}")
    print(f"  file   : {path}")

    # Judge from the function's own source, not the whole module: that is what
    # the runtime gate inspects, so the two cannot disagree.
    before = health(inspect.getsource(mu.PreTrainedModel.save_pretrained))
    print(f"  status : {before}")

    if before != "broken":
        if args.check:
            print("CHECK OK: no patch needed")
        else:
            print("OK: nothing to do (already healthy, or pre-5.14 shimmed path)")
        return 0

    if args.check:
        print("CHECK FAILED: sharded offloaded save is broken; run without --check")
        return 2

    text = path.read_text(encoding="utf-8")
    occurrences = text.count(BROKEN)
    if occurrences != 1:
        # Never guess: a changed upstream line means the fix must be re-derived
        # by reading it, not pattern-matched by this script.
        print(
            f"FAIL: expected exactly 1 occurrence of the marker, found "
            f"{occurrences}. Upstream has changed; re-derive the fix by hand.\n"
            f"       marker: {BROKEN}"
        )
        return 1

    if not args.no_backup:
        backup = path.with_suffix(path.suffix + ".orig")
        if not backup.exists():
            shutil.copy2(path, backup)
            print(f"  backup : {backup}")

    path.write_text(text.replace(BROKEN, FIXED, 1), encoding="utf-8")

    # Verify by re-reading the file from disk, not by trusting the write. The
    # module is already imported, so re-importing would return the cached one.
    after = health(path.read_text(encoding="utf-8"))
    if after == "broken":
        print("FAIL: wrote the patch but the file still reads as broken")
        return 1

    line_no = next(
        (
            i
            for i, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1)
            if FIXED in line
        ),
        None,
    )
    print(f"  patched: line {line_no}")
    print(f"  status : {after}")
    print("PATCHED: sharded offloaded save_pretrained repaired")
    return 0


if __name__ == "__main__":
    sys.exit(main())
