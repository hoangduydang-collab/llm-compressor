#!/usr/bin/env python3
"""Apply or verify the pinned Humming NVFP4 W4A8 source overlay."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from pipeline.hopper_nvfp4_w4a8.humming_patch import (
    PatchError,
    patch_humming_tree,
    patch_installed_humming,
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--root",
        type=Path,
        help="explicit site-packages root containing humming/ (tests/fixtures)",
    )
    parser.add_argument(
        "--check",
        action="store_true",
        help="validate only; exit 0 only when the complete overlay is present",
    )
    parser.add_argument("--json", type=Path, help="also write the JSON report")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        if args.root is None:
            report = patch_installed_humming(apply=not args.check)
        else:
            report = patch_humming_tree(args.root, apply=not args.check)
    except PatchError as exc:
        print(json.dumps({"error": str(exc), "status": "error"}, sort_keys=True))
        return 2

    payload = report.to_json()
    print(payload)
    if args.json is not None:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(payload + "\n", encoding="utf-8")
    if args.check and report.status != "patched":
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
