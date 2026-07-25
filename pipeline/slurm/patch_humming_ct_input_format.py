#!/usr/bin/env python3
"""Admit compressed-tensors ``pack-quantized`` in Humming's input schema.

A compressed-tensors GPTQ W4 checkpoint always carries the top-level format
``pack-quantized``. Humming's ``CompressedTensorsWeightSchema`` whitelists that
format, but ``CompressedTensorsInputSchema`` does not -- in 0.1.6, 0.1.10, and
0.1.11 alike. Any W4A8 compressed-tensors checkpoint therefore raises
``AssertionError`` while the first ``Linear`` is being constructed, so the
combination is unreachable by construction rather than rejected on purpose.

Inside ``CompressedTensorsInputSchema`` the format string is used for exactly
one thing::

    self.input_scale_key = "input_global_scale" if "nvfp4" in self.format else "input_scale"

and ``get_tensors_attrs`` requests an input-scale tensor only when the
activation is *not* dynamic (``dynamic is False`` or ``dynamic == "local"``).
For a dynamic per-token activation the added whitelist entry is therefore inert
beyond admission: no scale key is consulted and no checkpoint tensor is read.

This patcher edits an isolated side-install, never the shared venv, and is
idempotent. The resulting file hash is declared in
``pipeline/m3_humming_w4a8.py`` so the preflight integrity gate reports the
patch instead of silently tolerating a modified distribution.
"""

from __future__ import annotations

import argparse
import hashlib
import sys
from pathlib import Path

DEFAULT_SITE = Path("/mnt/nfs/hoangduy/venvs/humming-0.1.10-site")
RELATIVE_TARGET = "humming/schema/compressed_tensors.py"

ANCHOR = '''        assert self.format in [
            "int-quantized",
            "float-quantized",
            "naive-quantized",
            "nvfp4-pack-quantized",
            "mxfp4-pack-quantized",
        ]
        self.input_scale_key ='''

PATCHED = '''        assert self.format in [
            "int-quantized",
            "float-quantized",
            "naive-quantized",
            "pack-quantized",
            "nvfp4-pack-quantized",
            "mxfp4-pack-quantized",
        ]
        self.input_scale_key ='''


def target_path(site: Path) -> Path:
    return site / RELATIVE_TARGET


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def classify(source: str) -> str:
    """Return ``patched``, ``unpatched``, or ``unknown`` for one file body."""

    if PATCHED in source:
        return "patched"
    if ANCHOR in source:
        return "unpatched"
    return "unknown"


def apply_patch(site: Path, apply: bool) -> tuple[str, str]:
    """Apply or check the patch. Returns ``(status, sha256)``."""

    path = target_path(site)
    if not path.is_file():
        raise SystemExit(f"target not found: {path}")
    source = path.read_text(encoding="utf-8")
    status = classify(source)

    if status == "unknown":
        raise SystemExit(
            f"neither patched nor expected-unpatched content in {path}; "
            "refusing to guess"
        )
    if status == "patched":
        return "already patched", sha256(path)
    if not apply:
        return "NOT patched", sha256(path)

    if source.count(ANCHOR) != 1:
        raise SystemExit(
            f"expected exactly one anchor in {path}, found {source.count(ANCHOR)}"
        )
    path.write_text(source.replace(ANCHOR, PATCHED), encoding="utf-8")
    return "patched", sha256(path)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--site", type=Path, default=DEFAULT_SITE)
    parser.add_argument(
        "--check",
        action="store_true",
        help="report status without writing; exit nonzero if unpatched",
    )
    args = parser.parse_args(argv)

    status, digest = apply_patch(args.site, apply=not args.check)
    print(f"{RELATIVE_TARGET}: {status}")
    print(f"sha256: {digest}")
    if args.check and status != "already patched":
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
