"""Fail-closed identity check on a source snapshot before quantizing it.

Guards a mistake that has already been made: 64 GB of the GLM-5.3 **FP8** release
was downloaded before anyone noticed it was the wrong artifact. The two releases
are easy to confuse -- ``zai-org/GLM-5.3`` is the FP8 one (141 shards, carries a
``quantization_config``) and ``zai-org/GLM-5.3-BF16`` is the unquantized one (282
shards, no ``quantization_config``) -- and the naming is inverted from GLM-5.2,
where the unsuffixed repo was the BF16 release.

Quantizing the wrong one is not a graceful failure. AWQ needs BF16 to compute
smoothing scales and to pack int4, and llm-compressor has no
``FineGrainedFP8`` / ``weight_scale_inv`` read path anywhere in ``src/``, so a
block-scaled FP8 source cannot be consumed at all.

Exists as a module rather than an inline heredoc for a concrete reason: a heredoc
terminator must sit at column 0, which is impossible inside a YAML block scalar,
so the rendered container script never closed the heredoc. The launcher's gate 0
caught it (`syntax error: unexpected end of file`) before anything ran.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def check(
    snapshot: Path,
    *,
    expect_layers: int | None = None,
    expect_arch: str | None = None,
    require_unquantized: bool = False,
    expect_shards: int | None = None,
) -> list[str]:
    """Return a list of problems; empty means the snapshot is what was asked for."""
    problems: list[str] = []
    config_path = snapshot / "config.json"
    if not config_path.exists():
        return [f"no config.json under {snapshot}"]
    config = json.loads(config_path.read_text(encoding="utf-8"))

    if require_unquantized and config.get("quantization_config") is not None:
        method = (config.get("quantization_config") or {}).get("quant_method")
        problems.append(
            f"config.json HAS a quantization_config (quant_method={method!r}): this "
            "is a quantized release, which the AWQ path cannot consume"
        )
    if expect_layers is not None and config.get("num_hidden_layers") != expect_layers:
        problems.append(
            f"num_hidden_layers={config.get('num_hidden_layers')}, "
            f"expected {expect_layers}"
        )
    if expect_arch is not None and expect_arch not in (config.get("architectures") or []):
        problems.append(
            f"architectures={config.get('architectures')}, expected to contain "
            f"{expect_arch!r}"
        )
    if expect_shards is not None:
        found = len(list(snapshot.glob("*.safetensors")))
        if found != expect_shards:
            problems.append(f"{found} safetensors shards, expected {expect_shards}")
    return problems


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("snapshot", type=Path)
    parser.add_argument("--expect-layers", type=int, default=None)
    parser.add_argument("--expect-arch", default=None)
    parser.add_argument("--expect-shards", type=int, default=None)
    parser.add_argument("--require-unquantized", action="store_true")
    args = parser.parse_args(argv)

    problems = check(
        args.snapshot,
        expect_layers=args.expect_layers,
        expect_arch=args.expect_arch,
        require_unquantized=args.require_unquantized,
        expect_shards=args.expect_shards,
    )
    if problems:
        print("ABORT: wrong source artifact:", file=sys.stderr)
        for problem in problems:
            print(f"  - {problem}", file=sys.stderr)
        return 1
    print(f"==> source artifact verified: {args.snapshot}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
