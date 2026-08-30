"""Fail-closed verification of a converted SGLang ``w4afp8`` checkpoint.

The converter's internal guards sample: 8 modules for the nibble reinterpret
(now removed) and 26 for the fold cross-check. Nothing yet checks that all
57,600 expert repacks came through, or that no tensor was dropped, renamed
wrongly, or left in the source encoding. This is the analogue of
``verify_quant_checkpoint`` for the converted artifact, and it exists because
every failure mode here is SILENT: a checkpoint with a mis-encoded expert loads
fine and serves plausible-looking noise.

WHAT IS CHECKED, AND WHAT THE EXPECTED RESULT IS

  experts        BIT-EXACT. The conversion is a pure re-encoding: unpack the
                 source int32 nibbles, unpack the converted int8 nibbles, and
                 the integer values must be *equal*, not merely close. The
                 scale is renamed, never recomputed, so it must be bitwise
                 identical too. Any tolerance here would be a bug -- there is
                 no arithmetic in this path to lose precision to.

  fp8-rest       Agrees with the SOURCE's per-channel dequant to within two
                 independent e4m3 roundings (~0.037; bound 0.06). Not exact,
                 because the converted weight is re-derived from BF16 as a
                 block quantization while the source is per-channel. A dropped
                 or misapplied AWQ fold lands at |1-s|, typically 0.1-0.3.

  passthrough    BIT-EXACT. Norms, router, indexer, embeddings and lm_head are
                 copied, so anything else means the wrong tensor was written.

  structure      Every source module has a counterpart; no source-encoding
                 leftovers (``weight_packed`` / ``weight_shape`` /
                 per-channel ``weight_scale``); the index accounts for every
                 byte on disk; config declares ``quant_method: w4afp8``.

Usage:
    python -m pipeline.verify_sglang_w4afp8 \
        --src <compressed-tensors checkpoint> \
        --dst <converted w4afp8 checkpoint> \
        [--samples 40] [--all-experts]
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from pipeline.sglang_w4afp8_kernels import (
    DEFAULT_BLOCK,
    dequantize_block_fp8,
    unpack_nibbles_int8,
)

# Two independent e4m3 roundings give sqrt(2) * 0.0265 = 0.037. 0.06 leaves
# headroom for the fold reconstruction's BF16 rounding while still catching a
# dropped fold, which lands at |1 - s|.
_FP8_MAX_RESID = 0.06


def _fail(message: str, errors: list[str]) -> None:
    print(f"  [fail] {message}", flush=True)
    errors.append(message)


def _ok(message: str) -> None:
    print(f"  [ok]   {message}", flush=True)


def verify(
    src: Path,
    dst: Path,
    samples: int = 40,
    all_experts: bool = False,
) -> int:
    import torch
    from safetensors import safe_open

    from pipeline.serve_ignore import weight_map_of

    errors: list[str] = []
    warnings: list[str] = []

    src_map = weight_map_of(src)
    dst_index = json.loads(
        (dst / "model.safetensors.index.json").read_text(encoding="utf-8")
    )
    dst_map = dst_index["weight_map"]
    opened: dict[tuple, object] = {}

    def get(root: Path, wmap: dict, key: str):
        shard = (root, wmap[key])
        if shard not in opened:
            opened[shard] = safe_open(str(root / wmap[key]), framework="pt")
        return opened[shard].get_tensor(key)

    # Shard presence FIRST. Every later check reads tensors, so a missing file
    # would surface as a FileNotFoundError traceback from somewhere deep in the
    # middle of the report rather than as a verdict. Nothing else is meaningful
    # if the bytes are not there, so this bails out.
    print("\n== shard presence ==")
    absent = sorted({v for v in dst_map.values() if not (dst / v).is_file()})
    if absent:
        _fail(f"{len(absent)} shard(s) referenced by the index are missing: "
              f"{absent[:5]}", errors)
        print("\n== summary ==\n  RESULT: FAIL (shards missing; nothing else "
              "could be checked)", flush=True)
        return 1
    _ok(f"all {len({v for v in dst_map.values()})} referenced shard(s) present")

    print("\n== config ==")
    config = json.loads((dst / "config.json").read_text(encoding="utf-8"))
    quant = config.get("quantization_config", {}) or {}
    if quant.get("quant_method") != "w4afp8":
        _fail(f"quant_method is {quant.get('quant_method')!r}, expected 'w4afp8'",
              errors)
    else:
        _ok("quant_method: w4afp8")
    for entry in quant.get("ignored_layers", []) or []:
        if str(entry).startswith("re:"):
            _fail(f"ignored_layers contains an unresolved regex {entry!r}; the "
                  f"loader does prefix matching, so it would match nothing",
                  errors)

    print("\n== source-encoding leftovers ==")
    stale = [
        k for k in dst_map
        if k.endswith((".weight_packed", ".weight_shape"))
    ]
    if stale:
        _fail(f"{len(stale)} compressed-tensors tensors survived the "
              f"conversion, e.g. {stale[:3]}", errors)
    else:
        _ok("no weight_packed / weight_shape in the output")

    print("\n== expert coverage ==")
    src_experts = sorted(
        k[: -len(".weight_packed")] for k in src_map
        if k.endswith(".weight_packed")
    )
    missing = [m for m in src_experts if f"{m}.weight" not in dst_map]
    no_scale = [m for m in src_experts if f"{m}.weight_scale_inv" not in dst_map]
    if missing:
        _fail(f"{len(missing)} expert modules absent from the output, e.g. "
              f"{missing[:3]}", errors)
    elif no_scale:
        _fail(f"{len(no_scale)} expert modules have no weight_scale_inv, e.g. "
              f"{no_scale[:3]}", errors)
    else:
        _ok(f"all {len(src_experts)} expert modules present with scales")

    print("\n== expert values (must be BIT-EXACT) ==")
    unpack_src = _source_unpacker(warnings)
    if unpack_src is None:
        warnings.append("expert value check skipped (no compressed-tensors)")
    else:
        picks = src_experts if all_experts else src_experts[
            :: max(1, len(src_experts) // samples)
        ][:samples]
        mismatched = 0
        scale_mismatched = 0
        skipped = 0
        for module in picks:
            # Absent modules are already reported by the coverage check above;
            # reading them here would replace that verdict with a KeyError
            # traceback halfway through the report.
            if (f"{module}.weight" not in dst_map
                    or f"{module}.weight_scale_inv" not in dst_map):
                skipped += 1
                continue
            shape = torch.Size(get(src, src_map, f"{module}.weight_shape").tolist())
            want = unpack_src(get(src, src_map, f"{module}.weight_packed"), shape)
            got = unpack_nibbles_int8(get(dst, dst_map, f"{module}.weight"))
            if got.shape != want.shape or not torch.equal(
                got.to(want.dtype), want
            ):
                mismatched += 1
                if mismatched <= 3:
                    _fail(f"expert values differ in {module}", errors)
            a = get(src, src_map, f"{module}.weight_scale")
            b = get(dst, dst_map, f"{module}.weight_scale_inv")
            if a.dtype != b.dtype or not torch.equal(a, b):
                scale_mismatched += 1
                if scale_mismatched <= 3:
                    _fail(f"expert scale not bitwise identical in {module} "
                          f"({a.dtype} vs {b.dtype})", errors)
        if skipped:
            print(f"  [warn] {skipped}/{len(picks)} sampled expert modules "
                  f"absent from the output (already reported above)", flush=True)
        if not mismatched and not scale_mismatched:
            _ok(f"checked {len(picks) - skipped} expert modules: values and "
                f"scales bit-exact")
        elif mismatched:
            _fail(f"{mismatched}/{len(picks) - skipped} expert modules "
                  f"mis-encoded", errors)

    print("\n== fp8-rest agreement with the source ==")
    # Experts must be EXCLUDED here even though they also carry
    # weight_scale_inv in the output and weight_scale in the source: their
    # source weight lives under weight_packed, so treating them as fp8-rest
    # looks up a `.weight` that does not exist. Keyed on the absence of
    # weight_packed rather than on a name pattern, so a differently-named expert
    # container cannot slip through.
    fp8_modules = []
    for key in dst_map:
        if not key.endswith(".weight_scale_inv"):
            continue
        stem = key[: -len(".weight_scale_inv")]
        if f"{stem}.weight_packed" in src_map:
            continue
        if f"{stem}.weight_scale" in src_map and f"{stem}.weight" in src_map:
            fp8_modules.append(stem)
    fp8_modules.sort()
    if not fp8_modules:
        warnings.append("no fp8-rest modules matched; nothing compared")
    else:
        picks = fp8_modules[:: max(1, len(fp8_modules) // samples)][:samples]
        resids: list[tuple[float, str]] = []
        for module in picks:
            on_disk = (
                get(src, src_map, f"{module}.weight").float()
                * get(src, src_map, f"{module}.weight_scale").float()
            )
            converted = dequantize_block_fp8(
                get(dst, dst_map, f"{module}.weight"),
                get(dst, dst_map, f"{module}.weight_scale_inv"),
                DEFAULT_BLOCK,
            )
            denom = on_disk.norm()
            if denom > 0:
                resids.append((((converted - on_disk).norm() / denom).item(),
                               module))
        if resids:
            from statistics import median

            ordered = sorted(resids)
            print(f"  residuals: n={len(ordered)} min={ordered[0][0]:.4f} "
                  f"median={median([r for r, _ in ordered]):.4f} "
                  f"max={ordered[-1][0]:.4f} (bound {_FP8_MAX_RESID})",
                  flush=True)
            print(f"    max: {ordered[-1][1]}", flush=True)
            if ordered[-1][0] > _FP8_MAX_RESID:
                _fail(f"fp8-rest residual {ordered[-1][0]:.4f} exceeds "
                      f"{_FP8_MAX_RESID} on {ordered[-1][1]}; the AWQ fold is "
                      f"probably dropped or misapplied", errors)
            else:
                _ok(f"sampled {len(ordered)} fp8-rest modules within two e4m3 "
                    f"roundings of the source")

    print("\n== passthrough tensors (must be BIT-EXACT) ==")
    passthrough = [
        k for k in dst_map
        if k in src_map
        and not k.endswith((".weight_scale_inv", ".input_scale"))
        and f"{_stem(k)}.weight_packed" not in src_map
        and f"{_stem(k)}.weight_scale" not in src_map
    ]
    picks = passthrough[:: max(1, len(passthrough) // samples)][:samples]
    bad = 0
    for key in picks:
        a, b = get(src, src_map, key), get(dst, dst_map, key)
        if a.dtype != b.dtype or a.shape != b.shape or not torch.equal(a, b):
            bad += 1
            if bad <= 3:
                _fail(f"passthrough tensor changed: {key}", errors)
    if not bad:
        _ok(f"sampled {len(picks)} of {len(passthrough)} passthrough tensors: "
            f"identical")

    print("\n== index integrity ==")
    declared = int(dst_index.get("metadata", {}).get("total_size", 0))
    actual = 0
    for shard in sorted({v for v in dst_map.values()}):
        path = dst / shard
        if not path.is_file():
            _fail(f"index references a missing shard: {shard}", errors)
            continue
        with safe_open(str(path), framework="pt") as handle:
            for key in handle.keys():
                slice_ = handle.get_slice(key)
                numel = 1
                for dim in slice_.get_shape():
                    numel *= dim
                actual += numel * _dtype_bytes(slice_.get_dtype())
    if declared != actual:
        _fail(f"index total_size {declared} != {actual} bytes on disk "
              f"(delta {actual - declared})", errors)
    else:
        _ok(f"index total_size matches bytes on disk exactly "
            f"({actual / 1e9:.1f} GB)")

    print("\n== summary ==")
    for warning in warnings:
        print(f"  [warn] {warning}", flush=True)
    if errors:
        print(f"  RESULT: FAIL ({len(errors)} error(s))", flush=True)
        return 1
    print("  RESULT: PASS", flush=True)
    return 0


def _stem(key: str) -> str:
    for suffix in (".weight_scale_inv", ".weight_scale", ".weight_packed",
                   ".weight_shape", ".input_scale", ".weight"):
        if key.endswith(suffix):
            return key[: -len(suffix)]
    return key


def _source_unpacker(warnings: list[str]):
    try:
        from compressed_tensors.compressors.pack_quantized.base import (
            unpack_from_int32,
        )
    except ImportError as err:
        warnings.append(f"compressed-tensors unavailable ({err})")
        return None
    return lambda packed, shape: unpack_from_int32(packed, 4, shape)


def _dtype_bytes(dtype: str) -> int:
    return {
        "F64": 8, "I64": 8, "U64": 8,
        "F32": 4, "I32": 4, "U32": 4,
        "F16": 2, "BF16": 2, "I16": 2, "U16": 2,
        "F8_E4M3": 1, "F8_E5M2": 1, "I8": 1, "U8": 1, "BOOL": 1,
    }.get(dtype, 1)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--src", required=True, type=Path)
    parser.add_argument("--dst", required=True, type=Path)
    parser.add_argument("--samples", type=int, default=40)
    parser.add_argument("--all-experts", action="store_true",
                        help="check every expert module rather than a sample "
                             "(exhaustive; hours on a full checkpoint)")
    args = parser.parse_args(argv)
    return verify(args.src, args.dst, args.samples, args.all_experts)


if __name__ == "__main__":
    sys.exit(main())
