"""Post-hoc correctness verification for a GLM-5.2 compressed-tensors checkpoint.

Why this exists rather than reusing the in-run gates: the GLM-5.2 AWQ smoke
(`quant-glm52-awq-20260828t070917z`) saved a complete checkpoint and then exited 1
in its own post-save gates, because that run predates two fixes made the same day:

  * `pipeline/verify_quant_checkpoint.py` asserted a hardcoded MiniMax-M3 keep-bf16
    ignore list. Five of its nine entries name modules GLM-5.2 does not have
    (vision_tower, multi_modal_projector, patch_merge, block_sparse_moe, indexer),
    so a healthy GLM checkpoint collects five [FAIL] lines. Fixed by
    `--expect-ignore-preset glm52`.
  * `pipeline/m3_checkpoint_scale_audit.py` hardcoded M3's OFFSET norm form
    (`gain = 1 + weight`). GLM-5.2's norm class `GlmMoeDsaRMSNorm` is ORDINARY
    (`gain = weight`), registered in
    `llmcompressor.preflight.quantization.KNOWN_ORDINARY_NORM_CLASSES`, so the
    correct offset here is **0.0**. Auditing a healthy GLM fold with the M3 form
    scored it ~0.169 against a 0.02 threshold -- inside the M3 "lost fold" band,
    i.e. it mimics a catastrophic numerics failure.

So a green run of this script is the first real correctness statement about that
checkpoint; the failed exit code says nothing about the artifact.

`m3_checkpoint_scale_audit.py` is deliberately NOT used: its CLI requires a
four-way comparison (--base --reference --awq --gptq) built for the M3 matrix, and
the GPTQ arm does not exist yet.

WHAT IS AND IS NOT CHECKED HERE. This is structural and numerical verification of
the saved tensors:
  1. the persisted quantization_config matches the GLM keep-bf16 recipe;
  2. every routed expert in every quantized layer has the full set of packed
     weights and finite scales, with the layer range and expert count discovered
     from the checkpoint rather than assumed;
  3. sampled modules dequantize to values that agree with the base weights under a
     fitted per-column smoothing scale (this is what catches the r3-class bug where
     a distributed qparam broadcast left ~7/8 of scales uninitialized while every
     structural gate passed);
  4. the AWQ smoothing fold is explained by the norm-implied scale, in GLM's
     ordinary-norm form.
It is NOT a quality evaluation. Passing says the checkpoint is self-consistent and
faithful to the base weights, not that the quantized model scores well.
"""

from __future__ import annotations

import argparse
import glob
import os
import sys
from pathlib import Path

# GLM-5.2's norm class is ordinary (output * weight), so the smoothing fold is
# scale = weight_base / weight_folded with NO 1+ offset. Getting this wrong does
# not produce a small error -- it produces a fake catastrophic one.
GLM52_NORM_GAIN_OFFSET = 0.0

# Layers the smoke actually quantized (its ignore list keeps only 3 and 42).
# Layers that were never smoothed audit as scale == 1 and pass trivially, so a
# wider list is safe but slower.
DEFAULT_GATE_LAYERS = "3,42"


def resolve_base_snapshot(hf_cache: str, repo: str = "zai-org/GLM-5.2") -> Path | None:
    """Locate the base model snapshot inside an HF cache directory."""
    slug = "models--" + repo.replace("/", "--")
    hits = sorted(glob.glob(os.path.join(hf_cache, slug, "snapshots", "*")))
    for hit in hits:
        if Path(hit, "model.safetensors.index.json").exists():
            return Path(hit)
    return None


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ckpt", required=True, type=Path,
                        help="the saved checkpoint directory (contains config.json)")
    parser.add_argument("--base", type=Path, default=None,
                        help="base model snapshot; resolved from --hf-cache if omitted")
    parser.add_argument("--hf-cache", default=os.environ.get("HF_HOME", "/mnt/cephfs/.hf-cache"))
    parser.add_argument("--layers", default=DEFAULT_GATE_LAYERS,
                        help=f"comma-separated layers for the fold gate (default {DEFAULT_GATE_LAYERS})")
    parser.add_argument("--skip-fold-gate", action="store_true",
                        help="run only the structural/tensor verification")
    parser.add_argument("--skip-dequant", action="store_true",
                        help="skip the value-level dequant-vs-base comparison "
                             "(faster, but drops the check that catches "
                             "uninitialized-scale corruption)")
    args = parser.parse_args(argv)

    if not (args.ckpt / "config.json").exists():
        print(f"error: {args.ckpt}/config.json not found")
        return 2

    base = args.base or resolve_base_snapshot(args.hf_cache)
    if base is None:
        print(f"error: could not resolve a GLM-5.2 snapshot under {args.hf_cache}; "
              "pass --base explicitly")
        return 2
    print(f"==> checkpoint : {args.ckpt}")
    print(f"==> base       : {base}")
    print(f"==> norm form  : output * ({GLM52_NORM_GAIN_OFFSET:g} + weight)  "
          "(GlmMoeDsaRMSNorm is ordinary)")

    failures: list[str] = []

    # --- 1-3. structure + sampled tensors + dequant-vs-base -----------------
    from pipeline.verify_quant_checkpoint import _IGNORE_PRESETS, verify

    print("\n" + "=" * 70)
    print("STRUCTURAL + TENSOR VERIFICATION (glm52 keep-bf16 preset)")
    print("=" * 70)
    rc = verify(
        args.ckpt,
        check_tensors=True,
        dequant_base=None if args.skip_dequant else base,
        expect_ignore=_IGNORE_PRESETS["glm52"],
    )
    if rc != 0:
        failures.append("verify_quant_checkpoint returned non-zero")

    # --- 4. smoothing fold in GLM's ordinary-norm form ----------------------
    if not args.skip_fold_gate:
        print("\n" + "=" * 70)
        print("SMOOTH-FOLD GATE (ordinary-norm form, offset 0.0)")
        print("=" * 70)
        from pipeline.quantize import assert_smooth_fold_consistency

        layers = [int(p) for p in args.layers.split(",") if p.strip()]
        try:
            assert_smooth_fold_consistency(
                args.ckpt, base, layers,
                norm_gain_offset=GLM52_NORM_GAIN_OFFSET,
            )
        except Exception as exc:  # gate raises RuntimeError on a real failure
            print(f"[FAIL] smooth-fold gate: {exc}")
            failures.append("smooth-fold gate failed")

    print("\n" + "=" * 70)
    if failures:
        print("RESULT: FAILED")
        for item in failures:
            print(f"  - {item}")
        return 1
    print("RESULT: PASSED -- checkpoint is structurally sound, sampled tensors are "
          "finite and agree with the base weights, and the AWQ fold is explained.")
    print("NOTE: this is not a quality evaluation.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
