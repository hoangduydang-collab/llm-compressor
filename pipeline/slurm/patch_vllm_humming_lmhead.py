#!/usr/bin/env python3
"""Teach vLLM's prepare_humming_layer to handle ParallelLMHead (phase I.2).

WHY
---
Phase I cells B and C (HummingLinearKernel serving the EAGLE3 drafter's lm_head)
died at weight load on all 8 ranks:

    AttributeError: 'ParallelLMHead' object has no attribute 'input_size'

`prepare_humming_layer` (quantization/utils/humming_utils.py) re-derives shapes
from layer attributes that only LinearBase sets, even though its one caller —
HummingLinearKernel.process_weights_after_loading — already holds the shapes in
its MPLinearLayerConfig. The function is annotated `layer: LinearBase`, but the
MPLinear kernel path also serves quantized ParallelLMHead layers (that is how
Marlin serves this exact lm_head today), so the annotation is a lie upstream
never noticed: HummingLinearKernel sits below Marlin in _POSSIBLE_KERNELS[CUDA]
and Marlin always succeeds, so the path is unreachable by default AND broken
when reached.

THE FIX (verified against every downstream consumer)
-----------------------------------------------------
Three attribute reads, each with a clean VocabParallelEmbedding equivalent:

    LinearBase read              ParallelLMHead equivalent
    input_size_per_partition     embedding_dim        (hidden dim; never TP-split)
    output_partition_sizes       [num_embeddings_per_partition]
    has_bias                     getattr(layer, "bias", None) is not None

Everything else was checked layer-type agnostic before writing this:
  * VocabParallelEmbedding calls the same WNA16 scheme create_weights with
    (embedding_dim, [num_embeddings_per_partition], ...), so weight_packed /
    weight_scale carry the same names and input_dim/output_dim tags that
    convert_linear_layer_to_humming_standard expects; params_dtype is set
    (vocab_parallel_embedding.py:299).
  * The extra weight_shape param flows through convert_humming untouched.
  * HummingMethod.prepare_layer_meta takes shapes explicitly and only stashes
    humming_metas on the layer; forward_layer reads humming_metas + named
    tensors + locks.
  * Humming pads N internally (pad_n_to_multiple=256, 25008 -> 25088) but
    humming_gemm allocates its output at the VALID width
    (shape_n - pad_shape_n, ops/__init__.py:132), so the vocab-parallel logits
    gather stays aligned. No vocab-padding games needed, unlike Machete.
  * Cell C's serve.log shows all 8 LinearBase layers passed Humming prepare
    before lm_head raised, so the LinearBase path is load-proven; the fallbacks
    below only fire when the LinearBase attributes are absent, so this patch is
    strictly additive.

Numerics of a Humming forward on this path have never run anywhere — the serve
gates (WNA16 kernel set + accepted-length >= ACC_MIN) are the fail-closed check.

Usage:
    python pipeline/slurm/patch_vllm_humming_lmhead.py [--check] [--vllm-dir DIR]
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

REL = "model_executor/layers/quantization/utils/humming_utils.py"

MARKER = "LLMC_HUMMING_LMHEAD"

# Each (anchor, replacement) must appear exactly once; all three are inside
# prepare_humming_layer.
EDITS = [
    (
        """    # ReplicatedLinear has no TP partitioning and so does not set
    # input_size_per_partition; for it that is just input_size. Use hasattr
    # rather than getattr's default arg, which is evaluated eagerly and would
    # raise on layers lacking input_size (e.g. ParallelLMHead).
    if hasattr(layer, "input_size_per_partition"):
        input_size_per_partition = layer.input_size_per_partition
    else:
        input_size_per_partition = layer.input_size
    shape_k_stacks = [input_size_per_partition]
    shape_n_stacks = layer.output_partition_sizes
""",
        """    # LLMC_HUMMING_LMHEAD: this function is annotated LinearBase, but the
    # MPLinear kernel path also serves quantized ParallelLMHead layers (that is
    # how Marlin serves the EAGLE3 draft lm_head). ReplicatedLinear lacks
    # input_size_per_partition (use input_size); ParallelLMHead lacks both --
    # its equivalents are embedding_dim (the hidden dim, never TP-partitioned)
    # and num_embeddings_per_partition (the per-rank vocab shard).
    if hasattr(layer, "input_size_per_partition"):
        input_size_per_partition = layer.input_size_per_partition
    elif hasattr(layer, "input_size"):
        input_size_per_partition = layer.input_size
    else:
        input_size_per_partition = layer.embedding_dim
    if hasattr(layer, "output_partition_sizes"):
        output_partition_sizes = layer.output_partition_sizes
    else:
        output_partition_sizes = [layer.num_embeddings_per_partition]
    shape_k_stacks = [input_size_per_partition]
    shape_n_stacks = output_partition_sizes
""",
    ),
    (
        """        shape_n=sum(layer.output_partition_sizes),
""",
        """        shape_n=sum(output_partition_sizes),
""",
    ),
    (
        """        has_bias=layer.has_bias,
""",
        """        has_bias=getattr(layer, "has_bias", getattr(layer, "bias", None) is not None),
""",
    ),
]


def _vllm_dir() -> Path:
    env = os.environ.get("LLMC_VLLM_DIR")
    if env:
        return Path(env)
    try:
        import vllm  # noqa: PLC0415
    except Exception:
        raise SystemExit(
            "cannot import vllm; pass --vllm-dir explicitly or run under the serve venv"
        ) from None
    return Path(vllm.__file__).parent


def _atomic_write(path: Path, text: str) -> None:
    """Temp file + rename so a concurrently-importing vLLM worker can never
    observe a torn module file (same discipline as patch_vllm_m3_serve)."""
    tmp = path.with_suffix(path.suffix + ".llmc-tmp")
    tmp.write_text(text, encoding="utf-8")
    os.replace(tmp, path)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true", help="verify only, do not write")
    ap.add_argument("--vllm-dir", type=Path, default=None)
    args = ap.parse_args()

    vdir = args.vllm_dir or _vllm_dir()
    path = vdir / REL
    if not path.is_file():
        print(f"ERROR: not found: {path}")
        return 2

    text = path.read_text(encoding="utf-8")

    if MARKER in text:
        if any(anchor in text for anchor, _ in EDITS):
            print(f"ERROR: {path} contains BOTH patched and unpatched blocks")
            return 2
        missing = [rep for _, rep in EDITS if rep not in text]
        if missing:
            print(f"ERROR: marker present but {len(missing)} replacement block(s) absent")
            return 2
        print(f"already patched: {path}")
        return 0

    for i, (anchor, _) in enumerate(EDITS):
        n = text.count(anchor)
        if n != 1:
            print(
                f"ERROR: edit {i}: anchor appears {n} times in {path}; expected 1 "
                "(vLLM layout changed?) -- refusing to guess"
            )
            return 2

    if args.check:
        print(f"UNPATCHED: {path}")
        return 1

    for anchor, replacement in EDITS:
        text = text.replace(anchor, replacement)
    _atomic_write(path, text)
    print(f"patched: {path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
