#!/usr/bin/env python3
"""Make the EAGLE3 draft lm_head's vocab padding an env-gated knob (phase I, cell D).

WHY
---
The drafter is W4A16 compressed-tensors. vLLM's `choose_mp_linear_kernel` walks
`_POSSIBLE_KERNELS[CUDA]` in priority order and takes the first kernel whose
`can_implement` passes. For MiniMax-M3's EAGLE3 drafter at TP8 that produces a
*split*:

    8 of 9 linears  -> MacheteLinearKernel
    lm_head         -> MarlinLinearKernel

because `check_machete_supports_shape` requires `out_features % 128 == 0` and the
draft lm_head's per-rank output is 200064 / 8 = 25008, with `25008 % 128 == 48`.
lm_head is 153.6 M of the 254.3 M params read per rank per drafter forward -- 60% of
the drafter's weight traffic -- so the majority of the drafter runs on the older
kernel, and vLLM warns about it in the serve log:

    WARNING marlin_utils.py:237  Marlin requires thread-tile padding for some weight
    shapes in this model. Activations and/or outputs of the padded layers are
    padded/sliced on every forward; performance may be degraded.

`marlin_padded_nk(25008, 6144, 128)` picks (25024, 6144): only +16 columns (+0.064%
of bytes), so the cost is not bandwidth -- it is a pad/slice op and an extra launch
on every drafter forward.

WHAT THIS DOES
--------------
`llama_eagle3.py` constructs the draft lm_head without passing `padding_size`, so it
inherits `DEFAULT_VOCAB_PADDING_SIZE = 64`; 200064 % 64 == 0, hence no padding and
the awkward 25008. This patch makes that argument read an env var:

    padding_size=int(os.environ.get("LLMC_EAGLE3_LMHEAD_PAD", "64"))

With the var unset the value is 64 -- byte-identical behaviour to upstream, so the
patch is INERT for every run that does not opt in. Setting it to 1024 pads the draft
vocab to `pad_vocab_size(200064, 1024) == 200704`, i.e. 25088 per rank, which is
`% 128 == 0`, so Machete takes lm_head too and Marlin drops out of the process.

WHY THIS IS SAFE
----------------
vLLM's vocab padding is a first-class mechanism (it exists for models whose vocab is
not TP-divisible) and both draft logit paths mask the padded entries:

  * `LogitsProcessor._get_logits` (logits_processor.py:101-103)
        # Remove paddings in vocab (if any).
        logits = logits[..., : self.org_vocab_size]
  * `LogitsProcessor.get_top_tokens` (logits_processor.py:131-134), the vocab-parallel
    argmax path that avoids the all-gather:
        num_pad = lm_head.shard_indices.num_org_vocab_padding
        if num_pad > 0:
            logits[..., -num_pad:] = -float("inf")

`org_vocab_size` stays 200064 either way, so no padded id can be emitted. This is
also why the alternative -- physically padding the checkpoint to 200704 rows and
declaring `draft_vocab_size=200704` -- was rejected: that raises `org_vocab_size` to
200704, defeating both masks, and this drafter ships no `d2t`/`t2d` tensors (identity
draft->target mapping), so a padded id would be emitted verbatim as a real token.
Packed zeros decode to a 0.0 logit, which can win an argmax whenever every real
logit is negative.

The blast radius is the EAGLE3 draft lm_head only. The target model's lm_head is in
its checkpoint's `ignore` list (unquantized bf16) and never reaches this code path.

Usage:
    python pipeline/slurm/patch_vllm_eagle3_lmhead_pad.py [--check] [--vllm-dir DIR]
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

REL = "model_executor/models/llama_eagle3.py"

ANCHOR = """        self.lm_head = ParallelLMHead(
            self.config.draft_vocab_size,
            self.config.hidden_size,
            quant_config=get_draft_quant_config(vllm_config),
            prefix=maybe_prefix(prefix, "lm_head"),
        )
"""

# `os` is not imported by llama_eagle3.py, and editing the import block would make
# this patch's anchor fragile across vLLM releases. A local __import__ keeps the
# whole change to one self-contained inserted line.
PATCHED = """        self.lm_head = ParallelLMHead(
            self.config.draft_vocab_size,
            self.config.hidden_size,
            quant_config=get_draft_quant_config(vllm_config),
            prefix=maybe_prefix(prefix, "lm_head"),
            # llmc phase I: env-gated draft-vocab padding. Default 64 ==
            # DEFAULT_VOCAB_PADDING_SIZE, so unset is upstream behaviour. 1024 pads
            # 200064 -> 200704 (25088/rank, % 128 == 0) so Machete can take lm_head
            # instead of Marlin. Padded logits are masked by LogitsProcessor.
            padding_size=int(__import__("os").environ.get("LLMC_EAGLE3_LMHEAD_PAD", "64")),
        )
"""

MARKER = "LLMC_EAGLE3_LMHEAD_PAD"


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
    """Write via temp file + rename so a concurrently-importing vLLM worker can
    never observe a torn module file (same discipline as patch_vllm_m3_serve)."""
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
        if ANCHOR in text:
            print(f"ERROR: {path} contains BOTH patched and unpatched lm_head blocks")
            return 2
        print(f"already patched: {path}")
        return 0

    if ANCHOR not in text:
        print(
            f"ERROR: expected lm_head construction not found in {path} "
            "(vLLM layout changed?) -- refusing to guess"
        )
        return 2

    if text.count(ANCHOR) != 1:
        print(f"ERROR: lm_head anchor appears {text.count(ANCHOR)} times; expected 1")
        return 2

    if args.check:
        print(f"UNPATCHED: {path}")
        return 1

    _atomic_write(path, text.replace(ANCHOR, PATCHED))
    print(f"patched: {path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
