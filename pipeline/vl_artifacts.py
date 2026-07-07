"""Ensure VL processor artifacts exist in a quantized checkpoint for vLLM load.

``model.save_pretrained`` + ``tokenizer.save_pretrained`` do not write image-
processor configs (``preprocessor_config.json``, etc.). vLLM's MiniMax-M3 path
still initializes the multimodal budget and fails if those files are missing.
"""

from __future__ import annotations

from pathlib import Path

# Any one of these indicates processor artifacts were saved/copied.
_PROCESSOR_MARKERS = (
    "preprocessor_config.json",
    "processor_config.json",
)


def vl_processor_artifacts_present(ckpt: Path) -> bool:
    return any((ckpt / name).exists() for name in _PROCESSOR_MARKERS)


def ensure_vl_processor_artifacts(
    ckpt: Path,
    source: str,
    *,
    trust_remote_code: bool = True,
) -> list[str]:
    """Copy HF processor files from *source* into *ckpt* when missing.

    Returns the list of new file names written (empty if already present).
    """
    if vl_processor_artifacts_present(ckpt):
        return []

    from transformers import AutoProcessor

    processor = AutoProcessor.from_pretrained(source, trust_remote_code=trust_remote_code)
    before = {p.name for p in ckpt.iterdir() if p.is_file()}
    processor.save_pretrained(str(ckpt))
    after = {p.name for p in ckpt.iterdir() if p.is_file()}
    added = sorted(after - before)
    return added
