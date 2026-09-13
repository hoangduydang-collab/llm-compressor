"""Static regression checks for the CPU staging script."""

from __future__ import annotations

from pathlib import Path


_STAGE = Path(__file__).resolve().parents[1] / "k8s" / "stage-glm53-quality-eval.sh"


def test_no_gptq_parity_does_not_replace_an_empty_identity():
    text = _STAGE.read_text(encoding="utf-8")
    assert 'identities = ["glm-5.3-w4afp8-ours", "glm-5.3-w4afp8-phala", ours, phala]' in text
    assert 'if gptq:\n        identities.extend(["glm-5.3-w4afp8-gptq", gptq])' in text
