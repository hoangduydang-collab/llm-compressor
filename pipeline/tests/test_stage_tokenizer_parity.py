"""The step-3 tokenizer/template parity classifier, executed rather than grepped.

A static string assertion cannot tell an inverted condition from a correct one,
and this classifier decides whether three quality arms are comparable at all --
so the embedded block is extracted from the staging script and RUN against
fabricated tokenizer dirs.

Pinned behaviour:
  * a truncation-only tokenizer.json delta is INERT (measured: transformers
    resets backend truncation per call, so identical ids come out either way)
  * a vocab/merges/normalizer/pre_tokenizer/post_processor/decoder/added_tokens
    delta BLOCKS
  * a chat_template.jinja byte delta BLOCKS -- it is the prompt text itself
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

_STAGE = Path(__file__).resolve().parents[1] / "k8s" / "stage-glm53-quality-eval.sh"


def _extract_parity_block() -> str:
    """The python heredoc under 'step 3', by its own delimiters."""
    lines = _STAGE.read_text(encoding="utf-8").splitlines()
    start = None
    for i, ln in enumerate(lines):
        if ln.startswith('"$PY" - "$OURS" "$PHALA" "$GPTQ" <<') and "PY" in ln:
            start = i + 1
            break
    assert start is not None, "could not find the step-3 parity heredoc"
    end = next(i for i in range(start, len(lines)) if lines[i].strip() == "PY")
    return "\n".join(lines[start:end])


_VOCAB = {"a": 0, "b": 1, "c": 2}
_MERGES = [["a", "b"]]


def _tokenizer_json(*, truncation=None, vocab=None, normalizer="NFC"):
    return {
        "version": "1.0",
        "truncation": truncation,
        "padding": None,
        "added_tokens": [{"id": 0, "content": "<eos>"}],
        "normalizer": {"type": normalizer},
        "pre_tokenizer": {"type": "ByteLevel"},
        "post_processor": None,
        "decoder": {"type": "ByteLevel"},
        "model": {"type": "BPE", "vocab": vocab or _VOCAB, "merges": _MERGES},
    }


def _arm_dir(root: Path, name: str, tok: dict, *, chat_template="TEMPLATE") -> Path:
    d = root / name
    d.mkdir(parents=True, exist_ok=True)
    (d / "tokenizer.json").write_text(json.dumps(tok), encoding="utf-8")
    # identical across arms except where a test varies it
    (d / "tokenizer_config.json").write_text(
        json.dumps({"model_max_length": 128000, "is_local": name == "ours"}),
        encoding="utf-8")
    (d / "chat_template.jinja").write_text(chat_template, encoding="utf-8")
    (d / "generation_config.json").write_text(
        json.dumps({"temperature": 1.0}), encoding="utf-8")
    return d


def _run(tmp_path, ours_tok, phala_tok, gptq_tok, **kw):
    block = _extract_parity_block()
    ours = _arm_dir(tmp_path, "ours", ours_tok)
    phala = _arm_dir(tmp_path, "phala", phala_tok)
    gptq = _arm_dir(tmp_path, "gptq", gptq_tok, **kw)
    script = tmp_path / "parity.py"
    script.write_text(block, encoding="utf-8")
    proc = subprocess.run([sys.executable, str(script), str(ours), str(phala), str(gptq)],
                          capture_output=True, text=True)
    # The block is a GATE: it exits 1 when it blocks. Both outcomes print the
    # same JSON report, so the exit code is asserted per-case below rather than
    # here, and stdout is always the thing under test.
    assert proc.stdout, proc.stderr
    payload = json.loads(proc.stdout)
    payload["_exit"] = proc.returncode
    return payload


def test_truncation_only_delta_is_inert_and_does_not_block(tmp_path):
    """The real GPTQ case: 2048 truncation baked in by calibration."""
    res = _run(tmp_path,
               _tokenizer_json(truncation=None),
               _tokenizer_json(truncation=None),
               _tokenizer_json(truncation={"direction": "Right", "max_length": 2048,
                                           "strategy": "LongestFirst", "stride": 0}))
    assert res["_exit"] == 0, "an inert truncation delta must not fail the gate"
    assert res["blocking_differences"] == []
    assert res["tokenizer_json"]["gptq"]["inert_differences"] == ["truncation"]
    assert res["tokenizer_json"]["gptq"]["semantic_differences"] == []
    # the digests really do differ -- this is not a test of an equal-bytes case
    assert res["tokenizer_json"]["gptq"]["digest_equal"] is False
    assert res["tokenizer_json"]["phala"]["inert_differences"] == []


def test_a_vocab_delta_still_blocks(tmp_path):
    res = _run(tmp_path,
               _tokenizer_json(),
               _tokenizer_json(),
               _tokenizer_json(vocab={"a": 0, "b": 1, "zzz": 9}))
    assert res["_exit"] == 1, "a vocab delta must fail the gate"
    assert any(d.startswith("gptq:tokenizer.json:") for d in res["blocking_differences"])
    assert "model" in res["tokenizer_json"]["gptq"]["semantic_differences"]


def test_a_normalizer_delta_still_blocks(tmp_path):
    res = _run(tmp_path,
               _tokenizer_json(),
               _tokenizer_json(),
               _tokenizer_json(normalizer="NFKC"))
    assert res["_exit"] == 1, "a normalizer delta must fail the gate"
    assert "normalizer" in res["tokenizer_json"]["gptq"]["semantic_differences"]
    assert any("tokenizer.json" in d for d in res["blocking_differences"])


def test_a_chat_template_delta_blocks_outright(tmp_path):
    """Still a raw-digest blocker: the template IS the prompt text."""
    res = _run(tmp_path, _tokenizer_json(), _tokenizer_json(), _tokenizer_json(),
               chat_template="A DIFFERENT TEMPLATE")
    assert res["_exit"] == 1, "a chat-template delta must fail the gate"
    assert "gptq:chat_template.jinja" in res["blocking_differences"]


def test_identical_tokenizers_block_nothing(tmp_path):
    res = _run(tmp_path, _tokenizer_json(), _tokenizer_json(), _tokenizer_json())
    assert res["_exit"] == 0
    assert res["blocking_differences"] == []
    assert res["tokenizer_json"]["gptq"]["digest_equal"] is True
