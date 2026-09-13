"""Static regression checks for the CPU staging script."""

from __future__ import annotations

from pathlib import Path


_STAGE = Path(__file__).resolve().parents[1] / "k8s" / "stage-glm53-quality-eval.sh"
_GPTQ_CHECKPOINT = (
    "/mnt/cephfs/hoangduy/results/glm53-ep-gptq-w4afp8/full-ep8/"
    "20260912t183612z/output/304b8051cfb2b260b61ce0cbe330e02a98e73639-gptq-W4AFP8/"
    "20260912-184239/checkpoint"
)


def test_no_gptq_parity_does_not_replace_an_empty_identity():
    text = _STAGE.read_text(encoding="utf-8")
    assert 'identities = ["glm-5.3-w4afp8-ours", "glm-5.3-w4afp8-phala", ours, phala]' in text
    assert 'if gptq:\n        identities.extend(["glm-5.3-w4afp8-gptq", gptq])' in text


def test_gptq_staging_pins_candidate_and_writes_raw_digest_handoff():
    text = _STAGE.read_text(encoding="utf-8")
    assert f'GPTQ_CHECKPOINT="{_GPTQ_CHECKPOINT}"' in text
    assert '[ -n "$GPTQ" ] && [ "$GPTQ" != "$GPTQ_CHECKPOINT" ]' in text
    assert 'printf \'%s\\n\' "$GPTQ_SERVED_TOKENIZER_REVISION" > "$OUT/gptq-tokenizer-revision.txt"' in text


def test_gptq_arm_has_source_only_native_format_gate():
    arm = Path(__file__).resolve().parents[1] / "k8s" / "glm53_quality_arm.sh"
    text = arm.read_text(encoding="utf-8")
    assert 'if [ "$ARM" = "gptq" ]; then' in text
    assert 'native_sglang_manifest.json' in text
    assert '.native_mtp_incomplete.json' in text
    assert 'quant.get("quant_method") != "w4afp8"' in text
    assert 'quant.get("group_size") != 128' in text
    assert 'quant.get("weight_block_size") != [128, 128]' in text
