"""Tests for the probe comparison gate.

The gate's job is to distinguish "quantization error" from "the engine is
reading these bytes wrongly", so the tests that matter are the ones proving it
separates those two: small scatter must pass, a systematic shift must fail.
"""

from __future__ import annotations

import json

import pytest

from pipeline.compare_logit_probes import compare, main

PROMPTS = ["a", "b"]


def _probe(outputs, status="ok", prompts=None, model="m", quant=None):
    return {
        "model": model,
        "quantization": quant,
        "prompts": PROMPTS if prompts is None else prompts,
        "status": status,
        "outputs": outputs,
    }


def _out(top_logprobs, ids=(1, 2, 3), text="x"):
    return {
        "text": text,
        "output_ids": list(ids),
        "output_top_logprobs": top_logprobs,
        "input_top_logprobs": None,
    }


def _pos(pairs):
    return [[float(lp), int(tid)] for lp, tid in pairs]


def test_identical_probes_agree_perfectly():
    lp = [_pos([(-0.1, 5), (-2.0, 6)]), _pos([(-0.3, 7), (-1.5, 8)])]
    result = compare(_probe([_out(lp), _out(lp)]), _probe([_out(lp), _out(lp)]))
    assert result["top1_agreement"] == 1.0
    assert result["mean_logprob_delta"] == 0.0
    assert result["identical_sequences"] == 2


def test_small_scatter_passes_but_a_systematic_shift_fails(tmp_path):
    """The distinction the gate exists to make. Quantization perturbs logprobs
    slightly and keeps the ranking; a scale error shifts them all."""
    ref_lp = [_pos([(-0.10, 5), (-2.00, 6)])]
    jittered = [_pos([(-0.13, 5), (-2.05, 6)])]
    shifted = [_pos([(-5.10, 6), (-7.00, 5)])]

    ref = _probe([_out(ref_lp), _out(ref_lp)])
    ok = compare(ref, _probe([_out(jittered), _out(jittered)]))
    assert ok["top1_agreement"] == 1.0
    assert ok["mean_logprob_delta"] < 0.1

    bad = compare(ref, _probe([_out(shifted), _out(shifted)]))
    assert bad["top1_agreement"] == 0.0
    assert bad["mean_logprob_delta"] > 4.0


def test_only_shared_topk_tokens_are_compared():
    """A token in one arm's top-k and absent from the other has no comparable
    logprob; substituting -inf or the k-th value would invent a delta."""
    ref_lp = [_pos([(-0.1, 5), (-2.0, 6)])]
    test_lp = [_pos([(-0.15, 5), (-2.5, 99)])]  # token 6 absent, 99 new
    result = compare(_probe([_out(ref_lp)], prompts=["a"]),
                     _probe([_out(test_lp)], prompts=["a"]))
    # Only token 5 is shared, so exactly one pair should be compared.
    assert result["compared_logprob_pairs"] == 1
    assert result["mean_logprob_delta"] == pytest.approx(0.05, abs=1e-6)


def test_mismatched_prompts_are_refused():
    lp = [_pos([(-0.1, 5)])]
    with pytest.raises(ValueError, match="different prompts"):
        compare(_probe([_out(lp)], prompts=["a"]),
                _probe([_out(lp)], prompts=["z"]))


def test_incomplete_probe_is_refused(tmp_path):
    """Comparing a failed probe would report agreement on absent data."""
    lp = [_pos([(-0.1, 5)])]
    ref = tmp_path / "ref.json"
    test = tmp_path / "test.json"
    ref.write_text(json.dumps(_probe([_out(lp)], prompts=["a"])),
                   encoding="utf-8")
    test.write_text(json.dumps(_probe([_out(lp)], prompts=["a"],
                                      status="error")), encoding="utf-8")
    assert main(["--ref", str(ref), "--test", str(test)]) == 2


def test_missing_logprobs_fail_rather_than_pass_vacuously(tmp_path):
    ref = tmp_path / "ref.json"
    test = tmp_path / "test.json"
    empty = _out(None)
    ref.write_text(json.dumps(_probe([empty], prompts=["a"])), encoding="utf-8")
    test.write_text(json.dumps(_probe([empty], prompts=["a"])), encoding="utf-8")
    assert main(["--ref", str(ref), "--test", str(test)]) == 1


def test_cli_top1_threshold_gates_independently_of_the_delta_gate(tmp_path):
    """A NEAR-TIE flip: the ranking changes while the logprobs barely move.

    This is both the realistic case -- quantization flips rankings precisely
    where the top candidates were nearly tied -- and the only way to exercise
    the top-1 threshold in isolation. Flipping the ranking by swapping two
    well-separated logprobs would also blow the delta gate, so relaxing
    --min-top1 alone would still fail and the test would prove nothing about
    which gate fired.
    """
    ref_lp = [_pos([(-1.00, 5), (-1.01, 6)])]
    flipped = [_pos([(-1.00, 6), (-1.01, 5)])]
    ref = tmp_path / "ref.json"
    test = tmp_path / "test.json"
    ref.write_text(json.dumps(_probe([_out(ref_lp)], prompts=["a"])),
                   encoding="utf-8")
    test.write_text(json.dumps(_probe([_out(flipped)], prompts=["a"])),
                    encoding="utf-8")
    # Deltas are 0.01, far under any sane delta bound, so only top-1 decides.
    assert main(["--ref", str(ref), "--test", str(test),
                 "--min-top1", "0.8"]) == 1
    assert main(["--ref", str(ref), "--test", str(test),
                 "--min-top1", "0.0"]) == 0


def test_result_json_is_written(tmp_path):
    lp = [_pos([(-0.1, 5)])]
    ref = tmp_path / "ref.json"
    test = tmp_path / "test.json"
    outp = tmp_path / "sub" / "result.json"
    ref.write_text(json.dumps(_probe([_out(lp)], prompts=["a"])),
                   encoding="utf-8")
    test.write_text(json.dumps(_probe([_out(lp)], prompts=["a"])),
                    encoding="utf-8")
    assert main(["--ref", str(ref), "--test", str(test), "--out", str(outp)]) == 0
    written = json.loads(outp.read_text())
    assert written["top1_agreement"] == 1.0
