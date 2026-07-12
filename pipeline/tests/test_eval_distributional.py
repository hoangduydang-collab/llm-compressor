"""Tests for the MiniMax teacher-forced distributional fidelity probe."""

from __future__ import annotations

from collections import Counter
from types import SimpleNamespace

import pytest

from pipeline.evalsuite.distributional import (
    compare_distributional_records,
    normalize_prompt_logprobs,
)
from pipeline.evalsuite.probe_corpus import build_probe_corpus
from pipeline.m3_distributional_probe import probe_with_engine


class FakeTokenizer:
    def encode(self, text, add_special_tokens=False):
        del add_special_tokens
        return [index % 101 for index, _ in enumerate(text.split())]


def _record(
    *,
    observed_token_id=7,
    observed_logprob=-1.0,
    top=((10, -0.1), (11, -1.0)),
    corpus_sha256="corpus",
):
    return {
        "schema_version": 1,
        "corpus_sha256": corpus_sha256,
        "prompt_id": "short-0",
        "length_bucket": "short",
        "prompt_token_count": 8,
        "position": 1,
        "observed_token_id": observed_token_id,
        "observed_logprob": observed_logprob,
        "top_logprobs": [
            {"token_id": token_id, "logprob": logprob, "rank": rank}
            for rank, (token_id, logprob) in enumerate(top, start=1)
        ],
    }


def test_probe_corpus_has_fixed_nonoverlapping_length_buckets():
    texts = ["one two three " * 40_000]

    first = build_probe_corpus(texts, FakeTokenizer(), seed=42)
    second = build_probe_corpus(texts, FakeTokenizer(), seed=42)

    assert first == second
    assert Counter(row["length_bucket"] for row in first) == {
        "short": 8,
        "8k": 4,
        "32k": 2,
    }
    assert {len(row["prompt_token_ids"]) for row in first} == {2048, 8192, 32768}
    spans = sorted((row["start_token"], row["end_token"]) for row in first)
    assert all(left[1] <= right[0] for left, right in zip(spans, spans[1:]))


def test_probe_corpus_rejects_insufficient_tokens():
    with pytest.raises(ValueError, match="requires"):
        build_probe_corpus(["too short"], FakeTokenizer(), seed=42)


def test_distributional_metrics_use_observed_token_and_topk():
    reference = [_record(observed_logprob=-1.0)]
    candidate = [
        _record(
            observed_logprob=-1.5,
            top=((10, -0.2), (12, -0.9)),
        )
    ]

    result = compare_distributional_records(reference, candidate)

    assert result["paired_tokens"] == 1
    assert result["mean_observed_logprob_drift"] == pytest.approx(-0.5)
    assert result["top1_agreement"] == 1.0
    assert result["top5_jaccard"] == pytest.approx(1 / 3)
    assert result["bf16_top1_retained_top20"] == 1.0
    assert result["perplexity_ratio"] == pytest.approx(1.6487212707)
    assert "kl_divergence" not in result


@pytest.mark.parametrize(
    "candidate,match",
    [
        (_record(observed_token_id=99), "token identity"),
        (_record(corpus_sha256="other"), "corpus"),
    ],
)
def test_distributional_pairing_rejects_provenance_mismatch(candidate, match):
    with pytest.raises(ValueError, match=match):
        compare_distributional_records([_record()], [candidate])


def test_normalize_prompt_logprobs_supports_object_entries():
    output = SimpleNamespace(
        prompt_token_ids=[5, 7],
        prompt_logprobs=[
            None,
            {
                7: SimpleNamespace(logprob=-1.0, rank=2),
                10: SimpleNamespace(logprob=-0.1, rank=1),
            },
        ],
    )
    rows = normalize_prompt_logprobs(
        output,
        {
            "prompt_id": "short-0",
            "length_bucket": "short",
            "corpus_sha256": "corpus",
        },
    )

    assert len(rows) == 1
    assert rows[0]["observed_token_id"] == 7
    assert rows[0]["observed_logprob"] == -1.0
    assert rows[0]["top_logprobs"][0] == {
        "token_id": 10,
        "logprob": -0.1,
        "rank": 1,
    }



def test_probe_with_engine_writes_normalized_jsonl(tmp_path):
    class FakeEngine:
        def generate(self, prompts, sampling_params, use_tqdm=False):
            del sampling_params, use_tqdm
            token_ids = prompts[0]["prompt_token_ids"]
            entries = [None]
            for token_id in token_ids[1:]:
                entries.append(
                    {
                        token_id: {"logprob": -1.0, "rank": 2},
                        99: {"logprob": -0.1, "rank": 1},
                    }
                )
            return [
                SimpleNamespace(
                    prompt_token_ids=token_ids,
                    prompt_logprobs=entries,
                )
            ]

    corpus = [
        {
            "prompt_id": "short-0",
            "length_bucket": "short",
            "prompt_token_ids": [5, 7, 8],
        }
    ]
    output = tmp_path / "distributional_probe.jsonl"

    summary = probe_with_engine(
        FakeEngine(),
        sampling_params=object(),
        corpus=corpus,
        corpus_sha256="corpus",
        output_path=output,
    )

    assert summary["prompts"] == 1
    assert summary["tokens"] == 2
    assert summary["corpus_sha256"] == "corpus"
    assert len(output.read_text().splitlines()) == 2



def test_probe_with_engine_resumes_complete_prompts(tmp_path):
    class FakeEngine:
        def __init__(self):
            self.calls = 0

        def generate(self, prompts, sampling_params, use_tqdm=False):
            del sampling_params, use_tqdm
            self.calls += 1
            token_ids = prompts[0]["prompt_token_ids"]
            return [
                SimpleNamespace(
                    prompt_token_ids=token_ids,
                    prompt_logprobs=[
                        None,
                        {token_ids[1]: {"logprob": -1.0, "rank": 1}},
                    ],
                )
            ]

    corpus = [
        {
            "prompt_id": "short-0",
            "length_bucket": "short",
            "prompt_token_ids": [5, 7],
        }
    ]
    output = tmp_path / "probe.jsonl"
    first = FakeEngine()
    probe_with_engine(
        first,
        sampling_params=object(),
        corpus=corpus,
        corpus_sha256="corpus",
        output_path=output,
    )
    second = FakeEngine()

    summary = probe_with_engine(
        second,
        sampling_params=object(),
        corpus=corpus,
        corpus_sha256="corpus",
        output_path=output,
    )

    assert first.calls == 1
    assert second.calls == 0
    assert summary["resumed_prompts"] == 1
    assert len(output.read_text().splitlines()) == 1



def test_distributional_pairing_rejects_nonfinite_logprobs():
    with pytest.raises(ValueError, match="non-finite"):
        compare_distributional_records(
            [_record()],
            [_record(observed_logprob=float("nan"))],
        )
