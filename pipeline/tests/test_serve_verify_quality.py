"""CPU-only tests for offline serve generation and MiniMax-M3 quality fields."""

from __future__ import annotations

from pipeline.m3_quality_evidence import M3_QUALITY_CASES
from pipeline.serve_verify import _run_generation_smoke


class _Completion:
    def __init__(self, text: str):
        self.outputs = [type("Output", (), {"text": text})()]


class _FakeLLM:
    def __init__(self, outputs: list[str]):
        self._outputs = outputs
        self.prompts = None
        self.sampling_params = None

    def generate(self, prompts, sampling_params):
        self.prompts = prompts
        self.sampling_params = sampling_params
        return [_Completion(text) for text in self._outputs]


class _SamplingParams:
    def __init__(self, **kwargs):
        self.kwargs = kwargs


def test_m3_generation_runs_fixed_suite_and_rejects_garbage():
    llm = _FakeLLM(["arring" * 20, "4"])

    result = _run_generation_smoke(
        llm,
        is_m3=True,
        configured_prompt="ignored for M3",
        sampling_params_cls=_SamplingParams,
    )

    assert llm.prompts == [case.prompt for case in M3_QUALITY_CASES]
    assert llm.sampling_params.kwargs == {"max_tokens": 64, "temperature": 0.0}
    assert result["generation_completed"] is True
    assert result["quality_ok"] is False
    assert result["quality_cases"][0]["text"] == "arring" * 20
    assert result["sample_prompt"] == M3_QUALITY_CASES[0].prompt


def test_m3_generation_accepts_two_correct_outputs():
    llm = _FakeLLM(["Paris", "4"])

    result = _run_generation_smoke(
        llm,
        is_m3=True,
        configured_prompt="ignored",
        sampling_params_cls=_SamplingParams,
    )

    assert result["quality_ok"] is True
    assert result["generation_completed"] is True


def test_non_m3_generation_retains_configured_single_prompt():
    llm = _FakeLLM(["Paris"])

    result = _run_generation_smoke(
        llm,
        is_m3=False,
        configured_prompt="The capital of France is",
        sampling_params_cls=_SamplingParams,
    )

    assert llm.prompts == ["The capital of France is"]
    assert result["sample_output"] == "Paris"
    assert result["generation_completed"] is True
    assert result["quality_ok"] is None
    assert "quality_cases" not in result
