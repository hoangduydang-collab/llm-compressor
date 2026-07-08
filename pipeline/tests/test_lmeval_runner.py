"""Unit tests for per-task lm-eval kwargs (no GPU)."""

import os

from pipeline.config import EvalTask, PipelineConfig, ServeConfig
from pipeline._env import apply_lm_eval_sglang_compat, apply_sglang_compat_env
from pipeline.lmeval_runner import (
    model_args,
    per_task_limit,
    per_task_num_fewshot,
    sglang_model_args,
    vllm_model_args,
)


def test_per_task_num_fewshot_scalar_when_uniform():
    tasks = [
        EvalTask(name="a", num_fewshot=5),
        EvalTask(name="b", num_fewshot=5),
    ]
    assert per_task_num_fewshot(tasks) == 5


def test_per_task_num_fewshot_dict_when_mixed():
    tasks = [
        EvalTask(name="wikitext", num_fewshot=0),
        EvalTask(name="mmlu", num_fewshot=5),
    ]
    # Helper still reports dict form; evaluate_tasks uses per-task scalars instead.
    assert per_task_num_fewshot(tasks) == {"wikitext": 0, "mmlu": 5}


def test_merge_eval_results():
    from pipeline.lmeval_runner import _merge_eval_results

    merged: dict = {}
    _merge_eval_results(
        merged,
        {
            "results": {"wikitext": {"word_perplexity,none": 11.0}},
            "samples": {"wikitext": [{"doc_id": 0}]},
            "config": {"model": "vllm"},
        },
    )
    _merge_eval_results(
        merged,
        {
            "results": {"mmlu": {"acc,none": 0.8}},
            "samples": {"mmlu": [{"doc_id": 1}]},
        },
    )
    assert set(merged["results"]) == {"wikitext", "mmlu"}
    assert len(merged["samples"]["wikitext"]) == 1
    assert merged["config"] == {"model": "vllm"}


def test_merge_eval_results_accumulates_groups():
    """Group aggregates (mmlu, bbh) must survive across multiple task batches."""
    from pipeline.lmeval_runner import _merge_eval_results
    from pipeline.metrics_lmeval import task_results_from_batch

    merged: dict = {}
    _merge_eval_results(
        merged,
        {
            "results": {"mmlu_anatomy": {"acc,none": 0.6}},
            "groups": {"mmlu": {"acc,none": 0.65}},
            "group_subtasks": {"mmlu": ["mmlu_anatomy"]},
        },
    )
    _merge_eval_results(
        merged,
        {
            "results": {"bbh_boolean_expressions": {"exact_match,none": 0.7}},
            "groups": {"bbh": {"exact_match,none": 0.72}},
            "group_subtasks": {"bbh": ["bbh_boolean_expressions"]},
        },
    )
    assert set(merged["groups"]) == {"mmlu", "bbh"}
    assert set(merged["group_subtasks"]) == {"mmlu", "bbh"}
    # Both group aggregates remain resolvable from the merged dict.
    assert task_results_from_batch(merged, "mmlu") == {"acc,none": 0.65}
    assert task_results_from_batch(merged, "bbh") == {"exact_match,none": 0.72}


def test_per_task_limit_omitted_when_all_unlimited():
    tasks = [
        EvalTask(name="wikitext", limit=None),
        EvalTask(name="mmlu", limit=None),
    ]
    assert per_task_limit(tasks) is None


def test_per_task_limit_scalar_when_uniform():
    tasks = [
        EvalTask(name="a", limit=250),
        EvalTask(name="b", limit=250),
    ]
    assert per_task_limit(tasks) == 250


def test_per_task_limit_dict_when_mixed():
    tasks = [
        EvalTask(name="wikitext", limit=None),
        EvalTask(name="mmlu", limit=250),
    ]
    assert per_task_limit(tasks) == {"mmlu": 250}


def test_vllm_model_args():
    cfg = PipelineConfig()
    cfg.model.trust_remote_code = True
    cfg.serve.tensor_parallel_size = 2
    cfg.serve.max_model_len = 4096
    cfg.serve.disable_custom_all_reduce = True
    args = vllm_model_args(cfg, "/models/qwen")
    assert "pretrained=/models/qwen" in args
    assert "tensor_parallel_size=2" in args
    assert "max_model_len=4096" in args
    assert "trust_remote_code=True" in args
    assert "disable_custom_all_reduce=True" in args


def test_sglang_model_args_maps_serve_knobs():
    cfg = PipelineConfig()
    cfg.model.trust_remote_code = True
    cfg.serve = ServeConfig(
        tensor_parallel_size=8,
        max_model_len=8192,
        gpu_memory_utilization=0.85,
        kv_cache_dtype="fp8",
        sglang_kwargs={
            "quantization": "w4afp8",
            "disable_shared_experts_fusion": True,
        },
    )
    args = sglang_model_args(cfg, "/models/glm")
    assert "pretrained=/models/glm" in args
    assert "tp_size=8" in args
    assert "context_length=8192" in args
    assert "mem_fraction_static=0.85" in args
    assert "kv_cache_dtype=fp8_e4m3" in args
    assert "quantization=w4afp8" in args
    assert "disable_shared_experts_fusion=True" in args


def test_model_args_dispatches_on_backend():
    cfg = PipelineConfig()
    cfg.eval.backend = "sglang"
    cfg.serve.tensor_parallel_size = 4
    assert "tp_size=4" in model_args(cfg, "/m")
    cfg.eval.backend = "vllm"
    assert "tensor_parallel_size=4" in model_args(cfg, "/m")


def test_apply_sglang_compat_env_sets_sglang_keys(monkeypatch):
    for key in (
        "FLASHINFER_USE_CUDA_NORM",
        "SGLANG_ENABLE_JIT_DEEPGEMM",
        "SGLANG_DG_USE_NVRTC",
        "DG_JIT_USE_NVRTC",
        "DG_JIT_NVCC_COMPILER",
    ):
        monkeypatch.delenv(key, raising=False)
    applied = apply_sglang_compat_env()
    assert applied["SGLANG_ENABLE_JIT_DEEPGEMM"] == "0"
    assert os.environ["FLASHINFER_USE_CUDA_NORM"] == "1"


def test_apply_sglang_compat_env_keeps_nvcc_when_preset(monkeypatch):
    """env.sh may pre-set DG_JIT_NVCC_COMPILER; must not fall back to NVRTC."""
    nvcc129 = "/mnt/nfs/hoangduy/cuda-12.9/bin/nvcc"
    for key in (
        "FLASHINFER_USE_CUDA_NORM",
        "SGLANG_ENABLE_JIT_DEEPGEMM",
        "SGLANG_DG_USE_NVRTC",
        "DG_JIT_USE_NVRTC",
        "DG_JIT_NVCC_COMPILER",
        "CUDA_HOME",
        "WORK_ROOT",
    ):
        monkeypatch.delenv(key, raising=False)
    monkeypatch.setenv("DG_JIT_NVCC_COMPILER", nvcc129)
    monkeypatch.setenv("DG_JIT_USE_NVRTC", "0")
    monkeypatch.setenv("SGLANG_DG_USE_NVRTC", "0")

    def fake_version(path: str):
        return (12, 9) if path == nvcc129 else (12, 4)

    monkeypatch.setattr("pipeline._env._nvcc_version", fake_version)
    monkeypatch.setattr(
        "pipeline._env._iter_nvcc_candidates",
        lambda: [nvcc129, "/usr/local/cuda-12.4/bin/nvcc"],
    )

    applied = apply_sglang_compat_env()
    assert applied["DG_JIT_NVCC_COMPILER"] == nvcc129
    assert applied["DG_JIT_USE_NVRTC"] == "0"
    assert applied["SGLANG_DG_USE_NVRTC"] == "0"
    assert "SGLANG_DG_USE_NVRTC" not in applied or applied["SGLANG_DG_USE_NVRTC"] == "0"


def test_apply_lm_eval_sglang_compat_maps_max_tokens():
    try:
        from sglang.srt.sampling.sampling_params import SamplingParams
    except ImportError:
        return

    apply_lm_eval_sglang_compat()
    params = SamplingParams(max_tokens=32, temperature=0.0)
    assert params.max_new_tokens == 32
