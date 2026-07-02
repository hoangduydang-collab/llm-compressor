"""Unit tests for per-task lm-eval kwargs (no GPU)."""

from pipeline.config import EvalTask, PipelineConfig, ServeConfig
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
    args = vllm_model_args(cfg, "/models/qwen")
    assert "pretrained=/models/qwen" in args
    assert "tensor_parallel_size=2" in args
    assert "max_model_len=4096" in args
    assert "trust_remote_code=True" in args


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
