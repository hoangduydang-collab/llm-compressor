"""Unit tests for per-task lm-eval kwargs (no GPU)."""

import json
import os
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from pipeline._env import apply_lm_eval_sglang_compat, apply_sglang_compat_env
from pipeline.config import EvalTask, PipelineConfig, ServeConfig, load_config
from pipeline.lmeval_runner import (
    _prepare_vllm_runtime,
    model_args,
    per_task_limit,
    per_task_num_fewshot,
    sglang_model_args,
    vllm_model_args,
)


def test_m3_reasoning_r4_config_pins_generation_contract():
    cfg = load_config("pipeline/configs/eval_minimax_m3_reasoning_r4.yaml")

    assert cfg.eval.generation_seeds == [42, 1234, 4158]
    assert cfg.eval.enable_thinking is True
    assert cfg.eval.think_end_token == "</mm:think>"
    assert cfg.eval.gen_kwargs == {
        "temperature": 1.0,
        "top_p": 0.95,
        "do_sample": True,
        "max_gen_toks": 16384,
    }
    assert [task.name for task in cfg.eval.tasks] == [
        "gpqa_diamond",
        "mmlu_pro",
        "gsm8k",
        "aime_2025",
    ]
    assert [task.metric for task in cfg.eval.tasks] == [
        "exact_match,flexible-extract",
        "exact_match,custom-extract",
        "exact_match,strict-match",
        "exact_match,none",
    ]
    assert [task.num_fewshot for task in cfg.eval.tasks] == [0, 5, 8, 0]


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


def test_vllm_model_args_forwards_typed_vllm_kwargs():
    cfg = PipelineConfig()
    cfg.serve.vllm_kwargs = {
        "distributed_executor_backend": "ray",
        "enable_expert_parallel": True,
        "block_size": 128,
        "kv_cache_dtype": "fp8",
    }

    args = vllm_model_args(cfg, "/models/minimax-m3")

    assert "distributed_executor_backend=ray" in args
    assert "enable_expert_parallel=True" in args
    assert "block_size=128" in args
    assert "kv_cache_dtype=fp8" in args


def test_vllm_model_args_forwards_typed_serve_runtime_envelope():
    cfg = PipelineConfig()
    cfg.serve.enable_expert_parallel = True
    cfg.serve.block_size = 128
    cfg.serve.kv_cache_dtype = "fp8"

    args = vllm_model_args(cfg, "/models/minimax-m3")

    assert "enable_expert_parallel=True" in args
    assert "block_size=128" in args
    assert "kv_cache_dtype=fp8" in args


def test_vllm_model_args_explicit_kwargs_override_typed_envelope_once():
    cfg = PipelineConfig()
    cfg.serve.enable_expert_parallel = True
    cfg.serve.block_size = 128
    cfg.serve.kv_cache_dtype = "fp8"
    cfg.serve.vllm_kwargs = {
        "enable_expert_parallel": False,
        "block_size": 64,
        "kv_cache_dtype": "auto",
    }

    args = vllm_model_args(cfg, "/models/minimax-m3")

    assert args.count("enable_expert_parallel=") == 1
    assert args.count("block_size=") == 1
    assert args.count("kv_cache_dtype=") == 1
    assert "enable_expert_parallel=False" in args
    assert "block_size=64" in args
    assert "kv_cache_dtype=auto" in args


def test_prepare_vllm_runtime_reuses_minimax_runtime(monkeypatch, tmp_path):
    calls = []
    monkeypatch.setattr(
        "pipeline.m3_distributional_probe._prepare_minimax_runtime",
        lambda model, source: calls.append((model, source)) or {"prepared": True},
    )

    result = _prepare_vllm_runtime(str(tmp_path), "MiniMaxAI/MiniMax-M3")

    assert result == {"prepared": True}
    assert calls == [(tmp_path, "MiniMaxAI/MiniMax-M3")]


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


def test_sglang_model_args_thinking_harness():
    cfg = PipelineConfig()
    cfg.model.trust_remote_code = True
    cfg.eval.backend = "sglang"
    cfg.eval.enable_thinking = True
    cfg.eval.think_end_token = "</think>"
    args = sglang_model_args(cfg, "/models/glm")
    assert "enable_thinking=True" in args
    assert "think_end_token=</think>" in args


def test_sglang_model_args_no_thinking_when_unset():
    cfg = PipelineConfig()
    cfg.model.trust_remote_code = True
    cfg.eval.backend = "sglang"
    cfg.eval.enable_thinking = None
    cfg.eval.think_end_token = None
    args = sglang_model_args(cfg, "/models/glm")
    assert "enable_thinking" not in args
    assert "think_end_token" not in args


def test_leaderboard_bbh_config_loads_without_thinking():
    cfg_path = (
        Path(__file__).resolve().parents[1]
        / "configs"
        / "eval_glm52_w4afp8_leaderboard_bbh.yaml"
    )
    cfg = load_config(cfg_path)
    assert len(cfg.eval.tasks) == 1
    assert cfg.eval.tasks[0].name == "leaderboard_bbh"
    assert cfg.eval.tasks[0].metric == "acc_norm,none"
    assert cfg.eval.enable_thinking is None
    assert cfg.eval.think_end_token is None
    assert cfg.eval.fewshot_as_multiturn is True
    args = sglang_model_args(cfg, "/models/glm")
    assert "enable_thinking" not in args
    assert "think_end_token" not in args


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


def test_evaluate_tasks_passes_exact_samples(monkeypatch, tmp_path):
    from pipeline.lmeval_runner import evaluate_tasks

    sample_path = tmp_path / "samples.json"
    sample_path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "seed": 42,
                "tasks": {"mmlu_pro": {"mmlu_pro_math": [0, 4]}},
            }
        ),
        encoding="utf-8",
    )
    cfg = PipelineConfig()
    cfg.eval.samples_manifest = str(sample_path)
    task = EvalTask(name="mmlu_pro", limit=None)
    calls = []

    fake_lm = SimpleNamespace(clean=lambda: None)
    monkeypatch.setattr("pipeline.lmeval_runner._load_lm_model", lambda *_: fake_lm)
    monkeypatch.setitem(
        sys.modules,
        "lm_eval",
        SimpleNamespace(
            simple_evaluate=lambda **kwargs: (
                calls.append(kwargs) or {"results": {"mmlu_pro": {"acc,none": 1.0}}}
            )
        ),
    )

    evaluate_tasks("/model", cfg, [task])

    assert calls[0]["samples"] == {"mmlu_pro_math": [0, 4]}
    assert "limit" not in calls[0]


def test_evaluate_tasks_runs_paired_generation_seeds_with_one_model(monkeypatch):
    from pipeline.lmeval_runner import evaluate_tasks

    cfg = PipelineConfig()
    cfg.eval.generation_seeds = [42, 1234, 4158]
    cfg.eval.gen_kwargs = {
        "temperature": 1.0,
        "top_p": 0.95,
        "do_sample": True,
    }
    tasks = [EvalTask(name="gpqa", limit=None), EvalTask(name="aime", limit=None)]
    calls = []
    completed = []
    loads = []
    fake_lm = SimpleNamespace(clean=lambda: None)

    def fake_load(*_):
        loads.append(True)
        return fake_lm

    monkeypatch.setattr("pipeline.lmeval_runner._load_lm_model", fake_load)
    monkeypatch.setitem(
        sys.modules,
        "lm_eval",
        SimpleNamespace(
            simple_evaluate=lambda **kwargs: (
                calls.append(kwargs)
                or {"results": {kwargs["tasks"][0]: {"acc,none": 1.0}}}
            )
        ),
    )

    evaluate_tasks(
        "/model",
        cfg,
        tasks,
        on_task_complete=lambda task, seed, batch: completed.append((task.name, seed)),
    )

    expected = [(task, seed) for task in ("gpqa", "aime") for seed in (42, 1234, 4158)]
    assert len(loads) == 1
    assert [
        (call["tasks"][0], call["gen_kwargs"]["seed"]) for call in calls
    ] == expected
    assert completed == expected
    assert all(call["gen_kwargs"]["do_sample"] is True for call in calls)
    assert all(call["random_seed"] == 42 for call in calls)
    assert all(call["fewshot_random_seed"] == 42 for call in calls)


def test_evaluate_tasks_skips_completed_task_seed_pairs(monkeypatch):
    from pipeline.lmeval_runner import evaluate_tasks

    cfg = PipelineConfig()
    cfg.eval.generation_seeds = [42, 1234]
    cfg.eval.gen_kwargs = {"temperature": 1.0}
    calls = []
    monkeypatch.setattr(
        "pipeline.lmeval_runner._load_lm_model",
        lambda *_: SimpleNamespace(clean=lambda: None),
    )
    monkeypatch.setitem(
        sys.modules,
        "lm_eval",
        SimpleNamespace(
            simple_evaluate=lambda **kwargs: (
                calls.append(kwargs) or {"results": {"gpqa": {"acc,none": 1.0}}}
            )
        ),
    )

    evaluate_tasks(
        "/model",
        cfg,
        [EvalTask(name="gpqa", limit=None)],
        completed_task_seeds={("gpqa", 42)},
    )

    assert [call["gen_kwargs"]["seed"] for call in calls] == [1234]


def test_evaluate_tasks_rejects_limit_with_exact_samples(monkeypatch, tmp_path):
    from pipeline.lmeval_runner import evaluate_tasks

    sample_path = tmp_path / "samples.json"
    sample_path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "seed": 42,
                "tasks": {"mmlu_pro": {"mmlu_pro_math": [0]}},
            }
        ),
        encoding="utf-8",
    )
    cfg = PipelineConfig()
    cfg.eval.samples_manifest = str(sample_path)
    monkeypatch.setattr(
        "pipeline.lmeval_runner._load_lm_model",
        lambda *_: SimpleNamespace(clean=lambda: None),
    )
    monkeypatch.setitem(
        sys.modules,
        "lm_eval",
        SimpleNamespace(simple_evaluate=lambda **_: {}),
    )

    with pytest.raises(ValueError, match="mutually exclusive"):
        evaluate_tasks("/model", cfg, [EvalTask(name="mmlu_pro", limit=10)])
