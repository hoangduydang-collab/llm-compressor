"""Serve-handoff verification.

Loads the quantized checkpoint in vLLM (offline ``LLM``), confirms it
initializes and returns a sane completion, and records the resolved
quantization method + GPU SM. On Hopper (SM90) a W4A8/W4AFP8 MoE checkpoint
should select the CUTLASS W4A8 MoE path.
"""

import json
from pathlib import Path

from pipeline.config import PipelineConfig


def _read_quant_config(ckpt: Path) -> dict:
    cfg_path = ckpt / "config.json"
    if not cfg_path.exists():
        return {}
    with cfg_path.open("r", encoding="utf-8") as fh:
        data = json.load(fh)
    return data.get("quantization_config", {})


def verify_serve(cfg: PipelineConfig, ckpt: Path) -> dict:
    """Boot vLLM on ``ckpt`` and return a verification report dict."""
    report: dict = {
        "checkpoint": str(ckpt),
        "loaded": False,
        "quantization_config": _read_quant_config(ckpt),
    }

    try:
        import torch

        if torch.cuda.is_available():
            major, minor = torch.cuda.get_device_capability(0)
            report["compute_capability"] = f"{major}.{minor}"
            report["is_hopper_sm90"] = (major, minor) == (9, 0)
    except Exception:
        pass

    from vllm import LLM, SamplingParams

    s = cfg.serve
    llm_kwargs: dict = dict(
        model=str(ckpt),
        tensor_parallel_size=s.tensor_parallel_size,
        max_model_len=s.max_model_len,
        gpu_memory_utilization=s.gpu_memory_utilization,
        trust_remote_code=cfg.model.trust_remote_code,
        enforce_eager=False,
    )
    if s.enable_expert_parallel:
        llm_kwargs["enable_expert_parallel"] = True
    if s.block_size is not None:
        llm_kwargs["block_size"] = s.block_size
    if s.kv_cache_dtype is not None:
        llm_kwargs["kv_cache_dtype"] = s.kv_cache_dtype

    llm = LLM(**llm_kwargs)
    report["loaded"] = True

    out = llm.generate(
        [s.prompt], SamplingParams(max_tokens=64, temperature=0.0)
    )
    text = out[0].outputs[0].text
    report["sample_prompt"] = s.prompt
    report["sample_output"] = text
    report["sane_output"] = bool(text and text.strip())

    print("\n========== vLLM SERVE CHECK ==========")
    print(f"checkpoint: {ckpt}")
    print(f"quant: {report['quantization_config'].get('format')}")
    print(f"prompt: {s.prompt!r}")
    print(f"output: {text!r}")
    print("======================================\n")
    return report
