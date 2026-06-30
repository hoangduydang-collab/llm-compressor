"""Serve-handoff verification.

Loads the quantized checkpoint in vLLM (offline ``LLM``), confirms it
initializes and returns a sane completion, and records the resolved
quantization method + GPU SM. On Hopper (SM90) a W4A8/W4AFP8 MoE checkpoint
should select the CUTLASS W4A8 MoE path.

Before booting vLLM it runs a cheap preflight that catches a known hard kernel
constraint: the CUTLASS W4A8 MoE grouped-GEMM requires each routed expert's
intermediate width to be a multiple of 256. Models that violate this (e.g.
Qwen1.5-MoE-A2.7B, moe_intermediate_size=1408) cannot serve in W4A8/W4AFP8 and
are failed fast with an actionable message instead of after a full model load.
"""

import json
from pathlib import Path

from pipeline.config import PipelineConfig

# CUTLASS W4A8 MoE grouped-GEMM kernel constraint.
W4A8_MOE_INTERMEDIATE_MULTIPLE = 256


def _read_model_config(ckpt: Path) -> dict:
    cfg_path = ckpt / "config.json"
    if not cfg_path.exists():
        return {}
    with cfg_path.open("r", encoding="utf-8") as fh:
        return json.load(fh)


def _read_quant_config(ckpt: Path) -> dict:
    return _read_model_config(ckpt).get("quantization_config", {})


def _is_w4a8_moe_scheme(quant_config: dict) -> bool:
    """True if any config group is INT4 weights + 8-bit activations (W4A8/W4AFP8)."""
    groups = quant_config.get("config_groups", {})
    for group in groups.values():
        weights = group.get("weights") or {}
        acts = group.get("input_activations") or {}
        if weights.get("num_bits") == 4 and acts.get("num_bits") == 8:
            return True
    return False


_EXPERT_COUNT_KEYS = ("num_local_experts", "n_routed_experts", "num_experts")


def _moe_intermediate_sizes(model_config: dict) -> list[int]:
    """Collect routed-expert intermediate sizes from the (possibly nested) config.

    Different MoE archs name this differently:
      - Qwen/DeepSeek: ``moe_intermediate_size``
      - MiniMax-M3: ``intermediate_size`` inside ``text_config`` (alongside an
        expert-count key like ``num_local_experts``)
    We only treat ``intermediate_size`` as the routed-expert width when the
    (sub)config is actually an MoE (has an expert-count key), to avoid picking up
    a dense model's FFN size.
    """
    sizes: list[int] = []
    candidates = [model_config]
    for nested_key in ("text_config", "language_config", "llm_config"):
        nested = model_config.get(nested_key)
        if isinstance(nested, dict):
            candidates.append(nested)
    for cfg in candidates:
        if isinstance(cfg.get("moe_intermediate_size"), int):
            sizes.append(cfg["moe_intermediate_size"])
            continue
        is_moe = any(k in cfg for k in _EXPERT_COUNT_KEYS)
        if is_moe and isinstance(cfg.get("intermediate_size"), int):
            sizes.append(cfg["intermediate_size"])
    return sizes


def preflight_serve_check(cfg: PipelineConfig, ckpt: Path) -> tuple[bool, dict]:
    """Cheap static checks before booting vLLM.

    Returns ``(ok, info)``. ``ok=False`` means do not attempt to serve.
    """
    info: dict = {"preflight_ok": True, "preflight_reasons": []}
    model_config = _read_model_config(ckpt)
    quant_config = model_config.get("quantization_config", {})

    if not _is_w4a8_moe_scheme(quant_config):
        # Other schemes (W4A16, FP8, ...) don't use the W4A8 MoE kernel here.
        return True, info

    sizes = _moe_intermediate_sizes(model_config)
    if not sizes:
        info["preflight_reasons"].append(
            "could not determine moe_intermediate_size; skipping kernel geometry check"
        )
        return True, info

    s = cfg.serve
    mult = W4A8_MOE_INTERMEDIATE_MULTIPLE
    for size in sizes:
        # Fundamental requirement (holds regardless of parallelism layout).
        if size % mult != 0:
            info["preflight_ok"] = False
            info["preflight_reasons"].append(
                f"moe_intermediate_size={size} is not divisible by {mult}: the CUTLASS "
                f"W4A8 MoE kernel cannot serve this model in W4A8/W4AFP8. Options: pad "
                f"the expert intermediate dim up to a multiple of {mult}, or quantize "
                f"this model with a different scheme (e.g. W4A16 / FP8)."
            )
            continue
        # Under pure tensor parallelism the intermediate dim is sharded by TP;
        # expert parallelism keeps each expert's full width.
        if not s.enable_expert_parallel and s.tensor_parallel_size > 1:
            tp = s.tensor_parallel_size
            if size % tp != 0 or (size // tp) % mult != 0:
                info["preflight_ok"] = False
                info["preflight_reasons"].append(
                    f"with tensor_parallel_size={tp} and expert parallelism OFF, the "
                    f"per-partition expert width {size}//{tp}={size // tp if size % tp == 0 else 'n/a'} "
                    f"is not a multiple of {mult}. Enable expert parallelism "
                    f"(serve.enable_expert_parallel=true) so each expert keeps its full "
                    f"width {size}, or choose a TP that divides {size} into a multiple of {mult}."
                )

    return info["preflight_ok"], info


def verify_serve(cfg: PipelineConfig, ckpt: Path) -> dict:
    """Boot vLLM on ``ckpt`` and return a verification report dict."""
    report: dict = {
        "checkpoint": str(ckpt),
        "loaded": False,
        "ok": False,
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

    # Fail fast on known kernel-geometry incompatibilities before loading vLLM.
    preflight_ok, preflight_info = preflight_serve_check(cfg, ckpt)
    report.update(preflight_info)
    if not preflight_ok:
        print("\n========== SERVE PREFLIGHT FAILED ==========")
        for reason in preflight_info["preflight_reasons"]:
            print(f"- {reason}")
        print("(skipping vLLM load)")
        print("============================================\n")
        return report

    from pipeline._env import ensure_writable_caches

    changed = ensure_writable_caches()
    if changed:
        print(f"[pipeline] redirected caches: {changed}")

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
    report["ok"] = report["loaded"] and report["sane_output"]

    print("\n========== vLLM SERVE CHECK ==========")
    print(f"checkpoint: {ckpt}")
    print(f"quant: {report['quantization_config'].get('format')}")
    print(f"prompt: {s.prompt!r}")
    print(f"output: {text!r}")
    print("======================================\n")
    return report
