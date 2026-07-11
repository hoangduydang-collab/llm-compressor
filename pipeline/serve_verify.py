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
import os
from pathlib import Path

from pipeline.config import PipelineConfig

# FlashInfer routes MiniMax-M3's Gemma RMSNorm to a CuTe-DSL kernel
# (``rmsnorm_cute``) that fails to JIT-compile against nvidia-cutlass-dsl 4.5.2
# on Hopper (nanobind "Expected an MLIR object (got OpResultList)"), aborting all
# vLLM workers during profile_run. FlashInfer exposes a documented CUDA-JIT
# fallback via this env var; it is read at ``flashinfer.norm`` *import* time, so
# it must be set before vLLM imports FlashInfer. ``setdefault`` keeps any explicit
# override. See BUGS_AND_FIXES.md "FlashInfer gemma_rmsnorm CuTe-DSL".
os.environ.setdefault("FLASHINFER_USE_CUDA_NORM", "1")

# CUTLASS W4A8 MoE grouped-GEMM kernel constraint.
W4A8_MOE_INTERMEDIATE_MULTIPLE = 256


def _is_minimax_m3_checkpoint(ckpt: Path) -> bool:
    """True when ``ckpt/config.json`` looks like MiniMax-M3 / MiniMax-M3-VL."""
    cfg = _read_model_config(ckpt)
    if not cfg:
        return False
    model_type = str(cfg.get("model_type") or "").lower()
    archs = " ".join(cfg.get("architectures") or [])
    blob = f"{model_type} {archs}".lower()
    return "minimax" in blob


def apply_minimax_m3_serve_env(ckpt: Path) -> list[str]:
    """Apply MiniMax-M3-only serve env defaults before vLLM import.

    Currently sets ``VLLM_DISABLE_SHARED_EXPERTS_STREAM=1`` when unset. This is
    the production workaround for the HTTP async CUDA-graph IMA (h125 matrix:
    3/3 ready+chat with stream disabled). Non-M3 checkpoints are left alone so
    other models keep standard vLLM defaults. Explicit caller exports win.
    """
    applied: list[str] = []
    if not _is_minimax_m3_checkpoint(ckpt):
        return applied
    key = "VLLM_DISABLE_SHARED_EXPERTS_STREAM"
    if key not in os.environ:
        os.environ[key] = "1"
        applied.append(f"{key}=1")
    return applied


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


def _install_minimax_m3_site_diagnostics(
    ckpt: Path,
    *,
    diagnostic_installer=None,
    w4a8_patch_installer=None,
) -> dict:
    """Install M3 diagnostics for every scheme and execution patches for W4A8."""

    if not _is_minimax_m3_checkpoint(ckpt):
        return {"diagnostics": "skipped (not MiniMax-M3)", "w4a8_patches": False}
    if diagnostic_installer is None or w4a8_patch_installer is None:
        from pipeline.slurm.patch_vllm_m3_serve import (
            ensure_m3_quality_diagnostics,
            ensure_vllm_m3_patches,
        )

        diagnostic_installer = (
            diagnostic_installer or ensure_m3_quality_diagnostics
        )
        w4a8_patch_installer = (
            w4a8_patch_installer or ensure_vllm_m3_patches
        )
    diagnostic_status = diagnostic_installer()
    w4a8 = _is_w4a8_moe_scheme(_read_quant_config(ckpt))
    if w4a8:
        w4a8_patch_installer()
    return {"diagnostics": diagnostic_status, "w4a8_patches": w4a8}


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


def _read_weight_map(ckpt: Path) -> dict:
    """Return the safetensors index ``weight_map`` (tensor name -> shard) if present.

    Reading the index avoids loading any tensor data — it is a cheap way to see
    which parameter *names* the checkpoint actually stores.
    """
    idx = ckpt / "model.safetensors.index.json"
    if not idx.exists():
        return {}
    try:
        with idx.open("r", encoding="utf-8") as fh:
            return json.load(fh).get("weight_map", {}) or {}
    except Exception:
        return {}


def _layer_indices(keys, needle: str) -> set[int]:
    """Distinct ``layers.N.`` indices among ``keys`` that contain ``needle``."""
    import re as _re

    out: set[int] = set()
    for k in keys:
        if needle in k:
            m = _re.search(r"\.layers\.(\d+)\.", k)
            if m:
                out.add(int(m.group(1)))
    return out


def audit_m3_checkpoint(ckpt: Path, model_config: dict) -> dict:
    """Static (no model load) audit of the serve-side load/wiring suspects for the
    MiniMax-M3 ``"arring"`` garbage failure (BUGS_AND_FIXES.md
    "full-calib AWQ garbage output").

    Reputable community checkpoints on the *same* pipeline shape (transformers
    MiniMax-M3-VL export -> official/toncao vLLM) trace the garbage to serve-side
    load/wiring, not quant accuracy (see aquaman164 ``m3_official_loader.py``):

      * shared expert dropped/zero-loaded in every MoE layer — key-name mismatch
        (checkpoint ``mlp.shared_experts.*`` vs vLLM ``block_sparse_moe.shared_experts.*``)
        and/or ``n_shared_experts`` nested under ``text_config`` so the module is
        never built;
      * ``lm_head`` not loaded (``language_model.lm_head.weight`` vs top-level;
        ``tie_word_embeddings=False``) -> random logits.

    This reports what the checkpoint actually contains so those triggers can be
    confirmed/ruled out before any (expensive) re-quant. Cheap: reads config.json
    and the safetensors index only.
    """
    wm = _read_weight_map(ckpt)
    keys = list(wm.keys())
    text_cfg = model_config.get("text_config") or {}

    lm_head_keys = [k for k in keys if "lm_head" in k]
    shared_keys = [k for k in keys if "shared_experts" in k]
    mlp_shared = [k for k in shared_keys if ".mlp.shared_experts." in k]
    bsm_shared = [k for k in shared_keys if "block_sparse_moe.shared_experts." in k]
    routed_sample = next(
        (k for k in keys if ".mlp.experts." in k or "block_sparse_moe.experts." in k),
        None,
    )

    audit = {
        "tie_word_embeddings": model_config.get(
            "tie_word_embeddings", text_cfg.get("tie_word_embeddings")
        ),
        "n_shared_experts_top": model_config.get("n_shared_experts"),
        "n_shared_experts_text_config": text_cfg.get("n_shared_experts"),
        "lm_head_keys": lm_head_keys,
        "shared_expert_tensor_count": len(shared_keys),
        "shared_expert_key_style": (
            "mlp.shared_experts"
            if mlp_shared and not bsm_shared
            else "block_sparse_moe.shared_experts"
            if bsm_shared and not mlp_shared
            else "mixed/none"
        ),
        "shared_expert_sample_key": shared_keys[0] if shared_keys else None,
        "shared_expert_layers_covered": sorted(_layer_indices(keys, "shared_experts")),
        "routed_expert_sample_key": routed_sample,
        "index_present": bool(wm),
    }

    # Heuristic flags (the decisive proof is the runtime M3_MOE_PROBE, but these
    # cheap static signals point at the likely trigger).
    warnings: list[str] = []
    if not wm:
        warnings.append(
            "no model.safetensors.index.json — cannot audit key names statically"
        )
    if not lm_head_keys:
        warnings.append(
            "no 'lm_head' tensor in the checkpoint index - if tie_word_embeddings "
            "is False this means random logits (garbage). Confirm lm_head is saved."
        )
    if audit["shared_expert_key_style"] == "mlp.shared_experts":
        warnings.append(
            "shared experts stored as 'mlp.shared_experts.*' (transformers-VL "
            "naming). If vLLM's M3 model looks them up as "
            "'block_sparse_moe.shared_experts.*', the lookup misses -> shared "
            "expert loads as ZERO in every MoE layer -> garbage (aquaman164 "
            "m3_official_loader.py). Confirm at runtime with M3_MOE_PROBE=1."
        )
    if (
        audit["n_shared_experts_top"] in (None, 0)
        and audit["n_shared_experts_text_config"] not in (None, 0)
    ):
        warnings.append(
            "n_shared_experts is set under text_config but NOT top-level. If the "
            "serve arch reads top-level config, the shared-expert module is never "
            "built -> dropped in every MoE layer -> garbage. Force n_shared_experts "
            "at the level the vLLM M3 arch reads."
        )
    audit["warnings"] = warnings

    print("\n========== M3 CHECKPOINT QUALITY AUDIT (static) ==========")
    print(f"tie_word_embeddings: {audit['tie_word_embeddings']}")
    print(
        f"n_shared_experts: top={audit['n_shared_experts_top']} "
        f"text_config={audit['n_shared_experts_text_config']}"
    )
    print(f"lm_head keys: {lm_head_keys or 'NONE'}")
    print(
        f"shared experts: {len(shared_keys)} tensors, style="
        f"{audit['shared_expert_key_style']}, layers="
        f"{audit['shared_expert_layers_covered'][:3]}..."
        if shared_keys
        else "shared experts: NONE found in index"
    )
    print(f"shared sample key: {audit['shared_expert_sample_key']}")
    print(f"routed sample key: {routed_sample}")
    for w in warnings:
        print(f"  ! {w}")
    if not warnings:
        print("  (no static red flags; run M3_MOE_PROBE=1 to confirm at runtime)")
    print("==========================================================\n")
    return audit


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


def _print_serve_failure_hints(ckpt: Path, cfg: PipelineConfig) -> None:
    """Actionable hints when vLLM worker init fails (error is often in worker logs)."""
    import vllm

    print(f"vllm version: {getattr(vllm, '__version__', 'unknown')}")
    print(
        "The destroy_process_group() warnings are benign; the real error is usually "
        "a few hundred lines earlier from a worker rank."
    )
    print("On the cluster, grep the serve log for the worker root cause:")
    print("  grep -E 'Worker|ERROR|Error|OOM|out of memory|KeyError|size mismatch' \\")
    print("    serves/m3-awq-w4afp8/run.log | tail -40")
    print()
    print("Common causes for MiniMax-M3 W4AFP8 checkpoints:")
    print(
        "  0) CUDA graph capture IMA — vLLM **site-packages** must have all 4 patches "
        "(Worker_TP* are spawned; runtime hooks do not apply):\n"
        "       python pipeline/slurm/patch_vllm_m3_serve.py --check\n"
        "       grep -r 'llmc M3' \"$(python -c 'import vllm, pathlib; "
        "print(pathlib.Path(vllm.__file__).parent)')\" | head\n"
        "     See BUGS_AND_FIXES.md. Escape: ENFORCE_EAGER=1."
    )
    print(
        '  1) W4A8 MoE kernel rejects SWIGLUOAI_UNINTERLEAVE ("kernel does not '
        'support MoEActivation.SWIGLUOAI_UNINTERLEAVE"). No vLLM build wires this '
        "up; serve_verify patches it in-process via pipeline/vllm_m3_patches.py."
    )
    print(
        '  2) text_config.hidden_act is "silu" but vLLM requires "swigluoai" '
        "(transformers normalizes on load; serve_verify auto-restores from model.id)"
    )
    print(
        "  3) Missing vision_config.img_token_compression_config in saved config.json "
        "(quantize coercion hoists it; serve_verify auto-restores from model.id)"
    )
    print(
        "  4) Stock vLLM fuses quantized q/k/v with bf16 MSA indexer in one GEMM; our "
        "checkpoint keeps indexer bf16 (see quantization_config.ignore). Install:"
    )
    print("       bash pipeline/slurm/install_vllm_m3_serve.sh")
    print(
        "  5) GPU OOM during weight load / KV init on 8xH100 — retry with a smaller "
        "smoke config:"
    )
    print(
        "       MAX_MODEL_LEN=2048 GPU_UTIL=0.85 bash pipeline/slurm/run_serve_minimax_m3_detached.sh"
    )
    print(
        "  6) Per-expert linearized MoE layout (block_sparse_moe.experts.N.gate_proj) — "
        "requires compressed-tensors pack-quantized support in vLLM (same patch as 3)."
    )
    if cfg.model.auto_class == "AutoModelForImageTextToText":
        print(
            f"  7) VL processor files — ensure preprocessor_config.json exists in {ckpt} "
            "(serve_verify auto-copies from model.id)."
        )


def _run_generation_smoke(
    llm,
    *,
    is_m3: bool,
    configured_prompt: str,
    sampling_params_cls,
) -> dict:
    """Generate smoke outputs and assess M3 quality separately from readiness."""

    if is_m3:
        from pipeline.m3_quality_evidence import (
            M3_QUALITY_CASES,
            assess_quality_outputs,
        )

        prompts = [case.prompt for case in M3_QUALITY_CASES]
    else:
        prompts = [configured_prompt]
    sampling_params = sampling_params_cls(max_tokens=64, temperature=0.0)
    if is_m3:
        # Keep requests independent so one sequence cannot affect scheduling,
        # padding, or termination of the other reference-quality case.
        outputs = []
        for prompt in prompts:
            outputs.extend(llm.generate([prompt], sampling_params))
    else:
        outputs = llm.generate(prompts, sampling_params)
    texts = [item.outputs[0].text for item in outputs]
    generation_completed = len(texts) == len(prompts) and all(
        bool(text and text.strip()) for text in texts
    )
    result = {
        "sample_prompt": prompts[0],
        "sample_output": texts[0] if texts else "",
        "generation_completed": generation_completed,
        "sane_output": generation_completed,
        "quality_ok": None,
    }
    if is_m3:
        result.update(assess_quality_outputs(texts))
        for quality_case, request_output in zip(result["quality_cases"], outputs):
            completion = request_output.outputs[0]
            quality_case.update(
                {
                    "token_ids": list(getattr(completion, "token_ids", []) or []),
                    "finish_reason": getattr(completion, "finish_reason", None),
                    "stop_reason": getattr(completion, "stop_reason", None),
                }
            )
    return result


def verify_serve(cfg: PipelineConfig, ckpt: Path) -> dict:
    """Boot vLLM on ``ckpt`` and return a verification report dict."""
    report: dict = {
        "checkpoint": str(ckpt),
        "loaded": False,
        "ok": False,
        "quantization_config": _read_quant_config(ckpt),
        "disable_custom_all_reduce": cfg.serve.disable_custom_all_reduce,
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

    # MiniMax-M3-only env defaults must land before ``from vllm import ...`` so
    # Worker_TP* children inherit them. Non-M3 checkpoints are untouched.
    m3_env = apply_minimax_m3_serve_env(ckpt)
    if m3_env:
        print(f"[pipeline] MiniMax-M3 serve env defaults: {m3_env}")
    report["disable_shared_experts_stream"] = os.environ.get(
        "VLLM_DISABLE_SHARED_EXPERTS_STREAM"
    )

    if _is_minimax_m3_checkpoint(ckpt):
        try:
            site_status = _install_minimax_m3_site_diagnostics(ckpt)
            report["m3_site_diagnostics"] = site_status
            print(
                "[pipeline] M3 quality diagnostics (site-packages): "
                f"{site_status['diagnostics']}"
            )
            if site_status["w4a8_patches"]:
                print(
                    "[pipeline] verified vLLM M3 W4A8 serve patches in "
                    "site-packages (4/4)"
                )
        except Exception as exc:
            enabled = any(
                os.environ.get(name) == "1"
                for name in (
                    "M3_LOAD_AUDIT",
                    "M3_MOE_PROBE",
                    "M3_PARAM_FINGERPRINT",
                )
            )
            if enabled:
                raise
            print(f"[pipeline] M3 dormant diagnostics install skipped: {exc!r}")

    # VL checkpoints need preprocessor_config.json for vLLM multimodal init even
    # when running a text-only smoke prompt. Older quant runs may lack these files.
    if cfg.model.auto_class == "AutoModelForImageTextToText":
        from pipeline.minimax_m3_config import ensure_minimax_m3_vllm_serve_config
        from pipeline.vl_artifacts import ensure_vl_processor_artifacts

        cfg_patches = ensure_minimax_m3_vllm_serve_config(ckpt, cfg.model.id)
        if cfg_patches:
            print(f"[pipeline] patched checkpoint config for vLLM: {cfg_patches}")

        # Static serve-side load/wiring audit for the "arring" garbage failure
        # (shared-expert / lm_head naming; BUGS_AND_FIXES.md "full-calib AWQ garbage").
        report["quality_audit"] = audit_m3_checkpoint(ckpt, _read_model_config(ckpt))

        added = ensure_vl_processor_artifacts(
            ckpt, cfg.model.id, trust_remote_code=cfg.model.trust_remote_code
        )
        if added:
            print(
                f"[pipeline] copied VL processor artifacts from {cfg.model.id!r}: {added}"
            )

        # W4A8 MoE + SwiGLU-OAI (uninterleaved) is not wired up in stock/NVIDIA/
        # toncao vLLM builds. Patch the in-process vLLM before LLM() so the CUTLASS
        # W4A8 MoE kernel accepts M3's SWIGLUOAI_UNINTERLEAVE activation. No-op if
        # the checkpoint is not a W4A8 MoE scheme or the build lacks W4A8 MoE.
        if _is_w4a8_moe_scheme(_read_quant_config(ckpt)):
            from pipeline.vllm_m3_patches import (
                patch_vllm_w4a8_swigluoai_uninterleave,
                read_swiglu_params,
            )

            limit, alpha, beta = read_swiglu_params(ckpt, cfg.model.id)
            moe_patches = patch_vllm_w4a8_swigluoai_uninterleave(limit, alpha, beta)
            if moe_patches:
                print(
                    f"[pipeline] runtime vLLM W4A8 shim (main proc only): {moe_patches}"
                )

        if not cfg.serve.enforce_eager:
            print(
                "[pipeline] CUDA graphs ON — cudagraph patches 3–4 must be in "
                "site-packages (ensure_vllm_m3_patches above)"
            )

    from vllm import LLM, SamplingParams

    s = cfg.serve
    llm_kwargs: dict = dict(
        model=str(ckpt),
        tensor_parallel_size=s.tensor_parallel_size,
        max_model_len=s.max_model_len,
        gpu_memory_utilization=s.gpu_memory_utilization,
        trust_remote_code=cfg.model.trust_remote_code,
        enforce_eager=s.enforce_eager,
        disable_custom_all_reduce=s.disable_custom_all_reduce,
    )
    if s.enable_expert_parallel:
        llm_kwargs["enable_expert_parallel"] = True
    if s.block_size is not None:
        llm_kwargs["block_size"] = s.block_size
    if s.kv_cache_dtype is not None:
        llm_kwargs["kv_cache_dtype"] = s.kv_cache_dtype

    try:
        llm = LLM(**llm_kwargs)
    except Exception as exc:
        report["load_error"] = repr(exc)
        print("\n========== vLLM LOAD FAILED ==========")
        print(repr(exc))
        _print_serve_failure_hints(ckpt, cfg)
        print("======================================\n")
        return report
    report["loaded"] = True

    generation = _run_generation_smoke(
        llm,
        is_m3=_is_minimax_m3_checkpoint(ckpt),
        configured_prompt=s.prompt,
        sampling_params_cls=SamplingParams,
    )
    report.update(generation)
    report["ok"] = report["loaded"] and report["generation_completed"]

    print("\n========== vLLM SERVE CHECK ==========")
    print(f"checkpoint: {ckpt}")
    print(f"quant: {report['quantization_config'].get('format')}")
    if report.get("quality_cases"):
        for case in report["quality_cases"]:
            print(
                f"quality[{case['case_id']}]: passed={case['passed']} "
                f"prompt={case['prompt']!r} output={case['text']!r}"
            )
        print(f"quality_ok: {report['quality_ok']}")
    else:
        print(f"prompt: {report['sample_prompt']!r}")
        print(f"output: {report['sample_output']!r}")
    print("======================================\n")
    return report
