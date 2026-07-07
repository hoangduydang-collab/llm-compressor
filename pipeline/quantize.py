"""Quantize stage: load -> oneshot -> sanity-generate -> save compressed.

Produces a vLLM-servable ``pack-quantized`` compressed-tensors checkpoint in
``<run_dir>/checkpoint``.
"""

import json
from pathlib import Path

from pipeline.calibration import build_calibration_dataset
from pipeline.config import PipelineConfig
from pipeline.minimax_m3_config import apply_minimax_m3_config, patch_minimax_m3_for_text_calibration
from pipeline.recipe import build_recipe, describe_recipe
from pipeline import metrics, versioning


# Schemes whose weights are INT-packed and need explicit pack-quantized on save
# for vLLM to pick the right loader/kernel.
_PACK_QUANTIZED_SCHEMES = {"W4AFP8", "W4A8", "W4A16", "W4A16_ASYM"}


def _log_backbone_dtype(model) -> None:
    """Log resolved backbone dtype so fp32 linearize regressions are visible in logs."""
    text_cfg = getattr(model.config, "text_config", None)
    text_dtype = getattr(text_cfg, "dtype", None) if text_cfg is not None else None
    sample_param_dtype = None
    for name, module in model.named_modules():
        if ".mlp.experts." in name and name.endswith(".down_proj"):
            param = getattr(module, "weight", None)
            if param is not None:
                sample_param_dtype = param.dtype
                break
    print(
        "[pipeline] backbone dtype: "
        f"text_config.dtype={text_dtype} "
        f"sample_expert_weight.dtype={sample_param_dtype}"
    )


def _load_model_and_tokenizer(cfg: PipelineConfig):
    import transformers
    from transformers import AutoTokenizer
    from llmcompressor.utils import load_context

    m = cfg.model
    model_cls = getattr(transformers, m.auto_class)
    from_pretrained_kwargs: dict = {"trust_remote_code": m.trust_remote_code}
    if m.dtype and m.dtype != "auto":
        from_pretrained_kwargs["dtype"] = m.dtype
    if m.device_map is not None:
        from_pretrained_kwargs["device_map"] = m.device_map
    if m.offload_folder is not None:
        from_pretrained_kwargs["offload_folder"] = m.offload_folder
    if m.max_memory is not None:
        # YAML may give strings like "500e9"; coerce to float.
        from_pretrained_kwargs["max_memory"] = {
            k: float(v) for k, v in m.max_memory.items()
        }

    from_pretrained_kwargs = apply_minimax_m3_config(
        m.id, from_pretrained_kwargs, trust_remote_code=m.trust_remote_code
    )

    # load_context() patches from_pretrained so fused MoE experts load in a
    # linearized, quantizable layout (and handles offloaded loading).
    with load_context(model_cls):
        model = model_cls.from_pretrained(m.id, **from_pretrained_kwargs)
    _log_backbone_dtype(model)
    tokenizer = AutoTokenizer.from_pretrained(
        m.id, trust_remote_code=m.trust_remote_code
    )
    return model, tokenizer


def _persist_ignore_to_config(ckpt: Path, ignore: list[str]) -> None:
    """Ensure the recipe's ignore patterns survive into the saved config.

    llm-compressor prunes ignore patterns that didn't match a *quantized* module
    from the serialized ``quantization_config.ignore``. That silently drops
    entries for modules it (correctly) left unquantized -- e.g. the MoE router
    ``mlp.gate`` (and, for VL MoE, the vision tower / MSA indexer). Downstream
    loaders (vLLM) then treat those Linears as quantized and either fail to load
    or, worse, mis-load them -> broken routing -> garbage output. We re-add the
    intended ignore patterns so the on-disk config reflects what was actually
    skipped.
    """
    cfg_path = ckpt / "config.json"
    if not cfg_path.exists():
        return
    with cfg_path.open("r", encoding="utf-8") as fh:
        data = json.load(fh)
    qc = data.get("quantization_config")
    if not qc:
        return
    saved = list(qc.get("ignore", []))
    added = [p for p in ignore if p not in saved]
    if added:
        qc["ignore"] = saved + added
        with cfg_path.open("w", encoding="utf-8") as fh:
            json.dump(data, fh, indent=2)
        print(f"[pipeline] persisted ignore patterns to config: {added}")


def _sample_generation(model, tokenizer, prompt: str) -> str:
    from compressed_tensors.offload import dispatch_model

    dispatch_model(model)
    sample = tokenizer(prompt, return_tensors="pt")
    sample = {k: v.to(model.device) for k, v in sample.items()}
    output = model.generate(**sample, max_new_tokens=64)
    return tokenizer.decode(output[0])


def run_quantize(cfg: PipelineConfig, run_dir: Path) -> Path:
    """Execute the quantize stage. Returns the checkpoint directory."""
    from llmcompressor import oneshot

    model, tokenizer = _load_model_and_tokenizer(cfg)
    if patch_minimax_m3_for_text_calibration(model):
        print("[pipeline] patched MiniMax-M3 get_placeholder_mask for text-only calibration")

    ds = build_calibration_dataset(cfg.calibration, tokenizer)
    recipe = build_recipe(cfg.quantization)

    oneshot_kwargs: dict = dict(
        model=model,
        # Text-only calibration: pass the loaded tokenizer so oneshot does not
        # AutoProcessor.from_pretrained (M3 needs trust_remote_code for that).
        processor=tokenizer,
        trust_remote_code_model=cfg.model.trust_remote_code,
        dataset=ds,
        recipe=recipe,
        max_seq_length=cfg.calibration.max_seq_length,
        num_calibration_samples=cfg.calibration.num_samples,
        moe_calibrate_all_experts=cfg.calibration.moe_calibrate_all_experts,
    )
    if cfg.calibration.sequential_targets:
        oneshot_kwargs["sequential_targets"] = cfg.calibration.sequential_targets
    if cfg.calibration.pipeline:
        oneshot_kwargs["pipeline"] = cfg.calibration.pipeline

    # Capture llm-compressor's internal METRIC-level logs (GPTQ error/time, etc.)
    # to a per-run JSONL alongside the checkpoint.
    metrics_path = run_dir / "quant_metrics.jsonl"
    with metrics.capture_quant_metrics(metrics_path):
        oneshot(**oneshot_kwargs)

    # Sanity check: a quantized model should still produce coherent text.
    print("\n========== SAMPLE GENERATION ==========")
    try:
        print(_sample_generation(model, tokenizer, cfg.serve.prompt))
    except Exception as exc:  # generation issues should not lose the checkpoint
        print(f"[warn] sample generation failed: {exc}")
    print("=======================================\n")

    ckpt = versioning.checkpoint_dir(run_dir)
    save_kwargs: dict = {"save_compressed": True}
    if cfg.quantization.scheme in _PACK_QUANTIZED_SCHEMES:
        save_kwargs["quantization_format"] = "pack-quantized"
    model.save_pretrained(str(ckpt), **save_kwargs)
    tokenizer.save_pretrained(str(ckpt))

    # Re-add intended ignore patterns that llm-compressor pruned from the saved
    # config (e.g. the MoE router gate), so loaders treat them as unquantized.
    _persist_ignore_to_config(ckpt, cfg.quantization.ignore)

    versioning.write_recipe(run_dir, describe_recipe(cfg.quantization))

    # Summarize the captured internal metrics into metadata.json.
    summary = metrics.summarize_quant_metrics(metrics_path)
    versioning.update_metadata(run_dir, {"quant_metrics": summary})
    print(f"[pipeline] quant metrics: {summary}")

    return ckpt
