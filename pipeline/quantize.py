"""Quantize stage: load -> oneshot -> sanity-generate -> save compressed.

Produces a vLLM-servable ``pack-quantized`` compressed-tensors checkpoint in
``<run_dir>/checkpoint``.
"""

from pathlib import Path

from pipeline.calibration import build_calibration_dataset
from pipeline.config import PipelineConfig
from pipeline.recipe import build_recipe, describe_recipe
from pipeline import metrics, versioning


# Schemes whose weights are INT-packed and need explicit pack-quantized on save
# for vLLM to pick the right loader/kernel.
_PACK_QUANTIZED_SCHEMES = {"W4AFP8", "W4A8", "W4A16", "W4A16_ASYM"}


def _load_model_and_tokenizer(cfg: PipelineConfig):
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from llmcompressor.utils import load_context

    m = cfg.model
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

    # load_context() patches from_pretrained so fused MoE experts load in a
    # linearized, quantizable layout (and handles offloaded loading).
    with load_context():
        model = AutoModelForCausalLM.from_pretrained(m.id, **from_pretrained_kwargs)
    tokenizer = AutoTokenizer.from_pretrained(
        m.id, trust_remote_code=m.trust_remote_code
    )
    return model, tokenizer


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

    ds = build_calibration_dataset(cfg.calibration, tokenizer)
    recipe = build_recipe(cfg.quantization)

    oneshot_kwargs: dict = dict(
        model=model,
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

    versioning.write_recipe(run_dir, describe_recipe(cfg.quantization))

    # Summarize the captured internal metrics into metadata.json.
    summary = metrics.summarize_quant_metrics(metrics_path)
    versioning.update_metadata(run_dir, {"quant_metrics": summary})
    print(f"[pipeline] quant metrics: {summary}")

    return ckpt
