"""Quantize stage: load -> oneshot -> sanity-generate -> save compressed.

Produces a vLLM-servable ``pack-quantized`` compressed-tensors checkpoint in
``<run_dir>/checkpoint``.
"""

import json
from pathlib import Path

from pipeline.calibration import (
    CalibrationPartition,
    build_calibration_dataset_with_partition,
    calibration_partition_manifest,
)
from pipeline.config import PipelineConfig
from pipeline.distributed import DistributedContext
from pipeline.provenance import log_model_provenance
from pipeline.recipe import build_recipe, describe_recipe
from pipeline.vl_artifacts import ensure_vl_processor_artifacts
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
    from pipeline.minimax_m3_config import apply_minimax_m3_config

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


def _evidence_paths(
    run_dir: Path, dist_ctx: DistributedContext
) -> dict[str, Path]:
    return {
        "metrics": dist_ctx.rank_path(run_dir / "quant_metrics.jsonl"),
        "provenance": dist_ctx.rank_path(run_dir / "model_provenance.json"),
        "partition": dist_ctx.rank_path(run_dir / "calibration_partition.json"),
    }


def _persist_calibration_partition(
    run_dir: Path,
    dataset,
    partition: CalibrationPartition,
    dist_ctx: DistributedContext,
) -> Path:
    path = _evidence_paths(run_dir, dist_ctx)["partition"]
    manifest = calibration_partition_manifest(dataset, partition)
    manifest["distributed"] = dist_ctx.snapshot()
    path.write_text(
        json.dumps(
            manifest,
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    return path


def run_quantize(
    cfg: PipelineConfig,
    run_dir: Path,
    dist_ctx: DistributedContext | None = None,
    *,
    save_checkpoint: bool = True,
) -> Path:
    """Execute the quantize stage. Returns the checkpoint directory."""
    from llmcompressor import oneshot
    from pipeline.minimax_m3_config import (
        ensure_minimax_m3_vllm_serve_config,
        patch_minimax_m3_for_text_calibration,
        register_minimax_m3_awq_mappings,
    )

    dist_ctx = dist_ctx or DistributedContext()
    evidence_paths = _evidence_paths(run_dir, dist_ctx)

    model, tokenizer = _load_model_and_tokenizer(cfg)
    # Capture load/environment provenance BEFORE calibration: where the loaded
    # modeling code comes from (installed transformers vs trust_remote_code) and
    # whether sequential_targets match any module. A zero match count is the
    # direct cause of the sequential-trace collapse (single subgraph -> no
    # calibration -> un-smoothed weights). Written next to the checkpoint.
    log_model_provenance(
        model,
        cfg.calibration.sequential_targets,
        out_path=evidence_paths["provenance"],
    )
    if patch_minimax_m3_for_text_calibration(model):
        print("[pipeline] patched MiniMax-M3 get_placeholder_mask for text-only calibration")
        register_minimax_m3_awq_mappings()
        print("[pipeline] registered MiniMax-M3 AWQ mappings")

    ds, partition = build_calibration_dataset_with_partition(
        cfg.calibration, tokenizer
    )
    _persist_calibration_partition(run_dir, ds, partition, dist_ctx)
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
        num_calibration_samples=len(ds),
        shuffle_calibration_samples=False,
        moe_calibrate_all_experts=cfg.calibration.moe_calibrate_all_experts,
    )
    if cfg.calibration.sequential_targets:
        oneshot_kwargs["sequential_targets"] = cfg.calibration.sequential_targets
    if cfg.calibration.pipeline:
        oneshot_kwargs["pipeline"] = cfg.calibration.pipeline

    # Capture llm-compressor's internal METRIC-level logs (GPTQ error/time, etc.)
    # to a per-run JSONL alongside the checkpoint.
    metrics_path = evidence_paths["metrics"]
    with metrics.capture_quant_metrics(metrics_path):
        oneshot(**oneshot_kwargs)

    ckpt = versioning.checkpoint_dir(run_dir)
    if save_checkpoint:
        # compressed-tensors distributed saving is collective: every rank calls
        # model.save_pretrained, then only rank zero writes shared side artifacts.
        save_kwargs: dict = {"save_compressed": True}
        if cfg.quantization.scheme in _PACK_QUANTIZED_SCHEMES:
            save_kwargs["quantization_format"] = "pack-quantized"
        model.save_pretrained(str(ckpt), **save_kwargs)
        dist_ctx.barrier()

        if dist_ctx.is_source:
            tokenizer.save_pretrained(str(ckpt))

            # vLLM VL load needs image-processor configs; tokenizer.save_pretrained
            # alone does not write preprocessor_config.json.
            if cfg.model.auto_class == "AutoModelForImageTextToText":
                added = ensure_vl_processor_artifacts(
                    ckpt,
                    cfg.model.id,
                    trust_remote_code=cfg.model.trust_remote_code,
                )
                if added:
                    print(f"[pipeline] saved VL processor artifacts: {added}")

                cfg_patches = ensure_minimax_m3_vllm_serve_config(
                    ckpt, cfg.model.id
                )
                if cfg_patches:
                    print(
                        "[pipeline] patched saved config for vLLM serve: "
                        f"{cfg_patches}"
                    )

            # Preserve intended ignore patterns for downstream loaders.
            _persist_ignore_to_config(ckpt, cfg.quantization.ignore)
            versioning.write_recipe(run_dir, describe_recipe(cfg.quantization))
            print(f"[pipeline] saved checkpoint to {ckpt}")
        dist_ctx.barrier()
    else:
        # A partial-layer smoke is evidence only. The completion marker appears
        # only after every rank finishes calibration and reaches this barrier.
        dist_ctx.barrier()
        if dist_ctx.is_source:
            (run_dir / "smoke_complete.json").write_text(
                json.dumps(
                    {
                        "schema_version": 1,
                        "status": "complete",
                        "checkpoint_saved": False,
                        "distributed": dist_ctx.snapshot(),
                    },
                    indent=2,
                    sort_keys=True,
                )
                + "\n",
                encoding="utf-8",
            )
        dist_ctx.barrier()

    # Distributed generation is not part of calibration and can require a
    # different dispatch topology. Keep the existing local check only.
    if cfg.quantization.sample_generation and save_checkpoint and not dist_ctx.enabled:
        print("\n========== SAMPLE GENERATION ==========")
        try:
            print(_sample_generation(model, tokenizer, cfg.serve.prompt))
        except Exception as exc:  # generation issues should not lose the checkpoint
            print(f"[warn] sample generation failed: {exc}")
        print("=======================================\n")

    # Summarize the captured internal metrics into metadata.json.
    summary = metrics.summarize_quant_metrics(metrics_path)
    if dist_ctx.is_source:
        versioning.update_metadata(run_dir, {"quant_metrics": summary})
    print(f"[pipeline] quant metrics: {summary}")

    return ckpt
