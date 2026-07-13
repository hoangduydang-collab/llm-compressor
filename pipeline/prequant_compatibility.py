"""Static original-model compatibility gate for llm-compressor recipes.

The command builds only a meta model, invokes planner-only checks, writes a JSON
report, and exits before calibration data, model weights, GPUs, or forward hooks are
needed.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from pipeline.config import load_config
from pipeline.minimax_m3_config import (
    load_minimax_m3_vl_config,
    register_minimax_m3_awq_mappings,
)
from pipeline.recipe import build_recipe


def _build_meta_model(cfg):
    import transformers
    from accelerate import init_empty_weights

    from llmcompressor.modeling.moe.linearize import linearize_moe

    model_cfg = cfg.model
    if "minimax-m3" in model_cfg.id.lower():
        config = load_minimax_m3_vl_config(
            model_cfg.id, trust_remote_code=model_cfg.trust_remote_code
        )
        register_minimax_m3_awq_mappings()
    else:
        config = transformers.AutoConfig.from_pretrained(
            model_cfg.id, trust_remote_code=model_cfg.trust_remote_code
        )

    model_cls = getattr(transformers, model_cfg.auto_class)
    with init_empty_weights():
        model = model_cls.from_config(
            config, trust_remote_code=model_cfg.trust_remote_code
        )
        # Production loading exposes fused MoE weights as quantizable per-expert
        # Linears. Mirror that representation before target and AWQ resolution.
        linearize_moe(model)
    return model


def analyze_pipeline_config(config_path: str | Path, model_id: str | None = None):
    from llmcompressor.preflight.quantization import (
        analyze_quantization_compatibility,
    )

    cfg = load_config(config_path)
    if model_id is not None:
        cfg.model.id = model_id
    model = _build_meta_model(cfg)
    recipe = build_recipe(cfg.quantization)
    return analyze_quantization_compatibility(model, recipe)


def write_report(report, output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(f"{output}.tmp")
    temporary.write_text(
        json.dumps(report.to_dict(), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(output)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="check original model compatibility before quantization"
    )
    parser.add_argument("--config", required=True)
    parser.add_argument("--model-id")
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args(argv)

    report = analyze_pipeline_config(args.config, model_id=args.model_id)
    write_report(report, args.output)
    verdict = "PASS" if report.compatible else "FAIL"
    print(
        f"[prequant] {verdict}: quantized_modules="
        f"{report.quantized_module_count} awq_mappings={report.awq_mapping_count} "
        f"failures={len(report.failures)} warnings={len(report.warnings)} "
        f"report={args.output}"
    )
    return 0 if report.compatible else 2


if __name__ == "__main__":
    raise SystemExit(main())
