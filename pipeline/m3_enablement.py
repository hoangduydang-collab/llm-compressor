"""MiniMax-M3 MoE enablement probe (Stage 3, the M1/M2 critical path).

The pipeline relies on llm-compressor "linearizing" MoE experts so calibration
can route all tokens through all experts. This works automatically when the
experts module either implements the standard transformers MoE protocol
(``use_experts_implementation``) or has a registered ``LinearExperts2D`` class.

For a new arch like M3's ``minimax_m3_vl`` this may NOT hold, which is the hard
critical path called out in the execution plan. This module probes a loaded
model and reports exactly which expert modules are unrecognized and whether a
linear-experts class can be resolved for them, so we know up front whether a
custom module must be authored under ``src/llmcompressor/modeling/``.

Usage::

    python -m pipeline.m3_enablement --config pipeline/configs/minimax_m3.yaml
"""

import argparse
import json

from pipeline.config import load_config


def probe_moe_support(model) -> dict:
    """Inspect ``model`` and report MoE linearization readiness."""
    from llmcompressor.modeling.moe.linear_experts import LinearExperts2D
    from llmcompressor.modeling.moe.linearize import get_non_linearized_moes

    report: dict = {
        "architectures": list(getattr(model.config, "architectures", []) or []),
        "model_type": getattr(model.config, "model_type", None),
        "non_linearized_experts": [],
        "resolvable": True,
        "needs_custom_module": False,
    }

    non_linearized = get_non_linearized_moes(model)
    if not non_linearized:
        # Either already linear, or experts are recognized -> ready.
        report["status"] = "ready (no unrecognized experts)"
        return report

    for name, module in non_linearized:
        cls_name = module.__class__.__name__
        resolvable = True
        try:
            LinearExperts2D.get_linear_experts_cls(module.__class__)
        except Exception:
            resolvable = False
            report["resolvable"] = False
            report["needs_custom_module"] = True
        report["non_linearized_experts"].append(
            {"name": name, "class": cls_name, "auto_resolvable": resolvable}
        )

    report["status"] = (
        "needs custom LinearExperts2D registration"
        if report["needs_custom_module"]
        else "auto-linearizable on load"
    )
    return report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="probe MoE linearization support")
    parser.add_argument("--config", required=True)
    args = parser.parse_args(argv)

    cfg = load_config(args.config)

    from transformers import AutoModelForCausalLM

    from llmcompressor.utils import load_context

    print(f"[probe] loading {cfg.model.id} (this may take a while)...")
    with load_context():
        model = AutoModelForCausalLM.from_pretrained(
            cfg.model.id,
            trust_remote_code=cfg.model.trust_remote_code,
            dtype="auto",
        )

    report = probe_moe_support(model)
    print(json.dumps(report, indent=2))

    if report.get("needs_custom_module"):
        print(
            "\n[probe] ACTION REQUIRED: author a LinearExperts2D subclass for the "
            "above expert class and register it (see "
            "src/llmcompressor/modeling/moe/granitemoe.py for a template, and "
            "docs/developer-tutorials/add-moe-support.md)."
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
