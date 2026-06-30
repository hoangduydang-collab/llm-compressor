"""Config-driven quantization pipeline (M1/M2).

Turns any HF (MoE) checkpoint into a vLLM-servable W4AFP8 / W4A8 artifact via a
single YAML config, with a serve-handoff check, an accuracy gate, and versioned
outputs.

Entry point: ``python -m pipeline.run --config <yaml> [--stage ...]``.
"""

from pipeline.config import PipelineConfig, load_config

__all__ = ["PipelineConfig", "load_config"]
