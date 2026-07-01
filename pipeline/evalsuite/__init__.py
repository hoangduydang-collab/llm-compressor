"""Evaluation suite: static lm-eval, agentic tau2, and quant-vs-original comparison."""

from pipeline.evalsuite.compare import compare_eval_dirs
from pipeline.evalsuite.static import run_static_eval

__all__ = ["run_static_eval", "compare_eval_dirs"]
