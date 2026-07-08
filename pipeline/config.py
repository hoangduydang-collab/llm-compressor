"""Pipeline configuration schema and YAML loader.

A single YAML file fully describes a quantization run: which model, which
method x scheme, the calibration set, large-model offload, the serve-handoff
check, and the accuracy gate. Every field has a sensible default so a minimal
config (just ``model.id`` + ``quantization``) is enough to get going.
"""

from dataclasses import dataclass, field, fields, is_dataclass
from pathlib import Path
from typing import Any

import yaml

# Methods understood by ``pipeline.recipe.build_recipe``.
VALID_METHODS = {
    "gptq",
    "awq",
    "smoothquant+gptq",
    "smoothquant+awq",
    "autoround",
    "spinquant+gptq",
    "spinquant+awq",
    "quant_only",  # no calibration algorithm, just the QuantizationModifier
}

# Schemes that resolve to a compressed-tensors preset. Anything else is passed
# through verbatim and validated by compressed-tensors at recipe-build time.
KNOWN_SCHEMES = {"W4AFP8", "W4A8", "W4A16", "W8A8", "FP8_DYNAMIC", "FP8", "NVFP4"}


@dataclass
class ModelConfig:
    id: str = ""
    trust_remote_code: bool = False
    # transformers auto-class used to load the model. Causal LMs use the default;
    # VL / image-text-to-text MoEs (e.g. MiniMax-M3) need AutoModelForImageTextToText
    # so the full model (language backbone + vision tower) is loaded and saved.
    auto_class: str = "AutoModelForCausalLM"
    # Large-model loading. device_map="auto_offload" + offload_folder spills to
    # disk; max_memory caps per-device usage e.g. {"cpu": 500e9, 0: 70e9}.
    device_map: str | None = None
    offload_folder: str | None = None
    max_memory: dict[str, float] | None = None
    dtype: str = "auto"


@dataclass
class QuantizationConfig:
    method: str = "gptq"
    scheme: str = "W4AFP8"
    ignore: list[str] = field(default_factory=lambda: ["lm_head"])
    # Method-specific knobs.
    smoothquant_strength: float = 0.8
    awq_duo_scaling: bool = True
    gptq_dampening_frac: float | None = None
    # Post-quant sanity generation. Disable for very large offloaded models, where
    # autoregressive generation runs on CPU/disk (~minutes per token) and adds hours.
    sample_generation: bool = True


@dataclass
class CalibrationConfig:
    dataset_id: str = "HuggingFaceH4/ultrachat_200k"
    dataset_split: str = "train_sft"
    num_samples: int = 256
    max_seq_length: int = 2048
    seed: int = 42
    # MoE: route all tokens through all experts during calibration (default on).
    moe_calibrate_all_experts: bool = True
    # Optional: name(s) of the decoder layer class for layer-at-a-time onloading
    # of very large models, e.g. ["Qwen3MoeDecoderLayer"].
    sequential_targets: list[str] | None = None
    pipeline: str | None = None  # let llm-compressor pick by default


@dataclass
class ServeConfig:
    enabled: bool = True
    tensor_parallel_size: int = 1
    enable_expert_parallel: bool = False
    block_size: int | None = None
    kv_cache_dtype: str | None = None
    max_model_len: int = 4096
    gpu_memory_utilization: float = 0.9
    # Disable CUDA graphs. Useful to sidestep CUDA-graph/stream issues (e.g. the
    # W4A8 MoE stream-race in older vLLM) and for debugging.
    enforce_eager: bool = False
    # Skip vLLM custom all-reduce (NCCL fallback for that path only). Optional
    # escape hatch — does not disable M3's FlashInfer fused AR (BUGS_AND_FIXES.md).
    disable_custom_all_reduce: bool = False
    # Extra raw flags appended to ``vllm serve`` / passed to ``LLM(...)``.
    extra_args: list[str] = field(default_factory=list)
    # SGLang-only kwargs forwarded to ``sgl.Engine`` when ``eval.backend: sglang``
    # (e.g. ``quantization: w4afp8``, ``disable_shared_experts_fusion: true``).
    sglang_kwargs: dict[str, Any] = field(default_factory=dict)
    # When True, apply_sglang_compat_env() before loading SGLang (FlashInfer CUDA
    # norm, SGLANG_ENABLE_JIT_DEEPGEMM=0, NVRTC for DeepGEMM when nvcc is broken).
    sglang_compat_fallbacks: bool = False
    prompt: str = "The capital of France is"


@dataclass
class EvalTask:
    name: str  # lm-eval task name, e.g. "wikitext", "mmlu", "gsm8k"
    metric: str = "acc,none"  # metric key to read from lm-eval results
    num_fewshot: int = 0
    limit: int | None = 250
    # higher_is_better=False for perplexity-style metrics (e.g. word_perplexity).
    higher_is_better: bool = True


def full_static_tasks() -> list[EvalTask]:
    """Default static lm-eval suite (8 tasks, full splits unless overridden)."""
    return [
        EvalTask(name="wikitext", metric="word_perplexity,none", higher_is_better=False, limit=None),
        EvalTask(name="mmlu", metric="acc,none", num_fewshot=5, limit=None),
        EvalTask(name="arc_challenge", metric="acc_norm,none", num_fewshot=25, limit=None),
        EvalTask(name="hellaswag", metric="acc_norm,none", num_fewshot=10, limit=None),
        EvalTask(name="winogrande", metric="acc,none", num_fewshot=5, limit=None),
        EvalTask(name="gsm8k", metric="exact_match,strict-match", num_fewshot=5, limit=None),
        EvalTask(name="truthfulqa_mc2", metric="acc,none", num_fewshot=0, limit=None),
        EvalTask(name="bbh", metric="exact_match,get-answer", num_fewshot=3, limit=None),
    ]


@dataclass
class EvalConfig:
    enabled: bool = True
    # Path to a baseline metrics JSON (produced by an earlier run of the
    # unquantized model). If None, the gate records numbers but cannot pass/fail.
    baseline: str | None = None
    # Fraction of baseline that must be recovered (for higher-is-better metrics).
    recovery_threshold: float = 0.94
    # For perplexity metrics: max allowed relative increase over baseline.
    max_ppl_increase: float = 0.10
    backend: str = "vllm"  # lm-eval backend: vllm | sglang
    apply_chat_template: bool = False
    # lm-eval per-forward batch (SGLang). Use an int (e.g. 8) to avoid huge
    # auto-detected chunks on MMLU; "auto" probes and can stall on first batch.
    lm_eval_batch_size: str | int = "auto"
    # Per-sample logging for post-hoc flip-rate comparison (evalsuite).
    log_samples: bool = True
    samples_dir: str | None = None  # defaults to <out>/samples at runtime
    tasks: list[EvalTask] = field(default_factory=full_static_tasks)


@dataclass
class AgenticConfig:
    enabled: bool = False
    harness: str = "tau2"
    # Path to cloned tau2-bench repo (must contain .venv/bin/tau2).
    tau2_dir: str | None = None
    # Path to benchmarks-repo run_calibration.sh (tau2 launcher).
    calibration_script: str | None = None
    domain: str = "telecom"
    split: str = "small"
    num_tasks: int | None = None
    max_conc: int = 5
    num_trials: int = 1
    thinking: str = "off"  # on|off
    # Served agent endpoint (defaults derived from serve config at runtime).
    agent_base: str | None = None
    agent_model: str | None = None
    # User simulator (required for tau2; if unset agentic self-skips).
    user_base: str | None = None
    user_model: str | None = None
    user_key_file: str | None = None
    save_to: str = "evalsuite_agentic"
    max_steps: int = 50
    timeout: int = 900
    seed: int = 42


@dataclass
class CompareConfig:
    """Post-hoc comparison knobs (used by evalsuite.compare)."""
    # Per-task lm-eval metric keys treated as binary correctness for flip-rate.
    flip_task_metrics: dict[str, str] = field(
        default_factory=lambda: {
            "mmlu": "acc",
            "arc_challenge": "acc_norm",
            "hellaswag": "acc_norm",
            "winogrande": "acc",
            "gsm8k": "exact_match",
            "truthfulqa_mc2": "acc",
            "bbh": "exact_match",
        }
    )
    perplexity_tasks: list[str] = field(default_factory=lambda: ["wikitext"])
    perplexity_metric: str = "word_perplexity"
    agentic_reward_threshold: float = 1.0


@dataclass
class PipelineConfig:
    name: str = "run"
    model: ModelConfig = field(default_factory=ModelConfig)
    quantization: QuantizationConfig = field(default_factory=QuantizationConfig)
    calibration: CalibrationConfig = field(default_factory=CalibrationConfig)
    serve: ServeConfig = field(default_factory=ServeConfig)
    eval: EvalConfig = field(default_factory=EvalConfig)
    agentic: AgenticConfig = field(default_factory=AgenticConfig)
    compare: CompareConfig = field(default_factory=CompareConfig)
    # Root under which a per-run versioned artifact directory is written.
    output_dir: str = "./artifacts"

    def validate(self) -> None:
        if not self.model.id:
            raise ValueError("config.model.id is required")
        if self.quantization.method not in VALID_METHODS:
            raise ValueError(
                f"unknown quantization.method {self.quantization.method!r}; "
                f"valid: {sorted(VALID_METHODS)}"
            )
        if self.model.device_map == "auto_offload" and not self.model.offload_folder:
            raise ValueError(
                "model.offload_folder is required when device_map='auto_offload'"
            )

    @property
    def run_slug(self) -> str:
        """Stable, filesystem-safe identifier for this run."""
        model_tail = self.model.id.rstrip("/").split("/")[-1]
        method = self.quantization.method.replace("+", "-")
        return f"{model_tail}-{method}-{self.quantization.scheme}".replace("/", "_")


def _build(cls: type, data: dict[str, Any] | None) -> Any:
    """Recursively construct a (possibly nested) dataclass from a dict.

    Unknown keys raise, so typos in a YAML config fail loudly instead of being
    silently ignored.
    """
    data = data or {}
    if not is_dataclass(cls):
        return data

    field_map = {f.name: f for f in fields(cls)}
    unknown = set(data) - set(field_map)
    if unknown:
        raise ValueError(f"unknown keys for {cls.__name__}: {sorted(unknown)}")

    kwargs: dict[str, Any] = {}
    for key, value in data.items():
        f = field_map[key]
        # Nested dataclass field (e.g. PipelineConfig.model -> ModelConfig).
        if is_dataclass(f.type) and isinstance(value, dict):
            kwargs[key] = _build(f.type, value)
        # list[EvalTask]
        elif key == "tasks" and isinstance(value, list):
            kwargs[key] = [
                EvalTask(**t) if isinstance(t, dict) else t for t in value
            ]
        else:
            kwargs[key] = value
    return cls(**kwargs)


def load_config(path: str | Path) -> PipelineConfig:
    """Load and validate a pipeline config from a YAML file."""
    path = Path(path)
    with path.open("r", encoding="utf-8") as fh:
        raw = yaml.safe_load(fh) or {}

    cfg = _build(PipelineConfig, raw)
    cfg.validate()
    return cfg
