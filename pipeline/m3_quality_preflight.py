"""Executor-side dynamic preflight for the MiniMax-M3 quality matrix."""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import inspect
import json
import random
import subprocess
from pathlib import Path

import yaml

from pipeline.evalsuite.probe_corpus import build_probe_corpus
from pipeline.m3_checkpoint_diagnostics import diagnose_checkpoint
from pipeline.m3_distributional_probe import write_probe_corpus
from pipeline.m3_quality_eval import (
    build_profile_sample_manifests,
    load_matrix,
    resolve_task_aliases,
    validate_reasoning_config,
)
from pipeline.m3_serve_abi import analyze_checkpoint


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write(path: Path, data: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(data, indent=2), encoding="utf-8")
    temporary.replace(path)


_R4_TASK_CONTRACT = {
    "gpqa_diamond": {
        "installed_name": "gpqa_diamond_cot_zeroshot",
        "num_fewshot": 0,
        "metric": "exact_match,flexible-extract",
    },
    "mmlu_pro": {
        "installed_name": "mmlu_pro",
        "num_fewshot": 5,
        "metric": "exact_match,custom-extract",
    },
    "gsm8k": {
        "installed_name": "gsm8k_cot",
        "num_fewshot": 8,
        "metric": "exact_match,strict-match",
    },
    "aime_2025": {
        "installed_name": "aime25",
        "num_fewshot": 0,
        "metric": "exact_match,none",
    },
}
_R4_GENERATION = {
    "temperature": 1.0,
    "top_p": 0.95,
    "do_sample": True,
    "max_gen_toks": 16384,
}
_R4_SEEDS = [42, 1234, 4158]


def build_reasoning_harness_contract(
    *,
    revision: str,
    task_records: dict[str, dict],
    generation_seeds: list[int],
    gen_kwargs: dict,
) -> dict:
    """Validate and return the exact r4 lm-eval reasoning contract."""
    if revision != "0.4.12":
        raise ValueError(f"lm-eval revision must be 0.4.12, received {revision!r}")
    if set(task_records) != set(_R4_TASK_CONTRACT):
        raise ValueError("reasoning task set differs from the r4 harness contract")
    if generation_seeds != _R4_SEEDS:
        raise ValueError("generation_seeds must be exactly [42, 1234, 4158]")
    if gen_kwargs != _R4_GENERATION:
        raise ValueError(f"generation kwargs differ from r4 contract: {gen_kwargs!r}")

    for canonical, expected in _R4_TASK_CONTRACT.items():
        record = task_records[canonical]
        for field, expected_value in expected.items():
            if record.get(field) != expected_value:
                raise ValueError(
                    f"{canonical} {field} must be {expected_value!r}, "
                    f"received {record.get(field)!r}"
                )
        if expected["metric"] not in record.get("available_metric_filters", []):
            raise ValueError(
                f"{canonical} installed task does not expose metric/filter "
                f"{expected['metric']!r}"
            )
        if record.get("output_type") != "generate_until":
            raise ValueError(f"{canonical} output_type must be 'generate_until'")
        if canonical == "gpqa_diamond" and record.get("task_version") != "2.2":
            raise ValueError("gpqa_diamond task_version must be '2.2'")
        if not record.get("representative_prompt_sha256"):
            raise ValueError(f"{canonical} representative prompt is missing")
        if record.get("representative_prompt_sha256") != record.get(
            "repeat_prompt_sha256"
        ):
            raise ValueError(f"{canonical} representative prompt is unstable")

    gpqa = task_records["gpqa_diamond"]
    for field in ("displayed_choices", "correct_displayed_label"):
        if gpqa.get(field) != gpqa.get(f"repeat_{field}"):
            raise ValueError(f"gpqa_diamond {field} is unstable")

    return {
        "schema_version": 1,
        "valid": True,
        "lm_eval_version": revision,
        "tasks": task_records,
        "generation": {
            "generation_seeds": list(generation_seeds),
            "gen_kwargs": dict(gen_kwargs),
            "sampling_backend_note": (
                "vLLM samples because temperature>0; do_sample is recorded "
                "as the harness intent and is not passed to SamplingParams"
            ),
        },
    }


def _task_config_value(task, field: str):
    config = getattr(task, "config", None) or getattr(task, "_config", None)
    if isinstance(config, dict):
        return config.get(field)
    return getattr(config, field, None) if config is not None else None


def _metric_filter_keys(task) -> list[str]:
    metric_list = _task_config_value(task, "metric_list") or []
    filter_list = _task_config_value(task, "filter_list") or []

    def names(records, field: str) -> list[str]:
        values = []
        for record in records:
            value = record.get(field) if isinstance(record, dict) else None
            if value is not None:
                values.append(str(value))
        return values

    metrics = names(metric_list, "metric")
    filters = names(filter_list, "name") or ["none"]
    return sorted(
        f"{metric},{filter_name}" for metric in metrics for filter_name in filters
    )


def _loaded_task(manager, installed_name: str):
    tasks = manager.load([installed_name]).get("tasks") or {}
    if installed_name in tasks:
        return tasks[installed_name]
    if len(tasks) == 1:
        return next(iter(tasks.values()))
    raise ValueError(
        f"cannot select a single task object for {installed_name!r}: {sorted(tasks)}"
    )


def _representative_task_view(
    manager,
    tokenizer,
    *,
    installed_name: str,
    num_fewshot: int,
) -> dict:
    random.seed(42)
    task = _loaded_task(manager, installed_name)
    if hasattr(task, "set_fewshot_seed"):
        task.set_fewshot_seed(42)
    docs = task.eval_docs
    doc = docs[0] if hasattr(docs, "__getitem__") else next(iter(docs))
    formatter = task.fewshot_context
    parameters = inspect.signature(formatter).parameters
    optional = {
        "rnd": random.Random(42),
        "system_instruction": None,
        "apply_chat_template": True,
        "fewshot_as_multiturn": True,
        "chat_template": getattr(tokenizer, "chat_template", None),
        "tokenizer_name": getattr(tokenizer, "name_or_path", ""),
    }
    kwargs = {key: value for key, value in optional.items() if key in parameters}
    prompt = formatter(doc, num_fewshot, **kwargs)
    if isinstance(prompt, tuple):
        prompt = prompt[0]
    prompt_text = str(prompt)
    choice_formatter = getattr(task, "doc_to_choice", None)
    choices = choice_formatter(doc) if callable(choice_formatter) else None
    target = task.doc_to_target(doc)
    return {
        "task": task,
        "prompt_sha256": hashlib.sha256(prompt_text.encode()).hexdigest(),
        "displayed_choices": (
            [str(choice) for choice in choices] if choices is not None else None
        ),
        "correct_displayed_label": str(target) if target is not None else None,
    }


def inspect_reasoning_task_records(
    manager,
    tokenizer,
    *,
    resolved: dict[str, str],
    configured_tasks: list[dict],
    leaf_sizes: dict[str, dict[str, int]],
) -> dict[str, dict]:
    """Inspect resolved lm-eval objects twice and capture stable prompts."""
    configured = {task["name"]: task for task in configured_tasks}
    records = {}
    for canonical, installed in resolved.items():
        task_config = configured[canonical]
        first = _representative_task_view(
            manager,
            tokenizer,
            installed_name=installed,
            num_fewshot=int(task_config["num_fewshot"]),
        )
        second = _representative_task_view(
            manager,
            tokenizer,
            installed_name=installed,
            num_fewshot=int(task_config["num_fewshot"]),
        )
        task = first["task"]
        records[canonical] = {
            "canonical_name": canonical,
            "installed_name": installed,
            "output_type": str(getattr(task, "OUTPUT_TYPE", "")),
            "task_version": str(getattr(task, "VERSION", "")),
            "num_fewshot": int(task_config["num_fewshot"]),
            "metric": str(task_config["metric"]),
            "available_metric_filters": _metric_filter_keys(task),
            "dataset_path": str(_task_config_value(task, "dataset_path")),
            "dataset_name": str(_task_config_value(task, "dataset_name")),
            "representative_doc_id": 0,
            "representative_prompt_sha256": first["prompt_sha256"],
            "repeat_prompt_sha256": second["prompt_sha256"],
            "displayed_choices": first["displayed_choices"],
            "repeat_displayed_choices": second["displayed_choices"],
            "correct_displayed_label": first["correct_displayed_label"],
            "repeat_correct_displayed_label": second["correct_displayed_label"],
            "available_samples": sum(leaf_sizes[canonical].values()),
        }
    return records


def tokenizer_contract(tokenizer) -> dict[str, str]:
    """Fingerprint the tokenizer and the exact default chat rendering we serve."""
    payload = tokenizer.backend_tokenizer.to_str()
    chat = getattr(tokenizer, "chat_template", None) or ""
    rendered = tokenizer.apply_chat_template(
        [
            {
                "role": "user",
                "content": [{"type": "text", "text": "Harness contract probe."}],
            }
        ],
        add_generation_prompt=True,
        tokenize=False,
    )
    return {
        "tokenizer_sha256": hashlib.sha256(payload.encode()).hexdigest(),
        "chat_template_sha256": hashlib.sha256(chat.encode()).hexdigest(),
        "rendered_prompt_sha256": hashlib.sha256(rendered.encode()).hexdigest(),
    }


def compare_tokenizer_contracts(
    reference: dict[str, str], candidates: dict[str, dict[str, str]]
) -> dict:
    fields = tuple(reference)
    models = {}
    for label, candidate in candidates.items():
        mismatches = [
            field for field in fields if candidate.get(field) != reference[field]
        ]
        models[label] = {
            **candidate,
            "matches_reference": not mismatches,
            "mismatches": mismatches,
        }
    return {
        "valid": all(model["matches_reference"] for model in models.values()),
        "reference": reference,
        "models": models,
    }


def _task_split(task) -> str:
    config = getattr(task, "config", None) or getattr(task, "_config", None)
    dataset = getattr(task, "dataset", None)
    for name in ("test_split", "validation_split", "training_split"):
        split = getattr(config, name, None) if config is not None else None
        if split is None and isinstance(config, dict):
            split = config.get(name)
        if split and dataset is not None and split in dataset:
            return str(split)
    for split in ("test", "validation", "train"):
        if dataset is not None and split in dataset:
            return split
    raise ValueError(f"cannot identify evaluation split for {type(task).__name__}")


def inspect_leaf_sizes(manager, installed_task: str) -> dict[str, int]:
    loaded = manager.load([installed_task])
    tasks = loaded.get("tasks") or {}
    if not tasks:
        raise ValueError(f"lm-eval loaded no leaf tasks for {installed_task!r}")
    sizes = {}
    for leaf, task in tasks.items():
        eval_docs = task.eval_docs
        sizes[str(leaf)] = len(eval_docs)
    return sizes


def inspect_checkpoint_serving_abi(checkpoint: Path) -> dict:
    return analyze_checkpoint(checkpoint)


def require_valid_serving_abi(label: str, report: dict) -> None:
    if report.get("valid") is True:
        return
    examples = ", ".join(
        str(error.get("module") or error.get("code"))
        for error in (report.get("errors") or [])[:3]
    )
    raise ValueError(f"static serving ABI validation failed for {label}: {examples}")


def inspect_all_serving_abis(models, output_dir: Path) -> list[dict]:
    """Persist every static report and return invalid model summaries."""

    failures = []
    for model in models:
        report = inspect_checkpoint_serving_abi(model.path)
        _write(output_dir / f"{model.label}.json", report)
        if report.get("valid") is not True:
            failures.append({"label": model.label, "report": report})
    return failures


def require_all_serving_abis(failures: list[dict]) -> None:
    if not failures:
        return
    summaries = []
    for failure in failures:
        report = failure["report"]
        examples = ", ".join(
            str(error.get("module") or error.get("code"))
            for error in (report.get("errors") or [])[:3]
        )
        summaries.append(f"{failure['label']}: {examples}")
    raise ValueError("static serving ABI validation failed: " + "; ".join(summaries))


def run_preflight(matrix_path: Path, run_root: Path) -> dict:
    spec = load_matrix(matrix_path)
    raw = yaml.safe_load(spec.eval_config.read_text(encoding="utf-8"))
    validate_reasoning_config(raw)
    out = run_root / "preflight"
    out.mkdir(parents=True, exist_ok=True)
    for model in spec.models:
        if not model.path.is_dir() or not (model.path / "config.json").is_file():
            raise FileNotFoundError(
                f"checkpoint is missing or incomplete: {model.label}={model.path}"
            )

    abi_failures = inspect_all_serving_abis(spec.models, out / "serving_abi")
    require_all_serving_abis(abi_failures)

    from datasets import load_dataset
    from lm_eval.tasks import TaskManager
    from transformers import AutoTokenizer

    manager = TaskManager()
    available = set(manager.all_tasks)
    resolved = resolve_task_aliases(spec.task_aliases, available)
    leaf_sizes = {
        name: inspect_leaf_sizes(manager, installed)
        for name, installed in resolved.items()
    }
    try:
        revision = importlib.metadata.version("lm_eval")
    except importlib.metadata.PackageNotFoundError:
        revision = importlib.metadata.version("lm-eval")
    tokenizer = AutoTokenizer.from_pretrained(
        str(spec.model_source), trust_remote_code=True
    )
    task_records = inspect_reasoning_task_records(
        manager,
        tokenizer,
        resolved=resolved,
        configured_tasks=raw["eval"]["tasks"],
        leaf_sizes=leaf_sizes,
    )
    harness_contract = build_reasoning_harness_contract(
        revision=revision,
        task_records=task_records,
        generation_seeds=list(raw["eval"].get("generation_seeds") or []),
        gen_kwargs=dict(raw["eval"].get("gen_kwargs") or {}),
    )
    harness_contract_path = out / "harness_contract.json"
    _write(harness_contract_path, harness_contract)
    _quick_cap = spec.sampling.get("production_samples_per_task")
    manifests = build_profile_sample_manifests(
        resolved_tasks=resolved,
        leaf_sizes=leaf_sizes,
        mmlu_task="mmlu_pro",
        mmlu_total=int(spec.sampling["mmlu_pro_samples"]),
        seed=int(spec.sampling["seed"]),
        output_dir=out,
        harness_revision=revision,
        production_samples_per_task=(
            int(_quick_cap) if _quick_cap is not None else None
        ),
    )

    for task in raw["eval"]["tasks"]:
        task["name"] = resolved[task["name"]]
    resolved_config = out / "resolved_eval_config.yaml"
    resolved_config.write_text(yaml.safe_dump(raw, sort_keys=False), encoding="utf-8")
    _write(out / "resolved_tasks.json", {"aliases": resolved, "leaf_sizes": leaf_sizes})

    reference_tokenizer = tokenizer_contract(tokenizer)
    served_tokenizers = {
        model.label: tokenizer_contract(
            AutoTokenizer.from_pretrained(str(model.path), trust_remote_code=True)
        )
        for model in spec.models
    }
    tokenizer_report = compare_tokenizer_contracts(
        reference_tokenizer, served_tokenizers
    )
    _write(out / "tokenizer_contract.json", tokenizer_report)
    if tokenizer_report["valid"] is not True:
        mismatched = [
            label
            for label, contract in tokenizer_report["models"].items()
            if not contract["matches_reference"]
        ]
        raise ValueError(
            "served tokenizer/chat-template contract differs from official "
            f"MiniMax-M3 source: {', '.join(mismatched)}"
        )
    tokenizer_sha = reference_tokenizer["tokenizer_sha256"]
    if spec.probe.enabled:
        dataset_meta = {
            "id": "Salesforce/wikitext",
            "config": "wikitext-2-raw-v1",
            "split": "test",
            "revision": None,
            "text_column": "text",
        }
        dataset = load_dataset(
            dataset_meta["id"],
            dataset_meta["config"],
            split=dataset_meta["split"],
        )
        texts = dataset[dataset_meta["text_column"]]
        write_probe_corpus(
            out / "smoke_probe_corpus.json",
            build_probe_corpus(texts, tokenizer, seed=42, buckets={"short": (1, 2048)}),
            tokenizer_sha256=tokenizer_sha,
            dataset=dataset_meta,
            seed=42,
        )
        write_probe_corpus(
            out / "production_probe_corpus.json",
            build_probe_corpus(texts, tokenizer, seed=42),
            tokenizer_sha256=tokenizer_sha,
            dataset=dataset_meta,
            seed=42,
        )

    baseline_bytes = None
    diagnostics = {}
    for model in spec.models:
        report = diagnose_checkpoint(model.path, baseline_bytes=baseline_bytes)
        if model.label == spec.baseline_label:
            baseline_bytes = report["checkpoint_bytes"]
        diagnostics[model.label] = report
        _write(out / "checkpoint_diagnostics" / f"{model.label}.json", report)
    commit = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    production_manifest = manifests["production"]
    production_samples = json.loads(production_manifest.read_text(encoding="utf-8"))
    expected_question_counts = {
        task: sum(len(indices) for indices in leaves.values())
        for task, leaves in production_samples["tasks"].items()
    }
    run_manifest = {
        "schema_version": 1,
        "run_id": run_root.name,
        "git_commit": commit,
        "baseline_label": spec.baseline_label,
        "models": [model.label for model in spec.models],
        "expected_arms": [
            {"model_label": m, "shard": s} for m, s in spec.expected_arms
        ],
        "sample_manifest_sha256": _sha(production_manifest),
        "eval_config_sha256": _sha(resolved_config),
        "tokenizer_sha256": tokenizer_sha,
        "chat_template_sha256": reference_tokenizer["chat_template_sha256"],
        "rendered_prompt_sha256": reference_tokenizer["rendered_prompt_sha256"],
        "harness_contract_sha256": _sha(harness_contract_path),
        "generation_seeds": list(raw["eval"].get("generation_seeds") or []),
        "expected_question_counts": expected_question_counts,
        "resolved_tasks": resolved,
        "lm_eval_version": revision,
        "matrix_sha256": _sha(matrix_path),
    }
    _write(run_root / "run_manifest.json", run_manifest)
    return run_manifest


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--matrix", type=Path, required=True)
    parser.add_argument("--run-root", type=Path, required=True)
    args = parser.parse_args(argv)
    run_preflight(args.matrix, args.run_root)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
