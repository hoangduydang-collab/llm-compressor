"""Executor-side dynamic preflight for the MiniMax-M3 quality matrix."""
from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import subprocess
from dataclasses import asdict
from pathlib import Path

import yaml

from pipeline.evalsuite.probe_corpus import build_probe_corpus
from pipeline.m3_checkpoint_diagnostics import diagnose_checkpoint
from pipeline.m3_distributional_probe import write_probe_corpus
from pipeline.m3_serve_abi import analyze_checkpoint
from pipeline.m3_quality_eval import (
    build_profile_sample_manifests,
    load_matrix,
    resolve_task_aliases,
    validate_reasoning_config,
)


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write(path: Path, data: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2), encoding="utf-8")


def tokenizer_contract(tokenizer) -> dict[str, str]:
    """Fingerprint the tokenizer and the exact default chat rendering we serve."""
    payload = tokenizer.backend_tokenizer.to_str()
    chat = getattr(tokenizer, "chat_template", None) or ""
    rendered = tokenizer.apply_chat_template(
        [{"role": "user", "content": [{"type": "text", "text": "Harness contract probe."}]}],
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
        mismatches = [field for field in fields if candidate.get(field) != reference[field]]
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
    raise ValueError(
        f"static serving ABI validation failed for {label}: {examples}"
    )


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
    raise ValueError(
        "static serving ABI validation failed: " + "; ".join(summaries)
    )


def run_preflight(matrix_path: Path, run_root: Path) -> dict:
    spec = load_matrix(matrix_path)
    raw = yaml.safe_load(spec.eval_config.read_text(encoding="utf-8"))
    validate_reasoning_config(raw)
    out = run_root / "preflight"; out.mkdir(parents=True, exist_ok=True)
    for model in spec.models:
        if not model.path.is_dir() or not (model.path / "config.json").is_file():
            raise FileNotFoundError(f"checkpoint is missing or incomplete: {model.label}={model.path}")

    abi_failures = inspect_all_serving_abis(spec.models, out / "serving_abi")
    require_all_serving_abis(abi_failures)

    from datasets import load_dataset
    from lm_eval.tasks import TaskManager
    from transformers import AutoTokenizer

    manager = TaskManager()
    available = set(manager.all_tasks)
    resolved = resolve_task_aliases(spec.task_aliases, available)
    leaf_sizes = {name: inspect_leaf_sizes(manager, installed) for name, installed in resolved.items()}
    try:
        revision = importlib.metadata.version("lm_eval")
    except importlib.metadata.PackageNotFoundError:
        revision = importlib.metadata.version("lm-eval")
    _quick_cap = spec.sampling.get("production_samples_per_task")
    manifests = build_profile_sample_manifests(
        resolved_tasks=resolved, leaf_sizes=leaf_sizes, mmlu_task="mmlu_pro",
        mmlu_total=int(spec.sampling["mmlu_pro_samples"]),
        seed=int(spec.sampling["seed"]), output_dir=out, harness_revision=revision,
        production_samples_per_task=(
            int(_quick_cap) if _quick_cap is not None else None
        ),
    )

    for task in raw["eval"]["tasks"]:
        task["name"] = resolved[task["name"]]
    resolved_config = out / "resolved_eval_config.yaml"
    resolved_config.write_text(yaml.safe_dump(raw, sort_keys=False), encoding="utf-8")
    _write(out / "resolved_tasks.json", {"aliases": resolved, "leaf_sizes": leaf_sizes})

    tokenizer = AutoTokenizer.from_pretrained(str(spec.model_source), trust_remote_code=True)
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
    dataset_meta = {"id":"Salesforce/wikitext","config":"wikitext-2-raw-v1","split":"test","revision":None,"text_column":"text"}
    dataset = load_dataset(dataset_meta["id"], dataset_meta["config"], split=dataset_meta["split"])
    texts = dataset[dataset_meta["text_column"]]
    write_probe_corpus(out / "smoke_probe_corpus.json", build_probe_corpus(texts, tokenizer, seed=42, buckets={"short":(1,2048)}), tokenizer_sha256=tokenizer_sha, dataset=dataset_meta, seed=42)
    write_probe_corpus(out / "production_probe_corpus.json", build_probe_corpus(texts, tokenizer, seed=42), tokenizer_sha256=tokenizer_sha, dataset=dataset_meta, seed=42)

    baseline_bytes = None; diagnostics = {}
    for model in spec.models:
        report = diagnose_checkpoint(model.path, baseline_bytes=baseline_bytes)
        if model.label == spec.baseline_label: baseline_bytes = report["checkpoint_bytes"]
        diagnostics[model.label] = report
        _write(out / "checkpoint_diagnostics" / f"{model.label}.json", report)
    commit = subprocess.check_output(["git","rev-parse","HEAD"], text=True).strip()
    production_manifest = manifests["production"]
    run_manifest = {
        "schema_version":1, "run_id":run_root.name, "git_commit":commit,
        "baseline_label":spec.baseline_label,
        "models":[model.label for model in spec.models],
        "expected_arms":[{"model_label":m,"shard":s} for m,s in spec.expected_arms],
        "sample_manifest_sha256":_sha(production_manifest),
        "eval_config_sha256":_sha(resolved_config), "tokenizer_sha256":tokenizer_sha,
        "chat_template_sha256":reference_tokenizer["chat_template_sha256"],
        "rendered_prompt_sha256":reference_tokenizer["rendered_prompt_sha256"],
        "resolved_tasks":resolved, "lm_eval_version":revision,
        "matrix_sha256":_sha(matrix_path),
    }
    _write(run_root / "run_manifest.json", run_manifest)
    return run_manifest


def main(argv=None) -> int:
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--matrix",type=Path,required=True)
    parser.add_argument("--run-root",type=Path,required=True)
    args=parser.parse_args(argv)
    run_preflight(args.matrix,args.run_root)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
