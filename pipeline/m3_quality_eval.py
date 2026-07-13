"""Manifest, preflight, merge, gate, and report MiniMax-M3 quality matrices."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from pipeline.evalsuite.compare import compare_eval_dirs
from pipeline.evalsuite.distributional import compare_distributional_records
from pipeline.evalsuite.sampling import (
    build_stratified_indices,
    manifest_sha256,
)


@dataclass(frozen=True)
class ModelSpec:
    label: str
    path: Path
    kind: str
    nodes: int
    tensor_parallel_size: int
    distributed_executor_backend: str
    pipeline_parallel_size: int = 1


@dataclass(frozen=True)
class DeferredModelSpec:
    label: str
    path: Path
    kind: str
    reason: str
    revision: str | None = None


@dataclass(frozen=True)
class ShardSpec:
    name: str
    tasks: tuple[str, ...]
    distributional_probe: bool = False


@dataclass(frozen=True)
class ProbeSpec:
    total_tokens: int
    top_k: int
    max_overhead_seconds: float


@dataclass(frozen=True)
class GateThresholds:
    max_task_drop: float
    min_macro_recovery: float
    max_conditional_regression: float
    max_perplexity_increase: float
    max_degeneration_failures: int


@dataclass(frozen=True)
class MatrixSpec:
    name: str
    backend: str
    baseline_label: str
    model_source: Path
    eval_config: Path
    models: tuple[ModelSpec, ...]
    deferred_models: tuple[DeferredModelSpec, ...]
    shards: tuple[ShardSpec, ...]
    task_aliases: dict[str, tuple[str, ...]]
    sampling: dict[str, Any]
    probe: ProbeSpec
    gates: GateThresholds

    @property
    def expected_arms(self) -> tuple[tuple[str, str], ...]:
        return tuple(
            (model.label, shard.name)
            for model in self.models
            for shard in self.shards
        )

    @property
    def smoke_node_count(self) -> int:
        return sum(model.nodes for model in self.models)

    @property
    def production_node_count(self) -> int:
        return sum(model.nodes * len(self.shards) for model in self.models)


def _write_json(path: Path, data: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(data, indent=2), encoding="utf-8")
    temporary.replace(path)


def _read_json(path: Path) -> dict:
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"expected JSON object in {path}")
    return data


def _read_jsonl(path: Path) -> list[dict]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_matrix(path: str | Path) -> MatrixSpec:
    path = Path(path)
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict) or raw.get("schema_version") != 1:
        raise ValueError("unsupported MiniMax quality matrix schema")
    models = tuple(
        ModelSpec(
            str(item["label"]),
            Path(item["path"]),
            str(item["kind"]),
            int(item.get("nodes", 1)),
            int(item.get("tensor_parallel_size", 8)),
            str(item.get("distributed_executor_backend", "mp")),
            int(item.get("pipeline_parallel_size", 1)),
        )
        for item in raw.get("models") or []
    )
    deferred_models = tuple(
        DeferredModelSpec(
            label=str(item["label"]),
            path=Path(item["path"]),
            kind=str(item["kind"]),
            reason=str(item["reason"]),
            revision=str(item["revision"]) if item.get("revision") else None,
        )
        for item in raw.get("deferred_models") or []
    )
    if {model.label for model in models} & {model.label for model in deferred_models}:
        raise ValueError("active and deferred model labels must be disjoint")
    shards = tuple(
        ShardSpec(
            str(item["name"]),
            tuple(str(task) for task in item.get("tasks") or []),
            bool(item.get("distributional_probe", False)),
        )
        for item in raw.get("shards") or []
    )
    if not models or len({model.label for model in models}) != len(models):
        raise ValueError("matrix models must have unique non-empty labels")
    if not shards or len({shard.name for shard in shards}) != len(shards):
        raise ValueError("matrix shards must have unique non-empty names")
    baseline = str(raw["baseline_label"])
    if baseline not in {model.label for model in models}:
        raise ValueError(f"baseline label {baseline!r} is not a configured model")
    probe_raw = raw.get("probe") or {}
    gates_raw = raw.get("gates") or {}
    return MatrixSpec(
        name=str(raw["name"]),
        backend=str(raw.get("backend", "vllm")),
        baseline_label=baseline,
        model_source=Path(raw["model_source"]),
        eval_config=Path(raw["eval_config"]),
        models=models,
        deferred_models=deferred_models,
        shards=shards,
        task_aliases={
            str(name): tuple(str(alias) for alias in aliases)
            for name, aliases in (raw.get("task_aliases") or {}).items()
        },
        sampling=dict(raw.get("sampling") or {}),
        probe=ProbeSpec(
            int(probe_raw["total_tokens"]),
            int(probe_raw["top_k"]),
            float(probe_raw["max_overhead_seconds"]),
        ),
        gates=GateThresholds(
            float(gates_raw["max_task_drop"]),
            float(gates_raw["min_macro_recovery"]),
            float(gates_raw["max_conditional_regression"]),
            float(gates_raw["max_perplexity_increase"]),
            int(gates_raw["max_degeneration_failures"]),
        ),
    )



def validate_reasoning_config(raw: dict[str, Any]) -> None:
    """Reject MiniMax/lm-eval reasoning combinations before GPU allocation."""
    eval_raw = raw.get("eval") or {}
    if eval_raw.get("enable_thinking") is True:
        raise ValueError(
            "lm-eval 0.4.12 disallows enable_thinking=True for the mixed "
            "multiple-choice/loglikelihood and generative MiniMax suite; leave "
            "it unset to use the chat template's adaptive mode"
        )
    if eval_raw.get("think_end_token") != "</mm:think>":
        raise ValueError(
            "MiniMax-M3 quality evaluation requires think_end_token='</mm:think>'"
        )


def resolve_task_aliases(
    aliases: dict[str, tuple[str, ...]],
    available_tasks: set[str],
) -> dict[str, str]:
    resolved: dict[str, str] = {}
    for canonical, candidates in aliases.items():
        match = next(
            (candidate for candidate in candidates if candidate in available_tasks),
            None,
        )
        if match is None:
            raise ValueError(
                f"no installed lm-eval task resolves {canonical!r}; "
                f"tried {list(candidates)}"
            )
        resolved[canonical] = match
    return resolved


def validate_sample_indices(
    tasks: dict[str, dict[str, list[int]]],
    leaf_sizes: dict[str, dict[str, int]],
) -> None:
    """Validate exact indices against lm-eval's filtered leaf documents."""
    for task_name, leaves in tasks.items():
        sizes = leaf_sizes.get(task_name)
        if sizes is None:
            raise ValueError(f"sample manifest has no leaf sizes for task={task_name}")
        for leaf_name, indices in leaves.items():
            if leaf_name not in sizes:
                raise ValueError(
                    f"sample manifest has unknown leaf: task={task_name} "
                    f"leaf={leaf_name}"
                )
            size = int(sizes[leaf_name])
            maximum = max(indices) if indices else None
            minimum = min(indices) if indices else None
            if minimum is not None and (minimum < 0 or maximum >= size):
                raise ValueError(
                    f"sample index out of range: task={task_name} "
                    f"leaf={leaf_name} size={size} "
                    f"max_selected_index={maximum}"
                )


def build_exact_sample_manifest(
    *,
    task_name: str,
    leaf_sizes: dict[str, int],
    total: int,
    seed: int,
    output: Path,
    harness_revision: str,
) -> dict[str, Any]:
    manifest: dict[str, Any] = {
        "schema_version": 1,
        "seed": seed,
        "harness_revision": harness_revision,
        "selection": "seeded_proportional_stratified_without_replacement",
        "tasks": {
            task_name: build_stratified_indices(leaf_sizes, total, seed),
        },
    }
    manifest["sha256"] = manifest_sha256(manifest)
    _write_json(output, manifest)
    return manifest


def build_profile_sample_manifests(
    *,
    resolved_tasks: dict[str, str],
    leaf_sizes: dict[str, dict[str, int]],
    mmlu_task: str,
    mmlu_total: int,
    seed: int,
    output_dir: Path,
    harness_revision: str,
) -> dict[str, Path]:
    """Write exact tiny-smoke and MMLU-stratified production manifests."""
    output_dir.mkdir(parents=True, exist_ok=True)
    smoke_tasks: dict[str, dict[str, list[int]]] = {}
    for canonical, installed in resolved_tasks.items():
        sizes = leaf_sizes[canonical]
        if len(sizes) == 1:
            smoke_tasks[installed] = build_stratified_indices(
                sizes, min(2, sum(sizes.values())), seed
            )
        else:
            smoke_tasks[installed] = {
                leaf: build_stratified_indices({leaf: size}, 1, seed)[leaf]
                for leaf, size in sizes.items()
                if size > 0
            }
    production_name = resolved_tasks[mmlu_task]
    production_tasks = {
        production_name: build_stratified_indices(
            leaf_sizes[mmlu_task], mmlu_total, seed
        )
    }
    outputs = {}
    for profile, tasks in (("smoke", smoke_tasks), ("production", production_tasks)):
        installed_leaf_sizes = {
            resolved_tasks[canonical]: sizes
            for canonical, sizes in leaf_sizes.items()
        }
        validate_sample_indices(tasks, installed_leaf_sizes)
        data: dict[str, Any] = {
            "schema_version": 1,
            "seed": seed,
            "harness_revision": harness_revision,
            "selection": "seeded_proportional_stratified_without_replacement",
            "tasks": tasks,
        }
        data["sha256"] = manifest_sha256(data)
        path = output_dir / f"{profile}_sample_manifest.json"
        _write_json(path, data)
        outputs[profile] = path
    return outputs


def project_probe_overhead(
    *,
    smoke_tokens: int,
    smoke_elapsed_seconds: float,
    production_tokens: int,
    budget_seconds: float,
) -> dict[str, Any]:
    if smoke_tokens <= 0 or smoke_elapsed_seconds <= 0 or production_tokens <= 0:
        raise ValueError("probe timing inputs must be positive")
    tokens_per_second = smoke_tokens / smoke_elapsed_seconds
    projected = production_tokens / tokens_per_second
    return {
        "smoke_tokens": smoke_tokens,
        "smoke_elapsed_seconds": smoke_elapsed_seconds,
        "tokens_per_second": tokens_per_second,
        "production_tokens": production_tokens,
        "projected_seconds": projected,
        "budget_seconds": budget_seconds,
        "within_budget": projected <= budget_seconds,
    }



def validate_smoke_gate(spec: MatrixSpec, report: dict[str, Any]) -> dict[str, Any]:
    if report.get("profile") != "smoke":
        raise ValueError("smoke gate requires a profile=smoke report")
    reported_models = report.get("models") or {}
    expected = {model.label for model in spec.models}
    missing = sorted(expected - set(reported_models))
    extra = sorted(set(reported_models) - expected)
    root_sample_sha = report.get("sample_manifest_sha256")
    results: dict[str, dict[str, Any]] = {}

    for model in spec.models:
        evidence = reported_models.get(model.label)
        if not isinstance(evidence, dict):
            continue
        probe = evidence.get("probe") or {}
        smoke_tokens = int(probe.get("tokens", 0))
        smoke_elapsed_seconds = float(probe.get("elapsed_seconds", 0.0))
        if smoke_tokens > 0 and smoke_elapsed_seconds > 0:
            projection = project_probe_overhead(
                smoke_tokens=smoke_tokens,
                smoke_elapsed_seconds=smoke_elapsed_seconds,
                production_tokens=spec.probe.total_tokens,
                budget_seconds=spec.probe.max_overhead_seconds,
            )
        else:
            projection = {
                "smoke_tokens": smoke_tokens,
                "smoke_elapsed_seconds": smoke_elapsed_seconds,
                "production_tokens": spec.probe.total_tokens,
                "budget_seconds": spec.probe.max_overhead_seconds,
                "within_budget": False,
                "reason": "missing positive smoke probe timing",
            }
        checks = {
            "infrastructure": evidence.get("infrastructure_ok") is True,
            "artifacts": evidence.get("artifacts_valid") is True,
            "all_tasks_scored": int(evidence.get("tasks_scored", 0))
            == sum(len(shard.tasks) for shard in spec.shards),
            "sample_manifest": bool(root_sample_sha)
            and evidence.get("sample_manifest_sha256") == root_sample_sha,
            "empty_outputs": int(evidence.get("empty_count", 0)) == 0,
            "periodic_loops": int(evidence.get("periodic_loop_count", 0)) == 0,
            "distributed_world_size": int(
                evidence.get("distributed_world_size", 0)
            )
            == model.tensor_parallel_size * model.pipeline_parallel_size,
            "probe_budget": projection["within_budget"],
        }
        results[model.label] = {
            "passed": all(checks.values()),
            "checks": checks,
            "distributed_world_size": evidence.get("distributed_world_size"),
            "probe_projection": projection,
        }
    ready = not missing and not extra and len(results) == len(spec.models) and all(
        result["passed"] for result in results.values()
    )
    return {
        "ready_for_production": ready,
        "missing_models": missing,
        "unexpected_models": extra,
        "models": results,
    }


def _validate_arm_manifest(root_manifest: dict, arm_manifest: dict) -> None:
    fields = (
        ("run_id", "run ID"),
        ("git_commit", "git commit"),
        ("sample_manifest_sha256", "sample manifest"),
        ("eval_config_sha256", "eval config"),
        ("tokenizer_sha256", "tokenizer"),
        ("chat_template_sha256", "chat template"),
    )
    for field, label in fields:
        if arm_manifest.get(field) != root_manifest.get(field):
            raise ValueError(
                f"{label} mismatch for {arm_manifest.get('model_label')}/"
                f"{arm_manifest.get('shard')}"
            )


def _merge_model_arms(root: Path, model: str, arms: list[Path]) -> Path:
    destination = root / "merged" / model
    samples_destination = destination / "samples"
    samples_destination.mkdir(parents=True, exist_ok=True)
    aggregate: dict[str, dict] = {}
    samples: dict[str, dict[str, dict]] = {}
    distributional: Path | None = None
    health: dict[str, dict] = {}

    for arm in arms:
        for task, metrics in _read_json(arm / "aggregate.json").items():
            if task in aggregate and aggregate[task] != metrics:
                raise ValueError(f"conflicting aggregate for {model}/{task}")
            aggregate[task] = metrics
        for sample_path in sorted((arm / "samples").glob("*.jsonl")):
            task = sample_path.stem
            by_uid = samples.setdefault(task, {})
            for row in _read_jsonl(sample_path):
                uid = row.get("sample_uid")
                if not uid:
                    raise ValueError(f"sample without stable UID in {sample_path}")
                if uid in by_uid and by_uid[uid] != row:
                    raise ValueError(f"conflicting duplicate sample {uid} in {task}")
                by_uid[uid] = row
        health_dir = arm / "generation_health"
        if health_dir.is_dir():
            for health_path in health_dir.glob("*.json"):
                if health_path.stem in health:
                    raise ValueError(
                        f"duplicate generation health task {model}/{health_path.stem}"
                    )
                health[health_path.stem] = _read_json(health_path)
        probe = arm / "distributional_probe.jsonl"
        if probe.is_file():
            if distributional is not None:
                raise ValueError(f"multiple distributional probes for {model}")
            distributional = probe

    _write_json(destination / "aggregate.json", aggregate)
    for task, by_uid in samples.items():
        output = samples_destination / f"{task}.jsonl"
        with output.open("w", encoding="utf-8") as handle:
            for uid in sorted(by_uid):
                handle.write(json.dumps(by_uid[uid], ensure_ascii=False) + "\n")
    _write_json(destination / "generation_health.json", health)
    if distributional is not None:
        shutil.copy2(distributional, destination / "distributional_probe.jsonl")
    return destination


def _degeneration_failures(health: dict[str, dict]) -> int:
    return sum(
        int(task.get("empty_count", 0))
        + int(task.get("periodic_loop_count", 0))
        + int(task.get("nonfinite_metric_count", 0))
        for task in health.values()
    )


def validate_and_merge(root: str | Path) -> dict[str, Any]:
    root = Path(root)
    manifest = _read_json(root / "run_manifest.json")
    arms_by_model: dict[str, list[Path]] = {}
    failures = []
    for expected in manifest.get("expected_arms") or []:
        model = str(expected["model_label"])
        shard = str(expected["shard"])
        arm = root / "models" / model / "shards" / shard
        required = (
            arm / "arm_manifest.json",
            arm / "aggregate.json",
            arm / "return_code.txt",
            arm / "arm_complete.json",
        )
        missing = [str(path) for path in required if not path.is_file()]
        if missing:
            failures.append({"arm": f"{model}/{shard}", "missing": missing})
            continue
        if (arm / "return_code.txt").read_text().strip() != "0":
            failures.append({"arm": f"{model}/{shard}", "return_code": "nonzero"})
            continue
        arm_manifest = _read_json(arm / "arm_manifest.json")
        _validate_arm_manifest(manifest, arm_manifest)
        if arm_manifest.get("model_label") != model or arm_manifest.get("shard") != shard:
            raise ValueError(f"arm identity mismatch for {model}/{shard}")
        arms_by_model.setdefault(model, []).append(arm)
    if failures:
        result = {
            "schema_version": 1,
            "infrastructure_ok": False,
            "failures": failures,
            "comparisons": {},
        }
        _write_json(root / "matrix.json", result)
        return result

    merged = {
        model: _merge_model_arms(root, model, arms)
        for model, arms in arms_by_model.items()
    }
    baseline_label = str(manifest["baseline_label"])
    baseline = merged[baseline_label]
    comparisons: dict[str, dict] = {}
    for model, directory in merged.items():
        if model == baseline_label:
            continue
        comparison_dir = root / "comparisons" / model
        comparison = compare_eval_dirs(
            baseline,
            directory,
            out_dir=comparison_dir,
            label_a=baseline_label,
            label_b=model,
        )
        ref_probe = baseline / "distributional_probe.jsonl"
        candidate_probe = directory / "distributional_probe.jsonl"
        if ref_probe.is_file() and candidate_probe.is_file():
            comparison["distributional"] = compare_distributional_records(
                _read_jsonl(ref_probe), _read_jsonl(candidate_probe)
            )
        candidate_health = _read_json(directory / "generation_health.json")
        comparison["generation_health"] = {
            "tasks": candidate_health,
            "degeneration_failures": _degeneration_failures(candidate_health),
        }
        _write_json(comparison_dir / "compare.json", comparison)
        comparisons[model] = comparison
    result = {
        "schema_version": 1,
        "infrastructure_ok": True,
        "failures": [],
        "baseline_label": baseline_label,
        "models": list(merged),
        "comparisons": comparisons,
    }
    _write_json(root / "matrix.json", result)
    return result


def evaluate_gates(
    matrix: dict[str, Any],
    thresholds: GateThresholds,
) -> dict[str, Any]:
    infrastructure_ok = matrix.get("infrastructure_ok") is True
    model_results: dict[str, dict] = {}
    for model, comparison in (matrix.get("comparisons") or {}).items():
        tasks = [
            task
            for task in (comparison.get("tasks") or {}).values()
            if task.get("n_paired", 1) > 0 and task.get("kind") != "perplexity"
        ]
        deltas = [float(task["delta"]) for task in tasks if task.get("delta") is not None]
        recoveries = [
            float(task["score_recovery_ratio"])
            for task in tasks
            if task.get("score_recovery_ratio") is not None
        ]
        regressions = sum(
            int(task.get("regressions_a_correct_b_wrong", 0)) for task in tasks
        )
        baseline_correct = sum(
            int(task.get("both_correct", 0))
            + int(task.get("regressions_a_correct_b_wrong", 0))
            for task in tasks
        )
        max_drop = max((max(0.0, -delta) for delta in deltas), default=0.0)
        macro_recovery = sum(recoveries) / len(recoveries) if recoveries else None
        conditional_regression = (
            regressions / baseline_correct if baseline_correct else None
        )
        perplexity_ratio = (comparison.get("distributional") or {}).get(
            "perplexity_ratio"
        )
        perplexity_increase = (
            float(perplexity_ratio) - 1.0 if perplexity_ratio is not None else None
        )
        degeneration = int(
            (comparison.get("generation_health") or {}).get(
                "degeneration_failures", 0
            )
        )
        checks = {
            "max_task_drop": {
                "value": max_drop,
                "threshold": thresholds.max_task_drop,
                "passed": max_drop <= thresholds.max_task_drop,
            },
            "macro_recovery": {
                "value": macro_recovery,
                "threshold": thresholds.min_macro_recovery,
                "passed": macro_recovery is not None
                and macro_recovery >= thresholds.min_macro_recovery,
            },
            "conditional_regression": {
                "value": conditional_regression,
                "threshold": thresholds.max_conditional_regression,
                "passed": conditional_regression is not None
                and conditional_regression <= thresholds.max_conditional_regression,
            },
            "perplexity_increase": {
                "value": perplexity_increase,
                "threshold": thresholds.max_perplexity_increase,
                "passed": perplexity_increase is not None
                and perplexity_increase <= thresholds.max_perplexity_increase,
            },
            "degeneration_failures": {
                "value": degeneration,
                "threshold": thresholds.max_degeneration_failures,
                "passed": degeneration <= thresholds.max_degeneration_failures,
            },
        }
        checks["quality_ok"] = all(check["passed"] for check in checks.values())
        model_results[model] = checks
    return {
        "infrastructure_ok": infrastructure_ok,
        "quality_ok": infrastructure_ok
        and bool(model_results)
        and all(result["quality_ok"] for result in model_results.values()),
        "models": model_results,
    }


def build_launch_plan(
    spec: MatrixSpec,
    *,
    profile: str,
    smoke_gate: Path | None = None,
) -> dict[str, Any]:
    """Build an explicit concurrent resource plan; production requires smoke."""
    all_tasks = tuple(task for shard in spec.shards for task in shard.tasks)
    if profile == "smoke":
        arms = [
            {
                "model_label": model.label,
                "model_path": str(model.path),
                "model_kind": model.kind,
                "shard": "smoke",
                "nodes": model.nodes,
                "gpus_per_node": 8,
                "tensor_parallel_size": model.tensor_parallel_size,
                "pipeline_parallel_size": model.pipeline_parallel_size,
                "distributed_executor_backend": model.distributed_executor_backend,
                "tasks": list(all_tasks),
                "distributional_probe": True,
                "samples_per_task": 2,
                "probe_tokens": 2048,
            }
            for model in spec.models
        ]
    elif profile == "production":
        if smoke_gate is None or not smoke_gate.is_file():
            raise ValueError("production launch requires a smoke gate artifact")
        gate = _read_json(smoke_gate)
        if gate.get("ready_for_production") is not True:
            raise ValueError("production launch refused: smoke gate did not pass")
        arms = [
            {
                "model_label": model.label,
                "model_path": str(model.path),
                "model_kind": model.kind,
                "shard": shard.name,
                "nodes": model.nodes,
                "gpus_per_node": 8,
                "tensor_parallel_size": model.tensor_parallel_size,
                "pipeline_parallel_size": model.pipeline_parallel_size,
                "distributed_executor_backend": model.distributed_executor_backend,
                "tasks": list(shard.tasks),
                "distributional_probe": shard.distributional_probe,
                "samples_per_task": None,
                "probe_tokens": spec.probe.total_tokens if shard.distributional_probe else 0,
            }
            for model in spec.models
            for shard in spec.shards
        ]
    else:
        raise ValueError(f"unknown launch profile {profile!r}")
    return {
        "schema_version": 1,
        "profile": profile,
        "arms": arms,
        "max_parallel_arms": len(arms),
        "total_nodes": sum(arm["nodes"] for arm in arms),
    }


def _fmt_metric(value: object) -> str:
    if value is None:
        return "n/a"
    if isinstance(value, float):
        return f"{value:.6f}"
    return str(value)


def render_matrix_report(matrix: dict[str, Any], gates: dict[str, Any]) -> str:
    """Render the decision-facing quality and quantization-fidelity summary."""
    lines = [
        "# MiniMax-M3 Quality Matrix",
        "",
        f"Overall quality gate: **{'PASS' if gates.get('quality_ok') else 'FAIL'}**",
        "",
        "Baseline: `" + str(matrix.get("baseline_label", "unknown")) + "`",
        "",
        "| Model | Task | Accuracy delta | Flip rate | Conditional regression | Score recovery | Perplexity ratio | Degeneration failures | Gate |",
        "|---|---|---:|---:|---:|---:|---:|---:|---|",
    ]
    for model, comparison in (matrix.get("comparisons") or {}).items():
        model_gate = (gates.get("models") or {}).get(model, {})
        tasks = comparison.get("tasks") or {"n/a": {}}
        distributional = comparison.get("distributional") or {}
        health = comparison.get("generation_health") or {}
        for task_name, task in tasks.items():
            lines.append(
                "| " + " | ".join(
                    [
                        str(model),
                        str(task_name),
                        _fmt_metric(task.get("delta")),
                        _fmt_metric(task.get("flip_rate")),
                        _fmt_metric(task.get("conditional_regression_rate")),
                        _fmt_metric(task.get("score_recovery_ratio")),
                        _fmt_metric(distributional.get("perplexity_ratio")),
                        _fmt_metric(health.get("degeneration_failures")),
                        "PASS" if model_gate.get("quality_ok") else "FAIL",
                    ]
                ) + " |"
            )
    lines.extend(["", "Full machine-readable evidence: `matrix.json` and `gates.json`.", ""])
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    aggregate = subparsers.add_parser("aggregate")
    aggregate.add_argument("--root", type=Path, required=True)
    aggregate.add_argument("--matrix", type=Path, required=True)
    smoke_gate = subparsers.add_parser("smoke-gate")
    smoke_gate.add_argument("--matrix", type=Path, required=True)
    smoke_gate.add_argument("--report", type=Path, required=True)
    smoke_gate.add_argument("--out", type=Path, required=True)
    launch = subparsers.add_parser("launch-plan")
    launch.add_argument("--matrix", type=Path, required=True)
    launch.add_argument("--profile", choices=("smoke", "production"), required=True)
    launch.add_argument("--smoke-gate", type=Path)
    launch.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.command == "aggregate":
        spec = load_matrix(args.matrix)
        result = validate_and_merge(args.root)
        gates = evaluate_gates(result, spec.gates)
        _write_json(args.root / "gates.json", gates)
        (args.root / "report.md").write_text(
            render_matrix_report(result, gates), encoding="utf-8"
        )
        return 0 if gates["quality_ok"] else 1
    if args.command == "smoke-gate":
        result = validate_smoke_gate(load_matrix(args.matrix), _read_json(args.report))
        _write_json(args.out, result)
        return 0 if result["ready_for_production"] else 1
    if args.command == "launch-plan":
        result = build_launch_plan(
            load_matrix(args.matrix), profile=args.profile, smoke_gate=args.smoke_gate
        )
        _write_json(args.out, result)
        return 0
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
