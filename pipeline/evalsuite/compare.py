"""Post-hoc quantized-vs-original comparison with flip-rate metrics."""

from __future__ import annotations

import json
import math
from pathlib import Path

from pipeline.config import CompareConfig, PipelineConfig
from pipeline.evalsuite.stats import exact_mcnemar, paired_bootstrap


def _load_json(path: Path) -> dict | list | None:
    if not path.exists():
        return None
    with path.open("r", encoding="utf-8") as fh:
        return json.load(fh)


def _load_jsonl(path: Path) -> list[dict]:
    rows: list[dict] = []
    if not path.exists():
        return rows
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def _norm_sf(z: float) -> float:
    return 0.5 * math.erfc(z / math.sqrt(2))


def chi2_sf_df1(x: float) -> float:
    """Survival function for chi-square with 1 df."""
    if x <= 0:
        return 1.0
    return 2 * _norm_sf(math.sqrt(x))


def mcnemar_test(regressions: int, recoveries: int) -> dict:
    """Select exact McNemar for small samples, asymptotic for larger ones."""
    n_discordant = regressions + recoveries
    if n_discordant < 25:
        return exact_mcnemar(regressions, recoveries)
    stat = (abs(regressions - recoveries) - 1) ** 2 / n_discordant
    return {
        "method": "chi2_continuity",
        "statistic": stat,
        "p_value": chi2_sf_df1(stat),
        "discordant": n_discordant,
    }


def cohens_kappa(
    both_correct: int,
    both_wrong: int,
    regressions: int,
    recoveries: int,
) -> float | None:
    n = both_correct + both_wrong + regressions + recoveries
    if n == 0:
        return None
    p_o = (both_correct + both_wrong) / n
    p_a_yes = (both_correct + regressions) / n
    p_b_yes = (both_correct + recoveries) / n
    p_e = p_a_yes * p_b_yes + (1 - p_a_yes) * (1 - p_b_yes)
    if p_e == 1.0:
        return 1.0
    return (p_o - p_e) / (1 - p_e)


def _sample_key(row: dict, id_key: str | None) -> object:
    if id_key is not None:
        return row.get(id_key)
    return row.get("sample_uid", row.get("doc_id"))


def _rows_by_key(
    rows: list[dict],
    *,
    id_key: str | None,
    value_key: str,
) -> dict[object, dict]:
    mapped: dict[object, dict] = {}
    for row in rows:
        if row.get(value_key) is None:
            continue
        key = _sample_key(row, id_key)
        if key is None:
            raise ValueError("paired sample is missing a stable identity")
        if key in mapped:
            raise ValueError(f"duplicate paired sample identity: {key!r}")
        mapped[key] = row
    return mapped


def _delta_mean(values_a: list[float], values_b: list[float]) -> float:
    return (
        sum(b - a for a, b in zip(values_a, values_b, strict=True))
        / len(values_a)
    )


def _mean_second(_: list[float], values_b: list[float]) -> float:
    return sum(values_b) / len(values_b)


def _pair_binary(
    rows_a: list[dict],
    rows_b: list[dict],
    id_key: str | None = None,
    correct_key: str = "correct",
    *,
    seed: int = 42,
    iterations: int = 10_000,
) -> dict:
    map_a = _rows_by_key(rows_a, id_key=id_key, value_key=correct_key)
    map_b = _rows_by_key(rows_b, id_key=id_key, value_key=correct_key)
    keys_a = set(map_a)
    keys_b = set(map_b)
    shared = sorted(keys_a & keys_b, key=lambda value: str(value))

    both_correct = both_wrong = regressions = recoveries = 0
    correctness_a: list[float] = []
    correctness_b: list[float] = []
    flip_indicators: list[float] = []
    regression_indicators: list[float] = []
    for key in shared:
        ca = int(map_a[key][correct_key])
        cb = int(map_b[key][correct_key])
        correctness_a.append(float(ca))
        correctness_b.append(float(cb))
        flip_indicators.append(float(ca != cb))
        regression_indicators.append(float(ca == 1 and cb == 0))
        if ca and cb:
            both_correct += 1
        elif not ca and not cb:
            both_wrong += 1
        elif ca and not cb:
            regressions += 1
        else:
            recoveries += 1

    n = len(shared)
    acc_a = (both_correct + regressions) / n if n else None
    acc_b = (both_correct + recoveries) / n if n else None
    flips = regressions + recoveries
    baseline_correct = both_correct + regressions
    baseline_wrong = both_wrong + recoveries
    denominator = max(len(map_a), len(map_b))

    bootstrap = None
    if n:
        zeros = [0.0] * n
        bootstrap = {
            "accuracy_delta": paired_bootstrap(
                correctness_a,
                correctness_b,
                statistic=_delta_mean,
                seed=seed,
                iterations=iterations,
            ),
            "flip_rate": paired_bootstrap(
                zeros,
                flip_indicators,
                statistic=_mean_second,
                seed=seed + 1,
                iterations=iterations,
            ),
            "regression_rate": paired_bootstrap(
                zeros,
                regression_indicators,
                statistic=_mean_second,
                seed=seed + 2,
                iterations=iterations,
            ),
        }

    mcnemar = mcnemar_test(regressions, recoveries)
    kappa = cohens_kappa(both_correct, both_wrong, regressions, recoveries)
    agreement = (both_correct + both_wrong) / n if n else None

    return {
        "n_a": len(map_a),
        "n_b": len(map_b),
        "n_paired": n,
        "missing_in_a": len(keys_b - keys_a),
        "missing_in_b": len(keys_a - keys_b),
        "paired_coverage": n / denominator if denominator else None,
        "acc_a": acc_a,
        "acc_b": acc_b,
        "delta": (
            (acc_b - acc_a)
            if acc_a is not None and acc_b is not None
            else None
        ),
        "score_recovery_ratio": (acc_b / acc_a) if acc_a else None,
        "both_correct": both_correct,
        "both_wrong": both_wrong,
        "regressions_a_correct_b_wrong": regressions,
        "recoveries_a_wrong_b_correct": recoveries,
        "net_harmful_flips": regressions - recoveries,
        "flip_rate": flips / n if n else None,
        "regression_rate": regressions / n if n else None,
        "recovery_rate": recoveries / n if n else None,
        "conditional_regression_rate": (
            regressions / baseline_correct if baseline_correct else None
        ),
        "conditional_recovery_rate": (
            recoveries / baseline_wrong if baseline_wrong else None
        ),
        "agreement": agreement,
        "cohens_kappa": kappa,
        "mcnemar": mcnemar,
        "bootstrap": bootstrap,
    }


def _pair_perplexity(
    rows_a: list[dict],
    rows_b: list[dict],
    metric_key: str,
    *,
    seed: int = 42,
    iterations: int = 10_000,
) -> dict:
    map_a = _rows_by_key(rows_a, id_key=None, value_key="metric_value")
    map_b = _rows_by_key(rows_b, id_key=None, value_key="metric_value")
    keys_a = set(map_a)
    keys_b = set(map_b)
    shared = sorted(keys_a & keys_b, key=lambda value: str(value))
    denominator = max(len(map_a), len(map_b))
    if not shared:
        return {
            "n_a": len(map_a),
            "n_b": len(map_b),
            "n_paired": 0,
            "missing_in_a": len(keys_b - keys_a),
            "missing_in_b": len(keys_a - keys_b),
            "paired_coverage": 0.0 if denominator else None,
            "kind": "perplexity",
        }

    values_a = [float(map_a[key]["metric_value"]) for key in shared]
    values_b = [float(map_b[key]["metric_value"]) for key in shared]
    mean_a = sum(values_a) / len(values_a)
    mean_b = sum(values_b) / len(values_b)
    sorted_a = sorted(values_a)
    sorted_b = sorted(values_b)
    middle = len(values_a) // 2
    if len(values_a) % 2:
        median_a = sorted_a[middle]
        median_b = sorted_b[middle]
    else:
        median_a = (sorted_a[middle - 1] + sorted_a[middle]) / 2
        median_b = (sorted_b[middle - 1] + sorted_b[middle]) / 2
    relative_delta = (mean_b - mean_a) / mean_a if mean_a else None

    return {
        "n_a": len(map_a),
        "n_b": len(map_b),
        "n_paired": len(shared),
        "missing_in_a": len(keys_b - keys_a),
        "missing_in_b": len(keys_a - keys_b),
        "paired_coverage": len(shared) / denominator if denominator else None,
        "metric": metric_key,
        "mean_a": mean_a,
        "mean_b": mean_b,
        "median_a": median_a,
        "median_b": median_b,
        "delta": mean_b - mean_a,
        "rel_increase": relative_delta,
        "bootstrap": paired_bootstrap(
            values_a,
            values_b,
            statistic=_delta_mean,
            seed=seed,
            iterations=iterations,
        ),
        "kind": "perplexity",
    }


def _compare_task(
    dir_a: Path,
    dir_b: Path,
    task_name: str,
    cmp_cfg: CompareConfig,
    *,
    seed: int,
    iterations: int,
) -> dict:
    path_a = dir_a / "samples" / f"{task_name}.jsonl"
    path_b = dir_b / "samples" / f"{task_name}.jsonl"

    if task_name in cmp_cfg.perplexity_tasks:
        return _pair_perplexity(
            _load_jsonl(path_a),
            _load_jsonl(path_b),
            cmp_cfg.perplexity_metric,
            seed=seed,
            iterations=iterations,
        )

    return _pair_binary(
        _load_jsonl(path_a),
        _load_jsonl(path_b),
        seed=seed,
        iterations=iterations,
    )


def _compare_agentic(
    dir_a: Path,
    dir_b: Path,
    threshold: float,
    *,
    seed: int,
    iterations: int,
) -> dict | None:
    path_a = dir_a / "agentic_samples.jsonl"
    path_b = dir_b / "agentic_samples.jsonl"
    if not path_a.exists() or not path_b.exists():
        return None

    rows_a = _load_jsonl(path_a)
    rows_b = _load_jsonl(path_b)
    for r in rows_a:
        if "correct" not in r and "success" in r:
            r["correct"] = int(r["success"])
    for r in rows_b:
        if "correct" not in r and "success" in r:
            r["correct"] = int(r["success"])

    result = _pair_binary(
        rows_a,
        rows_b,
        id_key="task_id",
        correct_key="correct",
        seed=seed,
        iterations=iterations,
    )
    result["reward_threshold"] = threshold
    result["kind"] = "agentic"
    return result


def _micro_macro(task_results: dict[str, dict]) -> dict:
    scored = {
        k: v
        for k, v in task_results.items()
        if v.get("kind") != "perplexity" and v.get("n_paired", 0) > 0
    }
    if not scored:
        return {}

    total_n = sum(v["n_paired"] for v in scored.values())
    micro_flips = sum(
        v["regressions_a_correct_b_wrong"] + v["recoveries_a_wrong_b_correct"]
        for v in scored.values()
    )
    micro_regs = sum(v["regressions_a_correct_b_wrong"] for v in scored.values())
    micro_recs = sum(v["recoveries_a_wrong_b_correct"] for v in scored.values())

    macro_flip = sum(v["flip_rate"] for v in scored.values()) / len(scored)
    macro_delta = (
        sum(v["delta"] for v in scored.values() if v.get("delta") is not None)
        / len(scored)
    )

    return {
        "micro_flip_rate": micro_flips / total_n if total_n else None,
        "micro_regression_rate": micro_regs / total_n if total_n else None,
        "micro_recovery_rate": micro_recs / total_n if total_n else None,
        "macro_flip_rate": macro_flip,
        "macro_delta_acc": macro_delta,
        "tasks_compared": len(scored),
        "samples_paired": total_n,
    }


def compare_eval_dirs(
    dir_a: str | Path,
    dir_b: str | Path,
    out_dir: str | Path | None = None,
    cfg: PipelineConfig | None = None,
    label_a: str = "original",
    label_b: str = "quantized",
) -> dict:
    """Compare two evalsuite output directories (A=original, B=quantized)."""
    dir_a = Path(dir_a)
    dir_b = Path(dir_b)
    cmp_cfg = cfg.compare if cfg else CompareConfig()
    bootstrap_seed = cfg.eval.bootstrap_seed if cfg else 42
    bootstrap_iters = cfg.eval.bootstrap_iters if cfg else 10_000

    agg_a = _load_json(dir_a / "aggregate.json") or {}
    agg_b = _load_json(dir_b / "aggregate.json") or {}

    tasks = sorted(set(agg_a.keys()) | set(agg_b.keys()))
    task_results: dict[str, dict] = {}

    for task in tasks:
        sample_a = dir_a / "samples" / f"{task}.jsonl"
        sample_b = dir_b / "samples" / f"{task}.jsonl"
        if sample_a.exists() or sample_b.exists() or task in cmp_cfg.perplexity_tasks:
            task_results[task] = _compare_task(
                dir_a,
                dir_b,
                task,
                cmp_cfg,
                seed=bootstrap_seed,
                iterations=bootstrap_iters,
            )

    agentic = _compare_agentic(
        dir_a,
        dir_b,
        cmp_cfg.agentic_reward_threshold,
        seed=bootstrap_seed,
        iterations=bootstrap_iters,
    )

    report = {
        "label_a": label_a,
        "label_b": label_b,
        "dir_a": str(dir_a),
        "dir_b": str(dir_b),
        "aggregate_a": agg_a,
        "aggregate_b": agg_b,
        "tasks": task_results,
        "agentic": agentic,
        "summary": _micro_macro(task_results),
    }

    if out_dir:
        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        with (out_dir / "compare.json").open("w", encoding="utf-8") as fh:
            json.dump(report, fh, indent=2)

    return report
