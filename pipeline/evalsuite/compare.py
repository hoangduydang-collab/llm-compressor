"""Post-hoc quantized-vs-original comparison with flip-rate metrics."""

from __future__ import annotations

import json
import math
from pathlib import Path

from pipeline.config import CompareConfig, PipelineConfig


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
    """McNemar with continuity correction (regressions=A ok B fail, recoveries=A fail B ok)."""
    b, c = regressions, recoveries
    n_discordant = b + c
    if n_discordant == 0:
        return {"statistic": 0.0, "p_value": 1.0, "discordant": 0}
    stat = (abs(b - c) - 1) ** 2 / n_discordant
    return {"statistic": stat, "p_value": chi2_sf_df1(stat), "discordant": n_discordant}


def cohens_kappa(both_correct: int, both_wrong: int, regressions: int, recoveries: int) -> float | None:
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


def _pair_binary(
    rows_a: list[dict],
    rows_b: list[dict],
    id_key: str = "doc_id",
    correct_key: str = "correct",
) -> dict:
    map_a = {r[id_key]: r for r in rows_a if r.get(correct_key) is not None}
    map_b = {r[id_key]: r for r in rows_b if r.get(correct_key) is not None}
    shared = sorted(set(map_a) & set(map_b), key=lambda x: str(x))

    both_correct = both_wrong = regressions = recoveries = 0
    for key in shared:
        ca = int(map_a[key][correct_key])
        cb = int(map_b[key][correct_key])
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
    flip_rate = flips / n if n else None

    mcnemar = mcnemar_test(regressions, recoveries)
    kappa = cohens_kappa(both_correct, both_wrong, regressions, recoveries)
    agreement = (both_correct + both_wrong) / n if n else None

    return {
        "n_paired": n,
        "acc_a": acc_a,
        "acc_b": acc_b,
        "delta": (acc_b - acc_a) if acc_a is not None and acc_b is not None else None,
        "both_correct": both_correct,
        "both_wrong": both_wrong,
        "regressions_a_correct_b_wrong": regressions,
        "recoveries_a_wrong_b_correct": recoveries,
        "flip_rate": flip_rate,
        "regression_rate": regressions / n if n else None,
        "recovery_rate": recoveries / n if n else None,
        "agreement": agreement,
        "cohens_kappa": kappa,
        "mcnemar": mcnemar,
    }


def _pair_perplexity(rows_a: list[dict], rows_b: list[dict], metric_key: str) -> dict:
    map_a = {r["doc_id"]: r for r in rows_a if r.get("metric_value") is not None}
    map_b = {r["doc_id"]: r for r in rows_b if r.get("metric_value") is not None}
    shared = sorted(set(map_a) & set(map_b), key=lambda x: str(x))

    vals_a = [float(map_a[k]["metric_value"]) for k in shared]
    vals_b = [float(map_b[k]["metric_value"]) for k in shared]
    if not shared:
        return {"n_paired": 0}

    mean_a = sum(vals_a) / len(vals_a)
    mean_b = sum(vals_b) / len(vals_b)
    rel_increase = (mean_b - mean_a) / mean_a if mean_a else None

    return {
        "n_paired": len(shared),
        "metric": metric_key,
        "mean_a": mean_a,
        "mean_b": mean_b,
        "delta": mean_b - mean_a,
        "rel_increase": rel_increase,
        "kind": "perplexity",
    }


def _compare_task(
    dir_a: Path,
    dir_b: Path,
    task_name: str,
    cmp_cfg: CompareConfig,
) -> dict:
    path_a = dir_a / "samples" / f"{task_name}.jsonl"
    path_b = dir_b / "samples" / f"{task_name}.jsonl"

    if task_name in cmp_cfg.perplexity_tasks:
        return _pair_perplexity(
            _load_jsonl(path_a),
            _load_jsonl(path_b),
            cmp_cfg.perplexity_metric,
        )

    return _pair_binary(_load_jsonl(path_a), _load_jsonl(path_b))


def _compare_agentic(dir_a: Path, dir_b: Path, threshold: float) -> dict | None:
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

    result = _pair_binary(rows_a, rows_b, id_key="task_id", correct_key="correct")
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
    macro_delta = sum(v["delta"] for v in scored.values() if v.get("delta") is not None) / len(scored)

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

    agg_a = _load_json(dir_a / "aggregate.json") or {}
    agg_b = _load_json(dir_b / "aggregate.json") or {}

    tasks = sorted(set(agg_a.keys()) | set(agg_b.keys()))
    task_results: dict[str, dict] = {}

    for task in tasks:
        if (dir_a / "samples" / f"{task}.jsonl").exists() or task in cmp_cfg.perplexity_tasks:
            task_results[task] = _compare_task(dir_a, dir_b, task, cmp_cfg)

    agentic = _compare_agentic(dir_a, dir_b, cmp_cfg.agentic_reward_threshold)

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
