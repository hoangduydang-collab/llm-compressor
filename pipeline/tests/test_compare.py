"""Unit tests for evalsuite comparison metrics."""

import math

import pytest

from pipeline.config import PipelineConfig
from pipeline.evalsuite.compare import (
    chi2_sf_df1,
    cohens_kappa,
    compare_eval_dirs,
    mcnemar_test,
    _pair_binary,
    _pair_perplexity,
)


class TestMcNemar:
    def test_symmetric_flips_high_p(self):
        r = mcnemar_test(5, 5)
        assert r["discordant"] == 10
        assert r["p_value"] > 0.5

    def test_asymmetric_low_p(self):
        r = mcnemar_test(20, 2)
        assert r["p_value"] < 0.01

    def test_no_discordant(self):
        r = mcnemar_test(0, 0)
        assert r["p_value"] == 1.0


class TestCohensKappa:
    def test_perfect_agreement(self):
        k = cohens_kappa(10, 0, 0, 0)
        assert k == pytest.approx(1.0)

    def test_chance_agreement(self):
        # 50/50 split on both sides -> kappa ~ 0
        k = cohens_kappa(5, 5, 5, 5)
        assert k == pytest.approx(0.0)


class TestPairBinary:
    def _rows(self, correct_by_id: dict[int, int]) -> list[dict]:
        return [{"doc_id": k, "correct": v} for k, v in correct_by_id.items()]

    def test_identical_zero_flips(self):
        ids = {i: 1 if i % 2 == 0 else 0 for i in range(10)}
        r = _pair_binary(self._rows(ids), self._rows(ids))
        assert r["flip_rate"] == 0.0
        assert r["cohens_kappa"] == pytest.approx(1.0)
        assert r["regressions_a_correct_b_wrong"] == 0
        assert r["recoveries_a_wrong_b_correct"] == 0

    def test_all_flip(self):
        a = {i: 1 for i in range(4)}
        b = {i: 0 for i in range(4)}
        r = _pair_binary(self._rows(a), self._rows(b))
        assert r["flip_rate"] == 1.0
        assert r["regressions_a_correct_b_wrong"] == 4
        assert r["recoveries_a_wrong_b_correct"] == 0
        assert r["acc_a"] == 1.0
        assert r["acc_b"] == 0.0

    def test_mixed_flips(self):
        a = {0: 1, 1: 1, 2: 0, 3: 0}
        b = {0: 1, 1: 0, 2: 1, 3: 0}
        r = _pair_binary(self._rows(a), self._rows(b))
        assert r["both_correct"] == 1
        assert r["both_wrong"] == 1
        assert r["regressions_a_correct_b_wrong"] == 1
        assert r["recoveries_a_wrong_b_correct"] == 1
        assert r["flip_rate"] == 0.5

    def test_empty_overlap(self):
        a = self._rows({0: 1})
        b = self._rows({1: 0})
        r = _pair_binary(a, b)
        assert r["n_paired"] == 0
        assert r["flip_rate"] is None


    def test_reports_quantization_conditionals_and_recovery(self):
        a = {"a": 1, "b": 1, "c": 0, "d": 0}
        b = {"a": 1, "b": 0, "c": 1, "d": 0}

        r = _pair_binary(self._rows(a), self._rows(b), seed=7, iterations=200)

        assert r["conditional_regression_rate"] == 0.5
        assert r["conditional_recovery_rate"] == 0.5
        assert r["net_harmful_flips"] == 0
        assert r["score_recovery_ratio"] == 1.0
        assert r["bootstrap"]["accuracy_delta"]["iterations"] == 200

    def test_reports_unpaired_coverage(self):
        r = _pair_binary(self._rows({0: 1, 1: 0}), self._rows({1: 0, 2: 1}))
        assert r["n_a"] == 2
        assert r["n_b"] == 2
        assert r["missing_in_a"] == 1
        assert r["missing_in_b"] == 1
        assert r["paired_coverage"] == 0.5

    def test_rejects_duplicate_stable_ids(self):
        rows_a = [
            {"sample_uid": "same", "doc_id": 0, "correct": 1},
            {"sample_uid": "same", "doc_id": 1, "correct": 0},
        ]
        with pytest.raises(ValueError, match="duplicate"):
            _pair_binary(rows_a, [{"sample_uid": "same", "correct": 1}])


class TestPairPerplexity:
    def test_reports_paired_drift_and_bootstrap(self):
        rows_a = [
            {"sample_uid": "a", "doc_id": 0, "metric_value": 2.0},
            {"sample_uid": "b", "doc_id": 1, "metric_value": 4.0},
        ]
        rows_b = [
            {"sample_uid": "a", "doc_id": 8, "metric_value": 3.0},
            {"sample_uid": "b", "doc_id": 9, "metric_value": 5.0},
        ]

        result = _pair_perplexity(
            rows_a,
            rows_b,
            "word_perplexity",
            seed=3,
            iterations=25,
        )

        assert result["n_paired"] == 2
        assert result["mean_a"] == 3.0
        assert result["mean_b"] == 4.0
        assert result["delta"] == 1.0
        assert result["bootstrap"]["iterations"] == 25
        assert result["paired_coverage"] == 1.0


class TestCompareEvalDirs:
    def test_end_to_end_self_compare(self, tmp_path):
        import json

        for name in ("original", "quant"):
            d = tmp_path / name
            samples = d / "samples"
            samples.mkdir(parents=True)
            rows = [
                {"doc_id": 0, "correct": 1, "metric_value": 1.0},
                {"doc_id": 1, "correct": 0, "metric_value": 0.0},
                {"doc_id": 2, "correct": 1, "metric_value": 1.0},
            ]
            with (samples / "mmlu.jsonl").open("w") as fh:
                for row in rows:
                    fh.write(json.dumps(row) + "\n")
            with (d / "aggregate.json").open("w") as fh:
                json.dump({"mmlu": {"acc,none": 0.67}}, fh)

        out = tmp_path / "compare"
        cfg = PipelineConfig()
        cfg.eval.bootstrap_seed = 9
        cfg.eval.bootstrap_iters = 17
        report = compare_eval_dirs(
            tmp_path / "original",
            tmp_path / "quant",
            out_dir=out,
            cfg=cfg,
            label_a="original",
            label_b="quant",
        )

        assert report["tasks"]["mmlu"]["flip_rate"] == 0.0
        assert (
            report["tasks"]["mmlu"]["bootstrap"]["accuracy_delta"]["iterations"]
            == 17
        )
        assert (
            report["tasks"]["mmlu"]["bootstrap"]["accuracy_delta"]["seed"]
            == 9
        )
        assert report["summary"]["micro_flip_rate"] == 0.0
        assert (out / "compare.json").exists()


    def test_candidate_only_samples_are_not_silently_skipped(self, tmp_path):
        import json

        original = tmp_path / "original"
        candidate = tmp_path / "candidate"
        (original / "samples").mkdir(parents=True)
        (candidate / "samples").mkdir(parents=True)
        (original / "aggregate.json").write_text(
            json.dumps({"mmlu": {"acc,none": 0.0}}), encoding="utf-8"
        )
        (candidate / "aggregate.json").write_text(
            json.dumps({"mmlu": {"acc,none": 1.0}}), encoding="utf-8"
        )
        (candidate / "samples" / "mmlu.jsonl").write_text(
            json.dumps({"sample_uid": "x", "correct": 1}) + "\n",
            encoding="utf-8",
        )

        report = compare_eval_dirs(original, candidate)

        assert report["tasks"]["mmlu"]["n_a"] == 0
        assert report["tasks"]["mmlu"]["n_b"] == 1
        assert report["tasks"]["mmlu"]["missing_in_a"] == 1


class TestChi2:
    def test_monotone(self):
        assert chi2_sf_df1(0.0) == 1.0
        assert chi2_sf_df1(10.0) < chi2_sf_df1(1.0)
        assert chi2_sf_df1(10.0) < 0.01
