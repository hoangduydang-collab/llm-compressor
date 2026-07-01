"""Unit tests for evalsuite comparison metrics."""

import math

import pytest

from pipeline.evalsuite.compare import (
    chi2_sf_df1,
    cohens_kappa,
    compare_eval_dirs,
    mcnemar_test,
    _pair_binary,
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
        report = compare_eval_dirs(
            tmp_path / "original",
            tmp_path / "quant",
            out_dir=out,
            label_a="original",
            label_b="quant",
        )

        assert report["tasks"]["mmlu"]["flip_rate"] == 0.0
        assert report["summary"]["micro_flip_rate"] == 0.0
        assert (out / "compare.json").exists()


class TestChi2:
    def test_monotone(self):
        assert chi2_sf_df1(0.0) == 1.0
        assert chi2_sf_df1(10.0) < chi2_sf_df1(1.0)
        assert chi2_sf_df1(10.0) < 0.01
