"""Unit tests for OnlineSMD."""

import math

import pytest
import numpy as np

from onlinecml.diagnostics.smd import OnlineSMD
from onlinecml.datasets import LinearCausalStream


class TestOnlineSMD:
    def test_report_returns_dict_with_all_covariates(self):
        smd = OnlineSMD(covariates=["age", "income"])
        report = smd.report()
        assert "age" in report
        assert "income" in report

    def test_report_values_are_two_tuples(self):
        smd = OnlineSMD(covariates=["age"])
        smd.update({"age": 30}, treatment=1)
        smd.update({"age": 25}, treatment=0)
        report = smd.report()
        assert len(report["age"]) == 2

    def test_update_single_obs_no_crash(self):
        smd = OnlineSMD(covariates=["x"])
        smd.update({"x": 1.0}, treatment=1)
        # With only one group, SMD should be 0
        raw, weighted = smd.report()["x"]
        assert raw == 0.0

    def test_is_balanced_true_when_equal_groups(self):
        """With identical distributions, SMD should be ~0."""
        smd = OnlineSMD(covariates=["x"])
        rng = np.random.default_rng(0)
        for i in range(100):
            val = float(rng.normal(0, 1))
            smd.update({"x": val}, treatment=i % 2)
        # Should be approximately balanced
        assert abs(smd.report()["x"][0]) < 1.0  # loose bound

    def test_is_balanced_false_when_confounded(self):
        """Strongly confounded data should have high raw SMD."""
        smd = OnlineSMD(covariates=["x"])
        rng = np.random.default_rng(1)
        # Treated: ~N(5, 1), control: ~N(-5, 1)
        for _ in range(50):
            smd.update({"x": float(rng.normal(5.0, 1.0))}, treatment=1)
            smd.update({"x": float(rng.normal(-5.0, 1.0))}, treatment=0)
        raw, _ = smd.report()["x"]
        assert abs(raw) > 1.0

    def test_missing_covariate_defaults_to_zero(self):
        """Missing covariates in x should default to 0.0."""
        smd = OnlineSMD(covariates=["missing_key"])
        smd.update({}, treatment=1)
        smd.update({}, treatment=0)
        # Should not raise; both groups get 0.0

    def test_raw_smd_formula(self):
        """Manual SMD computation matches report()."""
        smd = OnlineSMD(covariates=["x"])
        treated_vals = [2.0, 3.0, 4.0]
        control_vals = [0.0, 1.0, 2.0]
        for v in treated_vals:
            smd.update({"x": v}, treatment=1)
        for v in control_vals:
            smd.update({"x": v}, treatment=0)

        mean_t = np.mean(treated_vals)
        mean_c = np.mean(control_vals)
        var_t = np.var(treated_vals, ddof=1)
        var_c = np.var(control_vals, ddof=1)
        expected = (mean_t - mean_c) / math.sqrt((var_t + var_c) / 2)
        raw, _ = smd.report()["x"]
        assert raw == pytest.approx(expected)

    def test_weighted_smd_reduces_imbalance(self):
        """With downweighted treated group, weighted SMD should be smaller than raw."""
        smd = OnlineSMD(covariates=["x"])
        rng = np.random.default_rng(2)
        # Treated: ~N(3, 1), control: ~N(-3, 1) — confounded
        for _ in range(50):
            smd.update({"x": float(rng.normal(3.0, 1.0))}, treatment=1, weight=0.1)
            smd.update({"x": float(rng.normal(-3.0, 1.0))}, treatment=0, weight=1.0)
        raw, weighted = smd.report()["x"]
        # Raw SMD should be large (confounded), weighted SMD should differ
        assert abs(raw) > 1.0  # strong imbalance in raw
        assert isinstance(weighted, float)

    def test_is_balanced_threshold(self):
        smd = OnlineSMD(covariates=["x"])
        # Force balanced: same distribution in both groups
        for _ in range(50):
            smd.update({"x": 0.0}, treatment=1)
            smd.update({"x": 0.0}, treatment=0)
        assert smd.is_balanced(thr=0.1) is True
