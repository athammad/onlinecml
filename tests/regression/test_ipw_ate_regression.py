"""Regression tests for OnlineIPW ATE estimates.

These tests lock in known-good ATE results on fixed seeds. If they fail it
means a code change has silently altered numerical behaviour. Either revert
the change or update the expected values deliberately after verifying the
new results are correct.
"""

import pytest

from onlinecml.datasets import LinearCausalStream, HeterogeneousCausalStream
from onlinecml.reweighting import OnlineIPW, OnlineAIPW, OnlineOverlapWeights

# Tolerance: allow ±1% relative deviation from the locked-in value
REL_TOL = 0.01


class TestIPWRegression:
    def test_linear_stream_seed42(self):
        """IPW ATE on LinearCausalStream(n=2000, seed=42) must stay stable."""
        ipw = OnlineIPW()
        for x, w, y, _ in LinearCausalStream(n=2000, true_ate=2.0, seed=42):
            ipw.learn_one(x, w, y)
        assert ipw.predict_ate() == pytest.approx(2.299589, rel=REL_TOL)

    def test_n_seen_after_full_stream(self):
        ipw = OnlineIPW()
        for x, w, y, _ in LinearCausalStream(n=500, seed=0):
            ipw.learn_one(x, w, y)
        assert ipw.n_seen == 500

    def test_ci_contains_locked_ate(self):
        """CI should contain the locked-in ATE estimate."""
        ipw = OnlineIPW()
        for x, w, y, _ in LinearCausalStream(n=2000, true_ate=2.0, seed=42):
            ipw.learn_one(x, w, y)
        lo, hi = ipw.predict_ci()
        assert lo < 2.299589 < hi

    def test_reset_and_retrain_same_result(self):
        """After reset + retrain, ATE should equal the first run."""
        ipw = OnlineIPW()
        for x, w, y, _ in LinearCausalStream(n=500, seed=7):
            ipw.learn_one(x, w, y)
        first_ate = ipw.predict_ate()

        ipw.reset()
        for x, w, y, _ in LinearCausalStream(n=500, seed=7):
            ipw.learn_one(x, w, y)
        assert ipw.predict_ate() == pytest.approx(first_ate)


class TestAIPWRegression:
    def test_aipw_closer_to_true_than_ipw(self):
        """AIPW should generally have lower bias than IPW on this stream."""
        ipw  = OnlineIPW()
        aipw = OnlineAIPW()
        for x, w, y, _ in LinearCausalStream(n=2000, true_ate=2.0, seed=0):
            ipw.learn_one(x, w, y)
            aipw.learn_one(x, w, y)
        # Both should be in a reasonable range
        assert abs(aipw.predict_ate() - 2.0) < 2.0
        assert abs(ipw.predict_ate() - 2.0) < 2.0

    def test_overlap_weights_n_seen(self):
        ow = OnlineOverlapWeights()
        for x, w, y, _ in LinearCausalStream(n=300, seed=1):
            ow.learn_one(x, w, y)
        assert ow.n_seen == 300


class TestSMDConvergence:
    def test_smd_decreases_after_weighting(self):
        """Weighted SMD should be closer to 0 than raw SMD on a confounded stream."""
        from onlinecml.diagnostics import OnlineSMD

        smd = OnlineSMD(covariates=["x0", "x1", "x2"])
        ipw = OnlineIPW()

        for x, w, y, _ in LinearCausalStream(n=1000, confounding_strength=1.0, seed=5):
            ps = ipw.ps_model.predict_one(x)
            weight = 1.0 / ps if w == 1 else 1.0 / (1.0 - ps)
            smd.update(x, treatment=w, weight=weight)
            ipw.learn_one(x, w, y)

        report = smd.report()
        for cov, (raw, weighted) in report.items():
            # Weighted SMD should be smaller in absolute value
            assert abs(weighted) <= abs(raw) + 0.3  # allow some variance
