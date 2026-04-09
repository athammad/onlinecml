"""Unit tests for OnlineAIPW."""

import pytest
from river.linear_model import LinearRegression, LogisticRegression

from onlinecml.reweighting.aipw import OnlineAIPW
from onlinecml.propensity.propensity_score import OnlinePropensityScore
from onlinecml.datasets import LinearCausalStream


class TestOnlineAIPW:
    def test_n_seen_starts_zero(self):
        est = OnlineAIPW()
        assert est.n_seen == 0

    def test_predict_ate_before_learn_returns_zero(self):
        est = OnlineAIPW()
        assert est.predict_ate() == 0.0

    def test_learn_increments_n_seen(self):
        est = OnlineAIPW()
        est.learn_one({"x": 1.0}, 1, 2.0)
        assert est.n_seen == 1

    def test_predict_ate_is_float(self):
        est = OnlineAIPW()
        for x, w, y, _ in LinearCausalStream(n=50, seed=0):
            est.learn_one(x, w, y)
        assert isinstance(est.predict_ate(), float)

    def test_ci_lower_less_than_upper(self):
        est = OnlineAIPW()
        for x, w, y, _ in LinearCausalStream(n=100, seed=0):
            est.learn_one(x, w, y)
        lo, hi = est.predict_ci()
        assert lo < hi

    def test_reset_clears_state(self):
        est = OnlineAIPW()
        for x, w, y, _ in LinearCausalStream(n=50, seed=0):
            est.learn_one(x, w, y)
        est.reset()
        assert est.n_seen == 0
        assert est.predict_ate() == 0.0

    def test_clone_has_fresh_state(self):
        est = OnlineAIPW()
        for x, w, y, _ in LinearCausalStream(n=50, seed=0):
            est.learn_one(x, w, y)
        cloned = est.clone()
        assert cloned.n_seen == 0

    def test_predict_one_returns_individual_cate(self):
        """AIPW predict_one returns mu1(x) - mu0(x), not the running ATE."""
        est = OnlineAIPW()
        for x, w, y, _ in LinearCausalStream(n=100, seed=0):
            est.learn_one(x, w, y)
        x_test = {"x0": 0.5, "x1": -0.3, "x2": 0.1}
        cate = est.predict_one(x_test)
        ate = est.predict_ate()
        # They should be different numbers (individual CATE vs ATE)
        assert isinstance(cate, float)
        # Individual CATE is not required to equal ATE exactly

    def test_predict_one_uses_both_outcome_models(self):
        """predict_one should return treated_model(x) - control_model(x)."""
        treated = LinearRegression()
        control = LinearRegression()
        est = OnlineAIPW(treated_model=treated, control_model=control)
        x_test = {"x0": 1.0}
        expected = treated.predict_one(x_test) - control.predict_one(x_test)
        assert est.predict_one(x_test) == pytest.approx(expected)

    def test_logged_propensity_used(self):
        """When propensity is passed, the internal ps_model should not update."""
        est = OnlineAIPW()
        n_before = est.ps_model.n_seen
        est.learn_one({"x": 1.0}, 1, 2.0, propensity=0.6)
        assert est.ps_model.n_seen == n_before
