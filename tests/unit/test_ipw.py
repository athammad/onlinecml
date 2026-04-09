"""Unit tests for OnlineIPW."""

import math

import pytest

from onlinecml.reweighting.ipw import OnlineIPW
from onlinecml.datasets import LinearCausalStream


class TestOnlineIPW:
    def test_n_seen_starts_zero(self):
        est = OnlineIPW()
        assert est.n_seen == 0

    def test_predict_ate_before_learn_returns_zero(self):
        est = OnlineIPW()
        assert est.predict_ate() == 0.0

    def test_learn_increments_n_seen(self):
        est = OnlineIPW()
        est.learn_one({"x": 1.0}, 1, 2.0)
        assert est.n_seen == 1

    def test_predict_ate_is_float(self):
        est = OnlineIPW()
        for x, w, y, _ in LinearCausalStream(n=50, seed=0):
            est.learn_one(x, w, y)
        assert isinstance(est.predict_ate(), float)

    def test_ate_direction_positive(self):
        """ATE should be positive for a positive true_ate."""
        est = OnlineIPW()
        for x, w, y, _ in LinearCausalStream(n=500, true_ate=5.0, seed=42):
            est.learn_one(x, w, y)
        assert est.predict_ate() > 0

    def test_predict_ci_returns_tuple_of_two_floats(self):
        est = OnlineIPW()
        for x, w, y, _ in LinearCausalStream(n=50, seed=0):
            est.learn_one(x, w, y)
        lo, hi = est.predict_ci()
        assert isinstance(lo, float)
        assert isinstance(hi, float)

    def test_ci_lower_less_than_upper(self):
        est = OnlineIPW()
        for x, w, y, _ in LinearCausalStream(n=100, seed=0):
            est.learn_one(x, w, y)
        lo, hi = est.predict_ci()
        assert lo < hi

    def test_ci_is_infinite_before_two_obs(self):
        est = OnlineIPW()
        est.learn_one({"x": 1.0}, 1, 2.0)
        lo, hi = est.predict_ci()
        assert lo == float("-inf")
        assert hi == float("inf")

    def test_reset_clears_ate(self):
        est = OnlineIPW()
        for x, w, y, _ in LinearCausalStream(n=50, seed=0):
            est.learn_one(x, w, y)
        est.reset()
        assert est.n_seen == 0
        assert est.predict_ate() == 0.0

    def test_clone_has_fresh_state(self):
        est = OnlineIPW()
        for x, w, y, _ in LinearCausalStream(n=50, seed=0):
            est.learn_one(x, w, y)
        cloned = est.clone()
        assert cloned.n_seen == 0
        assert cloned.predict_ate() == 0.0

    def test_weight_stats_has_expected_keys(self):
        est = OnlineIPW()
        est.learn_one({"x": 1.0}, 1, 2.0)
        ws = est.weight_stats
        assert "treated_mean" in ws
        assert "control_mean" in ws
        assert "n" in ws

    def test_logged_propensity_used(self):
        """When propensity is passed, the internal ps_model should not update."""
        est = OnlineIPW()
        n_before = est.ps_model.n_seen
        est.learn_one({"x": 1.0}, 1, 2.0, propensity=0.6)
        # Internal model should NOT have been updated since we passed propensity
        assert est.ps_model.n_seen == n_before

    def test_predict_one_returns_ate(self):
        """IPW predict_one always returns the running ATE."""
        est = OnlineIPW()
        for x, w, y, _ in LinearCausalStream(n=50, seed=0):
            est.learn_one(x, w, y)
        ate = est.predict_ate()
        assert est.predict_one({"x": 999.0}) == ate

    def test_normalize_flag(self):
        """Normalized IPW should run without errors."""
        est = OnlineIPW(normalize=True)
        for x, w, y, _ in LinearCausalStream(n=100, seed=0):
            est.learn_one(x, w, y)
        assert isinstance(est.predict_ate(), float)
        assert est.n_seen == 100
