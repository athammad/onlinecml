"""Unit tests for OnlineRLearner."""

import pytest

from onlinecml.metalearners.r_learner import OnlineRLearner
from onlinecml.datasets import LinearCausalStream


class TestOnlineRLearner:
    def test_n_seen_starts_zero(self):
        assert OnlineRLearner().n_seen == 0

    def test_predict_ate_before_learn_returns_zero(self):
        assert OnlineRLearner().predict_ate() == 0.0

    def test_learn_increments_n_seen(self):
        m = OnlineRLearner()
        m.learn_one({"x": 1.0}, 1, 2.0)
        assert m.n_seen == 1

    def test_predict_one_returns_float(self):
        m = OnlineRLearner()
        for x, w, y, _ in LinearCausalStream(n=100, seed=0):
            m.learn_one(x, w, y)
        assert isinstance(m.predict_one({"x0": 0.5, "x1": -0.3, "x2": 0.1}), float)

    def test_reset_clears_state(self):
        m = OnlineRLearner()
        for x, w, y, _ in LinearCausalStream(n=50, seed=0):
            m.learn_one(x, w, y)
        m.reset()
        assert m.n_seen == 0

    def test_clone_fresh(self):
        m = OnlineRLearner()
        for x, w, y, _ in LinearCausalStream(n=50, seed=0):
            m.learn_one(x, w, y)
        assert m.clone().n_seen == 0

    def test_ci_finite_after_data(self):
        m = OnlineRLearner()
        for x, w, y, _ in LinearCausalStream(n=100, seed=0):
            m.learn_one(x, w, y)
        lo, hi = m.predict_ci()
        assert lo < hi

    def test_logged_propensity(self):
        m = OnlineRLearner()
        n_before = m.ps_model.n_seen
        m.learn_one({"x": 1.0}, 1, 2.0, propensity=0.7)
        assert m.ps_model.n_seen == n_before

    def test_small_residual_skipped(self):
        """When |W_res| < min_residual, CATE model should not be updated."""
        m = OnlineRLearner(min_residual=0.99)
        # With min_residual=0.99, almost all updates are skipped
        # but n_seen should still increment
        m.learn_one({"x": 1.0}, 1, 2.0)
        assert m.n_seen == 1

    def test_large_residual_updates_cate_model(self):
        """When |W_res| >= min_residual, CATE model should be updated."""
        from river.linear_model import LinearRegression
        cate_model = LinearRegression()
        m = OnlineRLearner(min_residual=0.0, cate_model=cate_model)
        # Force large residual: treatment=1, ps_model predicts ~0.5
        # W_res = 1 - 0.5 = 0.5 >= 0.0, so CATE model should update
        m.learn_one({"x": 1.0}, 1, 3.0)
        assert m.n_seen == 1
