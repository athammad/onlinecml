"""Unit tests for OnlineXLearner."""

import pytest

from onlinecml.metalearners.x_learner import OnlineXLearner
from onlinecml.datasets import LinearCausalStream


class TestOnlineXLearner:
    def test_n_seen_starts_zero(self):
        assert OnlineXLearner().n_seen == 0

    def test_predict_ate_before_learn_returns_zero(self):
        assert OnlineXLearner().predict_ate() == 0.0

    def test_learn_increments_n_seen(self):
        m = OnlineXLearner()
        m.learn_one({"x": 1.0}, 1, 2.0)
        assert m.n_seen == 1

    def test_predict_one_returns_float(self):
        m = OnlineXLearner()
        for x, w, y, _ in LinearCausalStream(n=100, seed=0):
            m.learn_one(x, w, y)
        assert isinstance(m.predict_one({"x0": 0.5, "x1": -0.3, "x2": 0.1}), float)

    def test_reset_clears_state(self):
        m = OnlineXLearner()
        for x, w, y, _ in LinearCausalStream(n=50, seed=0):
            m.learn_one(x, w, y)
        m.reset()
        assert m.n_seen == 0

    def test_clone_fresh(self):
        m = OnlineXLearner()
        for x, w, y, _ in LinearCausalStream(n=50, seed=0):
            m.learn_one(x, w, y)
        assert m.clone().n_seen == 0

    def test_ci_finite_after_data(self):
        m = OnlineXLearner()
        for x, w, y, _ in LinearCausalStream(n=100, seed=0):
            m.learn_one(x, w, y)
        lo, hi = m.predict_ci()
        assert lo < hi

    def test_logged_propensity_skips_ps_model(self):
        m = OnlineXLearner()
        n_before = m.ps_model.n_seen
        m.learn_one({"x": 1.0}, 1, 2.0, propensity=0.6)
        assert m.ps_model.n_seen == n_before
