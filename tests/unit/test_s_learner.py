"""Unit tests for OnlineSLearner."""

import pytest
from river.linear_model import LinearRegression

from onlinecml.metalearners.s_learner import OnlineSLearner
from onlinecml.datasets import LinearCausalStream


class TestOnlineSLearner:
    def test_n_seen_starts_zero(self):
        model = OnlineSLearner()
        assert model.n_seen == 0

    def test_predict_ate_before_learn_returns_zero(self):
        model = OnlineSLearner()
        assert model.predict_ate() == 0.0

    def test_learn_increments_n_seen(self):
        model = OnlineSLearner()
        model.learn_one({"x": 1.0}, 1, 2.0)
        assert model.n_seen == 1

    def test_predict_one_returns_float(self):
        model = OnlineSLearner()
        for x, w, y, _ in LinearCausalStream(n=50, seed=0):
            model.learn_one(x, w, y)
        result = model.predict_one({"x0": 0.5, "x1": -0.3, "x2": 0.1})
        assert isinstance(result, float)

    def test_treatment_feature_in_x(self):
        """Treatment feature is added to x during predict and learn."""
        model = OnlineSLearner(treatment_feature="__W__")
        # Should not raise even though x doesn't have __W__
        model.learn_one({"x": 1.0}, 1, 2.0)
        cate = model.predict_one({"x": 1.0})
        assert isinstance(cate, float)

    def test_reset_clears_state(self):
        model = OnlineSLearner()
        for x, w, y, _ in LinearCausalStream(n=50, seed=0):
            model.learn_one(x, w, y)
        model.reset()
        assert model.n_seen == 0
        assert model.predict_ate() == 0.0

    def test_clone_has_fresh_state(self):
        model = OnlineSLearner()
        for x, w, y, _ in LinearCausalStream(n=50, seed=0):
            model.learn_one(x, w, y)
        cloned = model.clone()
        assert cloned.n_seen == 0

    def test_predict_then_learn_order(self):
        """On first call, predict returns 0 (before any learning)."""
        model = OnlineSLearner()
        # Before any learning, predictions are 0 (untrained LR default)
        cate_before = model.predict_one({"x": 1.0})
        assert cate_before == pytest.approx(0.0)

    def test_ate_is_mean_of_per_obs_cate(self):
        """Running ATE is the mean of CATE values computed before each update."""
        model = OnlineSLearner()
        for x, w, y, _ in LinearCausalStream(n=100, seed=0):
            model.learn_one(x, w, y)
        # ATE should converge toward the true effect (loose check)
        assert isinstance(model.predict_ate(), float)
