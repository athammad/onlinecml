"""Unit tests for OnlineTLearner."""

import warnings

import pytest
from river.linear_model import LinearRegression

from onlinecml.metalearners.t_learner import OnlineTLearner
from onlinecml.datasets import LinearCausalStream


class TestOnlineTLearner:
    def test_n_seen_starts_zero(self):
        model = OnlineTLearner()
        assert model.n_seen == 0

    def test_predict_ate_before_learn_returns_zero(self):
        model = OnlineTLearner()
        assert model.predict_ate() == 0.0

    def test_learn_increments_n_seen(self):
        model = OnlineTLearner()
        model.learn_one({"x": 1.0}, 1, 2.0)
        assert model.n_seen == 1

    def test_predict_one_returns_float(self):
        model = OnlineTLearner()
        for x, w, y, _ in LinearCausalStream(n=100, seed=0):
            model.learn_one(x, w, y)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            result = model.predict_one({"x0": 0.5, "x1": -0.3, "x2": 0.1})
        assert isinstance(result, float)

    def test_reset_clears_state(self):
        model = OnlineTLearner()
        for x, w, y, _ in LinearCausalStream(n=50, seed=0):
            model.learn_one(x, w, y)
        model.reset()
        assert model.n_seen == 0
        assert model.predict_ate() == 0.0

    def test_clone_has_fresh_state(self):
        model = OnlineTLearner()
        for x, w, y, _ in LinearCausalStream(n=50, seed=0):
            model.learn_one(x, w, y)
        cloned = model.clone()
        assert cloned.n_seen == 0

    def test_treated_model_updated_for_treated(self):
        """Only treated_model is updated when treatment==1."""
        treated = LinearRegression()
        control = LinearRegression()
        model = OnlineTLearner(treated_model=treated, control_model=control)
        model.learn_one({"x": 1.0}, 1, 3.0)
        # After one treated obs, treated model has seen data
        assert model._n_treated == 1
        assert model._n_control == 0

    def test_control_model_updated_for_control(self):
        """Only control_model is updated when treatment==0."""
        model = OnlineTLearner()
        model.learn_one({"x": 1.0}, 0, 1.0)
        assert model._n_treated == 0
        assert model._n_control == 1

    def test_warn_when_treated_arm_unseen(self):
        model = OnlineTLearner()
        # Only control obs seen — treated arm model has no data
        model.learn_one({"x": 1.0}, 0, 1.0)
        with pytest.warns(UserWarning, match="treated_model"):
            model.predict_one({"x": 1.0})

    def test_warn_when_control_arm_unseen(self):
        model = OnlineTLearner()
        model.learn_one({"x": 1.0}, 1, 2.0)
        with pytest.warns(UserWarning, match="control_model"):
            model.predict_one({"x": 1.0})

    def test_ci_lower_less_than_upper(self):
        model = OnlineTLearner()
        for x, w, y, _ in LinearCausalStream(n=100, seed=0):
            model.learn_one(x, w, y)
        lo, hi = model.predict_ci()
        assert lo < hi

    def test_propensity_weighting_via_learn_one(self):
        """Passing propensity should route sample weight to the model."""
        model = OnlineTLearner()
        # Just ensure no exception; River LR may ignore w silently
        model.learn_one({"x": 1.0}, 1, 2.0, propensity=0.7)
        model.learn_one({"x": 1.0}, 0, 1.0, propensity=0.3)
        assert model.n_seen == 2
