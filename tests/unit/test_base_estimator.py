"""Unit tests for BaseOnlineEstimator."""

import math

import pytest

from onlinecml.base.running_stats import RunningStats
from onlinecml.base.base_estimator import BaseOnlineEstimator


class _ConcreteEstimator(BaseOnlineEstimator):
    def __init__(self) -> None:
        self._n_seen: int = 0
        self._ate_stats: RunningStats = RunningStats()

    def learn_one(self, x, treatment, outcome, propensity=None):
        psi = float(outcome) * (2 * int(treatment) - 1)
        self._ate_stats.update(psi)
        self._n_seen += 1

    def predict_one(self, x):
        return self._ate_stats.mean


class TestBaseOnlineEstimator:
    def test_n_seen_starts_zero(self):
        est = _ConcreteEstimator()
        assert est.n_seen == 0

    def test_predict_ate_before_learn_returns_zero(self):
        est = _ConcreteEstimator()
        assert est.predict_ate() == 0.0

    def test_predict_ci_before_learn_returns_infinite_interval(self):
        est = _ConcreteEstimator()
        lo, hi = est.predict_ci()
        assert lo == float("-inf")
        assert hi == float("inf")

    def test_learn_one_increments_n_seen(self):
        est = _ConcreteEstimator()
        est.learn_one({"x": 1.0}, 1, 2.0)
        assert est.n_seen == 1

    def test_predict_ate_after_learn(self):
        est = _ConcreteEstimator()
        est.learn_one({"x": 1.0}, 1, 3.0)
        # psi = 3.0 * (2*1 - 1) = 3.0
        assert est.predict_ate() == pytest.approx(3.0)

    def test_predict_ci_after_two_obs(self):
        est = _ConcreteEstimator()
        est.learn_one({"x": 1.0}, 1, 2.0)
        est.learn_one({"x": 2.0}, 0, 1.0)
        lo, hi = est.predict_ci()
        assert math.isfinite(lo)
        assert math.isfinite(hi)
        assert lo < hi

    def test_smd_returns_none(self):
        est = _ConcreteEstimator()
        assert est.smd is None

    def test_reset_zeroes_state(self):
        est = _ConcreteEstimator()
        for i in range(5):
            est.learn_one({"x": float(i)}, i % 2, float(i))
        est.reset()
        assert est.n_seen == 0
        assert est.predict_ate() == 0.0

    def test_clone_returns_fresh_copy(self):
        est = _ConcreteEstimator()
        for i in range(5):
            est.learn_one({"x": float(i)}, i % 2, float(i))
        cloned = est.clone()
        assert cloned.n_seen == 0
        assert cloned.predict_ate() == 0.0

    def test_ci_lower_less_than_upper(self):
        est = _ConcreteEstimator()
        for i in range(10):
            est.learn_one({"x": float(i)}, i % 2, float(i))
        lo, hi = est.predict_ci()
        assert lo < hi
