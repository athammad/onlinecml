"""Unit-test-specific fixtures."""

import pytest
from river.linear_model import LinearRegression, LogisticRegression

from onlinecml.base.base_estimator import BaseOnlineEstimator
from onlinecml.base.running_stats import RunningStats


class _ConcreteEstimator(BaseOnlineEstimator):
    """Minimal concrete estimator for testing BaseOnlineEstimator."""

    def __init__(self) -> None:
        self._n_seen: int = 0
        self._ate_stats: RunningStats = RunningStats()

    def learn_one(self, x, treatment, outcome, propensity=None):
        self._ate_stats.update(float(outcome) * (2 * treatment - 1))
        self._n_seen += 1

    def predict_one(self, x):
        return self._ate_stats.mean


@pytest.fixture
def concrete_estimator():
    return _ConcreteEstimator()


@pytest.fixture
def regressor():
    return LinearRegression()


@pytest.fixture
def classifier():
    return LogisticRegression()
