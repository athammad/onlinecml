"""Unit tests for BasePolicy."""

import pytest

from onlinecml.base.base_policy import BasePolicy


class _ConcretePolicy(BasePolicy):
    def __init__(self, fixed_treatment: int = 1) -> None:
        self.fixed_treatment = fixed_treatment

    def choose(self, cate_score: float, step: int) -> tuple[int, float]:
        return self.fixed_treatment, 0.9


class TestBasePolicy:
    def test_choose_returns_tuple(self):
        policy = _ConcretePolicy(fixed_treatment=1)
        result = policy.choose(1.5, 0)
        assert isinstance(result, tuple)
        assert len(result) == 2

    def test_choose_treatment_and_propensity(self):
        policy = _ConcretePolicy(fixed_treatment=0)
        treatment, propensity = policy.choose(0.0, 10)
        assert treatment == 0
        assert propensity == pytest.approx(0.9)

    def test_update_is_noop(self):
        policy = _ConcretePolicy()
        # Should not raise
        policy.update(1.0)

    def test_reset_reinitializes(self):
        policy = _ConcretePolicy(fixed_treatment=1)
        policy.reset()
        treatment, _ = policy.choose(1.0, 0)
        assert treatment == 1
