"""Unit tests for DriftingCausalStream and UnbalancedCausalStream."""

import pytest

from onlinecml.datasets import DriftingCausalStream, UnbalancedCausalStream


class TestDriftingCausalStream:
    def test_len(self):
        assert len(DriftingCausalStream(n=100)) == 100

    def test_yields_four_tuple(self):
        for x, w, y, tau in DriftingCausalStream(n=1, seed=0):
            assert isinstance(x, dict)
            assert w in (0, 1)
            assert isinstance(y, float)
            assert isinstance(tau, float)

    def test_default_changepoint_is_half(self):
        stream = DriftingCausalStream(n=200)
        assert stream.changepoint == 100

    def test_true_cate_before_changepoint(self):
        """true_cate should equal true_ate before the changepoint."""
        stream = DriftingCausalStream(n=20, true_ate=3.0, shifted_ate=-1.0,
                                      changepoint=10, seed=1)
        obs = list(stream)
        for _, _, _, tau in obs[:10]:
            assert tau == pytest.approx(3.0)

    def test_true_cate_after_changepoint(self):
        """true_cate should equal shifted_ate from the changepoint onward."""
        stream = DriftingCausalStream(n=20, true_ate=3.0, shifted_ate=-1.0,
                                      changepoint=10, seed=1)
        obs = list(stream)
        for _, _, _, tau in obs[10:]:
            assert tau == pytest.approx(-1.0)

    def test_rereiterable_same_seed(self):
        stream = DriftingCausalStream(n=50, seed=42)
        first  = [(w, round(y, 6)) for _, w, y, _ in stream]
        second = [(w, round(y, 6)) for _, w, y, _ in stream]
        assert first == second

    def test_explicit_changepoint(self):
        stream = DriftingCausalStream(n=10, changepoint=3, true_ate=1.0,
                                      shifted_ate=5.0, seed=0)
        obs = list(stream)
        assert obs[2][3] == pytest.approx(1.0)
        assert obs[3][3] == pytest.approx(5.0)


class TestUnbalancedCausalStream:
    def test_len(self):
        assert len(UnbalancedCausalStream(n=50)) == 50

    def test_yields_four_tuple(self):
        for x, w, y, tau in UnbalancedCausalStream(n=1, seed=0):
            assert isinstance(x, dict)
            assert w in (0, 1)
            assert isinstance(y, float)
            assert isinstance(tau, float)

    def test_treatment_rate_low(self):
        """With treatment_rate=0.1, fewer than 30% of units should be treated."""
        stream = UnbalancedCausalStream(n=500, treatment_rate=0.1, seed=7)
        rate = sum(w for _, w, _, _ in stream) / 500
        assert rate < 0.3

    def test_treatment_rate_high(self):
        """With treatment_rate=0.9, more than 70% of units should be treated."""
        stream = UnbalancedCausalStream(n=500, treatment_rate=0.9, seed=7)
        rate = sum(w for _, w, _, _ in stream) / 500
        assert rate > 0.7

    def test_invalid_treatment_rate(self):
        with pytest.raises(ValueError):
            UnbalancedCausalStream(treatment_rate=0.0)
        with pytest.raises(ValueError):
            UnbalancedCausalStream(treatment_rate=1.0)
        with pytest.raises(ValueError):
            UnbalancedCausalStream(treatment_rate=1.5)

    def test_true_cate_constant(self):
        stream = UnbalancedCausalStream(n=10, true_ate=2.5, seed=0)
        for _, _, _, tau in stream:
            assert tau == pytest.approx(2.5)

    def test_rereiterable_same_seed(self):
        stream = UnbalancedCausalStream(n=30, seed=99)
        first  = [w for _, w, _, _ in stream]
        second = [w for _, w, _, _ in stream]
        assert first == second
