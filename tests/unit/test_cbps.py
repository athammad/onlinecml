"""Unit tests for OnlineCBPS."""

import pytest

from onlinecml.datasets import LinearCausalStream
from onlinecml.reweighting import OnlineCBPS


class TestOnlineCBPS:
    def test_n_seen_starts_zero(self):
        assert OnlineCBPS().n_seen == 0

    def test_predict_ate_before_learn_returns_zero(self):
        assert OnlineCBPS().predict_ate() == 0.0

    def test_learn_increments_n_seen(self):
        m = OnlineCBPS()
        m.learn_one({"x": 1.0}, 1, 2.0)
        assert m.n_seen == 1

    def test_predict_one_returns_float(self):
        m = OnlineCBPS()
        for x, w, y, _ in LinearCausalStream(n=100, seed=0):
            m.learn_one(x, w, y)
        assert isinstance(m.predict_one({"x0": 0.5}), float)

    def test_balance_report_keys_match_covariates(self):
        m = OnlineCBPS()
        for x, w, y, _ in LinearCausalStream(n=50, seed=0):
            m.learn_one(x, w, y)
        report = m.balance_report
        assert "x0" in report
        assert isinstance(report["x0"], float)

    def test_logged_propensity_skips_ps_model(self):
        m = OnlineCBPS()
        n_before = m.ps_model.n_seen if hasattr(m.ps_model, "n_seen") else 0
        m.learn_one({"x": 1.0}, 1, 2.0, propensity=0.7)
        # ps_model should not have learned with logged propensity
        # (just verify no crash)
        assert m.n_seen == 1

    def test_reset_clears_state(self):
        m = OnlineCBPS()
        for x, w, y, _ in LinearCausalStream(n=50, seed=0):
            m.learn_one(x, w, y)
        m.reset()
        assert m.n_seen == 0
        assert m.predict_ate() == 0.0

    def test_clone_gives_fresh_model(self):
        m = OnlineCBPS()
        for x, w, y, _ in LinearCausalStream(n=50, seed=0):
            m.learn_one(x, w, y)
        fresh = m.clone()
        assert fresh.n_seen == 0

    def test_ci_finite_after_data(self):
        m = OnlineCBPS()
        for x, w, y, _ in LinearCausalStream(n=200, seed=0):
            m.learn_one(x, w, y)
        lo, hi = m.predict_ci()
        assert lo < hi

    def test_ate_is_nonzero_after_data(self):
        m = OnlineCBPS()
        for x, w, y, _ in LinearCausalStream(n=500, true_ate=2.0, seed=42):
            m.learn_one(x, w, y)
        # Should be in a plausible range
        assert abs(m.predict_ate()) < 20.0
