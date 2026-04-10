"""Unit tests for OnlineOverlapWeights."""

import pytest

from onlinecml.reweighting.overlap_weights import OnlineOverlapWeights
from onlinecml.datasets import LinearCausalStream


class TestOnlineOverlapWeights:
    def test_n_seen_starts_zero(self):
        assert OnlineOverlapWeights().n_seen == 0

    def test_predict_ate_before_learn_returns_zero(self):
        assert OnlineOverlapWeights().predict_ate() == 0.0

    def test_learn_increments_n_seen(self):
        est = OnlineOverlapWeights()
        est.learn_one({"x": 1.0}, 1, 2.0)
        assert est.n_seen == 1

    def test_predict_one_returns_ate(self):
        est = OnlineOverlapWeights()
        for x, w, y, _ in LinearCausalStream(n=100, seed=0):
            est.learn_one(x, w, y)
        assert est.predict_one({"x": 99.0}) == est.predict_ate()

    def test_ci_finite(self):
        est = OnlineOverlapWeights()
        for x, w, y, _ in LinearCausalStream(n=100, seed=0):
            est.learn_one(x, w, y)
        lo, hi = est.predict_ci()
        assert lo < hi

    def test_reset(self):
        est = OnlineOverlapWeights()
        for x, w, y, _ in LinearCausalStream(n=50, seed=0):
            est.learn_one(x, w, y)
        est.reset()
        assert est.n_seen == 0

    def test_logged_propensity(self):
        est = OnlineOverlapWeights()
        n_before = est.ps_model.n_seen
        est.learn_one({"x": 1.0}, 1, 2.0, propensity=0.6)
        assert est.ps_model.n_seen == n_before
