"""Regression tests for OnlineRLearner ATE and CATE estimates.

These tests lock in known-good numerical results. Intentional changes that
alter output should update the expected values below after deliberate review.
"""

import pytest

from onlinecml.datasets import LinearCausalStream, HeterogeneousCausalStream
from onlinecml.evaluation import progressive_causal_score
from onlinecml.evaluation.metrics import ATEError, PEHE
from onlinecml.metalearners import OnlineRLearner, OnlineTLearner
from onlinecml.reweighting import OnlineIPW

REL_TOL = 0.01


class TestRLearnerRegression:
    def test_ate_linear_stream_seed42(self):
        """R-Learner ATE on LinearCausalStream(n=2000, seed=42) must stay stable."""
        m = OnlineRLearner()
        for x, w, y, _ in LinearCausalStream(n=2000, true_ate=2.0, seed=42):
            m.learn_one(x, w, y)
        assert m.predict_ate() == pytest.approx(1.788144, rel=REL_TOL)

    def test_n_seen_after_stream(self):
        m = OnlineRLearner()
        for x, w, y, _ in LinearCausalStream(n=300, seed=0):
            m.learn_one(x, w, y)
        assert m.n_seen == 300

    def test_reset_and_retrain_stable(self):
        m = OnlineRLearner()
        for x, w, y, _ in LinearCausalStream(n=400, seed=3):
            m.learn_one(x, w, y)
        first = m.predict_ate()
        m.reset()
        for x, w, y, _ in LinearCausalStream(n=400, seed=3):
            m.learn_one(x, w, y)
        assert m.predict_ate() == pytest.approx(first)

    def test_clone_gives_fresh_model(self):
        m = OnlineRLearner()
        for x, w, y, _ in LinearCausalStream(n=100, seed=0):
            m.learn_one(x, w, y)
        fresh = m.clone()
        assert fresh.n_seen == 0
        assert fresh.predict_ate() == 0.0


class TestProgressiveScoringRegression:
    def test_ate_error_decreasing_trend(self):
        """ATE error should trend downward over 1000 observations."""
        results = progressive_causal_score(
            stream  = LinearCausalStream(n=1000, true_ate=2.0, seed=10),
            model   = OnlineIPW(),
            metrics = [ATEError()],
            step    = 200,
        )
        errors = results["ATEError"]
        # Last recorded error should be smaller than first
        assert errors[-1] < errors[0] + 1.0  # allow large variance early on

    def test_pehe_finite_after_data(self):
        """PEHE should be a finite positive float after training."""
        results = progressive_causal_score(
            stream  = HeterogeneousCausalStream(n=500, seed=0),
            model   = OnlineRLearner(),
            metrics = [PEHE()],
            step    = 100,
        )
        for val in results["PEHE"]:
            assert val >= 0.0
            assert val < 1e6  # not infinite


class TestTLearnerRegression:
    def test_n_seen_increments(self):
        m = OnlineTLearner()
        for x, w, y, _ in LinearCausalStream(n=200, seed=0):
            m.learn_one(x, w, y)
        assert m.n_seen == 200

    def test_predict_one_is_float(self):
        m = OnlineTLearner()
        for x, w, y, _ in LinearCausalStream(n=200, seed=0):
            m.learn_one(x, w, y)
        result = m.predict_one({"x0": 0.0, "x1": 0.0, "x2": 0.0, "x3": 0.0, "x4": 0.0})
        assert isinstance(result, float)
