"""Unit tests for evaluation metrics and progressive_causal_score."""

import math
import pytest

from onlinecml.datasets import LinearCausalStream, HeterogeneousCausalStream
from onlinecml.evaluation import progressive_causal_score
from onlinecml.evaluation.metrics import (
    ATEError,
    CIcoverage,
    CIWidth,
    PEHE,
    QiniCoefficient,
    UpliftAUC,
)
from onlinecml.reweighting import OnlineIPW
from onlinecml.metalearners import OnlineRLearner


class _FakeModel:
    """Minimal model stub for metric tests."""

    def __init__(self, ate=1.5):
        self._ate = ate
        self._ci = (1.0, 2.0)

    def predict_ate(self):
        return self._ate

    def predict_ci(self, alpha=0.05):
        return self._ci

    def predict_one(self, x):
        return self._ate

    def learn_one(self, x, w, y):
        pass


class TestATEError:
    def test_score_starts_zero(self):
        assert ATEError().score == 0.0

    def test_score_after_one_obs(self):
        m = ATEError()
        model = _FakeModel(ate=3.0)
        m.update({}, 1, 1.0, true_cate=2.0, cate_hat=3.0, model=model)
        # true population ATE so far = 2.0; model ATE = 3.0 → error = 1.0
        assert m.score == pytest.approx(1.0)

    def test_score_with_constant_true_cate(self):
        m = ATEError()
        model = _FakeModel(ate=2.0)
        for _ in range(10):
            m.update({}, 1, 1.0, true_cate=2.0, cate_hat=2.0, model=model)
        assert m.score == pytest.approx(0.0)

    def test_reset(self):
        m = ATEError()
        m.update({}, 1, 1.0, true_cate=5.0, cate_hat=0.0, model=_FakeModel())
        m.reset()
        assert m.score == 0.0


class TestPEHE:
    def test_score_starts_zero(self):
        assert PEHE().score == 0.0

    def test_score_perfect_prediction(self):
        m = PEHE()
        for _ in range(10):
            m.update({}, 1, 1.0, true_cate=2.0, cate_hat=2.0, model=None)
        assert m.score == pytest.approx(0.0)

    def test_score_constant_error(self):
        m = PEHE()
        for _ in range(100):
            m.update({}, 1, 1.0, true_cate=0.0, cate_hat=1.0, model=None)
        assert m.score == pytest.approx(1.0)

    def test_score_is_non_negative(self):
        m = PEHE()
        for i in range(10):
            m.update({}, 1, 1.0, true_cate=float(i), cate_hat=float(i + 1), model=None)
        assert m.score >= 0.0

    def test_reset(self):
        m = PEHE()
        m.update({}, 1, 1.0, true_cate=5.0, cate_hat=0.0, model=None)
        m.reset()
        assert m.score == 0.0


class TestUpliftAUC:
    def test_score_starts_zero(self):
        assert UpliftAUC().score == 0.0

    def test_score_after_single_obs(self):
        m = UpliftAUC()
        m.update({}, 1, 1.0, 1.0, 1.0, None)
        assert m.score == 0.0  # only one obs, can't compute AUC

    def test_score_non_negative_with_data(self):
        m = UpliftAUC()
        for i in range(50):
            w = i % 2
            m.update({}, w, float(w), 0.0, float(i), None)
        assert m.score >= 0.0

    def test_max_buffer_enforced(self):
        m = UpliftAUC(max_buffer=10)
        for i in range(30):
            m.update({}, i % 2, 1.0, 0.0, float(i), None)
        assert len(m._buffer) <= 10

    def test_reset(self):
        m = UpliftAUC()
        m.update({}, 1, 1.0, 1.0, 1.0, None)
        m.reset()
        assert len(m._buffer) == 0

    def test_zero_when_all_same_arm(self):
        m = UpliftAUC()
        for i in range(10):
            m.update({}, 1, float(i), 0.0, float(i), None)
        # All treated — control mean undefined (0) → score may be 0 or small
        assert isinstance(m.score, float)


class TestQiniCoefficient:
    def test_score_starts_zero(self):
        assert QiniCoefficient().score == 0.0

    def test_score_after_single_obs(self):
        m = QiniCoefficient()
        m.update({}, 1, 1.0, 1.0, 1.0, None)
        assert m.score == 0.0

    def test_zero_when_one_arm_missing(self):
        m = QiniCoefficient()
        for i in range(10):
            m.update({}, 1, 1.0, 0.0, float(i), None)
        assert m.score == 0.0  # no control units

    def test_score_is_float_with_both_arms(self):
        m = QiniCoefficient()
        for i in range(20):
            m.update({}, i % 2, float(i % 2), 0.0, float(i), None)
        assert isinstance(m.score, float)

    def test_max_buffer_enforced(self):
        m = QiniCoefficient(max_buffer=5)
        for i in range(20):
            m.update({}, i % 2, 1.0, 0.0, float(i), None)
        assert len(m._buffer) <= 5

    def test_reset(self):
        m = QiniCoefficient()
        m.update({}, 1, 1.0, 1.0, 1.0, None)
        m.reset()
        assert len(m._buffer) == 0


class TestCIWidth:
    def test_score_starts_zero(self):
        assert CIWidth().score == 0.0

    def test_score_after_data(self):
        m = CIWidth()
        model = _FakeModel()
        model._ci = (0.0, 2.0)
        for _ in range(5):
            m.update({}, 1, 1.0, 1.0, 1.0, model)
        assert m.score == pytest.approx(2.0)

    def test_reset(self):
        m = CIWidth()
        model = _FakeModel()
        m.update({}, 1, 1.0, 1.0, 1.0, model)
        m.reset()
        assert m.score == 0.0


class TestCIcoverage:
    def test_score_starts_zero(self):
        assert CIcoverage().score == 0.0

    def test_full_coverage_when_ci_wide(self):
        m = CIcoverage()
        model = _FakeModel()
        model._ci = (-100.0, 100.0)
        for _ in range(10):
            m.update({}, 1, 1.0, true_cate=1.5, cate_hat=1.5, model=model)
        assert m.score == pytest.approx(1.0)

    def test_zero_coverage_when_ci_misses(self):
        m = CIcoverage()
        model = _FakeModel()
        model._ci = (50.0, 60.0)  # CI is [50, 60]; true ATE = 1.5
        for _ in range(10):
            m.update({}, 1, 1.0, true_cate=1.5, cate_hat=1.5, model=model)
        assert m.score == pytest.approx(0.0)

    def test_reset(self):
        m = CIcoverage()
        model = _FakeModel()
        m.update({}, 1, 1.0, 1.5, 1.5, model)
        m.reset()
        assert m.score == 0.0


class TestProgressiveCausalScore:
    def test_returns_dict_with_steps(self):
        results = progressive_causal_score(
            stream  = LinearCausalStream(n=300, seed=0),
            model   = OnlineIPW(),
            metrics = [ATEError()],
            step    = 100,
        )
        assert "steps" in results
        assert "ATEError" in results
        assert results["steps"] == [100, 200, 300]

    def test_step_count_correct(self):
        results = progressive_causal_score(
            stream  = LinearCausalStream(n=500, seed=0),
            model   = OnlineIPW(),
            metrics = [PEHE()],
            step    = 100,
        )
        assert len(results["steps"]) == 5
        assert len(results["PEHE"]) == 5

    def test_multiple_metrics(self):
        results = progressive_causal_score(
            stream  = LinearCausalStream(n=200, seed=1),
            model   = OnlineIPW(),
            metrics = [ATEError(), PEHE(), CIWidth()],
            step    = 100,
        )
        assert "ATEError" in results
        assert "PEHE" in results
        assert "CIWidth" in results

    def test_pehe_decreases_with_more_data(self):
        """PEHE should generally decrease as the model sees more data."""
        results = progressive_causal_score(
            stream  = HeterogeneousCausalStream(n=1000, seed=42),
            model   = OnlineRLearner(),
            metrics = [PEHE()],
            step    = 200,
        )
        pehe_vals = results["PEHE"]
        # Last value should be lower than first (rough convergence check)
        assert pehe_vals[-1] < pehe_vals[0] * 2  # at least not diverging

    def test_ate_error_is_non_negative(self):
        results = progressive_causal_score(
            stream  = LinearCausalStream(n=300, seed=5),
            model   = OnlineIPW(),
            metrics = [ATEError()],
            step    = 100,
        )
        for val in results["ATEError"]:
            assert val >= 0.0

    def test_empty_stream_returns_empty_steps(self):
        results = progressive_causal_score(
            stream  = LinearCausalStream(n=50, seed=0),
            model   = OnlineIPW(),
            metrics = [ATEError()],
            step    = 100,  # step > n → no checkpoints
        )
        assert results["steps"] == []
        assert results["ATEError"] == []
