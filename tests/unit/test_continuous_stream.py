"""Unit tests for ContinuousTreatmentStream."""

import pytest

from onlinecml.datasets import ContinuousTreatmentStream


class TestContinuousTreatmentStream:
    def test_len(self):
        assert len(ContinuousTreatmentStream(n=50)) == 50

    def test_yields_float_treatment(self):
        for x, w, y, marginal in ContinuousTreatmentStream(n=1, seed=0):
            assert isinstance(w, float)

    def test_yields_four_tuple(self):
        for x, w, y, marginal in ContinuousTreatmentStream(n=1, seed=0):
            assert isinstance(x, dict)
            assert isinstance(y, float)
            assert isinstance(marginal, float)

    def test_linear_marginal_constant(self):
        """Linear dose-response: marginal = true_effect for all obs."""
        stream = ContinuousTreatmentStream(
            n=20, dose_response="linear", true_effect=3.0, seed=0
        )
        for _, _, _, marginal in stream:
            assert marginal == pytest.approx(3.0)

    def test_quadratic_marginal_is_2aw(self):
        """Quadratic dose-response: marginal = 2 * true_effect * W."""
        stream = ContinuousTreatmentStream(
            n=20, dose_response="quadratic", true_effect=2.0, seed=0
        )
        for _, w, _, marginal in stream:
            assert marginal == pytest.approx(2.0 * 2.0 * w)

    def test_threshold_marginal_is_g(self):
        """Threshold: marginal equals g(W) = true_effect * (W > 0)."""
        stream = ContinuousTreatmentStream(
            n=20, dose_response="threshold", true_effect=1.5, seed=0
        )
        for _, w, _, marginal in stream:
            expected = 1.5 * float(w > 0.0)
            assert marginal == pytest.approx(expected)

    def test_normal_distribution(self):
        """w_distribution='normal' should yield float treatments."""
        stream = ContinuousTreatmentStream(n=10, w_distribution="normal", seed=1)
        for _, w, _, _ in stream:
            assert isinstance(w, float)

    def test_invalid_dose_response(self):
        with pytest.raises(ValueError):
            ContinuousTreatmentStream(dose_response="cubic")

    def test_invalid_w_distribution(self):
        with pytest.raises(ValueError):
            ContinuousTreatmentStream(w_distribution="beta")

    def test_rereiterable_same_seed(self):
        stream = ContinuousTreatmentStream(n=30, seed=7)
        first  = [round(w, 8) for _, w, _, _ in stream]
        second = [round(w, 8) for _, w, _, _ in stream]
        assert first == second

    def test_uniform_treatment_in_range(self):
        """Uniform treatment should stay near [w_min, w_max] without confounding."""
        stream = ContinuousTreatmentStream(
            n=200, w_min=-1.0, w_max=1.0, confounding_strength=0.0, seed=0
        )
        ws = [w for _, w, _, _ in stream]
        assert min(ws) > -3.0
        assert max(ws) < 3.0
