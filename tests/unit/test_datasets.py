"""Unit tests for LinearCausalStream and HeterogeneousCausalStream."""

import pytest

from onlinecml.datasets.linear_causal import LinearCausalStream
from onlinecml.datasets.heterogeneous_causal import HeterogeneousCausalStream


class TestLinearCausalStream:
    def test_length(self):
        stream = LinearCausalStream(n=100)
        assert len(stream) == 100

    def test_yields_n_items(self):
        stream = LinearCausalStream(n=50, seed=0)
        count = sum(1 for _ in stream)
        assert count == 50

    def test_x_is_dict(self):
        stream = LinearCausalStream(n=5, seed=0)
        for x, _, _, _ in stream:
            assert isinstance(x, dict)
            break

    def test_x_has_correct_keys(self):
        stream = LinearCausalStream(n=5, n_features=3, seed=0)
        for x, _, _, _ in stream:
            assert set(x.keys()) == {"x0", "x1", "x2"}
            break

    def test_treatment_is_0_or_1(self):
        stream = LinearCausalStream(n=50, seed=0)
        for _, w, _, _ in stream:
            assert w in (0, 1)

    def test_true_cate_is_constant(self):
        true_ate = 3.5
        stream = LinearCausalStream(n=50, true_ate=true_ate, seed=0)
        for _, _, _, tau in stream:
            assert tau == pytest.approx(true_ate)

    def test_seed_reproducibility(self):
        s1 = list(LinearCausalStream(n=20, seed=42))
        s2 = list(LinearCausalStream(n=20, seed=42))
        for (x1, w1, y1, t1), (x2, w2, y2, t2) in zip(s1, s2):
            assert x1 == x2
            assert w1 == w2
            assert y1 == pytest.approx(y2)

    def test_different_seeds_differ(self):
        s1 = list(LinearCausalStream(n=20, seed=1))
        s2 = list(LinearCausalStream(n=20, seed=2))
        outcomes1 = [y for _, _, y, _ in s1]
        outcomes2 = [y for _, _, y, _ in s2]
        assert outcomes1 != outcomes2

    def test_outcome_is_float(self):
        stream = LinearCausalStream(n=5, seed=0)
        for _, _, y, _ in stream:
            assert isinstance(y, float)
            break

    def test_reiterable_with_same_seed(self):
        stream = LinearCausalStream(n=10, seed=7)
        first_pass = list(stream)
        second_pass = list(stream)
        assert first_pass == second_pass


class TestHeterogeneousCausalStream:
    def test_cate_varies_nonlinear(self):
        stream = HeterogeneousCausalStream(n=50, heterogeneity="nonlinear", seed=0)
        cates = [tau for _, _, _, tau in stream]
        assert len(set(cates)) > 1

    def test_cate_varies_linear(self):
        stream = HeterogeneousCausalStream(n=50, heterogeneity="linear", seed=0)
        cates = [tau for _, _, _, tau in stream]
        assert len(set(cates)) > 1

    def test_cate_varies_step(self):
        stream = HeterogeneousCausalStream(n=50, heterogeneity="step", seed=0)
        cates = set(tau for _, _, _, tau in stream)
        # Step function produces exactly 2 distinct CATE values
        assert len(cates) == 2

    def test_x_dict_has_correct_keys(self):
        stream = HeterogeneousCausalStream(n=5, n_features=4, seed=0)
        for x, _, _, _ in stream:
            assert set(x.keys()) == {"x0", "x1", "x2", "x3"}
            break

    def test_seed_reproducibility(self):
        s1 = list(HeterogeneousCausalStream(n=20, seed=5))
        s2 = list(HeterogeneousCausalStream(n=20, seed=5))
        for (x1, w1, y1, t1), (x2, w2, y2, t2) in zip(s1, s2):
            assert x1 == x2
            assert y1 == pytest.approx(y2)

    def test_population_ate_linear(self):
        stream = HeterogeneousCausalStream(true_ate=3.0, heterogeneity="linear")
        assert stream.population_ate() == pytest.approx(3.0)

    def test_population_ate_nonlinear(self):
        stream = HeterogeneousCausalStream(true_ate=2.0, heterogeneity="nonlinear")
        assert stream.population_ate() == pytest.approx(2.0)

    def test_population_ate_step(self):
        stream = HeterogeneousCausalStream(true_ate=2.0, heterogeneity="step")
        assert stream.population_ate() == pytest.approx(2.5)

    def test_invalid_heterogeneity_raises(self):
        with pytest.raises(ValueError, match="heterogeneity"):
            HeterogeneousCausalStream(heterogeneity="invalid")

    def test_length(self):
        stream = HeterogeneousCausalStream(n=77, seed=0)
        assert len(stream) == 77
