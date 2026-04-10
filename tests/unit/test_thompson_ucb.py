"""Unit tests for ThompsonSampling, GaussianThompsonSampling, and UCB."""

import pytest

from onlinecml.policy.thompson_sampling import ThompsonSampling, GaussianThompsonSampling
from onlinecml.policy.ucb import UCB


class TestThompsonSampling:
    def test_choose_returns_binary(self):
        p = ThompsonSampling(seed=0)
        t, prop = p.choose(0.0, 0)
        assert t in (0, 1)
        assert 0.0 < prop <= 1.0

    def test_update_increments_posterior(self):
        p = ThompsonSampling(seed=0)
        p.choose(0.0, 0)
        alpha_before = list(p._alpha)
        p.update(reward=1.0)
        # Last chosen arm should have incremented alpha
        assert sum(p._alpha) > sum(alpha_before)

    def test_reset_resets_posteriors(self):
        p = ThompsonSampling(alpha_prior=2.0, beta_prior=3.0, seed=1)
        p.choose(0.0, 0)
        p.update(1.0)
        p.reset()
        assert p._alpha == [2.0, 2.0]
        assert p._beta == [3.0, 3.0]

    def test_seed_reproducibility(self):
        seq1 = [ThompsonSampling(seed=5).choose(0.0, i)[0] for i in range(10)]
        seq2 = [ThompsonSampling(seed=5).choose(0.0, i)[0] for i in range(10)]
        assert seq1 == seq2

    def test_positive_reward_updates_alpha(self):
        p = ThompsonSampling(seed=0)
        p._last_treatment = 1
        alpha_before = p._alpha[1]
        p.update(reward=1.0)
        assert p._alpha[1] == alpha_before + 1.0

    def test_negative_reward_updates_beta(self):
        p = ThompsonSampling(seed=0)
        p._last_treatment = 0
        beta_before = p._beta[0]
        p.update(reward=0.0)
        assert p._beta[0] == beta_before + 1.0


class TestGaussianThompsonSampling:
    def test_choose_returns_binary(self):
        p = GaussianThompsonSampling(seed=0)
        t, prop = p.choose(0.0, 0)
        assert t in (0, 1)

    def test_update_increments_n(self):
        p = GaussianThompsonSampling(seed=0)
        p.choose(0.0, 0)
        p.update(2.5)
        assert sum(p._n) == 1

    def test_reset_clears_state(self):
        p = GaussianThompsonSampling(prior_mean=1.0, seed=0)
        p.choose(0.0, 0)
        p.update(3.0)
        p.reset()
        assert p._n == [0, 0]
        assert p._sum_y == [0.0, 0.0]

    def test_seed_reproducibility(self):
        def run(seed):
            p = GaussianThompsonSampling(seed=seed)
            return [p.choose(0.0, i)[0] for i in range(10)]
        assert run(7) == run(7)

    def test_posterior_params_with_data(self):
        p = GaussianThompsonSampling(prior_mean=0.0, prior_std=1.0, noise_std=1.0)
        p._sum_y[1] = 10.0
        p._n[1] = 5
        mu, sigma = p._posterior_params(1)
        assert isinstance(mu, float)
        assert sigma > 0.0

    def test_gauss_sample_is_float(self):
        p = GaussianThompsonSampling(seed=0)
        sample = p._gauss_sample(0.0, 1.0)
        assert isinstance(sample, float)


class TestUCB:
    def test_choose_returns_binary(self):
        p = UCB()
        t, prop = p.choose(0.0, 0)
        assert t in (0, 1)

    def test_warmup_round_robin(self):
        p = UCB(min_pulls=2)
        # Warm-up: first 2*min_pulls calls hit the round-robin path
        results = []
        for i in range(6):
            t, prop = p.choose(0.0, i)
            results.append(t)
            p.update(1.0)
        # After warm-up, both arms should have been chosen at least once
        assert p._n_pulls[0] >= 1
        assert p._n_pulls[1] >= 1

    def test_update_increments_reward(self):
        p = UCB()
        p.choose(0.0, 0)
        arm = p._last_treatment
        p.update(reward=2.0)
        assert p._sum_reward[arm] == 2.0

    def test_reset_clears_state(self):
        p = UCB(confidence=2.0)
        for _ in range(10):
            t, _ = p.choose(0.0, _)
            p.update(1.0)
        p.reset()
        assert p._n_pulls == [0, 0]
        assert p._total_pulls == 0

    def test_high_confidence_no_crash(self):
        """High and low confidence UCB should both run without errors."""
        for conf in [0.0, 1.0, 10.0]:
            p = UCB(confidence=conf)
            for i in range(30):
                p.choose(0.0, i)
                p.update(1.0 if i % 2 == 0 else 0.0)
            assert p._total_pulls >= 0
