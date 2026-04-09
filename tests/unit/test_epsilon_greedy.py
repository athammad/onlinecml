"""Unit tests for EpsilonGreedy policy."""

import pytest

from onlinecml.policy.epsilon_greedy import EpsilonGreedy


class TestEpsilonGreedy:
    def test_choose_returns_binary_treatment(self):
        policy = EpsilonGreedy(seed=0)
        for step in range(10):
            treatment, _ = policy.choose(1.0, step)
            assert treatment in (0, 1)

    def test_propensity_in_unit_interval(self):
        policy = EpsilonGreedy(seed=0)
        for step in range(10):
            _, propensity = policy.choose(1.0, step)
            assert 0.0 <= propensity <= 1.0

    def test_epsilon_at_step_zero_is_eps_start(self):
        policy = EpsilonGreedy(eps_start=0.8, eps_end=0.05, decay=1000)
        eps = policy.current_epsilon(0)
        assert eps == pytest.approx(0.8)

    def test_epsilon_decays_over_steps(self):
        policy = EpsilonGreedy(eps_start=0.8, eps_end=0.05, decay=100)
        eps_early = policy.current_epsilon(0)
        eps_late = policy.current_epsilon(1000)
        assert eps_late < eps_early

    def test_epsilon_approaches_eps_end(self):
        policy = EpsilonGreedy(eps_start=0.8, eps_end=0.05, decay=100)
        eps = policy.current_epsilon(100_000)
        assert abs(eps - 0.05) < 0.001

    def test_exploit_positive_cate_chooses_treated(self):
        """With eps=0, positive CATE should always choose treatment 1."""
        # Use decay=1 to get epsilon very close to eps_end=0 quickly
        policy = EpsilonGreedy(eps_start=0.0, eps_end=0.0, decay=1, seed=42)
        treatment, _ = policy.choose(cate_score=5.0, step=10000)
        assert treatment == 1

    def test_exploit_negative_cate_chooses_control(self):
        policy = EpsilonGreedy(eps_start=0.0, eps_end=0.0, decay=1, seed=42)
        treatment, _ = policy.choose(cate_score=-5.0, step=10000)
        assert treatment == 0

    def test_seed_reproducibility(self):
        results_a = []
        results_b = []
        for seed_val in [7, 7]:
            policy = EpsilonGreedy(eps_start=0.5, eps_end=0.1, decay=100, seed=seed_val)
            seq = [policy.choose(1.0, i)[0] for i in range(20)]
            if not results_a:
                results_a = seq
            else:
                results_b = seq
        assert results_a == results_b

    def test_reset_resets_rng(self):
        policy = EpsilonGreedy(seed=42)
        seq1 = [policy.choose(1.0, i)[0] for i in range(10)]
        policy.reset()
        seq2 = [policy.choose(1.0, i)[0] for i in range(10)]
        assert seq1 == seq2
