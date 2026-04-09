"""Unit tests for RunningStats and WeightedRunningStats."""

import math

import numpy as np
import pytest

from onlinecml.base.running_stats import RunningStats, WeightedRunningStats


class TestRunningStats:
    def test_update_mean_three_values(self):
        stats = RunningStats()
        for x in [2.0, 4.0, 6.0]:
            stats.update(x)
        assert stats.mean == pytest.approx(4.0)

    def test_update_variance_matches_numpy(self):
        data = [1.0, 3.0, 5.0, 7.0, 9.0]
        stats = RunningStats()
        for x in data:
            stats.update(x)
        assert stats.variance == pytest.approx(np.var(data, ddof=1))

    def test_single_value_variance_zero(self):
        stats = RunningStats()
        stats.update(42.0)
        assert stats.variance == 0.0

    def test_std_is_sqrt_variance(self):
        stats = RunningStats()
        for x in [1.0, 2.0, 3.0, 4.0]:
            stats.update(x)
        assert stats.std == pytest.approx(math.sqrt(stats.variance))

    def test_n_increments(self):
        stats = RunningStats()
        for i in range(5):
            stats.update(float(i))
        assert stats.n == 5

    def test_reset_zeroes_state(self):
        stats = RunningStats()
        for x in [1.0, 2.0, 3.0]:
            stats.update(x)
        stats.reset()
        assert stats.n == 0
        assert stats.mean == 0.0
        assert stats.variance == 0.0

    def test_empty_stats_mean_zero(self):
        stats = RunningStats()
        assert stats.mean == 0.0
        assert stats.n == 0

    def test_variance_before_two_obs_is_zero(self):
        stats = RunningStats()
        stats.update(5.0)
        assert stats.variance == 0.0


class TestWeightedRunningStats:
    def test_uniform_weights_match_unweighted_mean(self):
        data = [1.0, 2.0, 3.0, 4.0, 5.0]
        wstats = WeightedRunningStats()
        stats = RunningStats()
        for x in data:
            wstats.update(x, w=1.0)
            stats.update(x)
        assert wstats.mean == pytest.approx(stats.mean)

    def test_nonuniform_mean_matches_numpy(self):
        data = [1.0, 2.0, 3.0]
        weights = [1.0, 2.0, 3.0]
        wstats = WeightedRunningStats()
        for x, w in zip(data, weights):
            wstats.update(x, w=w)
        expected = np.average(data, weights=weights)
        assert wstats.mean == pytest.approx(expected)

    def test_nonuniform_variance(self):
        data = [1.0, 3.0, 5.0]
        weights = [2.0, 1.0, 2.0]
        wstats = WeightedRunningStats()
        for x, w in zip(data, weights):
            wstats.update(x, w=w)
        # Population weighted variance
        mean = np.average(data, weights=weights)
        expected_var = np.average((np.array(data) - mean) ** 2, weights=weights)
        assert wstats.variance == pytest.approx(expected_var)

    def test_zero_weight_ignored(self):
        wstats = WeightedRunningStats()
        wstats.update(10.0, w=0.0)
        assert wstats.sum_weights == 0.0
        assert wstats.mean == 0.0

    def test_negative_weight_ignored(self):
        wstats = WeightedRunningStats()
        wstats.update(10.0, w=-1.0)
        assert wstats.sum_weights == 0.0

    def test_std_is_sqrt_variance(self):
        wstats = WeightedRunningStats()
        for x, w in zip([1.0, 2.0, 3.0], [1.0, 2.0, 1.0]):
            wstats.update(x, w=w)
        assert wstats.std == pytest.approx(math.sqrt(wstats.variance))

    def test_reset(self):
        wstats = WeightedRunningStats()
        wstats.update(5.0, w=2.0)
        wstats.reset()
        assert wstats.sum_weights == 0.0
        assert wstats.mean == 0.0

    def test_variance_no_data_is_zero(self):
        wstats = WeightedRunningStats()
        assert wstats.variance == 0.0
