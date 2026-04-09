"""Unit tests for ATETracker."""

import math

import numpy as np
import pytest

from onlinecml.diagnostics.ate_tracker import ATETracker


class TestATETracker:
    def test_update_increments_n(self):
        tracker = ATETracker()
        tracker.update(1.0)
        assert tracker.n == 1

    def test_ate_converges_to_true_value(self):
        """Feed 1000 draws from N(2.0, 1.0); ATE should converge to 2.0."""
        rng = np.random.default_rng(42)
        tracker = ATETracker()
        for val in rng.normal(2.0, 1.0, size=1000):
            tracker.update(float(val))
        assert abs(tracker.ate - 2.0) < 0.2

    def test_ci_before_two_obs_is_infinite(self):
        tracker = ATETracker()
        tracker.update(1.0)
        lo, hi = tracker.ci()
        assert lo == float("-inf")
        assert hi == float("inf")

    def test_ci_after_two_obs_is_finite(self):
        tracker = ATETracker()
        tracker.update(1.0)
        tracker.update(3.0)
        lo, hi = tracker.ci()
        assert math.isfinite(lo)
        assert math.isfinite(hi)

    def test_history_appended_when_log_every_1(self):
        tracker = ATETracker(log_every=1)
        for i in range(5):
            tracker.update(float(i))
        assert len(tracker.history) == 5

    def test_history_appended_every_n(self):
        tracker = ATETracker(log_every=3)
        for i in range(9):
            tracker.update(float(i))
        assert len(tracker.history) == 3

    def test_convergence_width_decreases(self):
        """CI width should generally decrease as more data is added."""
        tracker = ATETracker()
        rng = np.random.default_rng(0)
        for _ in range(5):
            tracker.update(float(rng.normal(0, 1)))
        w_early = tracker.convergence_width()
        for _ in range(1000):
            tracker.update(float(rng.normal(0, 1)))
        w_late = tracker.convergence_width()
        assert w_late < w_early

    def test_reset_clears_state(self):
        tracker = ATETracker()
        for i in range(10):
            tracker.update(float(i))
        tracker.reset()
        assert tracker.n == 0
        assert tracker.ate == 0.0
        assert tracker.history == []

    def test_plot_returns_axes(self):
        """Smoke test: plot() should return a matplotlib Axes."""
        tracker = ATETracker()
        for i in range(20):
            tracker.update(float(i))
        ax = tracker.plot()
        import matplotlib.axes
        assert isinstance(ax, matplotlib.axes.Axes)

    def test_ci_contains_true_ate(self):
        """95% CI should contain true ATE in the large-sample regime."""
        rng = np.random.default_rng(99)
        tracker = ATETracker()
        for val in rng.normal(3.0, 2.0, size=500):
            tracker.update(float(val))
        lo, hi = tracker.ci(alpha=0.05)
        assert lo < 3.0 < hi
