"""Unit tests for LiveLovePlot, OverlapChecker, and ConceptDriftMonitor."""

import pytest

from onlinecml.diagnostics.overlap_checker import OverlapChecker
from onlinecml.diagnostics.concept_drift_monitor import ConceptDriftMonitor
from onlinecml.diagnostics.live_love_plot import LiveLovePlot


class TestOverlapChecker:
    def test_initial_report(self):
        checker = OverlapChecker()
        r = checker.report()
        assert r["n_total"] == 0
        assert r["n_flagged"] == 0
        assert r["flag_rate"] == 0.0

    def test_extreme_ps_flagged(self):
        checker = OverlapChecker(ps_min=0.05, ps_max=0.95)
        checker.update(0.02, treatment=1)
        assert checker.report()["n_flagged"] == 1

    def test_normal_ps_not_flagged(self):
        checker = OverlapChecker(ps_min=0.05, ps_max=0.95)
        checker.update(0.5, treatment=1)
        assert checker.report()["n_flagged"] == 0

    def test_common_support_rate(self):
        checker = OverlapChecker()
        for _ in range(8):
            checker.update(0.4, treatment=1)
        checker.update(0.02, treatment=0)
        checker.update(0.98, treatment=0)
        r = checker.report()
        assert r["common_support_rate"] == pytest.approx(0.8)

    def test_is_overlap_adequate_true(self):
        checker = OverlapChecker()
        for _ in range(100):
            checker.update(0.5, treatment=1)
        assert checker.is_overlap_adequate()

    def test_is_overlap_adequate_false(self):
        checker = OverlapChecker()
        for _ in range(5):
            checker.update(0.01, treatment=0)  # all flagged
        assert not checker.is_overlap_adequate(max_flag_rate=0.01)

    def test_reset(self):
        checker = OverlapChecker()
        checker.update(0.5, treatment=1)
        checker.reset()
        assert checker.report()["n_total"] == 0

    def test_per_arm_mean(self):
        checker = OverlapChecker()
        checker.update(0.8, treatment=1)
        checker.update(0.2, treatment=0)
        r = checker.report()
        assert r["mean_ps_treated"] == pytest.approx(0.8)
        assert r["mean_ps_control"] == pytest.approx(0.2)


class TestConceptDriftMonitor:
    def test_n_seen_increments(self):
        m = ConceptDriftMonitor()
        m.update(1.0)
        assert m.n_seen == 1

    def test_no_drift_on_stationary(self):
        m = ConceptDriftMonitor(delta=0.002)
        for _ in range(100):
            m.update(1.0)
        # Constant stream should not trigger many drifts
        assert m.n_drifts <= 1  # allow at most 1 false alarm

    def test_drift_detected_is_bool(self):
        m = ConceptDriftMonitor()
        m.update(1.0)
        assert isinstance(m.drift_detected, bool)

    def test_n_drifts_starts_zero(self):
        assert ConceptDriftMonitor().n_drifts == 0

    def test_reset(self):
        m = ConceptDriftMonitor()
        for _ in range(10):
            m.update(1.0)
        m.reset()
        assert m.n_seen == 0
        assert m.n_drifts == 0


class TestLiveLovePlot:
    def test_update_no_crash(self):
        plot = LiveLovePlot(covariates=["x", "y"])
        plot.update({"x": 1.0, "y": 2.0}, treatment=1, weight=1.0)

    def test_render_returns_axes_or_none(self):
        import matplotlib
        matplotlib.use("Agg")
        plot = LiveLovePlot(covariates=["x"], update_every=1)
        plot.update({"x": 1.0}, treatment=1)
        plot.update({"x": -1.0}, treatment=0)
        result = plot.render()
        # Either returns Axes or None (if matplotlib unavailable)
        import matplotlib.axes
        assert result is None or isinstance(result, matplotlib.axes.Axes)

    def test_report_via_internal_smd(self):
        plot = LiveLovePlot(covariates=["a", "b"])
        for i in range(10):
            plot.update({"a": float(i), "b": float(i * 2)}, treatment=i % 2)
        report = plot._smd.report()
        assert "a" in report
        assert "b" in report

    def test_save_no_crash_when_no_fig(self):
        """save() should be a no-op when no figure has been rendered."""
        plot = LiveLovePlot(covariates=["x"])
        plot.save("/tmp/test_love_plot.png")  # should not raise
