"""Unit tests for matching methods and distance utilities."""

import math
import pytest

from onlinecml.matching.distance import (
    euclidean_distance,
    ps_distance,
    mahalanobis_distance,
    combined_distance,
)
from onlinecml.matching.online_matching import OnlineMatching
from onlinecml.matching.caliper_matching import OnlineCaliperMatching
from onlinecml.matching.kernel_matching import OnlineKernelMatching
from onlinecml.datasets import LinearCausalStream


class TestDistanceFunctions:
    def test_euclidean_same_point(self):
        assert euclidean_distance({"x": 1.0}, {"x": 1.0}) == pytest.approx(0.0)

    def test_euclidean_unit_distance(self):
        assert euclidean_distance({"x": 0.0}, {"x": 1.0}) == pytest.approx(1.0)

    def test_euclidean_multi_dim(self):
        # (3,4) -> dist = 5
        assert euclidean_distance({"x": 3.0, "y": 0.0}, {"x": 0.0, "y": 4.0}) == pytest.approx(5.0)

    def test_euclidean_missing_key_defaults_to_zero(self):
        # Missing "y" in second dict treated as 0
        d = euclidean_distance({"x": 1.0, "y": 2.0}, {"x": 1.0})
        assert d == pytest.approx(2.0)

    def test_ps_distance(self):
        assert ps_distance(0.3, 0.7) == pytest.approx(0.4)
        assert ps_distance(0.5, 0.5) == pytest.approx(0.0)

    def test_mahalanobis_no_cov_equals_euclidean(self):
        x = {"a": 3.0, "b": 0.0}
        y = {"a": 0.0, "b": 4.0}
        assert mahalanobis_distance(x, y) == pytest.approx(euclidean_distance(x, y))

    def test_mahalanobis_with_identity_cov(self):
        x = {"a": 1.0}
        y = {"a": 0.0}
        cov_inv = {"a": {"a": 1.0}}
        assert mahalanobis_distance(x, y, cov_inv=cov_inv) == pytest.approx(1.0)

    def test_combined_distance_ps_only(self):
        x = {"a": 1.0}
        y = {"a": 100.0}
        d = combined_distance(x, y, 0.3, 0.7, ps_weight=1.0)
        assert d == pytest.approx(ps_distance(0.3, 0.7))

    def test_combined_distance_euclidean_only(self):
        x = {"a": 0.0}
        y = {"a": 3.0}
        d = combined_distance(x, y, 0.5, 0.5, ps_weight=0.0)
        assert d == pytest.approx(euclidean_distance(x, y))


class TestOnlineMatching:
    def test_n_seen_starts_zero(self):
        assert OnlineMatching().n_seen == 0

    def test_learn_increments_n_seen(self):
        m = OnlineMatching()
        m.learn_one({"x": 1.0}, 1, 2.0)
        assert m.n_seen == 1

    def test_predict_one_before_both_buffers_filled(self):
        m = OnlineMatching()
        m.learn_one({"x": 1.0}, 1, 2.0)
        # No control obs yet, should return 0
        assert m.predict_one({"x": 1.0}) == 0.0

    def test_predict_one_after_both_arms(self):
        m = OnlineMatching()
        for x, w, y, _ in LinearCausalStream(n=100, seed=0):
            m.learn_one(x, w, y)
        assert isinstance(m.predict_one({"x0": 0.0, "x1": 0.0, "x2": 0.0}), float)

    def test_ate_after_full_stream(self):
        m = OnlineMatching(k=3)
        for x, w, y, _ in LinearCausalStream(n=200, seed=0):
            m.learn_one(x, w, y)
        assert isinstance(m.predict_ate(), float)

    def test_buffer_size_enforced(self):
        m = OnlineMatching(buffer_size=10)
        for x, w, y, _ in LinearCausalStream(n=50, seed=0):
            m.learn_one(x, w, y)
        assert len(m._treated_buffer) <= 10
        assert len(m._control_buffer) <= 10

    def test_reset(self):
        m = OnlineMatching()
        for x, w, y, _ in LinearCausalStream(n=50, seed=0):
            m.learn_one(x, w, y)
        m.reset()
        assert m.n_seen == 0

    def test_clone_fresh(self):
        m = OnlineMatching()
        for x, w, y, _ in LinearCausalStream(n=50, seed=0):
            m.learn_one(x, w, y)
        assert m.clone().n_seen == 0


class TestOnlineCaliperMatching:
    def test_common_support_rate_starts_zero(self):
        assert OnlineCaliperMatching().common_support_rate == 0.0

    def test_common_support_rate_when_caliper_large(self):
        m = OnlineCaliperMatching(caliper=1000.0)
        for x, w, y, _ in LinearCausalStream(n=100, seed=0):
            m.learn_one(x, w, y)
        # Large caliper: nearly all units should be matched
        assert m.common_support_rate > 0.0

    def test_common_support_rate_when_caliper_zero(self):
        m = OnlineCaliperMatching(caliper=0.0)
        for x, w, y, _ in LinearCausalStream(n=100, seed=0):
            m.learn_one(x, w, y)
        # Zero caliper: nothing should match
        assert m.common_support_rate == 0.0

    def test_reset(self):
        m = OnlineCaliperMatching()
        for x, w, y, _ in LinearCausalStream(n=50, seed=0):
            m.learn_one(x, w, y)
        m.reset()
        assert m.n_seen == 0
        assert m.common_support_rate == 0.0

    def test_predict_one_caliper_exceeded_returns_zero(self):
        m = OnlineCaliperMatching(caliper=0.0)
        m._treated_buffer.append(({"x": 100.0}, 5.0))
        m._control_buffer.append(({"x": -100.0}, 1.0))
        assert m.predict_one({"x": 0.0}) == 0.0

    def test_predict_one_within_caliper(self):
        m = OnlineCaliperMatching(caliper=1000.0)
        m._treated_buffer.append(({"x": 0.1}, 5.0))
        m._control_buffer.append(({"x": 0.0}, 1.0))
        assert m.predict_one({"x": 0.0}) == pytest.approx(4.0)


class TestOnlineKernelMatching:
    def test_n_seen_starts_zero(self):
        assert OnlineKernelMatching().n_seen == 0

    def test_predict_one_before_data(self):
        m = OnlineKernelMatching()
        assert m.predict_one({"x": 1.0}) == 0.0

    def test_ate_after_full_stream(self):
        m = OnlineKernelMatching(bandwidth=2.0)
        for x, w, y, _ in LinearCausalStream(n=200, seed=0):
            m.learn_one(x, w, y)
        assert isinstance(m.predict_ate(), float)

    def test_reset(self):
        m = OnlineKernelMatching()
        for x, w, y, _ in LinearCausalStream(n=50, seed=0):
            m.learn_one(x, w, y)
        m.reset()
        assert m.n_seen == 0
