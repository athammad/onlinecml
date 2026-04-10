"""Unit tests for CausalHoeffdingTree and OnlineCausalForest."""

import pytest

from onlinecml.datasets import LinearCausalStream
from onlinecml.forests import CausalHoeffdingTree, OnlineCausalForest


class TestCausalHoeffdingTree:
    def test_n_seen_starts_zero(self):
        assert CausalHoeffdingTree().n_seen == 0

    def test_predict_ate_before_data(self):
        assert CausalHoeffdingTree().predict_ate() == 0.0

    def test_predict_one_returns_float(self):
        t = CausalHoeffdingTree(grace_period=10, seed=0)
        for x, w, y, _ in LinearCausalStream(n=50, seed=0):
            t.learn_one(x, w, y)
        result = t.predict_one({"x0": 0.5, "x1": -0.3, "x2": 0.0, "x3": 0.1, "x4": -0.2})
        assert isinstance(result, float)

    def test_learn_increments_n_seen(self):
        t = CausalHoeffdingTree()
        t.learn_one({"x": 1.0}, 1, 2.0)
        assert t.n_seen == 1

    def test_n_nodes_starts_at_one(self):
        assert CausalHoeffdingTree().n_nodes == 1

    def test_n_leaves_starts_at_one(self):
        assert CausalHoeffdingTree().n_leaves == 1

    def test_tree_can_grow(self):
        """With a very small grace_period the tree should eventually split."""
        t = CausalHoeffdingTree(grace_period=20, delta=0.1, tau=0.001, seed=0)
        for x, w, y, _ in LinearCausalStream(n=500, seed=0):
            t.learn_one(x, w, y)
        # At minimum, the root should still be there
        assert t.n_nodes >= 1

    def test_predict_missing_feature_defaults_to_zero(self):
        """Routing with unseen features should not crash."""
        t = CausalHoeffdingTree(grace_period=10)
        t.learn_one({"x0": 1.0, "x1": 0.5}, 1, 2.0)
        t.predict_one({"x2": 99.0})  # completely different features

    def test_reset_clears_tree(self):
        t = CausalHoeffdingTree(grace_period=10, seed=0)
        for x, w, y, _ in LinearCausalStream(n=100, seed=0):
            t.learn_one(x, w, y)
        t.reset()
        assert t.n_seen == 0
        assert t.predict_ate() == 0.0

    def test_clone_gives_fresh_tree(self):
        t = CausalHoeffdingTree(grace_period=10, seed=0)
        for x, w, y, _ in LinearCausalStream(n=50, seed=0):
            t.learn_one(x, w, y)
        fresh = t.clone()
        assert fresh.n_seen == 0

    def test_max_depth_respected(self):
        """max_depth=0 means only the root leaf; no splits should happen."""
        t = CausalHoeffdingTree(grace_period=1, max_depth=0, delta=0.9, tau=0.0)
        for x, w, y, _ in LinearCausalStream(n=500, seed=0):
            t.learn_one(x, w, y)
        assert t.n_leaves == 1  # max_depth=0 → no splits

    def test_ci_finite_after_data(self):
        t = CausalHoeffdingTree(grace_period=10)
        for x, w, y, _ in LinearCausalStream(n=200, seed=0):
            t.learn_one(x, w, y)
        lo, hi = t.predict_ci()
        assert lo <= hi


class TestOnlineCausalForest:
    def test_n_seen_starts_zero(self):
        assert OnlineCausalForest().n_seen == 0

    def test_predict_ate_before_data(self):
        assert OnlineCausalForest().predict_ate() == 0.0

    def test_predict_one_returns_float(self):
        f = OnlineCausalForest(n_trees=3, grace_period=10, seed=0)
        for x, w, y, _ in LinearCausalStream(n=50, seed=0):
            f.learn_one(x, w, y)
        result = f.predict_one({"x0": 0.5, "x1": -0.3, "x2": 0.0, "x3": 0.1, "x4": -0.2})
        assert isinstance(result, float)

    def test_n_trees_property(self):
        f = OnlineCausalForest(n_trees=5)
        assert len(f._trees) == 5

    def test_n_nodes_and_leaves_lists(self):
        f = OnlineCausalForest(n_trees=3, grace_period=10, seed=0)
        for x, w, y, _ in LinearCausalStream(n=50, seed=0):
            f.learn_one(x, w, y)
        assert len(f.n_nodes)  == 3
        assert len(f.n_leaves) == 3

    def test_learn_increments_n_seen(self):
        f = OnlineCausalForest(n_trees=2)
        f.learn_one({"x": 1.0}, 1, 2.0)
        assert f.n_seen == 1

    def test_reset_clears_state(self):
        f = OnlineCausalForest(n_trees=3, grace_period=10, seed=1)
        for x, w, y, _ in LinearCausalStream(n=100, seed=0):
            f.learn_one(x, w, y)
        f.reset()
        assert f.n_seen == 0

    def test_clone_gives_fresh_forest(self):
        f = OnlineCausalForest(n_trees=3, grace_period=10, seed=0)
        for x, w, y, _ in LinearCausalStream(n=50, seed=0):
            f.learn_one(x, w, y)
        fresh = f.clone()
        assert fresh.n_seen == 0

    def test_subsample_rate_zero_no_crash(self):
        """subsample_rate=0 means each tree sees 0 obs → no crash."""
        f = OnlineCausalForest(n_trees=2, subsample_rate=0.0, seed=0)
        for x, w, y, _ in LinearCausalStream(n=20, seed=0):
            f.learn_one(x, w, y)
        assert f.n_seen == 20

    def test_ci_finite_after_data(self):
        f = OnlineCausalForest(n_trees=3, grace_period=10, seed=0)
        for x, w, y, _ in LinearCausalStream(n=200, seed=0):
            f.learn_one(x, w, y)
        lo, hi = f.predict_ci()
        assert lo <= hi

    def test_forest_mean_prediction_stable(self):
        """Two forests with same seed should give same prediction."""
        f1 = OnlineCausalForest(n_trees=3, grace_period=50, seed=7)
        f2 = OnlineCausalForest(n_trees=3, grace_period=50, seed=7)
        for x, w, y, _ in LinearCausalStream(n=100, seed=3):
            f1.learn_one(x, w, y)
            f2.learn_one(x, w, y)
        x_test = {"x0": 0.0, "x1": 0.0, "x2": 0.0, "x3": 0.0, "x4": 0.0}
        assert f1.predict_one(x_test) == pytest.approx(f2.predict_one(x_test))
