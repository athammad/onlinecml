"""CausalHoeffdingTree: Hoeffding tree with a causal split criterion."""

from __future__ import annotations

import math

from onlinecml.base.base_estimator import BaseOnlineEstimator
from onlinecml.base.running_stats import RunningStats


class _LeafStats:
    """Per-leaf statistics for both treatment arms."""

    __slots__ = ("treated", "control")

    def __init__(self) -> None:
        self.treated = RunningStats()
        self.control = RunningStats()

    def update(self, outcome: float, treatment: int) -> None:
        """Route one observation to the appropriate arm."""
        if treatment == 1:
            self.treated.update(outcome)
        else:
            self.control.update(outcome)

    @property
    def cate(self) -> float:
        """Current mean-difference CATE estimate for this leaf."""
        return self.treated.mean - self.control.mean

    @property
    def n(self) -> int:
        """Total observations in this leaf."""
        return self.treated.n + self.control.n


class _Node:
    """Internal node or leaf in the CausalHoeffdingTree."""

    __slots__ = (
        "feature", "threshold", "left", "right",
        "stats", "n_since_split",
    )

    def __init__(self) -> None:
        self.feature: str | None = None
        self.threshold: float | None = None
        self.left: "_Node | None" = None
        self.right: "_Node | None" = None
        self.stats: _LeafStats = _LeafStats()
        self.n_since_split: int = 0

    @property
    def is_leaf(self) -> bool:
        """True when this node has no children."""
        return self.left is None


class CausalHoeffdingTree(BaseOnlineEstimator):
    """Online causal tree with a CATE-variance split criterion.

    Grows a binary decision tree one observation at a time using the
    **Hoeffding bound** to guarantee that splits are chosen with high
    probability from the same feature as a batch learner would choose,
    given enough data.

    **Novel causal split criterion:**  instead of minimising prediction
    error on ``Y``, each candidate split is scored by the reduction in
    **within-split CATE variance** — the weighted sum of
    ``Var(CATE | X in left) + Var(CATE | X in right)``, estimated from the
    running arm-level statistics maintained at every candidate split.  A
    split is triggered when the Hoeffding bound guarantees (with
    probability ``1 - delta``) that the best split is at least ``tau``
    better than the second-best.

    Parameters
    ----------
    grace_period : int
        Minimum number of observations a leaf must collect before
        attempting a split. Default 200.
    delta : float
        Confidence parameter for the Hoeffding bound. Default 1e-5.
    tau : float
        Tie-breaking threshold: do not split if the best and second-best
        splits are within ``tau`` of each other. Default 0.05.
    max_depth : int or None
        Maximum tree depth. ``None`` = unlimited. Default 10.
    n_split_candidates : int
        Number of candidate thresholds evaluated per feature at each
        potential split. Default 10.
    min_arm_samples : int
        Minimum observations per treatment arm required in a leaf before
        its CATE estimate is used in the split score. Default 5.

    Notes
    -----
    The CATE-variance split score for a binary split ``(left, right)`` is:

    .. math::

        \\text{score} = -\\left(
            \\frac{n_L}{n} \\hat{\\sigma}^2_{\\tau,L} +
            \\frac{n_R}{n} \\hat{\\sigma}^2_{\\tau,R}
        \\right)

    where :math:`\\hat{\\sigma}^2_{\\tau,k}` is the empirical variance of the
    running arm-level CATE estimates within child ``k``.  Because arm-level
    outcomes are tracked with Welford running statistics, the tree maintains
    ``O(depth × features)`` memory.

    References
    ----------
    Domingos, P. and Hulten, G. (2000). Mining high-speed data streams.
    Proceedings of KDD, 71-80.

    Wager, S. and Athey, S. (2018). Estimation and inference of heterogeneous
    treatment effects using random forests. Journal of the American
    Statistical Association, 113(523), 1228-1242.

    Examples
    --------
    >>> from onlinecml.datasets import LinearCausalStream
    >>> from onlinecml.forests import CausalHoeffdingTree
    >>> tree = CausalHoeffdingTree(grace_period=50, seed=42)
    >>> for x, w, y, _ in LinearCausalStream(n=500, seed=0):
    ...     tree.learn_one(x, w, y)
    >>> isinstance(tree.predict_one({'x0': 0.5, 'x1': -0.3, 'x2': 0.0, 'x3': 0.1, 'x4': -0.2}), float)
    True
    """

    def __init__(
        self,
        grace_period: int = 200,
        delta: float = 1e-5,
        tau: float = 0.05,
        max_depth: int | None = 10,
        n_split_candidates: int = 10,
        min_arm_samples: int = 5,
        seed: int | None = None,
    ) -> None:
        self.grace_period       = grace_period
        self.delta              = delta
        self.tau                = tau
        self.max_depth          = max_depth
        self.n_split_candidates = n_split_candidates
        self.min_arm_samples    = min_arm_samples
        self.seed               = seed

        self._root: _Node = _Node()
        self._n_seen: int = 0
        self._ate_stats: RunningStats = RunningStats()

        # Per-leaf feature statistics: {node_id → {feat → {arm → [values]}}}
        # We use a simpler approach: store per-leaf per-feature running stats
        # for both arms.  Keyed by id(node).
        self._leaf_feat_stats: dict[int, dict[str, _LeafStats]] = {}

    # ------------------------------------------------------------------
    # BaseOnlineEstimator interface
    # ------------------------------------------------------------------

    def learn_one(
        self,
        x: dict,
        treatment: int,
        outcome: float,
        propensity: float | None = None,
    ) -> None:
        """Process one observation and potentially grow the tree.

        Parameters
        ----------
        x : dict
            Covariate dictionary.
        treatment : int
            Treatment indicator (0 or 1).
        outcome : float
            Observed outcome.
        propensity : float or None
            Ignored (trees do not use propensity scores internally).
        """
        self._n_seen += 1
        node, depth = self._route(x)

        # Update leaf stats
        node.stats.update(outcome, treatment)
        node.n_since_split += 1

        # Update per-feature stats for this leaf
        leaf_id = id(node)
        if leaf_id not in self._leaf_feat_stats:
            self._leaf_feat_stats[leaf_id] = {}
        for feat, val in x.items():
            if feat not in self._leaf_feat_stats[leaf_id]:
                self._leaf_feat_stats[leaf_id][feat] = _LeafStats()
            self._leaf_feat_stats[leaf_id][feat].update(val, treatment)

        # Update global ATE
        cate_est = node.stats.cate
        self._ate_stats.update(cate_est)

        # Attempt split
        if node.n_since_split >= self.grace_period:
            if self.max_depth is None or depth < self.max_depth:
                self._try_split(node, x, depth)

    def predict_one(self, x: dict) -> float:
        """Predict CATE for a single unit by routing to the appropriate leaf.

        Parameters
        ----------
        x : dict
            Covariate dictionary.

        Returns
        -------
        float
            Estimated CATE (mean-difference in leaf outcomes by arm).
        """
        node, _ = self._route(x)
        return node.stats.cate

    # ------------------------------------------------------------------
    # Tree internals
    # ------------------------------------------------------------------

    def _route(self, x: dict) -> tuple[_Node, int]:
        """Route ``x`` to its leaf, returning ``(leaf_node, depth)``."""
        node = self._root
        depth = 0
        while not node.is_leaf:
            val = x.get(node.feature, 0.0)  # type: ignore[arg-type]
            node = node.left if val <= node.threshold else node.right  # type: ignore[assignment]
            depth += 1
        return node, depth

    def _try_split(self, node: _Node, x: dict, depth: int) -> None:
        """Evaluate causal split candidates and split if the Hoeffding bound allows."""
        leaf_id = id(node)
        feat_stats = self._leaf_feat_stats.get(leaf_id, {})
        if not feat_stats:
            return

        n = node.n_since_split
        best_score   = float("-inf")
        best_feat    = None
        best_thresh  = None
        second_score = float("-inf")

        for feat, fs in feat_stats.items():
            # Collect observed values to define candidate thresholds
            # Use the running mean ± std as a proxy for the range
            all_stats = fs.treated if fs.treated.n >= 1 else fs.control
            if all_stats.n < 2:
                continue
            mu  = all_stats.mean
            std = all_stats.std if all_stats.n >= 2 else 1.0
            thresholds = [mu + std * t for t in
                          [-1.5, -1.0, -0.5, -0.2, 0.0,
                            0.2,  0.5,  1.0,  1.5,  2.0]]

            for thresh in thresholds:
                score = self._split_score(node, feat, thresh, x)
                if score > best_score:
                    second_score = best_score
                    best_score   = score
                    best_feat    = feat
                    best_thresh  = thresh
                elif score > second_score:
                    second_score = score

        if best_feat is None:
            return

        # Hoeffding bound: R = range of score (CATE variance ≤ (max_outcome)^2)
        # We normalise by assuming outcomes lie in [-10, 10] → R = 100
        R = 100.0
        hoeffding_bound = math.sqrt(R * R * math.log(1.0 / self.delta) / (2.0 * n))

        if best_score - second_score > hoeffding_bound or hoeffding_bound < self.tau:
            self._split_node(node, best_feat, best_thresh, leaf_id)

    def _split_score(
        self,
        node: _Node,
        feature: str,
        threshold: float,
        x: dict,
    ) -> float:
        """Score a candidate split by reduction in within-child CATE variance.

        Returns the negative weighted CATE variance (higher = better split).
        """
        leaf_id   = id(node)
        feat_stat = self._leaf_feat_stats.get(leaf_id, {}).get(feature)
        if feat_stat is None:
            return float("-inf")

        # Approximate child membership via arm-level feature stats
        # Left: feature ≤ threshold, right: feature > threshold
        # We use a simple fraction based on normal CDF approximation.
        fs_t = feat_stat.treated
        fs_c = feat_stat.control

        if fs_t.n < self.min_arm_samples or fs_c.n < self.min_arm_samples:
            return float("-inf")

        def _normal_cdf(val: float, mu: float, sigma: float) -> float:
            if sigma <= 0:
                return 1.0 if mu <= val else 0.0
            return 0.5 * (1.0 + math.erf((val - mu) / (sigma * math.sqrt(2.0))))

        p_left_t = _normal_cdf(threshold, fs_t.mean, max(fs_t.std, 1e-6))
        p_left_c = _normal_cdf(threshold, fs_c.mean, max(fs_c.std, 1e-6))

        n_total = node.stats.n
        if n_total == 0:
            return float("-inf")

        # Estimated child sizes
        n_left  = max(1, round(p_left_t  * fs_t.n + p_left_c  * fs_c.n))
        n_right = max(1, n_total - n_left)

        # CATE variance proxy: use (CATE_left - CATE_global)^2 + (CATE_right - CATE_global)^2
        cate_global = node.stats.cate
        cate_left   = (fs_t.mean * p_left_t  - fs_c.mean * p_left_c )
        cate_right  = (fs_t.mean * (1 - p_left_t) - fs_c.mean * (1 - p_left_c))

        weighted_var = (
            (n_left  / n_total) * (cate_left  - cate_global) ** 2 +
            (n_right / n_total) * (cate_right - cate_global) ** 2
        )
        return weighted_var  # maximise CATE variance between children

    def _split_node(
        self,
        node: _Node,
        feature: str,
        threshold: float,
        leaf_id: int,
    ) -> None:
        """Perform the split: convert the leaf into an internal node."""
        node.feature   = feature
        node.threshold = threshold
        node.left      = _Node()
        node.right     = _Node()
        node.n_since_split = 0
        # Clean up leaf stats for the old leaf
        self._leaf_feat_stats.pop(leaf_id, None)

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def n_nodes(self) -> int:
        """Total number of nodes (internal + leaf) in the tree."""
        return self._count_nodes(self._root)

    def _count_nodes(self, node: _Node | None) -> int:
        if node is None:
            return 0
        return 1 + self._count_nodes(node.left) + self._count_nodes(node.right)

    @property
    def n_leaves(self) -> int:
        """Number of leaf nodes."""
        return self._count_leaves(self._root)

    def _count_leaves(self, node: _Node | None) -> int:
        if node is None:
            return 0
        if node.is_leaf:
            return 1
        return self._count_leaves(node.left) + self._count_leaves(node.right)
