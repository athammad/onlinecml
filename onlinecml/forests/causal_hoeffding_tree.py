"""CausalHoeffdingTree: Hoeffding tree with a causal split criterion."""

from __future__ import annotations

import math

from onlinecml.base.base_estimator import BaseOnlineEstimator
from onlinecml.base.running_stats import RunningStats


class _LeafStats:
    """Per-leaf outcome statistics for both treatment arms."""

    __slots__ = ("treated", "control")

    def __init__(self) -> None:
        self.treated = RunningStats()
        self.control = RunningStats()

    def update(self, outcome: float, treatment: int) -> None:
        """Route one outcome to the appropriate arm."""
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


class _FeatureSplitStats:
    """Online sufficient statistics for one candidate split on a single feature.

    Tracks two sets of outcome statistics (one per child) by routing each
    incoming observation to the left child if ``x_j <= threshold`` and to
    the right child otherwise. The threshold is the running mean of the
    feature values seen so far — an unbiased estimate of the median for
    symmetric distributions.

    Note: because the threshold (mean) shifts as data arrives, early
    observations may be re-classified in hindsight. This is an approximation
    that works well in practice for the Hoeffding split criterion.
    """

    __slots__ = ("feat_stats", "left", "right")

    def __init__(self) -> None:
        self.feat_stats = RunningStats()  # tracks x_j values for threshold estimation
        self.left  = _LeafStats()         # outcomes for x_j <= threshold
        self.right = _LeafStats()         # outcomes for x_j >  threshold

    def update(self, feat_val: float, outcome: float, treatment: int) -> None:
        """Route one observation and update the appropriate child's outcome stats.

        Parameters
        ----------
        feat_val : float
            The value of this feature for the current observation.
        outcome : float
            Observed outcome.
        treatment : int
            Treatment indicator (0 or 1).
        """
        threshold = self.feat_stats.mean   # threshold = current running mean
        self.feat_stats.update(feat_val)   # update threshold estimate after routing
        if feat_val <= threshold:
            self.left.update(outcome, treatment)
        else:
            self.right.update(outcome, treatment)

    @property
    def threshold(self) -> float:
        """Current split threshold (running mean of the feature values)."""
        return self.feat_stats.mean


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

    **Novel causal split criterion:** instead of minimising prediction
    error on ``Y``, each candidate split is scored by the between-child
    **CATE variance** — how much the two children differ in their
    estimated treatment effects. A split is triggered when the Hoeffding
    bound guarantees (with probability ``1 - delta``) that the best
    feature is at least ``tau`` better than the second-best.

    For each leaf and each feature ``j``, outcome statistics are maintained
    separately for the two children defined by splitting at the running
    mean of ``x_j``. This requires ``O(depth × features)`` memory.

    Parameters
    ----------
    grace_period : int
        Minimum number of observations a leaf must collect before
        attempting a split. Default 200.
    delta : float
        Confidence parameter for the Hoeffding bound. Smaller values
        require a larger difference between best and second-best split
        before committing. Default 1e-5.
    tau : float
        Tie-breaking threshold: do not split if the best and second-best
        splits are within ``tau`` of each other. Default 0.05.
    max_depth : int or None
        Maximum tree depth. ``None`` = unlimited. Default 10.
    min_arm_samples : int
        Minimum observations per treatment arm required in each child
        before its CATE estimate is used in the split score. Default 5.
    outcome_range : float
        Upper bound on the absolute value of the split score. Used to
        calibrate the Hoeffding bound. Set to approximately
        ``max(|CATE|)`` for your problem. Default 10.0.

    Notes
    -----
    The CATE-variance split score for a binary split ``(left, right)`` is:

    .. math::

        \\text{score}(j) = \\frac{n_L}{n}(\\hat{\\tau}_L - \\hat{\\tau})^2
                         + \\frac{n_R}{n}(\\hat{\\tau}_R - \\hat{\\tau})^2

    where :math:`\\hat{\\tau}_k = \\bar{Y}_{1,k} - \\bar{Y}_{0,k}` is the
    mean-difference CATE estimate in child ``k``, and :math:`\\hat{\\tau}` is
    the overall leaf CATE. This score is **maximised** — we want children
    whose CATEs differ as much as possible from the leaf average.

    The split threshold for feature ``j`` is its running mean (an online
    approximation to the median). This uses ``O(d)`` memory per leaf.

    References
    ----------
    Domingos, P. and Hulten, G. (2000). Mining high-speed data streams.
    Proceedings of KDD, 71-80.

    Wager, S. and Athey, S. (2018). Estimation and inference of heterogeneous
    treatment effects using random forests. JASA, 113(523), 1228-1242.

    Examples
    --------
    >>> from onlinecml.datasets import HeterogeneousCausalStream
    >>> from onlinecml.forests import CausalHoeffdingTree
    >>> tree = CausalHoeffdingTree(grace_period=50, delta=0.01, seed=42)
    >>> for x, w, y, _ in HeterogeneousCausalStream(n=1000, seed=0):
    ...     tree.learn_one(x, w, y)
    >>> isinstance(tree.predict_one({'x0': 1.0, 'x1': 0.0, 'x2': 0.0, 'x3': 0.0, 'x4': 0.0}), float)
    True
    """

    def __init__(
        self,
        grace_period: int = 200,
        delta: float = 1e-5,
        tau: float = 0.05,
        max_depth: int | None = 10,
        min_arm_samples: int = 5,
        outcome_range: float = 10.0,
        seed: int | None = None,
    ) -> None:
        self.grace_period    = grace_period
        self.delta           = delta
        self.tau             = tau
        self.max_depth       = max_depth
        self.min_arm_samples = min_arm_samples
        self.outcome_range   = outcome_range
        self.seed            = seed

        self._root: _Node = _Node()
        self._n_seen: int = 0
        self._ate_stats: RunningStats = RunningStats()

        # Per-leaf per-feature split statistics: {id(node) → {feat → _FeatureSplitStats}}
        self._leaf_split_stats: dict[int, dict[str, _FeatureSplitStats]] = {}

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

        # Update leaf outcome stats
        node.stats.update(outcome, treatment)
        node.n_since_split += 1

        # Update per-feature split statistics for this leaf
        leaf_id = id(node)
        if leaf_id not in self._leaf_split_stats:
            self._leaf_split_stats[leaf_id] = {}
        for feat, val in x.items():
            if feat not in self._leaf_split_stats[leaf_id]:
                self._leaf_split_stats[leaf_id][feat] = _FeatureSplitStats()
            self._leaf_split_stats[leaf_id][feat].update(val, outcome, treatment)

        # Update global ATE estimate
        self._ate_stats.update(node.stats.cate)

        # Attempt split
        if node.n_since_split >= self.grace_period:
            if self.max_depth is None or depth < self.max_depth:
                self._try_split(node, depth)

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

    def _split_score(self, node: _Node, feat: str) -> float:
        """Compute the causal split score for ``feat`` at its running-mean threshold.

        Returns the weighted between-child CATE variance (higher = better split).
        Returns ``-inf`` if there is insufficient data to estimate either child's CATE.

        Parameters
        ----------
        node : _Node
            The leaf node to evaluate.
        feat : str
            Feature name to evaluate as a split candidate.

        Returns
        -------
        float
            Weighted CATE variance between the two children.
        """
        leaf_id = id(node)
        split_data = self._leaf_split_stats.get(leaf_id, {}).get(feat)
        if split_data is None:
            return float("-inf")

        left  = split_data.left
        right = split_data.right

        # Require enough per-arm data in both children
        if (left.treated.n  < self.min_arm_samples or
                left.control.n  < self.min_arm_samples or
                right.treated.n < self.min_arm_samples or
                right.control.n < self.min_arm_samples):
            return float("-inf")

        n_left  = left.n
        n_right = right.n
        n_total = n_left + n_right
        if n_total == 0:
            return float("-inf")

        cate_global = node.stats.cate
        cate_left   = left.cate
        cate_right  = right.cate

        # Weighted between-child CATE variance: maximise this
        score = (
            (n_left  / n_total) * (cate_left  - cate_global) ** 2 +
            (n_right / n_total) * (cate_right - cate_global) ** 2
        )
        return score

    def _try_split(self, node: _Node, depth: int) -> None:
        """Evaluate causal split candidates and split if the Hoeffding bound allows."""
        leaf_id    = id(node)
        feat_stats = self._leaf_split_stats.get(leaf_id, {})
        if not feat_stats:
            return

        n = node.n_since_split
        best_score   = float("-inf")
        best_feat    = None
        second_score = float("-inf")

        for feat in feat_stats:
            score = self._split_score(node, feat)
            if score > best_score:
                second_score = best_score
                best_score   = score
                best_feat    = feat
            elif score > second_score:
                second_score = score

        if best_feat is None or best_score <= 0:
            return

        # Hoeffding bound: R = outcome_range (max |CATE| estimate)
        # Score is a CATE variance, bounded by outcome_range^2
        R = self.outcome_range
        hoeffding_bound = math.sqrt(R * R * math.log(1.0 / self.delta) / (2.0 * n))

        gap = best_score - max(second_score, 0.0)
        if gap > hoeffding_bound or hoeffding_bound < self.tau:
            thresh = feat_stats[best_feat].threshold
            self._split_node(node, best_feat, thresh, leaf_id)

    def _split_node(
        self,
        node: _Node,
        feature: str,
        threshold: float,
        leaf_id: int,
    ) -> None:
        """Convert a leaf into an internal node."""
        node.feature   = feature
        node.threshold = threshold
        node.left      = _Node()
        node.right     = _Node()
        node.n_since_split = 0
        self._leaf_split_stats.pop(leaf_id, None)

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
