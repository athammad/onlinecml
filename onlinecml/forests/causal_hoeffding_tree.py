"""CausalHoeffdingTree: Hoeffding tree with a causal split criterion."""

from __future__ import annotations

import math
import random

from river.linear_model import LinearRegression, LogisticRegression

from onlinecml.base.base_estimator import BaseOnlineEstimator
from onlinecml.base.running_stats import RunningStats

# Precomputed z-scores for normal quantiles [0.10, 0.25, 0.50, 0.75, 0.90].
# Used to define candidate split thresholds from the running Gaussian of each feature.
# scipy.stats.norm.ppf([0.10, 0.25, 0.50, 0.75, 0.90])
_QUANTILE_ZSCORES: tuple[float, ...] = (-1.2816, -0.6745, 0.0, 0.6745, 1.2816)


class _LeafStats:
    """Per-child arm-wise outcome statistics for split-candidate scoring.

    Used exclusively inside ``_FeatureSplitStats`` to evaluate the causal
    split criterion.  Actual leaf predictions use the node's linear models.
    """

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
        """Mean-difference CATE estimate for this child."""
        return self.treated.mean - self.control.mean

    @property
    def n(self) -> int:
        """Total observations in both arms."""
        return self.treated.n + self.control.n


class _FeatureSplitStats:
    """Multi-threshold split statistics for one feature.

    Maintains ``len(_QUANTILE_ZSCORES)`` candidate split points, each defined
    by a z-score applied to the running Gaussian of the feature values.  Every
    incoming observation is routed to all candidates simultaneously, so each
    candidate independently accumulates left/right outcome stats.

    The best (threshold, score) pair is returned by ``best_split()`` at
    split-evaluation time.

    Notes
    -----
    All candidates receive every observation — no subsampling — so memory is
    ``O(len(_QUANTILE_ZSCORES))`` times the single-threshold cost.  With 5
    quantiles this is a 5× factor per leaf per feature.
    """

    __slots__ = ("feat_stats", "candidates")

    def __init__(self) -> None:
        self.feat_stats = RunningStats()
        self.candidates: list[tuple[_LeafStats, _LeafStats]] = [
            (_LeafStats(), _LeafStats()) for _ in _QUANTILE_ZSCORES
        ]

    def update(self, feat_val: float, outcome: float, treatment: int) -> None:
        """Route one observation to all candidate children.

        Parameters
        ----------
        feat_val : float
            Feature value for this observation.
        outcome : float
            Observed outcome.
        treatment : int
            Treatment indicator (0 or 1).
        """
        mu  = self.feat_stats.mean
        std = self.feat_stats.std if self.feat_stats.n >= 2 else 1.0
        self.feat_stats.update(feat_val)
        for (left, right), z in zip(self.candidates, _QUANTILE_ZSCORES):
            thresh = mu + std * z
            if feat_val <= thresh:
                left.update(outcome, treatment)
            else:
                right.update(outcome, treatment)

    def best_split(
        self,
        cate_global: float,
        min_arm_samples: int,
    ) -> tuple[float, float]:
        """Find the best candidate threshold and its CATE-variance split score.

        Parameters
        ----------
        cate_global : float
            Current leaf CATE (baseline for the between-child variance score).
        min_arm_samples : int
            Minimum per-arm observations required in each child to score a
            candidate.

        Returns
        -------
        score : float
            Best weighted between-child CATE variance. ``-inf`` if no candidate
            has sufficient per-arm data.
        threshold : float
            Feature threshold corresponding to the best score.
        """
        mu  = self.feat_stats.mean
        std = self.feat_stats.std if self.feat_stats.n >= 2 else 1.0

        best_score  = float("-inf")
        best_thresh = mu  # default: median (z=0)

        for (left, right), z in zip(self.candidates, _QUANTILE_ZSCORES):
            if (left.treated.n  < min_arm_samples or
                    left.control.n  < min_arm_samples or
                    right.treated.n < min_arm_samples or
                    right.control.n < min_arm_samples):
                continue

            n_left  = left.n
            n_right = right.n
            n_total = n_left + n_right
            if n_total == 0:
                continue

            score = (
                (n_left  / n_total) * (left.cate  - cate_global) ** 2 +
                (n_right / n_total) * (right.cate - cate_global) ** 2
            )
            if score > best_score:
                best_score  = score
                best_thresh = mu + std * z

        return best_score, best_thresh


class _Node:
    """Internal node or leaf in the CausalHoeffdingTree.

    Leaf nodes maintain three online models for doubly robust CATE estimation:
    a treated outcome model ``mu1``, a control outcome model ``mu0``, and a
    propensity score model ``ps``.  Predictions use ``mu1(x) - mu0(x)``
    (individual CATE) rather than the raw arm means.  The DR-corrected running
    ATE is stored in ``dr_stats`` and used as the global-leaf CATE baseline for
    split scoring.
    """

    __slots__ = (
        "feature", "threshold", "left", "right",
        "stats", "n_since_split",
        "treated_model", "control_model", "ps_model", "dr_stats",
    )

    def __init__(self) -> None:
        self.feature: str | None = None
        self.threshold: float | None = None
        self.left: "_Node | None" = None
        self.right: "_Node | None" = None
        self.stats: _LeafStats = _LeafStats()        # arm counts + raw mean-diff
        self.n_since_split: int = 0
        self.treated_model: LinearRegression = LinearRegression()
        self.control_model: LinearRegression = LinearRegression()
        self.ps_model: LogisticRegression    = LogisticRegression()
        self.dr_stats: RunningStats          = RunningStats()

    @property
    def is_leaf(self) -> bool:
        """True when this node has no children."""
        return self.left is None

    def predict_cate(self, x: dict, min_arm_samples: int) -> float:
        """Predict CATE using the leaf's linear models when enough data exists.

        Falls back to the DR-corrected running mean, then to raw
        mean-difference, when models are undertrained.

        Parameters
        ----------
        x : dict
            Covariate dictionary for the query point.
        min_arm_samples : int
            Minimum per-arm observations before using the linear models.

        Returns
        -------
        float
            CATE estimate for ``x``.
        """
        if (self.stats.treated.n >= min_arm_samples and
                self.stats.control.n >= min_arm_samples):
            return (self.treated_model.predict_one(x)
                    - self.control_model.predict_one(x))
        if self.dr_stats.n >= 2:
            return self.dr_stats.mean
        return self.stats.cate


class CausalHoeffdingTree(BaseOnlineEstimator):
    """Online causal tree with a CATE-variance split criterion.

    Grows a binary decision tree one observation at a time using the
    **Hoeffding bound** to guarantee that splits are chosen with high
    probability from the same feature as a batch learner would choose,
    given enough data.

    **Improvements over a naive causal tree:**

    - *Multi-threshold split search*: instead of a single running-mean
      threshold, evaluates 5 quantile-based candidates per feature and picks
      the best, improving split location accuracy.
    - *Linear leaf models*: each leaf maintains separate River
      ``LinearRegression`` models for the treated and control arms.
      ``predict_one`` returns ``mu1(x) - mu0(x)`` (individual CATE) rather
      than a flat leaf mean.
    - *Doubly robust leaf CATE*: the per-leaf ATE baseline used for split
      scoring is the running mean of the DR pseudo-outcome
      ``mu1 - mu0 + W(Y-mu1)/p - (1-W)(Y-mu0)/(1-p)``, correcting for
      within-leaf confounding.

    Parameters
    ----------
    grace_period : int
        Minimum observations a leaf must collect before attempting a split.
        Default 200.
    delta : float
        Confidence parameter for the Hoeffding bound. Default 1e-5.
    tau : float
        Tie-breaking threshold. Default 0.05.
    max_depth : int or None
        Maximum tree depth. ``None`` = unlimited. Default 10.
    min_arm_samples : int
        Minimum per-arm observations required per child for split scoring and
        for switching to linear-model predictions. Default 5.
    mtry : int or None
        Number of features randomly considered at each split attempt.
        ``None`` = all features. Default None.
    outcome_range : float
        Upper bound on ``|CATE|`` for calibrating the Hoeffding bound.
        Default 10.0.
    clip_ps : float
        Propensity score clipping bounds ``[clip_ps, 1 - clip_ps]`` for DR
        correction within leaves. Default 0.1.
    seed : int or None
        Random seed for the mtry RNG. Default None.

    Notes
    -----
    Split score (maximised):

    .. math::

        \\text{score}(j) = \\frac{n_L}{n}(\\hat{\\tau}_L - \\hat{\\tau})^2
                         + \\frac{n_R}{n}(\\hat{\\tau}_R - \\hat{\\tau})^2

    where :math:`\\hat{\\tau}` is the DR-corrected leaf CATE and
    :math:`\\hat{\\tau}_k = \\bar{Y}_{1,k} - \\bar{Y}_{0,k}` for child ``k``.

    References
    ----------
    Domingos, P. and Hulten, G. (2000). Mining high-speed data streams.
    KDD, 71-80.

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
        mtry: int | None = None,
        outcome_range: float = 10.0,
        clip_ps: float = 0.1,
        seed: int | None = None,
    ) -> None:
        self.grace_period    = grace_period
        self.delta           = delta
        self.tau             = tau
        self.max_depth       = max_depth
        self.min_arm_samples = min_arm_samples
        self.mtry            = mtry
        self.outcome_range   = outcome_range
        self.clip_ps         = clip_ps
        self.seed            = seed

        self._rng = random.Random(seed)
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

        Uses a predict-first-then-learn protocol: all three leaf models
        (treated, control, propensity) predict *before* being updated, so the
        DR pseudo-outcome is computed from out-of-sample predictions.

        Parameters
        ----------
        x : dict
            Covariate dictionary.
        treatment : int
            Treatment indicator (0 or 1).
        outcome : float
            Observed outcome.
        propensity : float or None
            If provided, uses this logged propensity instead of the leaf PS
            model for the DR correction.
        """
        self._n_seen += 1
        node, depth = self._route(x)

        # ── Predict-first: all models predict before any update ─────────────
        mu1 = node.treated_model.predict_one(x)
        mu0 = node.control_model.predict_one(x)

        if propensity is not None:
            p_hat = max(self.clip_ps, min(1.0 - self.clip_ps, propensity))
        else:
            raw = node.ps_model.predict_proba_one(x)
            p_hat = max(self.clip_ps, min(1.0 - self.clip_ps, raw.get(1, 0.5)))

        # ── DR pseudo-outcome ────────────────────────────────────────────────
        psi = (
            mu1 - mu0
            + treatment * (outcome - mu1) / p_hat
            - (1 - treatment) * (outcome - mu0) / (1.0 - p_hat)
        )

        # ── Update leaf-level statistics ─────────────────────────────────────
        node.stats.update(outcome, treatment)
        node.dr_stats.update(psi)
        node.n_since_split += 1

        # ── Learn-after: update models on current observation ────────────────
        if treatment == 1:
            node.treated_model.learn_one(x, outcome)
        else:
            node.control_model.learn_one(x, outcome)
        if propensity is None:
            node.ps_model.learn_one(x, treatment)

        # ── Update per-feature split statistics ──────────────────────────────
        leaf_id = id(node)
        if leaf_id not in self._leaf_split_stats:
            self._leaf_split_stats[leaf_id] = {}
        for feat, val in x.items():
            if feat not in self._leaf_split_stats[leaf_id]:
                self._leaf_split_stats[leaf_id][feat] = _FeatureSplitStats()
            self._leaf_split_stats[leaf_id][feat].update(val, outcome, treatment)

        # ── Update global ATE estimate (DR-corrected) ─────────────────────
        self._ate_stats.update(psi)

        # ── Attempt split ────────────────────────────────────────────────────
        if node.n_since_split >= self.grace_period:
            if self.max_depth is None or depth < self.max_depth:
                self._try_split(node, depth)

    def predict_one(self, x: dict) -> float:
        """Predict CATE for a single unit using the leaf's linear models.

        Parameters
        ----------
        x : dict
            Covariate dictionary.

        Returns
        -------
        float
            Estimated CATE: ``mu1(x) - mu0(x)`` from the leaf's linear models
            when enough per-arm data exists; DR-corrected mean otherwise.
        """
        node, _ = self._route(x)
        return node.predict_cate(x, self.min_arm_samples)

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

    def _try_split(self, node: _Node, depth: int) -> None:
        """Evaluate causal split candidates and split if the Hoeffding bound allows."""
        leaf_id    = id(node)
        feat_stats = self._leaf_split_stats.get(leaf_id, {})
        if not feat_stats:
            return

        # Use DR-corrected CATE as the global baseline τ
        cate_global = node.dr_stats.mean if node.dr_stats.n >= 2 else node.stats.cate

        n = node.n_since_split
        best_score   = float("-inf")
        best_feat    = None
        best_thresh  = 0.0
        second_score = float("-inf")

        # Optional mtry: randomly restrict candidate features
        candidates = list(feat_stats.keys())
        if self.mtry is not None and self.mtry < len(candidates):
            candidates = self._rng.sample(candidates, self.mtry)

        for feat in candidates:
            score, thresh = feat_stats[feat].best_split(cate_global, self.min_arm_samples)
            if score > best_score:
                second_score = best_score
                best_score   = score
                best_feat    = feat
                best_thresh  = thresh
            elif score > second_score:
                second_score = score

        if best_feat is None or best_score <= 0:
            return

        R = self.outcome_range
        hoeffding_bound = math.sqrt(R * R * math.log(1.0 / self.delta) / (2.0 * n))

        gap = best_score - max(second_score, 0.0)
        if gap > hoeffding_bound or hoeffding_bound < self.tau:
            self._split_node(node, best_feat, best_thresh, leaf_id)

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
        """Recursively count all nodes."""
        if node is None:
            return 0
        return 1 + self._count_nodes(node.left) + self._count_nodes(node.right)

    @property
    def n_leaves(self) -> int:
        """Number of leaf nodes."""
        return self._count_leaves(self._root)

    def _count_leaves(self, node: _Node | None) -> int:
        """Recursively count leaf nodes."""
        if node is None:
            return 0
        if node.is_leaf:
            return 1
        return self._count_leaves(node.left) + self._count_leaves(node.right)
