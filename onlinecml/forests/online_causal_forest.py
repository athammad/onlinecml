"""OnlineCausalForest: ensemble of CausalHoeffdingTrees with subsampling."""

from __future__ import annotations

import math
import random

from river.drift import ADWIN

from onlinecml.base.base_estimator import BaseOnlineEstimator
from onlinecml.base.running_stats import RunningStats
from onlinecml.forests.causal_hoeffding_tree import CausalHoeffdingTree


class OnlineCausalForest(BaseOnlineEstimator):
    """Ensemble of CausalHoeffdingTrees for online CATE estimation.

    Grows ``n_trees`` independent ``CausalHoeffdingTree`` instances in parallel.
    Each tree receives a random subsample of each observation (Poisson bootstrap,
    Oza 2001).  The forest CATE prediction is the mean of all tree predictions.

    Each tree is monitored for concept drift via an ADWIN detector on its
    normalised prediction signal.  On drift detection the affected tree is
    reset and starts growing from scratch, while the remaining trees continue
    uninterrupted.

    Parameters
    ----------
    n_trees : int
        Number of trees in the ensemble. Default 10.
    grace_period : int
        Grace period passed to each ``CausalHoeffdingTree``. Default 200.
    delta : float
        Hoeffding confidence parameter for each tree. Default 1e-5.
    tau : float
        Tie-breaking threshold for each tree. Default 0.05.
    max_depth : int or None
        Maximum tree depth. Default 10.
    subsample_rate : float
        Expected number of times each tree sees each observation (Poisson
        bootstrap ``lambda``). 1.0 = standard online bagging. Default 1.0.
    mtry : int or None
        Number of features randomly considered at each split attempt per tree.
        ``None`` = all features. ``int(sqrt(p))`` is a common choice when many
        features are informative. Default None.
    min_arm_samples : int
        Passed to each tree. Default 5.
    outcome_range : float
        Passed to each tree. Upper bound on ``|CATE|`` for calibrating the
        Hoeffding bound. Default 10.0.
    clip_ps : float
        Propensity clipping bounds for DR correction within leaves. Default 0.1.
    drift_detection : bool
        If ``True``, attach an ADWIN detector to each tree and reset trees on
        drift. Default True.
    seed : int or None
        Random seed for the subsampling RNG.

    Notes
    -----
    Online bagging (Oza 2001): each incoming observation is presented to tree
    ``k`` exactly ``Poisson(subsample_rate)`` times.

    Drift detection follows the ARF approach: each tree's prediction is
    normalised to ``[0, 1]`` using the running ``mean ± 3σ`` window and fed to
    ADWIN.  When ADWIN raises an alarm, the tree and its detector are reset.

    References
    ----------
    Oza, N.C. (2001). Online bagging and boosting. Proc. American Statistical
    Association, 229-234.

    Gomes, H.M. et al. (2017). Adaptive random forests for evolving data stream
    classification. Machine Learning, 106(9), 1469-1495.

    Examples
    --------
    >>> from onlinecml.datasets import LinearCausalStream
    >>> from onlinecml.forests import OnlineCausalForest
    >>> forest = OnlineCausalForest(n_trees=5, grace_period=50, seed=0)
    >>> for x, w, y, _ in LinearCausalStream(n=500, seed=0):
    ...     forest.learn_one(x, w, y)
    >>> isinstance(forest.predict_one({'x0': 0.5, 'x1': -0.3, 'x2': 0.0, 'x3': 0.1, 'x4': -0.2}), float)
    True
    """

    def __init__(
        self,
        n_trees: int = 10,
        grace_period: int = 200,
        delta: float = 1e-5,
        tau: float = 0.05,
        max_depth: int | None = 10,
        subsample_rate: float = 1.0,
        mtry: int | None = None,
        min_arm_samples: int = 5,
        outcome_range: float = 10.0,
        clip_ps: float = 0.1,
        drift_detection: bool = True,
        seed: int | None = None,
    ) -> None:
        self.n_trees         = n_trees
        self.grace_period    = grace_period
        self.delta           = delta
        self.tau             = tau
        self.max_depth       = max_depth
        self.subsample_rate  = subsample_rate
        self.mtry            = mtry
        self.min_arm_samples = min_arm_samples
        self.outcome_range   = outcome_range
        self.clip_ps         = clip_ps
        self.drift_detection = drift_detection
        self.seed            = seed

        self._rng = random.Random(seed)
        self._trees: list[CausalHoeffdingTree] = [
            self._new_tree(i) for i in range(n_trees)
        ]
        self._n_seen: int = 0
        self._ate_stats: RunningStats = RunningStats()

        # Per-tree drift monitoring
        self._drift_detectors: list[ADWIN] | None = (
            [ADWIN() for _ in range(n_trees)] if drift_detection else None
        )
        self._pred_stats: list[RunningStats] | None = (
            [RunningStats() for _ in range(n_trees)] if drift_detection else None
        )

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
        """Process one observation, updating all trees via online bagging.

        After each tree update, optionally checks for concept drift using its
        ADWIN detector and resets the tree if drift is detected.

        Parameters
        ----------
        x : dict
            Covariate dictionary.
        treatment : int
            Treatment indicator (0 or 1).
        outcome : float
            Observed outcome.
        propensity : float or None
            If provided, passed to each tree's DR correction.
        """
        self._n_seen += 1

        for i, tree in enumerate(self._trees):
            k = self._poisson(self.subsample_rate)
            for _ in range(k):
                tree.learn_one(x, treatment, outcome, propensity)

            # ── Drift detection ─────────────────────────────────────────────
            if self._drift_detectors is not None:
                pred  = tree.predict_one(x)
                stats = self._pred_stats[i]  # type: ignore[index]
                stats.update(pred)
                if stats.n > 1:
                    sigma      = stats.std or 1e-8
                    normalised = (pred - stats.mean) / (6.0 * sigma) + 0.5
                    normalised = max(0.0, min(1.0, normalised))
                    if self._drift_detectors[i].update(normalised):
                        self._trees[i]            = self._new_tree(i)
                        self._drift_detectors[i]  = ADWIN()
                        self._pred_stats[i]       = RunningStats()  # type: ignore[index]

        self._ate_stats.update(self.predict_one(x))

    def predict_one(self, x: dict) -> float:
        """Predict CATE as the mean across all tree predictions.

        Parameters
        ----------
        x : dict
            Covariate dictionary.

        Returns
        -------
        float
            Mean CATE across all trees. Returns ``0.0`` when untrained.
        """
        preds = [t.predict_one(x) for t in self._trees]
        return sum(preds) / len(preds) if preds else 0.0

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def n_nodes(self) -> list[int]:
        """Number of nodes in each tree."""
        return [t.n_nodes for t in self._trees]

    @property
    def n_leaves(self) -> list[int]:
        """Number of leaf nodes in each tree."""
        return [t.n_leaves for t in self._trees]

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _new_tree(self, index: int) -> CausalHoeffdingTree:
        """Instantiate a fresh ``CausalHoeffdingTree`` for position ``index``.

        Parameters
        ----------
        index : int
            Tree index, used to derive a per-tree seed from ``self.seed``.

        Returns
        -------
        CausalHoeffdingTree
            A freshly initialised tree with this forest's hyperparameters.
        """
        return CausalHoeffdingTree(
            grace_period    = self.grace_period,
            delta           = self.delta,
            tau             = self.tau,
            max_depth       = self.max_depth,
            min_arm_samples = self.min_arm_samples,
            mtry            = self.mtry,
            outcome_range   = self.outcome_range,
            clip_ps         = self.clip_ps,
            seed            = (self.seed + index) if self.seed is not None else None,
        )

    def _poisson(self, lam: float) -> int:
        """Draw from Poisson(lam) using Knuth's algorithm (lam ≤ 20).

        Parameters
        ----------
        lam : float
            Poisson rate parameter.

        Returns
        -------
        int
            A non-negative integer sample from Poisson(lam).
        """
        if lam <= 0:
            return 0
        L = math.exp(-lam)
        k = 0
        p = 1.0
        while p > L:
            k += 1
            p *= self._rng.random()
        return k - 1

    def reset(self) -> None:
        """Reset the forest to its initial (untrained) state."""
        fresh = self.clone()
        self.__dict__.update(fresh.__dict__)
