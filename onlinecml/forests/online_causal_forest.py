"""OnlineCausalForest: ensemble of CausalHoeffdingTrees with subsampling."""

from __future__ import annotations

import math
import random

from onlinecml.base.base_estimator import BaseOnlineEstimator
from onlinecml.base.running_stats import RunningStats
from onlinecml.forests.causal_hoeffding_tree import CausalHoeffdingTree


class OnlineCausalForest(BaseOnlineEstimator):
    """Ensemble of CausalHoeffdingTrees for online CATE estimation.

    Grows ``n_trees`` independent ``CausalHoeffdingTree`` instances
    in parallel. Each tree receives a random subsample of each observation
    (Poisson bootstrap / "online bagging"), following Oza (2001). The
    forest CATE prediction is the mean of all tree predictions.

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
        Expected fraction of observations each tree sees (Poisson
        bootstrap parameter ``lambda``). 1.0 = standard online bagging.
        Default 1.0.
    min_arm_samples : int
        Passed to each tree. Default 5.
    seed : int or None
        Random seed for the subsampling RNG.

    Notes
    -----
    Online bagging (Oza 2001): each incoming observation is presented to
    tree ``k`` exactly ``Poisson(subsample_rate)`` times.  This simulates
    drawing with replacement from the stream.  The Poisson weight is
    approximated by drawing from ``Poisson(lambda)`` using the RNG
    ``self._rng``.

    References
    ----------
    Oza, N.C. (2001). Online bagging and boosting. Proceedings of the
    American Statistical Association, 229-234.

    Wager, S. and Athey, S. (2018). Estimation and inference of heterogeneous
    treatment effects using random forests. Journal of the American
    Statistical Association, 113(523), 1228-1242.

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
        min_arm_samples: int = 5,
        seed: int | None = None,
    ) -> None:
        self.n_trees         = n_trees
        self.grace_period    = grace_period
        self.delta           = delta
        self.tau             = tau
        self.max_depth       = max_depth
        self.subsample_rate  = subsample_rate
        self.min_arm_samples = min_arm_samples
        self.seed            = seed

        self._rng = random.Random(seed)
        self._trees: list[CausalHoeffdingTree] = [
            CausalHoeffdingTree(
                grace_period    = grace_period,
                delta           = delta,
                tau             = tau,
                max_depth       = max_depth,
                min_arm_samples = min_arm_samples,
                seed            = (seed + i) if seed is not None else None,
            )
            for i in range(n_trees)
        ]
        self._n_seen: int = 0
        self._ate_stats: RunningStats = RunningStats()

    def learn_one(
        self,
        x: dict,
        treatment: int,
        outcome: float,
        propensity: float | None = None,
    ) -> None:
        """Process one observation, updating all trees via online bagging.

        Parameters
        ----------
        x : dict
            Covariate dictionary.
        treatment : int
            Treatment indicator (0 or 1).
        outcome : float
            Observed outcome.
        propensity : float or None
            Ignored (trees do not use propensity internally).
        """
        self._n_seen += 1

        for tree in self._trees:
            # Online bagging: Poisson(subsample_rate) replications
            k = self._poisson(self.subsample_rate)
            for _ in range(k):
                tree.learn_one(x, treatment, outcome)

        cate_hat = self.predict_one(x)
        self._ate_stats.update(cate_hat)

    def predict_one(self, x: dict) -> float:
        """Predict CATE for a single unit as the mean of all tree predictions.

        Parameters
        ----------
        x : dict
            Covariate dictionary.

        Returns
        -------
        float
            Mean CATE prediction across all trees. Returns ``0.0`` when no
            tree has learned any data yet.
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

    def _poisson(self, lam: float) -> int:
        """Draw from Poisson(lam) using Knuth's algorithm (lam ≤ 20)."""
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
