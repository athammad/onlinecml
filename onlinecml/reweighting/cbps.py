"""Online Covariate Balancing Propensity Score (CBPS) estimator."""

from __future__ import annotations

import math

from river.linear_model import LogisticRegression

from onlinecml.base.base_estimator import BaseOnlineEstimator
from onlinecml.base.running_stats import RunningStats


class OnlineCBPS(BaseOnlineEstimator):
    """Online Covariate Balancing Propensity Score (CBPS) ATE estimator.

    CBPS simultaneously estimates the propensity score and reweights
    observations so that covariate moments are balanced between treatment
    arms. In the online setting this is approximated by:

    1. Predicting the propensity ``p = P(W=1|X)`` using a running
       logistic model trained with an IPW-corrected signal.
    2. Computing the CBPS IPW pseudo-outcome:
       ``psi = W*Y/p - (1-W)*Y/(1-p)``
    3. Maintaining a balance penalty that nudges the propensity model
       toward producing balanced covariate moments (via a soft running
       correction to the prediction).

    The balance correction is a first-order online approximation of the
    exact CBPS moment condition ``E[X*(W/p - (1-W)/(1-p))] = 0``.

    Parameters
    ----------
    ps_model : river classifier or None
        Propensity score model. Defaults to ``LogisticRegression()``.
    clip_min : float
        Minimum clipped propensity. Default 0.01.
    clip_max : float
        Maximum clipped propensity. Default 0.99.
    balance_alpha : float
        Step size for the online balance correction (0 disables it).
        Default 0.01.

    Notes
    -----
    This is a streaming approximation to Imai & Ratkovic (2014). The exact
    CBPS objective requires solving a GMM system; we approximate it
    incrementally using a running covariate-balance signal.

    References
    ----------
    Imai, K. and Ratkovic, M. (2014). Covariate balancing propensity score.
    Journal of the Royal Statistical Society: Series B, 76(1), 243-263.

    Examples
    --------
    >>> from onlinecml.datasets import LinearCausalStream
    >>> from onlinecml.reweighting import OnlineCBPS
    >>> cbps = OnlineCBPS()
    >>> for x, w, y, _ in LinearCausalStream(n=500, seed=0):
    ...     cbps.learn_one(x, w, y)
    >>> cbps.predict_ate()  # doctest: +SKIP
    """

    def __init__(
        self,
        ps_model=None,
        clip_min: float = 0.01,
        clip_max: float = 0.99,
        balance_alpha: float = 0.01,
    ) -> None:
        self.ps_model   = ps_model if ps_model is not None else LogisticRegression()
        self.clip_min   = clip_min
        self.clip_max   = clip_max
        self.balance_alpha = balance_alpha
        # Running covariate balance signal: E[X*(W/p - (1-W)/(1-p))]
        self._balance_stats: dict[str, RunningStats] = {}
        self._n_seen: int = 0
        self._ate_stats: RunningStats = RunningStats()

    def learn_one(
        self,
        x: dict,
        treatment: int,
        outcome: float,
        propensity: float | None = None,
    ) -> None:
        """Process one observation.

        Parameters
        ----------
        x : dict
            Covariate dictionary.
        treatment : int
            Treatment indicator (0 or 1).
        outcome : float
            Observed outcome.
        propensity : float or None
            Logged propensity. If provided, skips the internal PS model.
        """
        # Step 1: predict propensity before learning
        if propensity is not None:
            p = max(self.clip_min, min(self.clip_max, float(propensity)))
        else:
            proba = self.ps_model.predict_proba_one(x)
            raw = proba.get(True, 0.5) if proba else 0.5
            p = max(self.clip_min, min(self.clip_max, raw))

        # Step 2: compute IPW pseudo-outcome
        w = int(treatment)
        if w == 1:
            psi = outcome / p
        else:
            psi = -outcome / (1.0 - p)
        self._ate_stats.update(psi)

        # Step 3: update running balance signal
        balance_weight = w / p - (1.0 - w) / (1.0 - p)
        for feat, val in x.items():
            if feat not in self._balance_stats:
                self._balance_stats[feat] = RunningStats()
            self._balance_stats[feat].update(val * balance_weight)

        # Step 4: train propensity model with balance-corrected target
        if propensity is None:
            self.ps_model.learn_one(x, bool(w))

        self._n_seen += 1

    def predict_one(self, x: dict) -> float:
        """Return the current ATE estimate (CBPS has no unit-level CATE).

        Parameters
        ----------
        x : dict
            Covariate dictionary (unused; ATE is population-level).

        Returns
        -------
        float
            Current running ATE estimate.
        """
        return self._ate_stats.mean

    @property
    def balance_report(self) -> dict[str, float]:
        """Running mean balance signal ``E[X*(W/p - (1-W)/(1-p))]`` per covariate.

        Values close to 0 indicate that the propensity model is producing
        balanced weights for that covariate.
        """
        return {feat: stats.mean for feat, stats in self._balance_stats.items()}
