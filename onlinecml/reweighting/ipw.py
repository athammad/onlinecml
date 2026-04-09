"""Online Inverse Probability Weighting (IPW) estimator."""

from river.linear_model import LogisticRegression

from onlinecml.base.base_estimator import BaseOnlineEstimator
from onlinecml.base.running_stats import RunningStats
from onlinecml.propensity.propensity_score import OnlinePropensityScore


class OnlineIPW(BaseOnlineEstimator):
    """Online Inverse Probability Weighting estimator for the ATE.

    Estimates the Average Treatment Effect via importance-weighted
    pseudo-outcomes, updated one observation at a time. The propensity
    score model is updated after each pseudo-outcome computation
    (predict-then-learn protocol).

    Parameters
    ----------
    ps_model : OnlinePropensityScore or None
        Propensity score model. If None, defaults to
        ``OnlinePropensityScore(LogisticRegression())``.
    clip_min : float
        Lower clip bound for propensity scores. Default 0.01.
    clip_max : float
        Upper clip bound for propensity scores. Default 0.99.
    normalize : bool
        If True, use normalized (stabilized) IPW weights by dividing
        by the running mean weight within each arm. Default False.

    Notes
    -----
    The IPW pseudo-outcome for observation ``i`` is:

    .. math::

        \\psi_i = \\frac{W_i Y_i}{\\hat{p}_i} - \\frac{(1-W_i) Y_i}{1 - \\hat{p}_i}

    The running mean of ``psi`` converges to the ATE under unconfoundedness,
    overlap, and SUTVA.

    **Predict-then-learn:** the propensity score is predicted *before*
    the classifier is updated on the current observation. This avoids
    look-ahead bias in the pseudo-outcome.

    **Limitation:** ``predict_one(x)`` returns the current running ATE
    estimate for any input ``x``. IPW does not produce individual CATE
    estimates. Use ``OnlineAIPW`` or meta-learners for individual CATE.

    **Cold start:** Before any training, propensity is 0.5, giving
    IPW weight = 2.0 regardless of treatment. The first ~50–100
    observations have high-variance weights.

    Examples
    --------
    >>> from onlinecml.datasets import LinearCausalStream
    >>> estimator = OnlineIPW()
    >>> for x, w, y, _ in LinearCausalStream(n=500, true_ate=2.0, seed=42):
    ...     estimator.learn_one(x, w, y)
    >>> abs(estimator.predict_ate()) < 5.0  # loose bound for short stream
    True
    """

    def __init__(
        self,
        ps_model: OnlinePropensityScore | None = None,
        clip_min: float = 0.01,
        clip_max: float = 0.99,
        normalize: bool = False,
    ) -> None:
        self.ps_model = ps_model if ps_model is not None else OnlinePropensityScore(
            LogisticRegression(), clip_min=clip_min, clip_max=clip_max
        )
        self.clip_min = clip_min
        self.clip_max = clip_max
        self.normalize = normalize
        # Non-constructor state
        self._n_seen: int = 0
        self._ate_stats: RunningStats = RunningStats()
        self._treated_weight_stats: RunningStats = RunningStats()
        self._control_weight_stats: RunningStats = RunningStats()

    def learn_one(
        self,
        x: dict,
        treatment: int,
        outcome: float,
        propensity: float | None = None,
    ) -> None:
        """Process one observation and update the ATE estimate.

        Parameters
        ----------
        x : dict
            Feature dictionary for this observation.
        treatment : int
            Treatment indicator (0 = control, 1 = treated).
        outcome : float
            Observed outcome.
        propensity : float or None
            If provided, uses this logged propensity instead of the
            internal model's prediction. Useful for off-policy evaluation.
        """
        # Step 1: predict propensity BEFORE updating the classifier
        if propensity is not None:
            p = max(self.clip_min, min(self.clip_max, propensity))
        else:
            p = self.ps_model.predict_one(x)

        # Step 2: compute raw IPW weights
        w_treated = 1.0 / p
        w_control = 1.0 / (1.0 - p)

        # Step 3: track weight stats (for normalization and diagnostics)
        self._treated_weight_stats.update(w_treated)
        self._control_weight_stats.update(w_control)

        # Step 4: normalize weights if requested
        if self.normalize:
            mean_wt = self._treated_weight_stats.mean
            mean_wc = self._control_weight_stats.mean
            w_treated = w_treated / mean_wt if mean_wt > 0 else w_treated
            w_control = w_control / mean_wc if mean_wc > 0 else w_control

        # Step 5: IPW pseudo-outcome
        psi = treatment * w_treated * outcome - (1 - treatment) * w_control * outcome

        # Step 6: update ATE tracker
        self._ate_stats.update(psi)
        self._n_seen += 1

        # Step 7: update propensity model AFTER pseudo-outcome
        if propensity is None:
            self.ps_model.learn_one(x, treatment)

    def predict_one(self, x: dict) -> float:
        """Return the current running ATE estimate.

        Parameters
        ----------
        x : dict
            Feature dictionary (not used; IPW produces no individual CATE).

        Returns
        -------
        float
            Current ATE estimate. IPW does not estimate individual CATE —
            the same value is returned for all inputs.
        """
        return self._ate_stats.mean

    @property
    def weight_stats(self) -> dict:
        """Summary statistics of IPW weights seen so far.

        Returns
        -------
        dict
            Dictionary with keys ``'treated_mean'``, ``'control_mean'``,
            and ``'n'``.
        """
        return {
            "treated_mean": self._treated_weight_stats.mean,
            "control_mean": self._control_weight_stats.mean,
            "n": self._n_seen,
        }
