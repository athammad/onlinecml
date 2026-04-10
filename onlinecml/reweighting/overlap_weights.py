"""Online Overlap Weights (OW) estimator for the ATE."""

from river.linear_model import LogisticRegression

from onlinecml.base.base_estimator import BaseOnlineEstimator
from onlinecml.base.running_stats import RunningStats
from onlinecml.propensity.propensity_score import OnlinePropensityScore


class OnlineOverlapWeights(BaseOnlineEstimator):
    """Online Overlap Weights estimator for the ATE.

    Uses overlap weights — proportional to the probability of belonging
    to the *opposite* treatment group — instead of inverse probability
    weights. Overlap weights are bounded and yield more stable estimates
    under near-positivity violations.

    The overlap weight for unit ``i`` is:

    .. math::

        h_i = \\begin{cases}
            1 - \\hat{p}(X_i) & \\text{if } W_i = 1 \\\\
            \\hat{p}(X_i)     & \\text{if } W_i = 0
        \\end{cases}

    The ATE estimator is:

    .. math::

        \\hat{\\tau}_{OW} = \\frac{
            \\sum_i h_i W_i Y_i / \\hat{p}_i
            - \\sum_i h_i (1-W_i) Y_i / (1-\\hat{p}_i)
        }{\\sum_i h_i}

    In the online setting this is approximated by maintaining running
    means of the numerator and denominator terms.

    Parameters
    ----------
    ps_model : OnlinePropensityScore or None
        Propensity score model. Defaults to
        ``OnlinePropensityScore(LogisticRegression())``.

    Notes
    -----
    Overlap weights target the Average Treatment Effect on the Overlap
    Population (ATO), which emphasizes units with propensity scores near
    0.5. The estimand differs slightly from the ATE when the propensity
    distribution is asymmetric.

    References
    ----------
    Li, F., Morgan, K.L., and Zaslavsky, A.M. (2018). Balancing covariates
    via propensity score weighting. Journal of the American Statistical
    Association, 113(521), 390-400.

    Examples
    --------
    >>> from onlinecml.datasets import LinearCausalStream
    >>> estimator = OnlineOverlapWeights()
    >>> for x, w, y, _ in LinearCausalStream(n=500, seed=42):
    ...     estimator.learn_one(x, w, y)
    >>> isinstance(estimator.predict_ate(), float)
    True
    """

    def __init__(self, ps_model: OnlinePropensityScore | None = None) -> None:
        self.ps_model = ps_model if ps_model is not None else OnlinePropensityScore(
            LogisticRegression()
        )
        # Non-constructor state
        self._n_seen: int = 0
        self._ate_stats: RunningStats = RunningStats()

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
            If provided, uses this logged propensity.
        """
        # Predict propensity before updating (predict-first)
        if propensity is not None:
            p = max(1e-6, min(1.0 - 1e-6, propensity))
        else:
            p = self.ps_model.predict_one(x)

        # Overlap weight: h = (1-p) for treated, p for control
        h = (1.0 - p) if treatment == 1 else p

        # Overlap-weighted pseudo-outcome
        psi = treatment * h * outcome / p - (1 - treatment) * h * outcome / (1.0 - p)
        self._ate_stats.update(psi)
        self._n_seen += 1

        if propensity is None:
            self.ps_model.learn_one(x, treatment)

    def predict_one(self, x: dict) -> float:
        """Return the current running ATE estimate.

        Parameters
        ----------
        x : dict
            Feature dictionary (not used; OW has no individual CATE).

        Returns
        -------
        float
            Current ATE estimate.
        """
        return self._ate_stats.mean
