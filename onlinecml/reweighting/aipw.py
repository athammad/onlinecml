"""Online Augmented Inverse Probability Weighting (AIPW) estimator."""

from river.linear_model import LinearRegression, LogisticRegression

from onlinecml.base.base_estimator import BaseOnlineEstimator
from onlinecml.base.running_stats import RunningStats
from onlinecml.propensity.propensity_score import OnlinePropensityScore


class OnlineAIPW(BaseOnlineEstimator):
    """Online Doubly Robust (AIPW) estimator for ATE and individual CATE.

    Estimates the Average Treatment Effect using the Augmented IPW
    (doubly robust) estimator. Maintains three online models: a propensity
    score, a treated outcome model, and a control outcome model. All models
    are updated using the predict-first-then-learn protocol.

    Parameters
    ----------
    ps_model : OnlinePropensityScore or None
        Propensity score model. Defaults to
        ``OnlinePropensityScore(LogisticRegression())``.
    treated_model : river.base.Regressor or None
        Outcome model for treated units ``E[Y|X, W=1]``. Defaults to
        ``LinearRegression()``.
    control_model : river.base.Regressor or None
        Outcome model for control units ``E[Y|X, W=0]``. Defaults to
        ``LinearRegression()``.
    clip_min : float
        Lower clip bound for propensity scores. Default 0.01.
    clip_max : float
        Upper clip bound for propensity scores. Default 0.99.

    Notes
    -----
    The doubly robust pseudo-outcome is:

    .. math::

        \\psi_i = \\hat{\\mu}_1(X_i) - \\hat{\\mu}_0(X_i)
                 + \\frac{W_i (Y_i - \\hat{\\mu}_1(X_i))}{\\hat{p}(X_i)}
                 - \\frac{(1-W_i)(Y_i - \\hat{\\mu}_0(X_i))}{1 - \\hat{p}(X_i)}

    The estimator is consistent if either the propensity score OR both
    outcome models are correctly specified (double robustness).

    **Predict-first-then-learn:** all three models predict *before* any
    of them are updated on the current observation. This approximates
    cross-fitting in the online setting.

    Unlike ``OnlineIPW``, ``predict_one(x)`` returns an individual CATE
    estimate: ``mu1(x) - mu0(x)``.

    Examples
    --------
    >>> from onlinecml.datasets import LinearCausalStream
    >>> estimator = OnlineAIPW()
    >>> for x, w, y, _ in LinearCausalStream(n=500, true_ate=2.0, seed=42):
    ...     estimator.learn_one(x, w, y)
    >>> abs(estimator.predict_ate()) < 5.0
    True
    """

    def __init__(
        self,
        ps_model: OnlinePropensityScore | None = None,
        treated_model=None,
        control_model=None,
        clip_min: float = 0.01,
        clip_max: float = 0.99,
    ) -> None:
        self.ps_model = ps_model if ps_model is not None else OnlinePropensityScore(
            LogisticRegression(), clip_min=clip_min, clip_max=clip_max
        )
        self.treated_model = treated_model if treated_model is not None else LinearRegression()
        self.control_model = control_model if control_model is not None else LinearRegression()
        self.clip_min = clip_min
        self.clip_max = clip_max
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
        """Process one observation and update all internal models.

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
            internal model. Useful for off-policy evaluation.
        """
        # Step 1: predict all models BEFORE any updates (predict-first)
        if propensity is not None:
            p = max(self.clip_min, min(self.clip_max, propensity))
        else:
            p = self.ps_model.predict_one(x)

        mu1 = self.treated_model.predict_one(x)
        mu0 = self.control_model.predict_one(x)

        # Step 2: compute doubly robust pseudo-outcome
        psi = (
            mu1
            - mu0
            + treatment * (outcome - mu1) / p
            - (1 - treatment) * (outcome - mu0) / (1.0 - p)
        )

        # Step 3: update ATE tracker
        self._ate_stats.update(psi)
        self._n_seen += 1

        # Step 4: update all models AFTER pseudo-outcome
        if propensity is None:
            self.ps_model.learn_one(x, treatment)
        if treatment == 1:
            self.treated_model.learn_one(x, outcome)
        else:
            self.control_model.learn_one(x, outcome)

    def predict_one(self, x: dict) -> float:
        """Predict the individual CATE for a single unit.

        Parameters
        ----------
        x : dict
            Feature dictionary for the unit.

        Returns
        -------
        float
            Estimated CATE: ``mu1(x) - mu0(x)``. Returns 0.0 before
            any observations are seen (both models predict 0 by default).
        """
        return self.treated_model.predict_one(x) - self.control_model.predict_one(x)
