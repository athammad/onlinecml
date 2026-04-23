"""Online X-Learner meta-learner for CATE estimation."""

from river.linear_model import LinearRegression, LogisticRegression

from onlinecml.base.base_estimator import BaseOnlineEstimator
from onlinecml.base.running_stats import RunningStats
from onlinecml.propensity.propensity_score import OnlinePropensityScore


class OnlineXLearner(BaseOnlineEstimator):
    """Online X-Learner for CATE estimation in unbalanced treatment groups.

    Implements a three-stage pipeline adapted for the online setting:

    1. **Stage 1 (T-Learner base):** Train two outcome models
       ``mu1(x) = E[Y|X, W=1]`` and ``mu0(x) = E[Y|X, W=0]``.

    2. **Stage 2 (Imputed effects):** For each treated unit, impute a
       control potential outcome: ``D1 = Y - mu0(X)``. For each control
       unit: ``D0 = mu1(X) - Y``.

    3. **Stage 3 (CATE models):** Train two CATE models — ``tau1`` on
       treated units' imputed effects and ``tau0`` on control units'. The
       final CATE is a propensity-weighted combination:
       ``CATE(x) = p(x) * tau0(x) + (1-p(x)) * tau1(x)``.

    Parameters
    ----------
    mu1_model : river.base.Regressor or None
        Outcome model for treated units. Defaults to ``LinearRegression()``.
    mu0_model : river.base.Regressor or None
        Outcome model for control units. Defaults to ``LinearRegression()``.
    tau1_model : river.base.Regressor or None
        CATE model trained on treated units' imputed effects.
        Defaults to ``LinearRegression()``.
    tau0_model : river.base.Regressor or None
        CATE model trained on control units' imputed effects.
        Defaults to ``LinearRegression()``.
    ps_model : river.base.Classifier, OnlinePropensityScore, or None
        Propensity score model for the weighted combination. Raw River
        classifiers and pipelines are automatically wrapped in
        ``OnlinePropensityScore``. If None, defaults to
        ``OnlinePropensityScore(LogisticRegression())``.

    Notes
    -----
    **Predict-first-then-learn:** All five models predict before any are
    updated. This approximates the cross-fitting required for stage 2.

    The X-Learner is most effective when treatment groups are substantially
    unbalanced — it leverages information from the larger group to improve
    CATE estimates for the smaller group.

    Examples
    --------
    >>> from onlinecml.datasets import LinearCausalStream
    >>> model = OnlineXLearner()
    >>> for x, w, y, _ in LinearCausalStream(n=500, seed=42):
    ...     model.learn_one(x, w, y)
    >>> isinstance(model.predict_one({"x0": 0.5}), float)
    True
    """

    def __init__(
        self,
        mu1_model=None,
        mu0_model=None,
        tau1_model=None,
        tau0_model=None,
        ps_model: OnlinePropensityScore | None = None,
    ) -> None:
        self.mu1_model = mu1_model if mu1_model is not None else LinearRegression()
        self.mu0_model = mu0_model if mu0_model is not None else LinearRegression()
        self.tau1_model = tau1_model if tau1_model is not None else LinearRegression()
        self.tau0_model = tau0_model if tau0_model is not None else LinearRegression()
        self.ps_model = (
            ps_model if isinstance(ps_model, OnlinePropensityScore)
            else OnlinePropensityScore(
                ps_model if ps_model is not None else LogisticRegression()
            )
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
            If provided, used for the weighted combination instead of
            the internal propensity model.
        """
        # Stage 1: predict all models before any updates (predict-first)
        mu1 = self.mu1_model.predict_one(x)
        mu0 = self.mu0_model.predict_one(x)

        # Stage 2: compute imputed individual treatment effect
        if treatment == 1:
            d = outcome - mu0   # treated: observed - counterfactual control
        else:
            d = mu1 - outcome   # control: counterfactual treated - observed

        # Stage 3: predict CATE models (before updating them)
        tau1 = self.tau1_model.predict_one(x)
        tau0 = self.tau0_model.predict_one(x)

        # Propensity for weighted combination
        if propensity is not None:
            p = max(1e-6, min(1.0 - 1e-6, propensity))
        else:
            p = self.ps_model.predict_one(x)

        # Weighted combination: p * tau0 + (1-p) * tau1
        cate = p * tau0 + (1.0 - p) * tau1
        self._ate_stats.update(cate)
        self._n_seen += 1

        # Update all models after pseudo-outcome computation
        if treatment == 1:
            self.mu1_model.learn_one(x, outcome)
            self.tau1_model.learn_one(x, d)
        else:
            self.mu0_model.learn_one(x, outcome)
            self.tau0_model.learn_one(x, d)

        if propensity is None:
            self.ps_model.learn_one(x, treatment)

    def predict_one(self, x: dict) -> float:
        """Predict the CATE for a single unit.

        Parameters
        ----------
        x : dict
            Feature dictionary for the unit.

        Returns
        -------
        float
            Estimated CATE: propensity-weighted combination of
            ``tau0(x)`` and ``tau1(x)``.
        """
        p = self.ps_model.predict_one(x)
        tau1 = self.tau1_model.predict_one(x)
        tau0 = self.tau0_model.predict_one(x)
        return p * tau0 + (1.0 - p) * tau1
