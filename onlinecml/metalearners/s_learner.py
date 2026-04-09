"""Online S-Learner (single-model) meta-learner for CATE estimation."""

from river.linear_model import LinearRegression

from onlinecml.base.base_estimator import BaseOnlineEstimator
from onlinecml.base.running_stats import RunningStats


class OnlineSLearner(BaseOnlineEstimator):
    """Online S-Learner for CATE estimation via a single augmented model.

    Trains a single outcome model on features augmented with the treatment
    indicator. CATE is estimated as the difference in predictions when the
    treatment indicator is set to 1 vs 0, holding covariates fixed.

    Parameters
    ----------
    model : river.base.Regressor or None
        A River regressor. Must support ``learn_one(x, y)`` and
        ``predict_one(x)``. Defaults to ``LinearRegression()``.
    treatment_feature : str
        Name of the synthetic treatment feature added to ``x``.
        Default ``'__treatment__'``. Change if this key conflicts with
        existing feature names.

    Notes
    -----
    **Predict-then-learn:** At each step, CATE is predicted *before* the
    model is updated. This avoids look-ahead bias in the running ATE.

    **Limitation:** Because treatment is just another feature, the
    S-Learner can under-regularize the treatment effect and produce
    biased CATE estimates when treatment is rare or the model is
    mis-specified. For better CATE estimates, prefer ``OnlineTLearner``
    or ``OnlineRLearner``.

    Examples
    --------
    >>> from onlinecml.datasets import LinearCausalStream
    >>> from river.linear_model import LinearRegression
    >>> model = OnlineSLearner(LinearRegression())
    >>> for x, w, y, _ in LinearCausalStream(n=200, seed=42):
    ...     model.learn_one(x, w, y)
    >>> isinstance(model.predict_ate(), float)
    True
    """

    def __init__(
        self,
        model=None,
        treatment_feature: str = "__treatment__",
    ) -> None:
        self.model = model if model is not None else LinearRegression()
        self.treatment_feature = treatment_feature
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
        """Process one observation and update the outcome model.

        Parameters
        ----------
        x : dict
            Feature dictionary for this observation.
        treatment : int
            Treatment indicator (0 = control, 1 = treated).
        outcome : float
            Observed outcome.
        propensity : float or None
            Not used by S-Learner; included for API compatibility.
        """
        # Step 1: predict CATE before updating the model (predict-then-learn)
        x_treated = {**x, self.treatment_feature: 1}
        x_control = {**x, self.treatment_feature: 0}
        cate = self.model.predict_one(x_treated) - self.model.predict_one(x_control)

        # Step 2: update running ATE
        self._ate_stats.update(cate)
        self._n_seen += 1

        # Step 3: update outcome model with the actual treatment assignment
        x_augmented = {**x, self.treatment_feature: treatment}
        self.model.learn_one(x_augmented, outcome)

    def predict_one(self, x: dict) -> float:
        """Predict the CATE for a single unit.

        Parameters
        ----------
        x : dict
            Feature dictionary for the unit.

        Returns
        -------
        float
            Estimated CATE: ``model(x, W=1) - model(x, W=0)``.
        """
        x_treated = {**x, self.treatment_feature: 1}
        x_control = {**x, self.treatment_feature: 0}
        return self.model.predict_one(x_treated) - self.model.predict_one(x_control)
