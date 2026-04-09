"""Online T-Learner (two-model) meta-learner for CATE estimation."""

import warnings

from river.linear_model import LinearRegression

from onlinecml.base.base_estimator import BaseOnlineEstimator
from onlinecml.base.running_stats import RunningStats


class OnlineTLearner(BaseOnlineEstimator):
    """Online T-Learner for CATE estimation via two separate outcome models.

    Trains one model on treated units and one on control units. CATE is
    estimated as the difference in predictions from the two models.

    Parameters
    ----------
    treated_model : river.base.Regressor or None
        Model for treated units ``E[Y | X, W=1]``. Defaults to
        ``LinearRegression()``.
    control_model : river.base.Regressor or None
        Model for control units ``E[Y | X, W=0]``. Defaults to
        ``LinearRegression()``. Must be a different object from
        ``treated_model``.

    Notes
    -----
    **Predict-then-learn:** CATE is predicted from both models *before*
    either model is updated. This avoids look-ahead bias.

    **IPW correction:** If ``propensity`` is passed to ``learn_one``,
    the sample weight ``1/p`` (treated) or ``1/(1-p)`` (control) is
    passed as ``w`` to the model's ``learn_one`` call. This corrects
    for treatment imbalance. If the River model does not support sample
    weights, the weight is silently ignored.

    **Single-arm cold start:** If only one treatment arm has been seen,
    the other model returns 0 (River default for untrained regression).
    A ``UserWarning`` is emitted on ``predict_one`` when either model
    has seen no data.

    Examples
    --------
    >>> from onlinecml.datasets import LinearCausalStream
    >>> from river.linear_model import LinearRegression
    >>> model = OnlineTLearner(LinearRegression(), LinearRegression())
    >>> for x, w, y, _ in LinearCausalStream(n=200, seed=42):
    ...     model.learn_one(x, w, y)
    >>> isinstance(model.predict_one({"x0": 0.5}), float)
    True
    """

    def __init__(
        self,
        treated_model=None,
        control_model=None,
    ) -> None:
        self.treated_model = treated_model if treated_model is not None else LinearRegression()
        self.control_model = control_model if control_model is not None else LinearRegression()
        # Non-constructor state
        self._n_seen: int = 0
        self._ate_stats: RunningStats = RunningStats()
        self._n_treated: int = 0
        self._n_control: int = 0

    def learn_one(
        self,
        x: dict,
        treatment: int,
        outcome: float,
        propensity: float | None = None,
    ) -> None:
        """Process one observation and update the appropriate arm model.

        Parameters
        ----------
        x : dict
            Feature dictionary for this observation.
        treatment : int
            Treatment indicator (0 = control, 1 = treated).
        outcome : float
            Observed outcome.
        propensity : float or None
            If provided, computes IPW weight ``1/p`` (treated) or
            ``1/(1-p)`` (control) and passes it as sample weight ``w``
            to the model's ``learn_one`` call.
        """
        # Step 1: predict CATE before updating either model (predict-then-learn)
        cate = self.treated_model.predict_one(x) - self.control_model.predict_one(x)

        # Step 2: update running ATE
        self._ate_stats.update(cate)
        self._n_seen += 1

        # Step 3: compute sample weight from propensity if provided
        if propensity is not None:
            p = max(1e-6, min(1.0 - 1e-6, propensity))
            w = 1.0 / p if treatment == 1 else 1.0 / (1.0 - p)
        else:
            w = 1.0

        # Step 4: update the appropriate arm model
        if treatment == 1:
            self._n_treated += 1
            try:
                self.treated_model.learn_one(x, outcome, w=w)
            except TypeError:
                self.treated_model.learn_one(x, outcome)
        else:
            self._n_control += 1
            try:
                self.control_model.learn_one(x, outcome, w=w)
            except TypeError:
                self.control_model.learn_one(x, outcome)

    def predict_one(self, x: dict) -> float:
        """Predict the CATE for a single unit.

        Parameters
        ----------
        x : dict
            Feature dictionary for the unit.

        Returns
        -------
        float
            Estimated CATE: ``treated_model(x) - control_model(x)``.

        Warns
        -----
        UserWarning
            If either arm model has not seen any data yet, the prediction
            from that model defaults to 0 (River's untrained regressor
            default), which may bias the CATE estimate.
        """
        if self._n_treated == 0:
            warnings.warn(
                "treated_model has not seen any data yet; CATE estimate may be biased.",
                UserWarning,
                stacklevel=2,
            )
        if self._n_control == 0:
            warnings.warn(
                "control_model has not seen any data yet; CATE estimate may be biased.",
                UserWarning,
                stacklevel=2,
            )
        return self.treated_model.predict_one(x) - self.control_model.predict_one(x)
