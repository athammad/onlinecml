"""Online R-Learner meta-learner via Robinson transformation."""

from river.linear_model import LinearRegression, LogisticRegression

from onlinecml.base.base_estimator import BaseOnlineEstimator
from onlinecml.base.running_stats import RunningStats
from onlinecml.propensity.propensity_score import OnlinePropensityScore


class OnlineRLearner(BaseOnlineEstimator):
    """Online R-Learner for CATE via the Robinson (1988) transformation.

    Estimates CATE by orthogonalizing the treatment assignment and outcome
    with respect to their conditional means. The residualized targets are:

    .. math::

        \\tilde{W}_i = W_i - \\hat{p}(X_i)

        \\tilde{Y}_i = Y_i - \\hat{m}(X_i)

    The CATE model is then trained on the pseudo-outcome
    ``tilde_Y / tilde_W`` weighted by ``tilde_W^2``:

    .. math::

        \\hat{\\tau}(x) = \\arg\\min_\\tau
        \\mathbb{E}[(\\tilde{Y}_i - \\tau(X_i) \\tilde{W}_i)^2]

    This approach is the theoretical foundation of Double Machine Learning
    (DML) and produces nearly oracle-rate CATE estimates when both nuisance
    models are consistent.

    Parameters
    ----------
    ps_model : river.base.Classifier, OnlinePropensityScore, or None
        Propensity score model ``P(W=1|X)``. Raw River classifiers and
        pipelines are automatically wrapped in ``OnlinePropensityScore``.
        If None, defaults to ``OnlinePropensityScore(LogisticRegression())``.
    outcome_model : river.base.Regressor or None
        Outcome model ``E[Y|X]``. Defaults to ``LinearRegression()``.
    cate_model : river.base.Regressor or None
        CATE model trained on Robinson-residualized targets.
        Defaults to ``LinearRegression()``.

    Notes
    -----
    **Predict-first-then-learn:** All three models predict before any are
    updated. This is the natural online approximation to cross-fitting.

    **Residual weighting:** The CATE model is updated with sample weight
    ``w = tilde_W^2`` when the River model supports sample weights. When
    ``|tilde_W|`` is very small (near 0.05), the update is skipped to
    avoid noisy updates from nearly-deterministic treatment assignments.

    **Connection to DML:** The R-Learner is equivalent to Partially Linear
    DML (Robinson 1988, Chernozhukov et al. 2018) when the CATE model is
    linear.

    References
    ----------
    Robinson, P.M. (1988). Root-N-consistent semiparametric regression.
    Econometrica, 56(4), 931-954.

    Nie, X. and Wager, S. (2021). Quasi-oracle estimation of heterogeneous
    treatment effects. Biometrika, 108(2), 299-319.

    Examples
    --------
    >>> from onlinecml.datasets import LinearCausalStream
    >>> from river.linear_model import LinearRegression, LogisticRegression
    >>> model = OnlineRLearner(
    ...     ps_model=None,
    ...     outcome_model=LinearRegression(),
    ...     cate_model=LinearRegression(),
    ... )
    >>> for x, w, y, _ in LinearCausalStream(n=500, seed=42):
    ...     model.learn_one(x, w, y)
    >>> isinstance(model.predict_ate(), float)
    True
    """

    def __init__(
        self,
        ps_model: "OnlinePropensityScore | object | None" = None,
        outcome_model=None,
        cate_model=None,
        min_residual: float = 0.05,
    ) -> None:
        self.ps_model = (
            ps_model if isinstance(ps_model, OnlinePropensityScore)
            else OnlinePropensityScore(
                ps_model if ps_model is not None else LogisticRegression()
            )
        )
        self.outcome_model = outcome_model if outcome_model is not None else LinearRegression()
        self.cate_model = cate_model if cate_model is not None else LinearRegression()
        self.min_residual = min_residual
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
        """Process one observation using the Robinson transformation.

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
            internal model.
        """
        # Predict all nuisance models before any updates
        if propensity is not None:
            p_hat = max(1e-6, min(1.0 - 1e-6, propensity))
        else:
            p_hat = self.ps_model.predict_one(x)

        m_hat = self.outcome_model.predict_one(x)
        cate_hat = self.cate_model.predict_one(x)

        # Robinson residuals
        w_res = treatment - p_hat      # W_tilde
        y_res = outcome - m_hat        # Y_tilde

        # Update running ATE from current CATE estimate
        self._ate_stats.update(cate_hat)
        self._n_seen += 1

        # Update CATE model with Robinson pseudo-outcome
        # weight = w_res^2 (residual-weighted regression)
        if abs(w_res) >= self.min_residual:
            # Pseudo-outcome: y_res / w_res (Robinson's rearrangement)
            pseudo_outcome = y_res / w_res
            w_sample = w_res ** 2
            try:
                self.cate_model.learn_one(x, pseudo_outcome, w=w_sample)
            except TypeError:
                self.cate_model.learn_one(x, pseudo_outcome)

        # Update nuisance models
        if propensity is None:
            self.ps_model.learn_one(x, treatment)
        self.outcome_model.learn_one(x, outcome)

    def predict_one(self, x: dict) -> float:
        """Predict the CATE for a single unit.

        Parameters
        ----------
        x : dict
            Feature dictionary for the unit.

        Returns
        -------
        float
            Estimated CATE from the Robinson-residualized model.
        """
        return self.cate_model.predict_one(x)
