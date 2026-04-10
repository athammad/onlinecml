"""Abstract base class for all online causal estimators."""

import abc

import scipy.stats
from river.base import Base

from onlinecml.base.running_stats import RunningStats


class BaseOnlineEstimator(Base):
    """Abstract base class for all online causal estimators.

    Every estimator in OnlineCML inherits from this class. It provides
    the standard interface for online causal inference: processing one
    observation at a time, estimating the Average Treatment Effect (ATE),
    and reporting confidence intervals.

    Inherits from ``river.base.Base`` (not ``river.base.Estimator``) to
    avoid signature conflicts: our ``learn_one`` takes ``(x, treatment,
    outcome, propensity)`` while River's Estimator expects ``(x, y)``.

    Notes
    -----
    Subclasses must implement ``learn_one`` and ``predict_one``.

    All constructor parameters must be stored as ``self.param_name``
    (matching the parameter name exactly) so that ``clone()`` and
    ``_get_params()`` work correctly.

    Non-constructor state (``_n_seen``, ``_ate_stats``) is initialized
    in each concrete ``__init__``. It is intentionally NOT cloned —
    ``clone()`` returns a fresh estimator with zero observations.
    ``reset()`` re-initializes all state by calling ``__init__`` again.

    The ``predict_ci`` method returns a confidence interval for the ATE
    (mean CATE), not for individual CATE predictions. This uses a normal
    approximation via the central limit theorem applied to the running
    pseudo-outcome variance.
    """

    @abc.abstractmethod
    def learn_one(
        self,
        x: dict,
        treatment: int,
        outcome: float,
        propensity: float | None = None,
    ) -> None:
        """Process one observation and update the estimator.

        Parameters
        ----------
        x : dict
            Feature dictionary for this observation.
        treatment : int
            Treatment indicator (0 = control, 1 = treated).
        outcome : float
            Observed outcome for this unit.
        propensity : float or None
            Known or logged propensity P(W=1|X). If None, the estimator
            will use its internal propensity model.
        """

    @abc.abstractmethod
    def predict_one(self, x: dict) -> float:
        """Predict the CATE for a single unit.

        Parameters
        ----------
        x : dict
            Feature dictionary for the unit.

        Returns
        -------
        float
            Estimated Conditional Average Treatment Effect for this unit.
        """

    def predict_ate(self) -> float:
        """Return the current running ATE estimate.

        Returns
        -------
        float
            The current Average Treatment Effect estimate. Returns 0.0
            before any observations have been processed.
        """
        return self._ate_stats.mean

    def predict_ci(self, alpha: float = 0.05) -> tuple[float, float]:
        """Return a confidence interval for the ATE estimate.

        Uses a normal approximation via the central limit theorem applied
        to the running variance of per-observation pseudo-outcomes.

        Parameters
        ----------
        alpha : float
            Significance level. Default 0.05 gives a 95% CI.

        Returns
        -------
        lower : float
            Lower bound of the confidence interval.
        upper : float
            Upper bound of the confidence interval.

        Notes
        -----
        Returns ``(-inf, inf)`` before at least 2 observations are seen.
        The CI is for the ATE (mean CATE), not for individual CATE predictions.
        """
        n = self._ate_stats.n
        if n < 2:
            return (float("-inf"), float("inf"))
        ate = self._ate_stats.mean
        se = (self._ate_stats.variance / n) ** 0.5
        z = scipy.stats.norm.ppf(1.0 - alpha / 2.0)
        return (ate - z * se, ate + z * se)

    def reset(self) -> None:
        """Reset the estimator to its initial (untrained) state.

        Equivalent to creating a fresh instance with the same constructor
        arguments. All learned state is discarded.
        """
        fresh = self.clone()
        self.__dict__.update(fresh.__dict__)

    @property
    def n_seen(self) -> int:
        """Number of observations processed so far."""
        return self._n_seen

    @property
    def smd(self) -> dict | None:
        """Current Standardized Mean Difference per covariate.

        Returns None by default. Subclasses that maintain per-group
        statistics (e.g. IPW-based estimators) override this property.

        Returns
        -------
        dict or None
            Mapping from covariate name to (raw_smd, weighted_smd), or
            None if this estimator does not track balance diagnostics.
        """
        return None
