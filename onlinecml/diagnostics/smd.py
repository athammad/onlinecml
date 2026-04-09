"""Online Standardized Mean Difference (SMD) for covariate balance diagnostics."""

import math

from river.base import Base

from onlinecml.base.running_stats import RunningStats, WeightedRunningStats


class OnlineSMD(Base):
    """Online covariate balance diagnostics via Standardized Mean Difference.

    Tracks the raw and IPW-weighted SMD for a set of covariates, updated
    one observation at a time. Used to monitor whether treatment and control
    groups are comparable in covariate distributions.

    Parameters
    ----------
    covariates : list of str
        Names of covariates to track. Must match keys in the feature dicts
        passed to ``update``.

    Notes
    -----
    SMD for a covariate is defined as:

    .. math::

        \\text{SMD} = \\frac{\\bar{X}_T - \\bar{X}_C}{\\sqrt{(s_T^2 + s_C^2) / 2}}

    where ``s^2`` is the sample variance within each group. Returns 0.0
    when either group has fewer than 2 observations.

    Raw SMD uses unweighted ``RunningStats``; weighted SMD uses
    ``WeightedRunningStats`` with West's (1979) algorithm (population
    variance). The weighted SMD is used by ``is_balanced``.

    This class does NOT inherit from ``BaseOnlineEstimator`` — it is a
    standalone diagnostic tool.

    Examples
    --------
    >>> smd = OnlineSMD(covariates=["age", "income"])
    >>> smd.update({"age": 30, "income": 50000}, treatment=1, weight=1.2)
    >>> smd.update({"age": 45, "income": 70000}, treatment=0, weight=0.8)
    >>> report = smd.report()
    >>> "age" in report
    True
    """

    def __init__(self, covariates: list[str]) -> None:
        self.covariates = covariates
        # Lazily initialized on first update call
        self._stats: dict[str, dict] = {}

    def _init_covariate(self, cov: str) -> None:
        """Initialize tracking stats for a covariate.

        Parameters
        ----------
        cov : str
            Covariate name to initialize.
        """
        self._stats[cov] = {
            "raw_treated": RunningStats(),
            "raw_control": RunningStats(),
            "weighted_treated": WeightedRunningStats(),
            "weighted_control": WeightedRunningStats(),
        }

    def update(self, x: dict, treatment: int, weight: float = 1.0) -> None:
        """Update balance statistics with one observation.

        Parameters
        ----------
        x : dict
            Feature dictionary. Missing covariates default to 0.0.
        treatment : int
            Treatment indicator (0 = control, 1 = treated).
        weight : float
            Importance weight for this observation (e.g. IPW weight).
            Default 1.0 (unweighted).
        """
        for cov in self.covariates:
            if cov not in self._stats:
                self._init_covariate(cov)
            val = float(x.get(cov, 0.0))
            s = self._stats[cov]
            if treatment == 1:
                s["raw_treated"].update(val)
                s["weighted_treated"].update(val, w=weight)
            else:
                s["raw_control"].update(val)
                s["weighted_control"].update(val, w=weight)

    @staticmethod
    def _compute_smd(stats_t: RunningStats, stats_c: RunningStats) -> float:
        """Compute SMD between two groups using their running statistics.

        Parameters
        ----------
        stats_t : RunningStats
            Running stats for the treated group.
        stats_c : RunningStats
            Running stats for the control group.

        Returns
        -------
        float
            SMD value. Returns 0.0 if either group has fewer than 2 obs.
        """
        if stats_t.n < 2 or stats_c.n < 2:
            return 0.0
        pooled_var = (stats_t.variance + stats_c.variance) / 2.0
        if pooled_var <= 0.0:
            return 0.0
        return (stats_t.mean - stats_c.mean) / math.sqrt(pooled_var)

    @staticmethod
    def _compute_weighted_smd(
        stats_t: WeightedRunningStats, stats_c: WeightedRunningStats
    ) -> float:
        """Compute weighted SMD between two groups.

        Parameters
        ----------
        stats_t : WeightedRunningStats
            Weighted running stats for the treated group.
        stats_c : WeightedRunningStats
            Weighted running stats for the control group.

        Returns
        -------
        float
            Weighted SMD value. Returns 0.0 if either group has no weight mass.
        """
        if stats_t.sum_weights <= 0.0 or stats_c.sum_weights <= 0.0:
            return 0.0
        pooled_var = (stats_t.variance + stats_c.variance) / 2.0
        if pooled_var <= 0.0:
            return 0.0
        return (stats_t.mean - stats_c.mean) / math.sqrt(pooled_var)

    def report(self) -> dict[str, tuple[float, float]]:
        """Return the current SMD for each tracked covariate.

        Returns
        -------
        dict
            Mapping from covariate name to ``(raw_smd, weighted_smd)``.
            Covariates with insufficient data return ``(0.0, 0.0)``.
        """
        result = {}
        for cov in self.covariates:
            if cov not in self._stats:
                result[cov] = (0.0, 0.0)
            else:
                s = self._stats[cov]
                raw = self._compute_smd(s["raw_treated"], s["raw_control"])
                weighted = self._compute_weighted_smd(
                    s["weighted_treated"], s["weighted_control"]
                )
                result[cov] = (raw, weighted)
        return result

    def is_balanced(self, thr: float = 0.1) -> bool:
        """Check whether all covariates are balanced after weighting.

        Parameters
        ----------
        thr : float
            Maximum absolute weighted SMD threshold. Default 0.1
            (the conventional "well-balanced" threshold).

        Returns
        -------
        bool
            True if all covariates have ``|weighted_smd| < thr``.
        """
        return all(abs(smd_val) < thr for _, smd_val in self.report().values())
