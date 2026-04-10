"""Online positivity / overlap checker for propensity score diagnostics."""

from onlinecml.base.running_stats import RunningStats


class OverlapChecker:
    """Monitors propensity score distributions for positivity violations.

    Tracks the distribution of predicted propensity scores per treatment
    arm and raises warnings when extreme scores are detected. Reports
    the proportion of units in the common support region.

    Parameters
    ----------
    ps_min : float
        Lower positivity threshold. PS values below this are flagged.
        Default 0.05.
    ps_max : float
        Upper positivity threshold. PS values above this are flagged.
        Default 0.95.

    Notes
    -----
    A unit is in *common support* if its propensity score satisfies
    ``ps_min < p < ps_max``. Units outside this region have unreliable
    causal estimates.

    The ``report()`` method returns a summary including:
    - Mean PS per arm
    - Proportion flagged per arm
    - Overall common support rate

    Examples
    --------
    >>> checker = OverlapChecker(ps_min=0.05, ps_max=0.95)
    >>> checker.update(propensity=0.3, treatment=1)
    >>> checker.update(propensity=0.02, treatment=0)
    >>> checker.report()['n_flagged']
    1
    """

    def __init__(self, ps_min: float = 0.05, ps_max: float = 0.95) -> None:
        self.ps_min = ps_min
        self.ps_max = ps_max
        self._ps_stats = [RunningStats(), RunningStats()]   # [control, treated]
        self._n_flagged: int = 0
        self._n_total: int = 0

    def update(self, propensity: float, treatment: int) -> None:
        """Record a propensity score observation.

        Parameters
        ----------
        propensity : float
            Predicted propensity score ``P(W=1|X)`` for this unit.
        treatment : int
            Observed treatment indicator (0 or 1). Used to track
            per-arm PS distributions.
        """
        arm = int(bool(treatment))
        self._ps_stats[arm].update(propensity)
        self._n_total += 1
        if propensity < self.ps_min or propensity > self.ps_max:
            self._n_flagged += 1

    def report(self) -> dict:
        """Return a summary of the propensity score distribution.

        Returns
        -------
        dict
            Keys:

            - ``'n_total'`` — total observations seen
            - ``'n_flagged'`` — observations outside ``[ps_min, ps_max]``
            - ``'flag_rate'`` — proportion flagged
            - ``'common_support_rate'`` — ``1 - flag_rate``
            - ``'mean_ps_treated'`` — mean PS in the treated arm
            - ``'mean_ps_control'`` — mean PS in the control arm
        """
        flag_rate = self._n_flagged / self._n_total if self._n_total > 0 else 0.0
        return {
            "n_total": self._n_total,
            "n_flagged": self._n_flagged,
            "flag_rate": flag_rate,
            "common_support_rate": 1.0 - flag_rate,
            "mean_ps_treated": self._ps_stats[1].mean,
            "mean_ps_control": self._ps_stats[0].mean,
        }

    def is_overlap_adequate(self, max_flag_rate: float = 0.05) -> bool:
        """Return True if the positivity violation rate is acceptable.

        Parameters
        ----------
        max_flag_rate : float
            Maximum tolerable fraction of flagged units. Default 0.05.

        Returns
        -------
        bool
            True if fewer than ``max_flag_rate`` of units are flagged.
        """
        r = self.report()
        return r["flag_rate"] <= max_flag_rate

    def reset(self) -> None:
        """Reset all statistics."""
        self._ps_stats = [RunningStats(), RunningStats()]
        self._n_flagged = 0
        self._n_total = 0
