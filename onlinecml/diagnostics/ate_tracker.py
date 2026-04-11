"""Running ATE tracker with confidence intervals and convergence plotting."""

import math
from typing import TYPE_CHECKING

import scipy.stats

from onlinecml.base.running_stats import EWMAStats, RunningStats

if TYPE_CHECKING:
    import matplotlib.axes


class ATETracker:
    """Tracks the running ATE estimate with online confidence intervals.

    Maintains a running mean and variance of per-observation pseudo-outcomes
    and optionally records history for convergence plotting.

    Parameters
    ----------
    log_every : int
        Append a history entry every ``log_every`` observations. Default 1
        (log every observation). Set to a larger value for long streams.
    warmup : int
        Number of initial pseudo-outcomes to skip when accumulating the ATE
        estimate. History is not recorded during warmup. Default 0.
    forgetting_factor : float
        Controls how quickly old pseudo-outcomes are forgotten.
        ``1.0`` = cumulative Welford mean (no forgetting, default).
        Values < 1.0 (e.g. 0.95–0.99) switch to EWMA so the tracker
        adapts to concept drift. ``alpha = 1 - forgetting_factor``.

    Notes
    -----
    Unlike ``BaseOnlineEstimator``, this is a standalone diagnostic tool
    that users instantiate separately and feed pseudo-outcomes into. It is
    not tied to any specific estimation method.

    When ``forgetting_factor < 1.0``, the internal Welford state is replaced
    by an EWMA (``EWMAStats``). The CI formula remains ``mean ± z * sqrt(var/n)``
    where ``var`` and ``n`` come from the EWMA estimates.

    Examples
    --------
    >>> tracker = ATETracker(log_every=10)
    >>> for pseudo_outcome in [1.5, 2.3, 1.8, 2.1]:
    ...     tracker.update(pseudo_outcome)
    >>> abs(tracker.ate - 1.925) < 1e-10
    True
    """

    def __init__(
        self,
        log_every: int = 1,
        warmup: int = 0,
        forgetting_factor: float = 1.0,
    ) -> None:
        self.log_every = log_every
        self.warmup = warmup
        self.forgetting_factor = forgetting_factor
        # Delegate statistics to the appropriate backend
        self._stats: RunningStats | EWMAStats = (
            EWMAStats(alpha=1.0 - forgetting_factor)
            if forgetting_factor < 1.0
            else RunningStats()
        )
        self._n_total: int = 0  # includes warmup observations
        self._history: list[tuple[int, float, float, float]] = []

    def update(self, pseudo_outcome: float) -> None:
        """Incorporate one pseudo-outcome into the running ATE estimate.

        Parameters
        ----------
        pseudo_outcome : float
            Per-observation pseudo-outcome (e.g. IPW score, DR score, or
            per-obs CATE estimate). The running mean of these values
            converges to the ATE under the relevant identification assumptions.
        """
        self._n_total += 1
        if self._n_total <= self.warmup:
            return

        self._stats.update(pseudo_outcome)

        if self._stats.n % self.log_every == 0:
            lo, hi = self.ci()
            self._history.append((self._n_total, self._stats.mean, lo, hi))

    def reset(self) -> None:
        """Reset all state to the initial (empty) condition."""
        self._stats.reset()
        self._n_total = 0
        self._history = []

    @property
    def ate(self) -> float:
        """Current ATE estimate (running mean of pseudo-outcomes).

        Returns 0.0 before any observations are seen (or during warmup).
        """
        return self._stats.mean

    @property
    def n(self) -> int:
        """Number of pseudo-outcomes processed (excluding warmup)."""
        return self._stats.n

    @property
    def history(self) -> list[tuple[int, float, float, float]]:
        """Recorded history as a list of ``(step, ate, ci_lower, ci_upper)`` tuples."""
        return list(self._history)

    def ci(self, alpha: float = 0.05) -> tuple[float, float]:
        """Return a confidence interval for the current ATE estimate.

        Uses a normal approximation via the central limit theorem.

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
        """
        n = self._stats.n
        if n < 2:
            return (float("-inf"), float("inf"))
        variance = self._stats.variance
        se = math.sqrt(variance / n)
        z = scipy.stats.norm.ppf(1.0 - alpha / 2.0)
        mean = self._stats.mean
        return (mean - z * se, mean + z * se)

    def convergence_width(self, alpha: float = 0.05) -> float:
        """Return the current confidence interval width.

        Useful as an early-stopping criterion: stop collecting data when
        the CI width falls below a target threshold.

        Parameters
        ----------
        alpha : float
            Significance level for the CI. Default 0.05.

        Returns
        -------
        float
            Width of the current CI (``upper - lower``). Returns ``inf``
            before 2 observations are seen.
        """
        lo, hi = self.ci(alpha)
        return hi - lo

    def plot(self, ax: "matplotlib.axes.Axes | None" = None) -> "matplotlib.axes.Axes":
        """Plot the ATE convergence curve with shaded confidence band.

        Parameters
        ----------
        ax : matplotlib.axes.Axes or None
            Axes to plot on. If None, creates a new figure.

        Returns
        -------
        matplotlib.axes.Axes
            The axes with the convergence plot.
        """
        import matplotlib.pyplot as plt

        if ax is None:
            _, ax = plt.subplots()

        if not self._history:
            return ax

        steps = [h[0] for h in self._history]
        ates = [h[1] for h in self._history]
        lows = [h[2] for h in self._history]
        highs = [h[3] for h in self._history]

        ax.plot(steps, ates, label="ATE estimate", color="steelblue")
        ax.fill_between(steps, lows, highs, alpha=0.2, color="steelblue", label="95% CI")
        ax.axhline(self._stats.mean, linestyle="--", color="gray", linewidth=0.8, label="Current ATE")
        ax.set_xlabel("Observations")
        ax.set_ylabel("ATE estimate")
        ax.set_title("ATE Convergence")
        ax.legend()
        return ax
