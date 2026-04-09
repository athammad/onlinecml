"""Running ATE tracker with confidence intervals and convergence plotting."""

import math
from typing import TYPE_CHECKING

import scipy.stats

if TYPE_CHECKING:
    import matplotlib.axes


class ATETracker:
    """Tracks the running ATE estimate with online confidence intervals.

    Maintains a running mean and variance of per-observation pseudo-outcomes
    using Welford's algorithm, and optionally records history for convergence
    plotting.

    Parameters
    ----------
    log_every : int
        Append a history entry every ``log_every`` observations. Default 1
        (log every observation). Set to a larger value for long streams.

    Notes
    -----
    Unlike ``BaseOnlineEstimator``, this is a standalone diagnostic tool
    that users instantiate separately and feed pseudo-outcomes into. It is
    not tied to any specific estimation method.

    ``ATETracker`` internal Welford state is NOT part of the constructor,
    so it is not preserved by ``clone()``. This is intentional — the tracker
    is always created fresh.

    Examples
    --------
    >>> tracker = ATETracker(log_every=10)
    >>> for pseudo_outcome in [1.5, 2.3, 1.8, 2.1]:
    ...     tracker.update(pseudo_outcome)
    >>> abs(tracker.ate - 1.925) < 1e-10
    True
    """

    def __init__(self, log_every: int = 1) -> None:
        self.log_every = log_every
        # Welford state — not constructor params (intentionally not cloned)
        self._n: int = 0
        self._mean: float = 0.0
        self._M2: float = 0.0
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
        self._n += 1
        delta = pseudo_outcome - self._mean
        self._mean += delta / self._n
        delta2 = pseudo_outcome - self._mean
        self._M2 += delta * delta2

        if self._n % self.log_every == 0:
            lo, hi = self.ci()
            self._history.append((self._n, self._mean, lo, hi))

    def reset(self) -> None:
        """Reset all state to the initial (empty) condition."""
        self._n = 0
        self._mean = 0.0
        self._M2 = 0.0
        self._history = []

    @property
    def ate(self) -> float:
        """Current ATE estimate (running mean of pseudo-outcomes).

        Returns 0.0 before any observations are seen.
        """
        return self._mean

    @property
    def n(self) -> int:
        """Number of pseudo-outcomes processed."""
        return self._n

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
        if self._n < 2:
            return (float("-inf"), float("inf"))
        variance = self._M2 / (self._n - 1)
        se = math.sqrt(variance / self._n)
        z = scipy.stats.norm.ppf(1.0 - alpha / 2.0)
        return (self._mean - z * se, self._mean + z * se)

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
        ax.axhline(self._mean, linestyle="--", color="gray", linewidth=0.8, label="Current ATE")
        ax.set_xlabel("Observations")
        ax.set_ylabel("ATE estimate")
        ax.set_title("ATE Convergence")
        ax.legend()
        return ax
