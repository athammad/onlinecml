"""Real-time Love Plot for visualizing covariate balance."""

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import matplotlib.axes
    import matplotlib.figure


class LiveLovePlot:
    """Real-time Love Plot for monitoring covariate balance online.

    Displays raw and weighted Standardized Mean Differences (SMD) for a
    set of covariates. Updates the plot every ``update_every`` steps.
    A vertical reference line at ``|SMD| = 0.1`` marks the conventional
    "well-balanced" threshold.

    Parameters
    ----------
    covariates : list of str
        Names of covariates to display (in order).
    update_every : int
        Redraw the plot every ``update_every`` calls to ``update``.
        Default 100.
    balance_threshold : float
        Reference line position. Default 0.1.

    Notes
    -----
    This class wraps ``OnlineSMD`` internally. Users can either pass
    feature dicts directly to ``update`` or maintain an external
    ``OnlineSMD`` instance and call ``render`` with its report.

    The plot is only rendered when ``matplotlib`` is available. If
    ``matplotlib`` is not installed, calls to ``update`` and ``render``
    are no-ops.

    Examples
    --------
    >>> plot = LiveLovePlot(covariates=["age", "income"], update_every=50)
    >>> plot.update({"age": 30, "income": 50000}, treatment=1, weight=1.2)
    >>> ax = plot.render()
    """

    def __init__(
        self,
        covariates: list[str],
        update_every: int = 100,
        balance_threshold: float = 0.1,
    ) -> None:
        self.covariates = covariates
        self.update_every = update_every
        self.balance_threshold = balance_threshold
        self._n: int = 0
        self._fig = None
        self._ax = None
        # Internal SMD tracker
        from onlinecml.diagnostics.smd import OnlineSMD
        self._smd = OnlineSMD(covariates=covariates)

    def update(self, x: dict, treatment: int, weight: float = 1.0) -> None:
        """Update the covariate balance statistics.

        Parameters
        ----------
        x : dict
            Feature dictionary for this observation.
        treatment : int
            Treatment indicator (0 or 1).
        weight : float
            Importance weight for this observation. Default 1.0.
        """
        self._smd.update(x, treatment, weight=weight)
        self._n += 1
        if self._n % self.update_every == 0:
            self.render()

    def render(self, ax: "matplotlib.axes.Axes | None" = None) -> "matplotlib.axes.Axes | None":
        """Render the Love Plot from current SMD data.

        Parameters
        ----------
        ax : matplotlib.axes.Axes or None
            Axes to render on. If None, creates or reuses internal axes.

        Returns
        -------
        matplotlib.axes.Axes or None
            The rendered axes, or None if matplotlib is unavailable.
        """
        try:
            import matplotlib.pyplot as plt
        except ImportError:
            return None

        report = self._smd.report()
        if not report:
            return None

        if ax is None:
            if self._fig is None:
                self._fig, self._ax = plt.subplots(figsize=(8, max(3, len(self.covariates) * 0.5)))
            ax = self._ax

        ax.cla()
        covs = list(report.keys())
        raw_smds = [report[c][0] for c in covs]
        weighted_smds = [report[c][1] for c in covs]
        y_pos = list(range(len(covs)))

        ax.scatter(raw_smds, y_pos, marker="o", label="Raw SMD", color="steelblue", zorder=3)
        ax.scatter(
            weighted_smds, y_pos, marker="^", label="Weighted SMD", color="darkorange", zorder=3
        )
        ax.axvline(
            self.balance_threshold, linestyle="--", color="red", linewidth=0.8, label="Threshold"
        )
        ax.axvline(
            -self.balance_threshold, linestyle="--", color="red", linewidth=0.8
        )
        ax.axvline(0, linestyle="-", color="black", linewidth=0.5)
        ax.set_yticks(y_pos)
        ax.set_yticklabels(covs)
        ax.set_xlabel("Standardized Mean Difference")
        ax.set_title(f"Love Plot (n={self._n})")
        ax.legend(loc="lower right")

        if self._fig is not None:
            self._fig.tight_layout()
            plt.pause(0.001)

        return ax

    def save(self, path: str) -> None:
        """Save the current plot to a file.

        Parameters
        ----------
        path : str
            File path (e.g. ``'balance.png'``).
        """
        if self._fig is not None:
            self._fig.savefig(path, bbox_inches="tight")
