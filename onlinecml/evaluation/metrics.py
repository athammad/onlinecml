"""Causal evaluation metrics for progressive scoring."""

import math


class ATEError:
    """Running absolute error between the estimated and true ATE.

    Accumulates the true CATE via a running mean (so it works for both
    constant-ATE and heterogeneous streams) and computes
    ``|model.predict_ate() - mean(true_cate)|`` at each checkpoint.

    Examples
    --------
    >>> m = ATEError()
    >>> m.score
    0.0
    """

    def __init__(self) -> None:
        self._n: int = 0
        self._sum_true: float = 0.0
        self._last_ate_hat: float = 0.0

    def update(
        self,
        x: dict,
        w: int,
        y: float,
        true_cate: float,
        cate_hat: float,
        model,  # noqa: ANN001
    ) -> None:
        """Accumulate one observation.

        Parameters
        ----------
        x : dict
            Covariate dict (unused by this metric).
        w : int
            Treatment indicator (unused by this metric).
        y : float
            Observed outcome (unused by this metric).
        true_cate : float
            True CATE for this unit. Used to build a running mean of the
            population ATE.
        cate_hat : float
            Predicted CATE (unused by this metric; uses model.predict_ate()).
        model :
            The causal estimator. Must implement ``predict_ate() -> float``.
        """
        self._n += 1
        self._sum_true += true_cate
        self._last_ate_hat = model.predict_ate()

    @property
    def score(self) -> float:
        """Current |ATE_hat - ATE_true|.

        Returns ``0.0`` before any data has been seen.
        """
        if self._n == 0:
            return 0.0
        return abs(self._last_ate_hat - self._sum_true / self._n)

    def reset(self) -> None:
        """Reset all accumulated state."""
        self.__init__()  # type: ignore[misc]


class PEHE:
    """Running Precision in Estimation of Heterogeneous Effects.

    Computes ``sqrt(mean((cate_hat - cate_true)^2))`` incrementally using
    Welford's algorithm. The ``cate_hat`` passed by ``progressive_causal_score``
    is the predict-before-learn CATE prediction from ``model.predict_one(x)``.

    References
    ----------
    Hill, J. (2011). Bayesian nonparametric modeling for causal inference.
    Journal of Computational and Graphical Statistics, 20(1), 217-240.

    Examples
    --------
    >>> m = PEHE()
    >>> m.score
    0.0
    """

    def __init__(self) -> None:
        self._n: int = 0
        self._mean_sq: float = 0.0

    def update(
        self,
        x: dict,
        w: int,
        y: float,
        true_cate: float,
        cate_hat: float,
        model,  # noqa: ANN001
    ) -> None:
        """Accumulate one observation.

        Parameters
        ----------
        x : dict
            Covariate dict (unused by this metric).
        w : int
            Treatment indicator (unused by this metric).
        y : float
            Observed outcome (unused by this metric).
        true_cate : float
            True CATE for this unit.
        cate_hat : float
            Predicted CATE from ``model.predict_one(x)`` before learning.
        model :
            The causal estimator (unused by this metric).
        """
        self._n += 1
        err_sq = (cate_hat - true_cate) ** 2
        self._mean_sq += (err_sq - self._mean_sq) / self._n

    @property
    def score(self) -> float:
        """Current PEHE (sqrt of running mean squared CATE error).

        Returns ``0.0`` before any data has been seen.
        """
        if self._n == 0:
            return 0.0
        return math.sqrt(self._mean_sq)

    def reset(self) -> None:
        """Reset all accumulated state."""
        self.__init__()  # type: ignore[misc]


class UpliftAUC:
    """Area under the uplift curve (AUUC).

    Accumulates ``(cate_hat, treatment, outcome)`` triples. At each call to
    ``score``, it sorts the buffer by predicted CATE descending, computes the
    cumulative uplift curve, and returns the area via the trapezoidal rule,
    normalized to ``[0, 1]``.

    The uplift at depth ``k`` is::

        uplift(k) = mean_outcome_treated(top_k) - mean_outcome_control(top_k)

    Parameters
    ----------
    max_buffer : int
        Maximum number of recent observations to retain. Older observations
        are dropped to keep memory bounded. Default 5000.

    Examples
    --------
    >>> m = UpliftAUC()
    >>> m.score
    0.0
    """

    def __init__(self, max_buffer: int = 5000) -> None:
        self.max_buffer = max_buffer
        self._buffer: list[tuple[float, int, float]] = []

    def update(
        self,
        x: dict,
        w: int,
        y: float,
        true_cate: float,
        cate_hat: float,
        model,  # noqa: ANN001
    ) -> None:
        """Accumulate one observation.

        Parameters
        ----------
        x : dict
            Covariate dict (unused by this metric).
        w : int
            Treatment indicator (0 or 1).
        y : float
            Observed outcome.
        true_cate : float
            True CATE (unused by this metric).
        cate_hat : float
            Predicted CATE used to rank units.
        model :
            The causal estimator (unused by this metric).
        """
        self._buffer.append((cate_hat, w, y))
        if len(self._buffer) > self.max_buffer:
            self._buffer.pop(0)

    @property
    def score(self) -> float:
        """Current AUUC, normalized to ``[0, 1]``.

        Returns ``0.0`` when fewer than two observations have been seen or
        when all units are in one arm.
        """
        if len(self._buffer) < 2:
            return 0.0
        sorted_buf = sorted(self._buffer, key=lambda t: t[0], reverse=True)
        n = len(sorted_buf)

        cum_t_sum, cum_c_sum = 0.0, 0.0
        cum_t_n,   cum_c_n   = 0,   0
        uplift_vals = []

        for _, w, y in sorted_buf:
            if w == 1:
                cum_t_sum += y
                cum_t_n   += 1
            else:
                cum_c_sum += y
                cum_c_n   += 1
            mean_t = cum_t_sum / cum_t_n if cum_t_n > 0 else 0.0
            mean_c = cum_c_sum / cum_c_n if cum_c_n > 0 else 0.0
            uplift_vals.append(mean_t - mean_c)

        if not uplift_vals:
            return 0.0
        # Trapezoidal AUC over depth fractions [0, 1]
        auc = sum(uplift_vals) / n
        # Normalize by the range of outcomes so score is on a comparable scale
        all_y = [y for _, _, y in sorted_buf]
        y_range = max(all_y) - min(all_y)
        if y_range == 0.0:
            return 0.0
        return max(0.0, auc / y_range)

    def reset(self) -> None:
        """Reset all accumulated state."""
        self._buffer.clear()


class QiniCoefficient:
    """Qini coefficient (area under the Qini curve).

    The Qini curve plots cumulative incremental gains vs. cumulative population
    fraction when units are ranked by predicted CATE descending. The Qini
    coefficient is the area under this curve minus the area under the random
    policy line, normalized by the maximum achievable Qini.

    Parameters
    ----------
    max_buffer : int
        Maximum number of recent observations to retain. Default 5000.

    References
    ----------
    Radcliffe, N.J. (2007). Using control groups to target on predicted lift.
    Direct Marketing Analytics Journal, 14-21.

    Examples
    --------
    >>> m = QiniCoefficient()
    >>> m.score
    0.0
    """

    def __init__(self, max_buffer: int = 5000) -> None:
        self.max_buffer = max_buffer
        self._buffer: list[tuple[float, int, float]] = []

    def update(
        self,
        x: dict,
        w: int,
        y: float,
        true_cate: float,
        cate_hat: float,
        model,  # noqa: ANN001
    ) -> None:
        """Accumulate one observation.

        Parameters
        ----------
        x : dict
            Covariate dict (unused by this metric).
        w : int
            Treatment indicator (0 or 1).
        y : float
            Observed outcome.
        true_cate : float
            True CATE (unused by this metric).
        cate_hat : float
            Predicted CATE used to rank units.
        model :
            The causal estimator (unused by this metric).
        """
        self._buffer.append((cate_hat, w, y))
        if len(self._buffer) > self.max_buffer:
            self._buffer.pop(0)

    @property
    def score(self) -> float:
        """Current normalized Qini coefficient.

        Returns ``0.0`` when fewer than two observations have been seen or
        when either arm is empty.
        """
        if len(self._buffer) < 2:
            return 0.0
        sorted_buf = sorted(self._buffer, key=lambda t: t[0], reverse=True)
        n = len(sorted_buf)

        n_t_total = sum(1 for _, w, _ in sorted_buf if w == 1)
        n_c_total = n - n_t_total
        if n_t_total == 0 or n_c_total == 0:
            return 0.0

        cum_t, cum_c = 0, 0
        qini_vals = [0.0]  # starts at 0

        for _, w, _ in sorted_buf:
            if w == 1:
                cum_t += 1
            else:
                cum_c += 1
            # Qini at this depth: treated_rate - control_rate * (n_t_total / n_c_total)
            qini_vals.append(cum_t / n_t_total - cum_c / n_c_total)

        # Area under Qini curve (trapezoidal)
        depths = [i / n for i in range(n + 1)]
        auc = sum(
            0.5 * (qini_vals[i] + qini_vals[i + 1]) * (depths[i + 1] - depths[i])
            for i in range(n)
        )
        # Normalize: maximum area is 0.5 (perfect model)
        return auc / 0.5

    def reset(self) -> None:
        """Reset all accumulated state."""
        self._buffer.clear()


class CIWidth:
    """Running mean width of the confidence interval on the ATE.

    Computes the mean of ``(upper - lower)`` for each CI returned by
    ``model.predict_ci(alpha)`` at each observation.

    Parameters
    ----------
    alpha : float
        Significance level for the CI. Default 0.05 (95% CI).

    Examples
    --------
    >>> m = CIWidth()
    >>> m.score
    0.0
    """

    def __init__(self, alpha: float = 0.05) -> None:
        self.alpha = alpha
        self._n: int = 0
        self._mean_width: float = 0.0

    def update(
        self,
        x: dict,
        w: int,
        y: float,
        true_cate: float,
        cate_hat: float,
        model,  # noqa: ANN001
    ) -> None:
        """Accumulate one observation.

        Parameters
        ----------
        x : dict
            Covariate dict (unused by this metric).
        w : int
            Treatment indicator (unused by this metric).
        y : float
            Observed outcome (unused by this metric).
        true_cate : float
            True CATE (unused by this metric).
        cate_hat : float
            Predicted CATE (unused by this metric).
        model :
            The causal estimator. Must implement ``predict_ci(alpha) -> tuple``.
        """
        lo, hi = model.predict_ci(alpha=self.alpha)
        width = hi - lo
        self._n += 1
        self._mean_width += (width - self._mean_width) / self._n

    @property
    def score(self) -> float:
        """Current mean CI width.

        Returns ``0.0`` before any data has been seen.
        """
        return self._mean_width

    def reset(self) -> None:
        """Reset all accumulated state."""
        self.__init__(alpha=self.alpha)  # type: ignore[misc]


class CIcoverage:
    """Empirical coverage of the ATE confidence interval.

    At each checkpoint, checks whether the true population ATE (running mean
    of ``true_cate``) falls within ``model.predict_ci(alpha)``. Returns the
    running fraction of checkpoints where the CI covered the true ATE.

    Parameters
    ----------
    alpha : float
        Significance level. Default 0.05 (95% CI; target coverage = 0.95).

    Examples
    --------
    >>> m = CIcoverage()
    >>> m.score
    0.0
    """

    def __init__(self, alpha: float = 0.05) -> None:
        self.alpha = alpha
        self._n_obs: int = 0
        self._sum_true: float = 0.0
        self._n_checks: int = 0
        self._n_covered: int = 0

    def update(
        self,
        x: dict,
        w: int,
        y: float,
        true_cate: float,
        cate_hat: float,
        model,  # noqa: ANN001
    ) -> None:
        """Accumulate one observation and check CI coverage.

        Parameters
        ----------
        x : dict
            Covariate dict (unused by this metric).
        w : int
            Treatment indicator (unused by this metric).
        y : float
            Observed outcome (unused by this metric).
        true_cate : float
            True CATE. Used to estimate the population ATE via running mean.
        cate_hat : float
            Predicted CATE (unused by this metric).
        model :
            The causal estimator. Must implement ``predict_ci(alpha) -> tuple``.
        """
        self._n_obs += 1
        self._sum_true += true_cate
        true_ate = self._sum_true / self._n_obs
        lo, hi = model.predict_ci(alpha=self.alpha)
        self._n_checks += 1
        if lo <= true_ate <= hi:
            self._n_covered += 1

    @property
    def score(self) -> float:
        """Fraction of checkpoints where the CI covered the true ATE.

        Returns ``0.0`` before any data has been seen.
        """
        if self._n_checks == 0:
            return 0.0
        return self._n_covered / self._n_checks

    def reset(self) -> None:
        """Reset all accumulated state."""
        self.__init__(alpha=self.alpha)  # type: ignore[misc]
