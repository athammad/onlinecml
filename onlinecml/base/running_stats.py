"""Online running statistics using Welford's and West's algorithms."""

import math


class RunningStats:
    """Online mean and variance using Welford's single-pass algorithm.

    Computes the sample mean, variance, and standard deviation of a
    stream of scalar values without storing the data.

    Examples
    --------
    >>> stats = RunningStats()
    >>> for x in [2.0, 4.0, 6.0]:
    ...     stats.update(x)
    >>> stats.mean
    4.0
    >>> stats.n
    3
    """

    def __init__(self) -> None:
        self._n: int = 0
        self._mean: float = 0.0
        self._M2: float = 0.0

    def update(self, x: float) -> None:
        """Update statistics with a new observation.

        Parameters
        ----------
        x : float
            The new scalar value to incorporate.
        """
        self._n += 1
        delta = x - self._mean
        self._mean += delta / self._n
        delta2 = x - self._mean
        self._M2 += delta * delta2

    def reset(self) -> None:
        """Reset all state to the initial (empty) condition."""
        self._n = 0
        self._mean = 0.0
        self._M2 = 0.0

    @property
    def n(self) -> int:
        """Number of observations seen."""
        return self._n

    @property
    def mean(self) -> float:
        """Current sample mean. Returns 0.0 before any observations."""
        return self._mean

    @property
    def variance(self) -> float:
        """Current sample variance (ddof=1). Returns 0.0 when n < 2."""
        if self._n < 2:
            return 0.0
        return self._M2 / (self._n - 1)

    @property
    def std(self) -> float:
        """Current sample standard deviation. Returns 0.0 when n < 2."""
        return math.sqrt(self.variance)


class WeightedRunningStats:
    """Online weighted mean and variance using West's (1979) algorithm.

    Computes the weighted mean and population-weighted variance of a
    stream of scalar values with associated importance weights.

    Notes
    -----
    Returns population-weighted variance (S / sum_w), not sample variance.
    This is appropriate for SMD computation where we normalize by pooled
    standard deviation, not for statistical inference.

    References
    ----------
    West, D.H.D. (1979). Updating mean and variance estimates: an improved
    method. Communications of the ACM, 22(9), 532-535.

    Examples
    --------
    >>> stats = WeightedRunningStats()
    >>> stats.update(2.0, w=1.0)
    >>> stats.update(4.0, w=2.0)
    >>> stats.mean  # (2*1 + 4*2) / 3 = 10/3
    3.3333333333333335
    """

    def __init__(self) -> None:
        self._sum_w: float = 0.0
        self._mean: float = 0.0
        self._S: float = 0.0  # weighted sum of squared deviations

    def update(self, x: float, w: float = 1.0) -> None:
        """Update statistics with a new weighted observation.

        Parameters
        ----------
        x : float
            The new scalar value to incorporate.
        w : float
            Non-negative importance weight. Silently ignored if w <= 0.
        """
        if w <= 0.0:
            return
        self._sum_w += w
        mean_old = self._mean
        self._mean += (w / self._sum_w) * (x - mean_old)
        self._S += w * (x - mean_old) * (x - self._mean)

    def reset(self) -> None:
        """Reset all state to the initial (empty) condition."""
        self._sum_w = 0.0
        self._mean = 0.0
        self._S = 0.0

    @property
    def sum_weights(self) -> float:
        """Sum of all weights seen so far."""
        return self._sum_w

    @property
    def mean(self) -> float:
        """Current weighted mean. Returns 0.0 before any observations."""
        return self._mean

    @property
    def variance(self) -> float:
        """Population-weighted variance (S / sum_w). Returns 0.0 when sum_w <= 0."""
        if self._sum_w <= 0.0:
            return 0.0
        return self._S / self._sum_w

    @property
    def std(self) -> float:
        """Population-weighted standard deviation. Returns 0.0 when sum_w <= 0."""
        return math.sqrt(self.variance)
