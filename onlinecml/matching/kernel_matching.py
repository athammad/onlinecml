"""Online kernel-weighted matching for CATE estimation."""

import math
from collections import deque
from collections.abc import Callable

from onlinecml.base.base_estimator import BaseOnlineEstimator
from onlinecml.base.running_stats import RunningStats
from onlinecml.matching.distance import DistanceFn, euclidean_distance


def _gaussian_kernel(distance: float, bandwidth: float) -> float:
    """Gaussian (RBF) kernel weight.

    Parameters
    ----------
    distance : float
        Distance between two units.
    bandwidth : float
        Bandwidth parameter (sigma).

    Returns
    -------
    float
        Kernel weight in (0, 1].
    """
    return math.exp(-0.5 * (distance / bandwidth) ** 2)


class OnlineKernelMatching(BaseOnlineEstimator):
    """Online kernel-weighted matching for CATE estimation.

    Instead of selecting K discrete neighbors, uses all units in the
    opposite arm's buffer with weights determined by a kernel function.
    The CATE for a treated unit is:

    .. math::

        \\hat{\\tau}(x) = Y_i - \\frac{\\sum_j K(d(x, x_j)) Y_j}{\\sum_j K(d(x, x_j))}

    Parameters
    ----------
    bandwidth : float
        Bandwidth of the kernel. Smaller values → sharper matching.
        Default 1.0.
    buffer_size : int
        Maximum number of units to retain per arm. Default 200.
    distance_fn : callable or None
        Distance function ``f(x1, x2) -> float``.
        Defaults to ``euclidean_distance``.
    kernel_fn : callable or None
        Kernel function ``f(distance, bandwidth) -> float``.
        Defaults to Gaussian kernel.

    Examples
    --------
    >>> from onlinecml.datasets import LinearCausalStream
    >>> matcher = OnlineKernelMatching(bandwidth=1.5, buffer_size=100)
    >>> for x, w, y, _ in LinearCausalStream(n=300, seed=42):
    ...     matcher.learn_one(x, w, y)
    >>> isinstance(matcher.predict_ate(), float)
    True
    """

    def __init__(
        self,
        bandwidth: float = 1.0,
        buffer_size: int = 200,
        distance_fn: DistanceFn | None = None,
        kernel_fn: Callable[[float, float], float] | None = None,
    ) -> None:
        self.bandwidth = bandwidth
        self.buffer_size = buffer_size
        self.distance_fn = distance_fn if distance_fn is not None else euclidean_distance
        self.kernel_fn = kernel_fn if kernel_fn is not None else _gaussian_kernel
        # Non-constructor state
        self._n_seen: int = 0
        self._ate_stats: RunningStats = RunningStats()
        self._treated_buffer: deque = deque()
        self._control_buffer: deque = deque()

    def _kernel_weighted_mean(self, x: dict, buffer: deque) -> float | None:
        """Compute the kernel-weighted mean outcome from a buffer.

        Parameters
        ----------
        x : dict
            Query feature dictionary.
        buffer : deque
            Buffer of ``(features, outcome)`` tuples.

        Returns
        -------
        float or None
            Kernel-weighted mean outcome, or None if buffer is empty
            or all weights are zero.
        """
        if not buffer:
            return None
        total_weight = 0.0
        weighted_sum = 0.0
        for bx, by in buffer:
            d = self.distance_fn(x, bx)
            w = self.kernel_fn(d, self.bandwidth)
            total_weight += w
            weighted_sum += w * by
        if total_weight <= 0.0:
            return None
        return weighted_sum / total_weight

    def learn_one(
        self,
        x: dict,
        treatment: int,
        outcome: float,
        propensity: float | None = None,
    ) -> None:
        """Process one observation and update the CATE estimate.

        Parameters
        ----------
        x : dict
            Feature dictionary for this observation.
        treatment : int
            Treatment indicator (0 = control, 1 = treated).
        outcome : float
            Observed outcome.
        propensity : float or None
            Not used; included for API compatibility.
        """
        opposite_buffer = self._control_buffer if treatment == 1 else self._treated_buffer
        neighbor_mean = self._kernel_weighted_mean(x, opposite_buffer)
        self._n_seen += 1

        if neighbor_mean is not None:
            cate = outcome - neighbor_mean if treatment == 1 else neighbor_mean - outcome
            self._ate_stats.update(cate)

        if treatment == 1:
            self._treated_buffer.append((x, outcome))
            if len(self._treated_buffer) > self.buffer_size:
                self._treated_buffer.popleft()
        else:
            self._control_buffer.append((x, outcome))
            if len(self._control_buffer) > self.buffer_size:
                self._control_buffer.popleft()

    def predict_one(self, x: dict) -> float:
        """Predict the CATE for a single unit via kernel-weighted matching.

        Parameters
        ----------
        x : dict
            Feature dictionary for the unit.

        Returns
        -------
        float
            Estimated CATE. Returns 0.0 if either buffer is empty.
        """
        y_t = self._kernel_weighted_mean(x, self._treated_buffer)
        y_c = self._kernel_weighted_mean(x, self._control_buffer)
        if y_t is None or y_c is None:
            return 0.0
        return y_t - y_c
