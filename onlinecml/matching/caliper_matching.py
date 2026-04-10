"""Online caliper (maximum-distance) matching for CATE estimation."""

from collections import deque

from onlinecml.base.base_estimator import BaseOnlineEstimator
from onlinecml.base.running_stats import RunningStats
from onlinecml.matching.distance import DistanceFn, euclidean_distance


class OnlineCaliperMatching(BaseOnlineEstimator):
    """Online matching with a maximum distance threshold (caliper).

    Extends nearest-neighbor matching by rejecting matches that exceed
    a maximum distance threshold. Units that cannot be matched within
    the caliper are tracked separately. Reports the proportion of units
    in common support.

    Parameters
    ----------
    caliper : float
        Maximum allowable distance for a match. Observations whose
        nearest neighbor is farther than ``caliper`` are counted as
        unmatched. Default 1.0.
    buffer_size : int
        Maximum number of units to retain in each arm's buffer. Default 200.
    distance_fn : callable or None
        Distance function ``f(x1, x2) -> float``.
        Defaults to ``euclidean_distance``.

    Notes
    -----
    The ``common_support_rate`` property returns the proportion of
    observations that were successfully matched (distance ≤ caliper).
    A high unmatched rate indicates a positivity violation.

    Examples
    --------
    >>> from onlinecml.datasets import LinearCausalStream
    >>> matcher = OnlineCaliperMatching(caliper=2.0, buffer_size=100)
    >>> for x, w, y, _ in LinearCausalStream(n=300, seed=42):
    ...     matcher.learn_one(x, w, y)
    >>> isinstance(matcher.common_support_rate, float)
    True
    """

    def __init__(
        self,
        caliper: float = 1.0,
        buffer_size: int = 200,
        distance_fn: DistanceFn | None = None,
    ) -> None:
        self.caliper = caliper
        self.buffer_size = buffer_size
        self.distance_fn = distance_fn if distance_fn is not None else euclidean_distance
        # Non-constructor state
        self._n_seen: int = 0
        self._n_matched: int = 0
        self._ate_stats: RunningStats = RunningStats()
        self._treated_buffer: deque = deque()
        self._control_buffer: deque = deque()

    def _find_nearest(self, x: dict, buffer: deque) -> tuple[float, float] | None:
        """Find the nearest neighbor in a buffer, return (distance, outcome).

        Parameters
        ----------
        x : dict
            Query feature dictionary.
        buffer : deque
            Buffer of ``(features, outcome)`` tuples.

        Returns
        -------
        tuple of (float, float) or None
            ``(distance, outcome)`` of the nearest neighbor, or None if
            the buffer is empty.
        """
        if not buffer:
            return None
        best_dist = float("inf")
        best_y = 0.0
        for bx, by in buffer:
            d = self.distance_fn(x, bx)
            if d < best_dist:
                best_dist = d
                best_y = by
        return (best_dist, best_y)

    def learn_one(
        self,
        x: dict,
        treatment: int,
        outcome: float,
        propensity: float | None = None,
    ) -> None:
        """Process one observation with caliper-constrained matching.

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
        result = self._find_nearest(x, opposite_buffer)
        self._n_seen += 1

        if result is not None:
            dist, neighbor_y = result
            if dist <= self.caliper:
                cate = outcome - neighbor_y if treatment == 1 else neighbor_y - outcome
                self._ate_stats.update(cate)
                self._n_matched += 1

        # Add to appropriate buffer
        if treatment == 1:
            self._treated_buffer.append((x, outcome))
            if len(self._treated_buffer) > self.buffer_size:
                self._treated_buffer.popleft()
        else:
            self._control_buffer.append((x, outcome))
            if len(self._control_buffer) > self.buffer_size:
                self._control_buffer.popleft()

    def predict_one(self, x: dict) -> float:
        """Predict CATE for a single unit via caliper-constrained matching.

        Parameters
        ----------
        x : dict
            Feature dictionary for the unit.

        Returns
        -------
        float
            Estimated CATE. Returns 0.0 if either buffer is empty or if
            the nearest neighbors exceed the caliper.
        """
        res_t = self._find_nearest(x, self._treated_buffer)
        res_c = self._find_nearest(x, self._control_buffer)
        if res_t is None or res_c is None:
            return 0.0
        dist_t, y_t = res_t
        dist_c, y_c = res_c
        if dist_t > self.caliper or dist_c > self.caliper:
            return 0.0
        return y_t - y_c

    @property
    def common_support_rate(self) -> float:
        """Proportion of observations successfully matched within the caliper.

        Returns
        -------
        float
            Value in [0, 1]. Returns 0.0 before any observations are seen.
        """
        if self._n_seen == 0:
            return 0.0
        return self._n_matched / self._n_seen
