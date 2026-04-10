"""Online K-nearest-neighbor matching for CATE estimation."""

import heapq
from collections import deque

from onlinecml.base.base_estimator import BaseOnlineEstimator
from onlinecml.base.running_stats import RunningStats
from onlinecml.matching.distance import DistanceFn, euclidean_distance


class OnlineMatching(BaseOnlineEstimator):
    """Online K-nearest-neighbor matching estimator for CATE.

    Maintains separate sliding-window buffers for treated and control
    units. For each new observation, finds the K nearest neighbors in
    the opposite treatment arm and computes a matched CATE estimate via
    IPW-corrected neighbor averaging.

    Parameters
    ----------
    k : int
        Number of nearest neighbors to match. Default 1.
    buffer_size : int
        Maximum number of units to retain in each arm's buffer.
        Older units are dropped when the buffer is full (FIFO).
        Default 200.
    distance_fn : callable or None
        Distance function ``f(x1, x2) -> float``. Defaults to
        ``euclidean_distance``.

    Notes
    -----
    The buffer implements Sliding Window Nearest Neighbor (SWINN) matching.
    With finite ``buffer_size``, older observations may be dropped. This
    provides implicit adaptation to concept drift at the cost of match
    quality early in the stream.

    The per-observation CATE estimate is:

    .. math::

        \\hat{\\tau}_i = Y_i - \\frac{1}{K} \\sum_{j \\in \\mathcal{N}(i)} Y_j

    where ``N(i)`` is the K nearest neighbors in the opposite arm.

    **Predict-then-match:** The CATE estimate is computed from the current
    buffer *before* the new observation is added.

    Examples
    --------
    >>> from onlinecml.datasets import LinearCausalStream
    >>> matcher = OnlineMatching(k=3, buffer_size=100)
    >>> for x, w, y, _ in LinearCausalStream(n=500, seed=42):
    ...     matcher.learn_one(x, w, y)
    >>> isinstance(matcher.predict_ate(), float)
    True
    """

    def __init__(
        self,
        k: int = 1,
        buffer_size: int = 200,
        distance_fn: DistanceFn | None = None,
    ) -> None:
        self.k = k
        self.buffer_size = buffer_size
        self.distance_fn = distance_fn if distance_fn is not None else euclidean_distance
        # Non-constructor state
        self._n_seen: int = 0
        self._ate_stats: RunningStats = RunningStats()
        self._treated_buffer: deque = deque()   # list of (x, y) tuples
        self._control_buffer: deque = deque()

    def _find_knn(self, x: dict, buffer: deque, k: int) -> list[float]:
        """Find K nearest neighbor outcomes in a buffer.

        Parameters
        ----------
        x : dict
            Query feature dictionary.
        buffer : deque
            Buffer of ``(features, outcome)`` tuples to search.
        k : int
            Number of neighbors to return.

        Returns
        -------
        list of float
            Outcomes of the K nearest neighbors. Returns an empty list
            if the buffer is empty.
        """
        if not buffer:
            return []
        distances = [(self.distance_fn(x, bx), by) for bx, by in buffer]
        k_nearest = heapq.nsmallest(k, distances, key=lambda t: t[0])
        return [y for _, y in k_nearest]

    def learn_one(
        self,
        x: dict,
        treatment: int,
        outcome: float,
        propensity: float | None = None,
    ) -> None:
        """Process one observation and update the matched CATE estimate.

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
        # Match against the opposite arm's buffer (predict-then-add)
        if treatment == 1:
            neighbor_outcomes = self._find_knn(x, self._control_buffer, self.k)
        else:
            neighbor_outcomes = self._find_knn(x, self._treated_buffer, self.k)

        if neighbor_outcomes:
            neighbor_mean = sum(neighbor_outcomes) / len(neighbor_outcomes)
            # Treated: cate = Y - neighbor_mean; Control: cate = neighbor_mean - Y
            cate = outcome - neighbor_mean if treatment == 1 else neighbor_mean - outcome
            self._ate_stats.update(cate)

        self._n_seen += 1

        # Add current obs to the appropriate buffer
        if treatment == 1:
            self._treated_buffer.append((x, outcome))
            if len(self._treated_buffer) > self.buffer_size:
                self._treated_buffer.popleft()
        else:
            self._control_buffer.append((x, outcome))
            if len(self._control_buffer) > self.buffer_size:
                self._control_buffer.popleft()

    def predict_one(self, x: dict) -> float:
        """Predict the CATE for a single unit via nearest-neighbor matching.

        Finds the K nearest treated and control neighbors in the buffer,
        and returns the difference in their mean outcomes.

        Parameters
        ----------
        x : dict
            Feature dictionary for the unit.

        Returns
        -------
        float
            Estimated CATE. Returns 0.0 if either buffer is empty.
        """
        treated_nn = self._find_knn(x, self._treated_buffer, self.k)
        control_nn = self._find_knn(x, self._control_buffer, self.k)
        if not treated_nn or not control_nn:
            return 0.0
        return sum(treated_nn) / len(treated_nn) - sum(control_nn) / len(control_nn)
