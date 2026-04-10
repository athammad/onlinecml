"""Concept drift monitor for online causal estimates."""

from river.drift import ADWIN


class ConceptDriftMonitor:
    """Monitors the ATE estimate stream for structural breaks (concept drift).

    Wraps River's ADWIN (Adaptive Windowing) drift detector to identify
    changes in the distribution of per-observation pseudo-outcomes, which
    signal a shift in the underlying treatment effect.

    Parameters
    ----------
    delta : float
        ADWIN confidence parameter. Smaller values reduce false alarm
        rate at the cost of slower detection. Default 0.002.

    Notes
    -----
    ADWIN maintains an adaptive window over a data stream and raises a
    drift signal when the mean of the earlier and later sub-windows
    differ by more than the statistical threshold.

    When drift is detected, ``drift_detected`` returns True and
    ``n_drifts`` increments. The estimator being monitored should be
    reset after drift is detected (the monitor does not do this
    automatically — it only signals).

    References
    ----------
    Bifet, A. and Gavalda, R. (2007). Learning from time-changing data
    with adaptive windowing. Proceedings of the 7th SIAM International
    Conference on Data Mining, 443-448.

    Examples
    --------
    >>> monitor = ConceptDriftMonitor(delta=0.002)
    >>> for pseudo_outcome in [1.0, 1.1, 0.9, 1.0, 5.0, 5.1, 4.9]:
    ...     monitor.update(pseudo_outcome)
    >>> monitor.n_drifts >= 0
    True
    """

    def __init__(self, delta: float = 0.002) -> None:
        self.delta = delta
        self._detector = ADWIN(delta=delta)
        self._n_drifts: int = 0
        self._drift_detected: bool = False
        self._n_seen: int = 0

    def update(self, pseudo_outcome: float) -> None:
        """Feed one pseudo-outcome to the drift detector.

        Parameters
        ----------
        pseudo_outcome : float
            Per-observation pseudo-outcome (e.g. IPW score or CATE estimate).
            Drift in this stream indicates a shift in the treatment effect.
        """
        self._detector.update(pseudo_outcome)
        self._n_seen += 1
        if self._detector.drift_detected:
            self._n_drifts += 1
            self._drift_detected = True
        else:
            self._drift_detected = False

    @property
    def drift_detected(self) -> bool:
        """True if drift was detected on the most recent ``update`` call."""
        return self._drift_detected

    @property
    def n_drifts(self) -> int:
        """Total number of drift events detected since initialization."""
        return self._n_drifts

    @property
    def n_seen(self) -> int:
        """Total number of observations processed."""
        return self._n_seen

    def reset(self) -> None:
        """Reset the detector and drift counters."""
        self._detector = ADWIN(delta=self.delta)
        self._n_drifts = 0
        self._drift_detected = False
        self._n_seen = 0
