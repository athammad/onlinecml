"""Online propensity score estimation wrapping any River classifier."""

from river.base import Base, Classifier


class OnlinePropensityScore(Base):
    """Online propensity score estimator wrapping any River classifier.

    Estimates P(W=1 | X) incrementally, one observation at a time. The
    wrapped classifier is updated after each call to ``learn_one``.
    Predicted probabilities are clipped to ``[clip_min, clip_max]`` to
    prevent extreme importance weights.

    This class does NOT inherit from ``BaseOnlineEstimator`` because it
    is a helper component, not a CATE estimator. It is used internally
    by ``OnlineIPW``, ``OnlineAIPW``, and the meta-learners.

    Parameters
    ----------
    classifier : river.base.Classifier
        Any River binary classifier with ``learn_one`` and
        ``predict_proba_one`` methods.
    clip_min : float
        Lower clip bound for predicted probabilities. Default 0.01.
    clip_max : float
        Upper clip bound for predicted probabilities. Default 0.99.

    Notes
    -----
    Before any observations are seen, ``predict_one`` returns 0.5
    (uniform prior). This means early IPW weights equal 2.0 regardless
    of treatment — the variance is high during warm-up (~50–100 obs).

    River's ``predict_proba_one`` returns ``{True: p, False: 1-p}``
    with boolean keys. This class accesses the probability via
    ``proba[True]``.

    Examples
    --------
    >>> from river.linear_model import LogisticRegression
    >>> ps = OnlinePropensityScore(LogisticRegression())
    >>> ps.predict_one({"age": 30, "income": 50000})
    0.5
    >>> ps.learn_one({"age": 30, "income": 50000}, treatment=1)
    >>> ps.n_seen
    1
    """

    def __init__(
        self,
        classifier: Classifier,
        clip_min: float = 0.01,
        clip_max: float = 0.99,
    ) -> None:
        self.classifier = classifier
        self.clip_min = clip_min
        self.clip_max = clip_max
        self._n_seen: int = 0

    def learn_one(self, x: dict, treatment: int) -> None:
        """Update the propensity model with one observation.

        Parameters
        ----------
        x : dict
            Feature dictionary for this observation.
        treatment : int
            Treatment indicator (0 = control, 1 = treated).
        """
        self.classifier.learn_one(x, bool(treatment))
        self._n_seen += 1

    def predict_one(self, x: dict) -> float:
        """Predict P(W=1 | X) for a single unit.

        Parameters
        ----------
        x : dict
            Feature dictionary for the unit.

        Returns
        -------
        float
            Estimated propensity score, clipped to ``[clip_min, clip_max]``.
            Returns 0.5 before any training observations are seen.
        """
        if self._n_seen == 0:
            return 0.5
        proba = self.classifier.predict_proba_one(x)
        p = proba.get(True, 0.5)
        return max(self.clip_min, min(self.clip_max, p))

    def ipw_weight(self, x: dict, treatment: int) -> float:
        """Compute the inverse probability weight for this observation.

        Parameters
        ----------
        x : dict
            Feature dictionary for the unit.
        treatment : int
            Treatment indicator (0 or 1).

        Returns
        -------
        float
            IPW weight: ``1/p`` for treated units, ``1/(1-p)`` for control.
        """
        p = self.predict_one(x)
        return 1.0 / p if treatment == 1 else 1.0 / (1.0 - p)

    def overlap_weight(self, x: dict, treatment: int) -> float:
        """Compute the overlap (trimming) weight for this observation.

        Overlap weights are bounded and proportional to the probability
        of being in the opposite treatment group, providing more stable
        estimates under near-positivity violations.

        Parameters
        ----------
        x : dict
            Feature dictionary for the unit.
        treatment : int
            Treatment indicator (0 or 1).

        Returns
        -------
        float
            Overlap weight: ``(1-p)`` for treated, ``p`` for control.
        """
        p = self.predict_one(x)
        return (1.0 - p) if treatment == 1 else p

    @property
    def n_seen(self) -> int:
        """Number of observations processed."""
        return self._n_seen
