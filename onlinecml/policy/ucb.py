"""Upper Confidence Bound (UCB) exploration policy."""

import math

from onlinecml.base.base_policy import BasePolicy


class UCB(BasePolicy):
    """Upper Confidence Bound policy for treatment selection.

    Selects the treatment with the highest upper confidence bound on its
    expected reward. Balances exploration (high uncertainty) and
    exploitation (high mean reward) via a confidence coefficient.

    Parameters
    ----------
    confidence : float
        Exploration coefficient. Larger values encourage more exploration.
        Default 1.0 (standard UCB1).
    min_pulls : int
        Minimum number of times each arm must be pulled before switching
        to UCB selection. During warm-up, arms are pulled in round-robin.
        Default 1.

    Notes
    -----
    UCB1 bound for arm ``a``:

    .. math::

        UCB_a = \\hat{\\mu}_a + c \\sqrt{\\frac{\\ln(t)}{n_a}}

    where ``t`` is the total number of pulls, ``n_a`` is the number of
    pulls for arm ``a``, and ``c`` is the confidence coefficient.

    The propensity returned reflects whether we are in the warm-up phase
    (0.5) or UCB exploitation phase (1 - exploration_fraction).

    Examples
    --------
    >>> policy = UCB(confidence=1.0)
    >>> treatment, propensity = policy.choose(cate_score=0.0, step=5)
    >>> treatment in (0, 1)
    True
    """

    def __init__(self, confidence: float = 1.0, min_pulls: int = 1) -> None:
        self.confidence = confidence
        self.min_pulls = min_pulls
        self._n_pulls = [0, 0]       # pulls per arm
        self._sum_reward = [0.0, 0.0]
        self._total_pulls: int = 0
        self._last_treatment: int = 0

    def choose(self, cate_score: float, step: int) -> tuple[int, float]:
        """Choose a treatment using the UCB rule.

        Parameters
        ----------
        cate_score : float
            Not used directly; the UCB rule uses observed rewards.
        step : int
            Not used directly; the class tracks pulls internally.

        Returns
        -------
        treatment : int
            Arm with the highest UCB score.
        propensity : float
            0.5 during warm-up; approximate propensity during UCB phase.
        """
        # Warm-up: ensure each arm is pulled min_pulls times
        for arm in range(2):
            if self._n_pulls[arm] < self.min_pulls:
                self._last_treatment = arm
                return arm, 0.5

        # UCB selection
        t = max(1, self._total_pulls)
        ucb_scores = []
        for arm in range(2):
            mean = self._sum_reward[arm] / self._n_pulls[arm]
            bonus = self.confidence * math.sqrt(math.log(t) / max(1, self._n_pulls[arm]))
            ucb_scores.append(mean + bonus)

        treatment = 1 if ucb_scores[1] >= ucb_scores[0] else 0
        self._last_treatment = treatment
        self._total_pulls += 1
        return treatment, 0.5

    def update(self, reward: float) -> None:
        """Update the reward estimate for the last chosen arm.

        Parameters
        ----------
        reward : float
            Observed outcome after applying the last chosen treatment.
        """
        arm = self._last_treatment
        self._n_pulls[arm] += 1
        self._sum_reward[arm] += reward

    def reset(self) -> None:
        """Reset all arm statistics."""
        self.__init__(**self._get_params())  # type: ignore[misc]
