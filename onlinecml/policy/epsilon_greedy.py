"""Epsilon-greedy exploration policy with exponential decay."""

import math
import random

from onlinecml.base.base_policy import BasePolicy


class EpsilonGreedy(BasePolicy):
    """Epsilon-greedy treatment policy with exponential epsilon decay.

    Randomly explores (assigns random treatment) with probability epsilon,
    and exploits (assigns the treatment with the highest estimated effect)
    with probability 1 - epsilon. Epsilon decays exponentially from
    ``eps_start`` toward ``eps_end`` over time.

    Parameters
    ----------
    eps_start : float
        Initial exploration rate. Default 0.5.
    eps_end : float
        Minimum exploration rate (asymptote). Default 0.05.
    decay : int
        Decay timescale in steps. Larger values = slower decay. Default 2000.
    seed : int or None
        Random seed for reproducibility. Uses standard library ``random``
        to avoid numpy serialization issues.

    Notes
    -----
    Epsilon at step ``t`` is:

    .. math::

        \\epsilon_t = \\epsilon_{\\text{end}} +
        (\\epsilon_{\\text{start}} - \\epsilon_{\\text{end}}) \\cdot
        e^{-t / \\text{decay}}

    **Explore:** With probability ``eps``, a random treatment is chosen
    with propensity 0.5.

    **Exploit:** With probability ``1 - eps``, the treatment with the
    higher estimated CATE is chosen. The propensity of the chosen
    treatment under the greedy policy is ``1 - eps`` (since we always
    choose the same arm when exploiting).

    Examples
    --------
    >>> policy = EpsilonGreedy(eps_start=0.5, eps_end=0.05, decay=100, seed=0)
    >>> treatment, propensity = policy.choose(cate_score=1.5, step=0)
    >>> treatment in (0, 1)
    True
    >>> 0.0 < propensity <= 1.0
    True
    """

    def __init__(
        self,
        eps_start: float = 0.5,
        eps_end: float = 0.05,
        decay: int = 2000,
        seed: int | None = None,
    ) -> None:
        self.eps_start = eps_start
        self.eps_end = eps_end
        self.decay = decay
        self.seed = seed
        # Standard library random — not numpy (avoids serialization issues)
        self._rng = random.Random(seed)
        # Internal arm reward tracking (used when cate_score is not provided)
        self._arm_sums: list[float] = [0.0, 0.0]
        self._arm_counts: list[int] = [0, 0]
        self._last_arm: int | None = None

    def choose(self, cate_score: float, step: int) -> tuple[int, float]:
        """Choose a treatment assignment.

        Parameters
        ----------
        cate_score : float
            Current CATE estimate. Positive = treatment beneficial.
        step : int
            Current time step (used for epsilon decay).

        Returns
        -------
        treatment : int
            Chosen treatment (0 or 1).
        propensity : float
            Probability of the chosen treatment under this policy.
        """
        eps = self.eps_end + (self.eps_start - self.eps_end) * math.exp(-step / self.decay)

        if self._rng.random() < eps:
            # Explore: random treatment, uniform propensity
            treatment = self._rng.randint(0, 1)
            self._last_arm = treatment
            return treatment, 0.5

        # Exploit: use cate_score if non-zero (causal model available),
        # else fall back to internal arm reward estimates.
        if cate_score == 0.0:
            mu1 = self._arm_sums[1] / self._arm_counts[1] if self._arm_counts[1] > 0 else 0.5
            mu0 = self._arm_sums[0] / self._arm_counts[0] if self._arm_counts[0] > 0 else 0.5
            cate_score = mu1 - mu0
        treatment = 1 if cate_score > 0 else 0
        self._last_arm = treatment
        propensity = 1.0 - eps
        return treatment, propensity

    def update(self, reward: float) -> None:
        """Update internal arm reward estimate after observing a reward.

        Parameters
        ----------
        reward : float
            Observed reward for the arm chosen in the most recent
            ``choose`` call. No-op if ``choose`` has not been called yet.
        """
        if self._last_arm is not None:
            self._arm_sums[self._last_arm] += reward
            self._arm_counts[self._last_arm] += 1

    def current_epsilon(self, step: int) -> float:
        """Return the current epsilon value at a given step.

        Parameters
        ----------
        step : int
            Current time step.

        Returns
        -------
        float
            Epsilon at this step.
        """
        return self.eps_end + (self.eps_start - self.eps_end) * math.exp(-step / self.decay)
