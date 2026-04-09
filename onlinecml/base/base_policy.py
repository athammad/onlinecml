"""Abstract base class for causal exploration policies."""

import abc

from river.base import Base


class BasePolicy(Base):
    """Abstract base class for treatment exploration policies.

    A policy decides which treatment to assign and with what probability,
    given a CATE score estimate and the current step count. Subclasses
    implement different exploration-exploitation trade-offs.

    All policies follow River conventions: constructor parameters are
    stored as instance attributes with the same name, enabling clone()
    and _get_params() to work correctly.

    Notes
    -----
    This class intentionally does NOT inherit from River's bandit Policy
    because our interface operates on CATE scores (continuous real-valued
    estimates of causal effects) rather than bandit arm indices. The
    returned propensity is the probability of the chosen treatment under
    the policy, which is used for IPW correction downstream.
    """

    @abc.abstractmethod
    def choose(self, cate_score: float, step: int) -> tuple[int, float]:
        """Choose a treatment assignment given the current CATE estimate.

        Parameters
        ----------
        cate_score : float
            Current CATE estimate for the unit. Positive values suggest
            treatment is beneficial; negative values suggest control.
        step : int
            The current time step (used for decay schedules).

        Returns
        -------
        treatment : int
            The chosen treatment assignment (0 or 1).
        propensity : float
            The probability of the chosen treatment under this policy.
            Used for IPW correction in downstream estimators.
        """

    def update(self, reward: float) -> None:
        """Update policy state after observing a reward.

        Parameters
        ----------
        reward : float
            The observed outcome after applying the chosen treatment.
            Default implementation is a no-op; override for adaptive policies.
        """

    def reset(self) -> None:
        """Reset the policy to its initial (untrained) state."""
        self.__init__(**self._get_params())  # type: ignore[misc]
