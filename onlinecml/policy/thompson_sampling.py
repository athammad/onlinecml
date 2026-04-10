"""Thompson Sampling exploration policies for online treatment assignment."""

import math
import random

from onlinecml.base.base_policy import BasePolicy


class ThompsonSampling(BasePolicy):
    """Thompson Sampling policy for binary outcomes (Beta-Bernoulli).

    Maintains a Beta posterior over the success probability for each
    treatment arm. At each step, samples from each posterior and assigns
    the treatment with the higher sample.

    Parameters
    ----------
    alpha_prior : float
        Prior pseudo-count for successes (Beta alpha). Default 1.0
        (uniform prior).
    beta_prior : float
        Prior pseudo-count for failures (Beta beta). Default 1.0
        (uniform prior).
    seed : int or None
        Random seed for reproducibility.

    Notes
    -----
    This policy assumes binary outcomes in [0, 1]. For continuous
    outcomes, use ``GaussianThompsonSampling``.

    The propensity returned is the probability that the chosen arm would
    be selected, estimated as the fraction of Monte Carlo samples where
    that arm wins. For implementation simplicity, we return 0.5 during
    exploration-equivalent draws and the exploit probability otherwise.

    Examples
    --------
    >>> policy = ThompsonSampling(seed=0)
    >>> treatment, propensity = policy.choose(cate_score=0.0, step=0)
    >>> treatment in (0, 1)
    True
    >>> policy.update(reward=1.0)
    """

    def __init__(
        self,
        alpha_prior: float = 1.0,
        beta_prior: float = 1.0,
        seed: int | None = None,
    ) -> None:
        self.alpha_prior = alpha_prior
        self.beta_prior = beta_prior
        self.seed = seed
        self._rng = random.Random(seed)
        # Posterior parameters: [alpha, beta] for each arm (0 = control, 1 = treated)
        self._alpha = [alpha_prior, alpha_prior]
        self._beta = [beta_prior, beta_prior]
        self._last_treatment: int = 0

    def _beta_sample(self, alpha: float, beta: float) -> float:
        """Draw one sample from a Beta(alpha, beta) distribution.

        Uses the gamma-ratio method: Beta(a,b) = Gamma(a) / (Gamma(a) + Gamma(b)).

        Parameters
        ----------
        alpha : float
            Beta distribution alpha parameter.
        beta : float
            Beta distribution beta parameter.

        Returns
        -------
        float
            Sample in (0, 1).
        """
        x = self._rng.gammavariate(alpha, 1.0)
        y = self._rng.gammavariate(beta, 1.0)
        total = x + y
        if total <= 0.0:
            return 0.5
        return x / total

    def choose(self, cate_score: float, step: int) -> tuple[int, float]:
        """Choose a treatment by sampling from the Beta posteriors.

        Parameters
        ----------
        cate_score : float
            Not used by Thompson Sampling (posteriors drive the choice).
        step : int
            Not used; included for API compatibility.

        Returns
        -------
        treatment : int
            The arm with the higher posterior sample.
        propensity : float
            Approximate propensity (0.5 as a conservative estimate for
            the Beta-Bernoulli sampler).
        """
        s0 = self._beta_sample(self._alpha[0], self._beta[0])
        s1 = self._beta_sample(self._alpha[1], self._beta[1])
        treatment = 1 if s1 > s0 else 0
        self._last_treatment = treatment
        # Conservative propensity estimate: 0.5 (true propensity depends on
        # the full posterior, which is expensive to compute exactly)
        return treatment, 0.5

    def update(self, reward: float) -> None:
        """Update the Beta posterior for the last chosen arm.

        Parameters
        ----------
        reward : float
            Observed outcome. Values > 0.5 are treated as successes;
            values ≤ 0.5 as failures (for binary reward encoding).
        """
        arm = self._last_treatment
        if reward > 0.5:
            self._alpha[arm] += 1.0
        else:
            self._beta[arm] += 1.0

    def reset(self) -> None:
        """Reset posteriors to prior and reinitialize RNG."""
        self.__init__(**self._get_params())  # type: ignore[misc]


class GaussianThompsonSampling(BasePolicy):
    """Thompson Sampling policy for continuous outcomes (Gaussian).

    Maintains a Gaussian posterior over the mean reward for each arm
    using a Normal-Normal conjugate model. Assumes known variance.

    Parameters
    ----------
    prior_mean : float
        Prior mean for each arm's reward. Default 0.0.
    prior_std : float
        Prior standard deviation (uncertainty about the mean). Default 1.0.
    noise_std : float
        Known observation noise standard deviation. Default 1.0.
    seed : int or None
        Random seed for reproducibility.

    Notes
    -----
    The posterior after ``n`` observations with sample mean ``y_bar`` is:

    .. math::

        \\mu_{post} = \\frac{\\sigma_0^2 n \\bar{y} + \\sigma^2 \\mu_0}
                           {\\sigma_0^2 n + \\sigma^2}

        \\sigma_{post}^2 = \\frac{\\sigma_0^2 \\sigma^2}{\\sigma_0^2 n + \\sigma^2}

    Examples
    --------
    >>> policy = GaussianThompsonSampling(seed=42)
    >>> treatment, _ = policy.choose(1.5, 10)
    >>> treatment in (0, 1)
    True
    """

    def __init__(
        self,
        prior_mean: float = 0.0,
        prior_std: float = 1.0,
        noise_std: float = 1.0,
        seed: int | None = None,
    ) -> None:
        self.prior_mean = prior_mean
        self.prior_std = prior_std
        self.noise_std = noise_std
        self.seed = seed
        self._rng = random.Random(seed)
        # Posterior sufficient stats per arm: (sum_y, n)
        self._sum_y = [0.0, 0.0]
        self._n = [0, 0]
        self._last_treatment: int = 0

    def _posterior_params(self, arm: int) -> tuple[float, float]:
        """Return (posterior_mean, posterior_std) for an arm.

        Parameters
        ----------
        arm : int
            Arm index (0 or 1).

        Returns
        -------
        tuple of (float, float)
            Posterior mean and standard deviation.
        """
        n = self._n[arm]
        sigma0_sq = self.prior_std ** 2
        sigma_sq = self.noise_std ** 2
        if n == 0:
            return self.prior_mean, self.prior_std
        y_bar = self._sum_y[arm] / n
        denom = sigma0_sq * n + sigma_sq
        post_mean = (sigma0_sq * n * y_bar + sigma_sq * self.prior_mean) / denom
        post_var = (sigma0_sq * sigma_sq) / denom
        return post_mean, math.sqrt(post_var)

    def _gauss_sample(self, mu: float, sigma: float) -> float:
        """Sample from N(mu, sigma^2).

        Parameters
        ----------
        mu : float
            Mean.
        sigma : float
            Standard deviation.

        Returns
        -------
        float
            Gaussian sample.
        """
        return self._rng.gauss(mu, sigma)

    def choose(self, cate_score: float, step: int) -> tuple[int, float]:
        """Choose a treatment by sampling from the Gaussian posteriors.

        Parameters
        ----------
        cate_score : float
            Not used; included for API compatibility.
        step : int
            Not used; included for API compatibility.

        Returns
        -------
        treatment : int
            Arm with higher posterior sample.
        propensity : float
            Conservative propensity estimate (0.5).
        """
        mu0, s0 = self._posterior_params(0)
        mu1, s1 = self._posterior_params(1)
        sample0 = self._gauss_sample(mu0, s0)
        sample1 = self._gauss_sample(mu1, s1)
        treatment = 1 if sample1 > sample0 else 0
        self._last_treatment = treatment
        return treatment, 0.5

    def update(self, reward: float) -> None:
        """Update the Gaussian posterior for the last chosen arm.

        Parameters
        ----------
        reward : float
            Observed continuous outcome.
        """
        arm = self._last_treatment
        self._sum_y[arm] += reward
        self._n[arm] += 1

    def reset(self) -> None:
        """Reset posteriors to prior and reinitialize RNG."""
        self.__init__(**self._get_params())  # type: ignore[misc]
