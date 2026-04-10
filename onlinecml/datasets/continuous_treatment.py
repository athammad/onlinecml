"""Continuous-treatment (dose-response) data stream."""

import math
from typing import Iterator

import numpy as np


class ContinuousTreatmentStream:
    """Synthetic streaming dataset with a continuous treatment (dose-response).

    Generates observations where the treatment ``W`` is a continuous random
    variable (uniform or normal) rather than binary. The outcome follows a
    dose-response model ``Y = X @ beta + g(W) + noise``, where ``g`` is a
    known dose-response function.

    The fourth element yielded per observation is the **marginal causal
    effect** ``dE[Y]/dW`` at the observed dose ``W``:

    - ``'linear'``    → ``g(W) = true_effect * W``;          marginal = ``true_effect``
    - ``'quadratic'`` → ``g(W) = true_effect * W^2``;        marginal = ``2 * true_effect * W``
    - ``'threshold'`` → ``g(W) = true_effect * (W > 0.0)``; marginal = ``0`` (non-differentiable, yields ``g(W)``)

    Parameters
    ----------
    n : int
        Number of observations. Default 1000.
    n_features : int
        Number of covariates. Default 5.
    true_effect : float
        Scaling of the dose-response function. Default 1.0.
    dose_response : str
        One of ``'linear'``, ``'quadratic'``, ``'threshold'``. Default ``'linear'``.
    w_distribution : str
        Treatment distribution: ``'uniform'`` or ``'normal'``. Default ``'uniform'``.
    w_min : float
        Lower bound for uniform treatment draw. Default -1.0.
    w_max : float
        Upper bound for uniform treatment draw. Default 1.0.
    w_mean : float
        Mean for normal treatment draw. Default 0.0.
    w_std : float
        Standard deviation for normal treatment draw. Default 1.0.
    confounding_strength : float
        How much ``X`` shifts the expected treatment value. 0 = exogenous.
        Default 0.3.
    noise_std : float
        Outcome noise standard deviation. Default 1.0.
    seed : int or None
        Random seed for reproducibility.

    Examples
    --------
    >>> stream = ContinuousTreatmentStream(n=200, dose_response='linear', seed=0)
    >>> for x, w, y, marginal in stream:
    ...     assert isinstance(w, float)
    ...     assert isinstance(marginal, float)
    """

    _DOSE_RESPONSES = ("linear", "quadratic", "threshold")
    _DISTRIBUTIONS  = ("uniform", "normal")

    def __init__(
        self,
        n: int = 1000,
        n_features: int = 5,
        true_effect: float = 1.0,
        dose_response: str = "linear",
        w_distribution: str = "uniform",
        w_min: float = -1.0,
        w_max: float = 1.0,
        w_mean: float = 0.0,
        w_std: float = 1.0,
        confounding_strength: float = 0.3,
        noise_std: float = 1.0,
        seed: int | None = None,
    ) -> None:
        if dose_response not in self._DOSE_RESPONSES:
            raise ValueError(f"dose_response must be one of {self._DOSE_RESPONSES}.")
        if w_distribution not in self._DISTRIBUTIONS:
            raise ValueError(f"w_distribution must be one of {self._DISTRIBUTIONS}.")
        self.n = n
        self.n_features = n_features
        self.true_effect = true_effect
        self.dose_response = dose_response
        self.w_distribution = w_distribution
        self.w_min = w_min
        self.w_max = w_max
        self.w_mean = w_mean
        self.w_std = w_std
        self.confounding_strength = confounding_strength
        self.noise_std = noise_std
        self.seed = seed

    # ------------------------------------------------------------------

    def _g(self, w: float) -> float:
        """Dose-response function value at dose ``w``."""
        if self.dose_response == "linear":
            return self.true_effect * w
        if self.dose_response == "quadratic":
            return self.true_effect * w ** 2
        # threshold
        return self.true_effect * float(w > 0.0)

    def _marginal(self, w: float) -> float:
        """Marginal causal effect dE[Y]/dW at dose ``w``."""
        if self.dose_response == "linear":
            return self.true_effect
        if self.dose_response == "quadratic":
            return 2.0 * self.true_effect * w
        # threshold is non-differentiable; yield g(W) as the "effect"
        return self._g(w)

    # ------------------------------------------------------------------

    def __iter__(self) -> Iterator[tuple[dict, float, float, float]]:
        """Iterate over the stream, yielding one observation at a time.

        Yields
        ------
        x : dict
            Feature dictionary ``{'x0': ..., 'x1': ..., ...}``.
        w : float
            Continuous treatment dose.
        y : float
            Observed outcome.
        marginal_effect : float
            True marginal causal effect ``dE[Y]/dW`` at the observed dose.
        """
        rng = np.random.default_rng(self.seed)
        beta = rng.standard_normal(self.n_features)
        norm = math.sqrt(self.n_features)

        for _ in range(self.n):
            X_i = rng.standard_normal(self.n_features)
            x_effect = float(X_i @ beta) / norm  # covariate signal

            # Treatment draw (possibly confounded)
            if self.w_distribution == "uniform":
                shift = self.confounding_strength * x_effect * (self.w_max - self.w_min) * 0.5
                W_i = float(rng.uniform(self.w_min + shift, self.w_max + shift))
            else:
                shift = self.confounding_strength * x_effect * self.w_std
                W_i = float(rng.normal(self.w_mean + shift, self.w_std))

            eps = rng.normal(0.0, self.noise_std)
            Y_i = x_effect + self._g(W_i) + eps
            x_dict = {f"x{j}": float(X_i[j]) for j in range(self.n_features)}
            yield x_dict, W_i, Y_i, self._marginal(W_i)

    def __len__(self) -> int:
        """Total number of observations in this stream."""
        return self.n
