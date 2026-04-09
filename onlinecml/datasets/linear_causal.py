"""Linear causal data-generating process for streaming benchmarks."""

import math
from typing import Iterator

import numpy as np


class LinearCausalStream:
    """Synthetic streaming dataset with a linear DGP and constant ATE.

    Generates a stream of (features, treatment, outcome, true_cate) tuples
    from a linear data-generating process with confounding. The true CATE
    is constant (equal to ``true_ate``) for all units.

    Parameters
    ----------
    n : int
        Number of observations to generate. Default 1000.
    n_features : int
        Number of covariates. Default 5.
    true_ate : float
        The true constant Average Treatment Effect. Default 2.0.
    confounding_strength : float
        Controls how strongly covariates predict treatment assignment.
        0.0 = no confounding (RCT), 1.0 = strong confounding. Default 0.5.
    noise_std : float
        Standard deviation of the outcome noise. Default 1.0.
    seed : int or None
        Random seed for reproducibility. If None, results are random.

    Notes
    -----
    Data-generating process:

    - ``beta ~ N(0, I_p)`` — fixed coefficient vector per stream iteration
    - ``X_i ~ N(0, I_p)`` — independent covariates
    - ``logit(P(W=1|X)) = confounding_strength * X @ beta / sqrt(p)``
    - ``W_i ~ Bernoulli(sigmoid(logit_p))``
    - ``Y_i = X @ beta + W_i * true_ate + eps``, ``eps ~ N(0, noise_std^2)``

    The coefficient vector ``beta`` is re-sampled once per ``__iter__``
    call. Re-iterating with the same seed gives the same stream.

    Because CATE is constant, ``true_cate == true_ate`` for all units.

    Examples
    --------
    >>> stream = LinearCausalStream(n=500, n_features=3, true_ate=2.0, seed=42)
    >>> for x, w, y, tau in stream:
    ...     pass  # process one observation at a time
    >>> len(stream)
    500
    """

    def __init__(
        self,
        n: int = 1000,
        n_features: int = 5,
        true_ate: float = 2.0,
        confounding_strength: float = 0.5,
        noise_std: float = 1.0,
        seed: int | None = None,
    ) -> None:
        self.n = n
        self.n_features = n_features
        self.true_ate = true_ate
        self.confounding_strength = confounding_strength
        self.noise_std = noise_std
        self.seed = seed

    def __iter__(self) -> Iterator[tuple[dict, int, float, float]]:
        """Iterate over the stream, yielding one observation at a time.

        Yields
        ------
        x : dict
            Feature dictionary with keys ``'x0'``, ``'x1'``, ...
        treatment : int
            Treatment indicator (0 or 1).
        outcome : float
            Observed outcome.
        true_cate : float
            The true CATE for this unit (equals ``true_ate`` for all units).
        """
        rng = np.random.default_rng(self.seed)
        beta = rng.standard_normal(self.n_features)
        norm = math.sqrt(self.n_features)

        for _ in range(self.n):
            X_i = rng.standard_normal(self.n_features)
            logit_p = self.confounding_strength * (X_i @ beta) / norm
            p = 1.0 / (1.0 + math.exp(-logit_p))
            W_i = int(rng.binomial(1, p))
            eps = rng.normal(0.0, self.noise_std)
            Y_i = float(X_i @ beta) + W_i * self.true_ate + eps
            x_dict = {f"x{j}": float(X_i[j]) for j in range(self.n_features)}
            yield x_dict, W_i, Y_i, float(self.true_ate)

    def __len__(self) -> int:
        """Total number of observations in this stream."""
        return self.n
