"""Causal data stream with extreme propensity scores for positivity stress tests."""

import math
from typing import Iterator

import numpy as np


class UnbalancedCausalStream:
    """Synthetic streaming dataset with extreme treatment probabilities.

    Generates observations where most units are assigned to one arm, creating
    near-positivity violations. Designed to stress-test overlap diagnostics
    and the stability of IPW-based estimators under extreme propensity scores.

    Parameters
    ----------
    n : int
        Number of observations to generate. Default 1000.
    n_features : int
        Number of covariates. Default 5.
    true_ate : float
        True Average Treatment Effect. Default 2.0.
    treatment_rate : float
        Target marginal probability of treatment. Values close to 0 or 1
        create the most severe positivity violations. Default 0.1.
    confounding_strength : float
        Controls how strongly X predicts treatment on top of the marginal
        imbalance. 0.0 = only marginal imbalance, 1.0 = strong additional
        confounding. Default 1.5.
    noise_std : float
        Standard deviation of outcome noise. Default 1.0.
    seed : int or None
        Random seed for reproducibility.

    Notes
    -----
    The logit for treatment assignment is:

    .. math::

        \\text{logit}(P(W=1|X)) = \\text{logit}(\\text{treatment\\_rate})
            + \\text{confounding\\_strength} \\cdot X\\beta / \\sqrt{p}

    This ensures that the marginal treatment rate is approximately
    ``treatment_rate`` while adding covariate-driven confounding.

    Examples
    --------
    >>> stream = UnbalancedCausalStream(n=500, treatment_rate=0.1, seed=42)
    >>> rates = [w for _, w, _, _ in stream]
    >>> abs(sum(rates) / len(rates) - 0.1) < 0.1  # roughly 10% treated
    True
    """

    def __init__(
        self,
        n: int = 1000,
        n_features: int = 5,
        true_ate: float = 2.0,
        treatment_rate: float = 0.1,
        confounding_strength: float = 1.5,
        noise_std: float = 1.0,
        seed: int | None = None,
    ) -> None:
        if not 0.0 < treatment_rate < 1.0:
            raise ValueError("treatment_rate must be strictly between 0 and 1.")
        self.n = n
        self.n_features = n_features
        self.true_ate = true_ate
        self.treatment_rate = treatment_rate
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
            True CATE for this unit (constant, equals ``true_ate``).
        """
        rng = np.random.default_rng(self.seed)
        beta = rng.standard_normal(self.n_features)
        norm = math.sqrt(self.n_features)
        # Intercept that sets marginal treatment rate
        intercept = math.log(self.treatment_rate / (1.0 - self.treatment_rate))

        for _ in range(self.n):
            X_i = rng.standard_normal(self.n_features)
            logit_p = intercept + self.confounding_strength * (X_i @ beta) / norm
            p = 1.0 / (1.0 + math.exp(-logit_p))
            W_i = int(rng.binomial(1, p))
            eps = rng.normal(0.0, self.noise_std)
            Y_i = float(X_i @ beta) + W_i * self.true_ate + eps
            x_dict = {f"x{j}": float(X_i[j]) for j in range(self.n_features)}
            yield x_dict, W_i, Y_i, float(self.true_ate)

    def __len__(self) -> int:
        """Total number of observations in this stream."""
        return self.n
