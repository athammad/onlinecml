"""Causal data stream with a known ATE shift at a changepoint."""

import math
from typing import Iterator

import numpy as np


class DriftingCausalStream:
    """Synthetic streaming dataset where the ATE shifts at a known changepoint.

    Generates observations from a linear DGP with confounding. The true ATE
    is ``true_ate`` for the first ``changepoint`` observations, then shifts
    to ``shifted_ate`` for the remainder. The changepoint is announced via the
    ``changepoint`` attribute so that downstream drift monitors can be evaluated
    against the known ground truth.

    Parameters
    ----------
    n : int
        Total number of observations to generate. Default 2000.
    n_features : int
        Number of covariates. Default 5.
    true_ate : float
        ATE before the changepoint. Default 2.0.
    shifted_ate : float
        ATE after the changepoint. Default -1.0.
    changepoint : int or None
        Index (0-based) at which the ATE shifts. Defaults to ``n // 2``.
    confounding_strength : float
        Strength of confounding. 0.0 = RCT, 1.0 = strong. Default 0.5.
    noise_std : float
        Standard deviation of outcome noise. Default 1.0.
    seed : int or None
        Random seed for reproducibility.

    Notes
    -----
    The true CATE returned per observation reflects the current ATE segment:
    ``true_ate`` before the changepoint and ``shifted_ate`` after.

    Examples
    --------
    >>> stream = DriftingCausalStream(n=1000, true_ate=2.0, shifted_ate=-1.0, seed=0)
    >>> for x, w, y, tau in stream:
    ...     pass  # tau shifts from 2.0 to -1.0 at step 500
    """

    def __init__(
        self,
        n: int = 2000,
        n_features: int = 5,
        true_ate: float = 2.0,
        shifted_ate: float = -1.0,
        changepoint: int | None = None,
        confounding_strength: float = 0.5,
        noise_std: float = 1.0,
        seed: int | None = None,
    ) -> None:
        self.n = n
        self.n_features = n_features
        self.true_ate = true_ate
        self.shifted_ate = shifted_ate
        self.changepoint = changepoint if changepoint is not None else n // 2
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
            True CATE for this unit. Equals ``true_ate`` before the changepoint
            and ``shifted_ate`` from the changepoint onward.
        """
        rng = np.random.default_rng(self.seed)
        beta = rng.standard_normal(self.n_features)
        norm = math.sqrt(self.n_features)

        for i in range(self.n):
            ate = self.true_ate if i < self.changepoint else self.shifted_ate
            X_i = rng.standard_normal(self.n_features)
            logit_p = self.confounding_strength * (X_i @ beta) / norm
            p = 1.0 / (1.0 + math.exp(-logit_p))
            W_i = int(rng.binomial(1, p))
            eps = rng.normal(0.0, self.noise_std)
            Y_i = float(X_i @ beta) + W_i * ate + eps
            x_dict = {f"x{j}": float(X_i[j]) for j in range(self.n_features)}
            yield x_dict, W_i, Y_i, float(ate)

    def __len__(self) -> int:
        """Total number of observations in this stream."""
        return self.n
