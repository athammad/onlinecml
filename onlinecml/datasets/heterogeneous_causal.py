"""Heterogeneous CATE causal data-generating process for streaming benchmarks."""

import math
from typing import Iterator

import numpy as np


class HeterogeneousCausalStream:
    """Synthetic streaming dataset with heterogeneous treatment effects.

    Generates a stream of (features, treatment, outcome, true_cate) tuples
    where the CATE varies across units according to a specified functional
    form. Designed for benchmarking CATE estimators.

    Parameters
    ----------
    n : int
        Number of observations to generate. Default 1000.
    n_features : int
        Number of covariates (at least 2 required for nonlinear DGP).
        Default 5.
    true_ate : float
        Base treatment effect. The marginal average ``E[tau(X)] = true_ate``
        for all heterogeneity types when covariates are standard normal.
        Default 2.0.
    heterogeneity : str
        Functional form of the CATE:

        - ``'linear'`` — ``tau(X) = true_ate * (1 + 0.5 * X[0])``
        - ``'nonlinear'`` — ``tau(X) = true_ate + X[0] + sin(X[1]) * 0.5``
        - ``'step'`` — ``tau(X) = 2 * true_ate if X[0] > 0 else 0.5 * true_ate``

        Default ``'nonlinear'``.
    confounding_strength : float
        Controls how strongly covariates predict treatment assignment.
        Default 0.5.
    noise_std : float
        Standard deviation of the outcome noise. Default 1.0.
    seed : int or None
        Random seed for reproducibility.

    Notes
    -----
    For standard-normal covariates, ``E[tau(X)] = true_ate`` for all
    three heterogeneity types:

    - ``'linear'``: ``E[true_ate * (1 + 0.5*X0)] = true_ate * (1 + 0) = true_ate``
    - ``'nonlinear'``: ``E[true_ate + X0 + sin(X1)*0.5] = true_ate + 0 + 0 = true_ate``
    - ``'step'``: ``E[2*tau * 1{X0>0} + 0.5*tau * 1{X0<=0}] = true_ate * 1.25``
      (step DGP does NOT have population ATE exactly equal to ``true_ate``)

    Use ``population_ate()`` to get the theoretical marginal ATE.

    Examples
    --------
    >>> stream = HeterogeneousCausalStream(n=500, true_ate=2.0, seed=42)
    >>> x, w, y, tau = next(iter(stream))
    >>> isinstance(tau, float)
    True
    """

    def __init__(
        self,
        n: int = 1000,
        n_features: int = 5,
        true_ate: float = 2.0,
        heterogeneity: str = "nonlinear",
        confounding_strength: float = 0.5,
        noise_std: float = 1.0,
        seed: int | None = None,
    ) -> None:
        if heterogeneity not in ("linear", "nonlinear", "step"):
            raise ValueError(
                f"heterogeneity must be 'linear', 'nonlinear', or 'step'; got {heterogeneity!r}"
            )
        if n_features < 2 and heterogeneity == "nonlinear":
            raise ValueError("nonlinear heterogeneity requires at least 2 features")
        self.n = n
        self.n_features = n_features
        self.true_ate = true_ate
        self.heterogeneity = heterogeneity
        self.confounding_strength = confounding_strength
        self.noise_std = noise_std
        self.seed = seed

    def _tau(self, X_i: np.ndarray) -> float:
        """Compute the individual CATE for a feature vector.

        Parameters
        ----------
        X_i : np.ndarray
            Feature vector of shape ``(n_features,)``.

        Returns
        -------
        float
            Individual CATE for this unit.
        """
        if self.heterogeneity == "linear":
            return float(self.true_ate * (1.0 + 0.5 * X_i[0]))
        elif self.heterogeneity == "nonlinear":
            return float(self.true_ate + X_i[0] + math.sin(X_i[1]) * 0.5)
        else:  # 'step'
            return float(2.0 * self.true_ate if X_i[0] > 0 else 0.5 * self.true_ate)

    def population_ate(self) -> float:
        """Return the theoretical marginal ATE, ``E[tau(X)]``.

        For ``'linear'`` and ``'nonlinear'`` DGPs with standard-normal
        covariates, this equals ``true_ate``. For ``'step'``, it equals
        ``1.25 * true_ate``.

        Returns
        -------
        float
            Theoretical population average treatment effect.
        """
        if self.heterogeneity in ("linear", "nonlinear"):
            return float(self.true_ate)
        else:  # 'step': E[tau] = 2*tau*0.5 + 0.5*tau*0.5 = 1.25*tau
            return float(1.25 * self.true_ate)

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
            The true individual CATE for this unit.
        """
        rng = np.random.default_rng(self.seed)
        beta = rng.standard_normal(self.n_features)
        norm = math.sqrt(self.n_features)

        for _ in range(self.n):
            X_i = rng.standard_normal(self.n_features)
            logit_p = self.confounding_strength * (X_i @ beta) / norm
            p = 1.0 / (1.0 + math.exp(-logit_p))
            W_i = int(rng.binomial(1, p))
            tau_i = self._tau(X_i)
            eps = rng.normal(0.0, self.noise_std)
            Y_i = float(X_i @ beta) + W_i * tau_i + eps
            x_dict = {f"x{j}": float(X_i[j]) for j in range(self.n_features)}
            yield x_dict, W_i, Y_i, tau_i

    def __len__(self) -> int:
        """Total number of observations in this stream."""
        return self.n
