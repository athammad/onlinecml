"""Distance functions for online matching methods."""

import math
from collections.abc import Callable


def euclidean_distance(x: dict, y: dict) -> float:
    """Compute the Euclidean distance between two feature dicts.

    Only features present in both dicts are used. Missing features are
    treated as 0.

    Parameters
    ----------
    x : dict
        First feature dictionary.
    y : dict
        Second feature dictionary.

    Returns
    -------
    float
        Euclidean distance.
    """
    keys = set(x) | set(y)
    return math.sqrt(sum((x.get(k, 0.0) - y.get(k, 0.0)) ** 2 for k in keys))


def ps_distance(p_x: float, p_y: float) -> float:
    """Compute the absolute difference in propensity scores.

    Parameters
    ----------
    p_x : float
        Propensity score for unit x.
    p_y : float
        Propensity score for unit y.

    Returns
    -------
    float
        Absolute propensity score distance ``|p_x - p_y|``.
    """
    return abs(p_x - p_y)


def mahalanobis_distance(x: dict, y: dict, cov_inv: dict | None = None) -> float:
    """Compute the Mahalanobis distance between two feature dicts.

    If no inverse covariance matrix is provided, falls back to scaled
    Euclidean distance (divides each dimension by its variance proxy = 1).

    Parameters
    ----------
    x : dict
        First feature dictionary.
    y : dict
        Second feature dictionary.
    cov_inv : dict or None
        Inverse covariance matrix as a nested dict
        ``{feature: {feature: value}}``. If None, uses identity matrix
        (equivalent to Euclidean distance).

    Returns
    -------
    float
        Mahalanobis distance.
    """
    keys = sorted(set(x) | set(y))
    diff = [x.get(k, 0.0) - y.get(k, 0.0) for k in keys]
    if cov_inv is None:
        return math.sqrt(sum(d * d for d in diff))
    # d^T Sigma^{-1} d
    result = 0.0
    for i, ki in enumerate(keys):
        row = cov_inv.get(ki, {})
        for j, kj in enumerate(keys):
            result += diff[i] * row.get(kj, 0.0) * diff[j]
    return math.sqrt(max(0.0, result))


def combined_distance(
    x: dict,
    y: dict,
    p_x: float,
    p_y: float,
    ps_weight: float = 0.5,
) -> float:
    """Compute a weighted combination of Euclidean and PS distance.

    Parameters
    ----------
    x : dict
        First feature dictionary.
    y : dict
        Second feature dictionary.
    p_x : float
        Propensity score for unit x.
    p_y : float
        Propensity score for unit y.
    ps_weight : float
        Weight on the PS distance component (0 to 1). Default 0.5.

    Returns
    -------
    float
        Combined distance.
    """
    return (1.0 - ps_weight) * euclidean_distance(x, y) + ps_weight * ps_distance(p_x, p_y)


DistanceFn = Callable[[dict, dict], float]
