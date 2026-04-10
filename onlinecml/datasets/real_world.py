"""Real-world causal inference benchmark dataset loaders.

All loaders return River-compatible iterators that yield
``(x_dict, treatment, outcome, true_cate)`` tuples, where ``true_cate``
is ``None`` for datasets without known individual treatment effects.

Datasets are downloaded on first use and cached in ``~/.onlinecml/data/``.
"""

import csv
import io
import os
import urllib.request
from pathlib import Path
from typing import Iterator

_CACHE_DIR = Path.home() / ".onlinecml" / "data"


def _get_cache_path(filename: str) -> Path:
    """Return the cache path for a dataset file, creating the directory."""
    _CACHE_DIR.mkdir(parents=True, exist_ok=True)
    return _CACHE_DIR / filename


def _download(url: str, dest: Path) -> None:
    """Download a file from a URL to a local path.

    Parameters
    ----------
    url : str
        URL to download from.
    dest : Path
        Local file path to write to.
    """
    with urllib.request.urlopen(url) as response:  # noqa: S310
        content = response.read()
    dest.write_bytes(content)


# ---------------------------------------------------------------------------
# LaLonde (1986) — National Supported Work Program
# ---------------------------------------------------------------------------

_LALONDE_URL = (
    "https://raw.githubusercontent.com/microsoft/EconML/main/"
    "notebooks/LaLonde_data/lalonde_nsw_dw.csv"
)


def load_lalonde(shuffle: bool = False, seed: int | None = None) -> Iterator:
    """Load the LaLonde (1986) National Supported Work dataset.

    A classic benchmark for causal inference. The treatment is
    participation in a job training program (NSW). The outcome is
    real earnings in 1978.

    Parameters
    ----------
    shuffle : bool
        If True, shuffle the rows before iterating. Default False.
    seed : int or None
        Random seed for shuffling.

    Yields
    ------
    x : dict
        Covariates: age, education, black, hispanic, married, nodegree,
        re74 (earnings 1974), re75 (earnings 1975).
    treatment : int
        1 = participated in job training, 0 = control.
    outcome : float
        Real earnings in 1978.
    true_cate : None
        Individual CATE is not known for observational studies.

    Notes
    -----
    The dataset is downloaded on first use and cached in
    ``~/.onlinecml/data/``.

    References
    ----------
    LaLonde, R.J. (1986). Evaluating the econometric evaluations of
    training programs with experimental data. American Economic Review,
    76(4), 604-620.
    """
    cache = _get_cache_path("lalonde_nsw_dw.csv")
    if not cache.exists():
        try:
            _download(_LALONDE_URL, cache)
        except Exception as e:
            raise RuntimeError(
                f"Failed to download LaLonde dataset: {e}\n"
                f"Try manually placing the file at {cache}"
            ) from e

    rows = []
    with open(cache, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(row)

    if shuffle:
        import random
        rng = random.Random(seed)
        rng.shuffle(rows)

    for row in rows:
        x = {
            "age": float(row["age"]),
            "educ": float(row["educ"]),
            "black": float(row["black"]),
            "hisp": float(row.get("hisp", row.get("hispanic", 0))),
            "married": float(row["married"]),
            "nodegree": float(row["nodegree"]),
            "re74": float(row["re74"]),
            "re75": float(row["re75"]),
        }
        treatment = int(float(row["treat"]))
        outcome = float(row["re78"])
        yield x, treatment, outcome, None


# ---------------------------------------------------------------------------
# IHDP — Infant Health and Development Program
# ---------------------------------------------------------------------------

_IHDP_URL = (
    "https://raw.githubusercontent.com/AMLab-Amsterdam/CEVAE/master/"
    "datasets/IHDP/csv/ihdp_npci_1.csv"
)


def load_ihdp(split: int = 1, shuffle: bool = False, seed: int | None = None) -> Iterator:
    """Load the IHDP (Infant Health Development Program) dataset.

    Semi-synthetic benchmark for CATE estimation. True individual
    treatment effects are available for this dataset.

    Parameters
    ----------
    split : int
        Dataset split (1–10). Default 1.
    shuffle : bool
        If True, shuffle rows before iterating.
    seed : int or None
        Random seed for shuffling.

    Yields
    ------
    x : dict
        25 covariates (x1–x25).
    treatment : int
        1 = received intensive intervention, 0 = control.
    outcome : float
        Observed outcome (cognitive test score).
    true_cate : float
        True individual treatment effect for this unit.

    References
    ----------
    Hill, J.L. (2011). Bayesian nonparametric modeling for causal
    inference. Journal of Computational and Graphical Statistics,
    20(1), 217-240.
    """
    filename = f"ihdp_npci_{split}.csv"
    cache = _get_cache_path(filename)
    url = _IHDP_URL.replace("ihdp_npci_1.csv", filename)

    if not cache.exists():
        try:
            _download(url, cache)
        except Exception as e:
            raise RuntimeError(
                f"Failed to download IHDP split {split}: {e}\n"
                f"Try manually placing the file at {cache}"
            ) from e

    rows = []
    with open(cache, newline="") as f:
        reader = csv.reader(f)
        header = next(reader)
        for row in reader:
            rows.append(row)

    if shuffle:
        import random
        rng = random.Random(seed)
        rng.shuffle(rows)

    # IHDP CSV columns: treatment, y_factual, y_cfactual, mu0, mu1, x1..x25
    for row in rows:
        treatment = int(float(row[0]))
        y_factual = float(row[1])
        y_cf = float(row[2])
        mu0 = float(row[3])
        mu1 = float(row[4])
        true_cate = mu1 - mu0
        x = {f"x{i+1}": float(row[5 + i]) for i in range(25)}
        yield x, treatment, y_factual, true_cate


# ---------------------------------------------------------------------------
# News — High-dimensional text dataset
# ---------------------------------------------------------------------------

def load_news(n: int | None = None, seed: int | None = None) -> Iterator:
    """Load a synthetic high-dimensional approximation of the News dataset.

    Because the original News dataset requires proprietary preprocessing,
    this loader generates a synthetic version with matching statistical
    properties: ~3000 features (sparse), binary treatment, continuous
    outcome, and heterogeneous treatment effects.

    Parameters
    ----------
    n : int or None
        Number of observations. Default 5000.
    seed : int or None
        Random seed.

    Yields
    ------
    x : dict
        Sparse feature dictionary (100 non-zero features out of 3000).
    treatment : int
        Binary treatment indicator.
    outcome : float
        Continuous outcome.
    true_cate : float
        True individual CATE.

    Notes
    -----
    This is a synthetic proxy. For the original dataset processing code,
    see Johansson et al. (2016).
    """
    import math
    import numpy as np

    n = n or 5000
    rng = np.random.default_rng(seed)
    n_features = 3000
    n_nonzero = 100

    for _ in range(n):
        # Sparse feature vector: 100 non-zero out of 3000
        idx = rng.choice(n_features, size=n_nonzero, replace=False)
        vals = rng.exponential(1.0, size=n_nonzero)
        x = {f"f{i}": float(v) for i, v in zip(idx, vals)}

        # Confounded treatment
        score = float(sum(vals[:10])) - 5.0
        p = 1.0 / (1.0 + math.exp(-0.3 * score))
        treatment = int(rng.binomial(1, p))

        # Heterogeneous CATE: depends on first non-zero feature
        tau = float(vals[0]) * 0.5
        outcome = score * 0.1 + treatment * tau + rng.normal(0, 0.5)
        yield x, treatment, float(outcome), tau


# ---------------------------------------------------------------------------
# Twins — Twin births dataset
# ---------------------------------------------------------------------------

_TWINS_URL = (
    "https://raw.githubusercontent.com/AMLab-Amsterdam/CEVAE/master/"
    "datasets/twins.csv"
)


def load_twins(shuffle: bool = False, seed: int | None = None) -> Iterator:
    """Load the Twin births dataset.

    Each observation is a pair of twins. The treatment is being the
    heavier twin (weight ≥ 2000g). The outcome is 1-year mortality.
    True individual treatment effects are not available.

    Parameters
    ----------
    shuffle : bool
        If True, shuffle rows before iterating.
    seed : int or None
        Random seed for shuffling.

    Yields
    ------
    x : dict
        30 covariates (birth characteristics, demographics).
    treatment : int
        1 = heavier twin, 0 = lighter twin.
    outcome : float
        1-year mortality (0 = alive, 1 = deceased).
    true_cate : None
        Not available for real-world twin data.

    References
    ----------
    Louizos, C. et al. (2017). Causal effect inference with deep
    latent-variable models. NeurIPS 2017.
    """
    cache = _get_cache_path("twins.csv")

    if not cache.exists():
        try:
            _download(_TWINS_URL, cache)
        except Exception as e:
            raise RuntimeError(
                f"Failed to download Twins dataset: {e}\n"
                f"Try manually placing the file at {cache}"
            ) from e

    rows = []
    with open(cache, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(row)

    if shuffle:
        import random
        rng = random.Random(seed)
        rng.shuffle(rows)

    for row in rows:
        try:
            treatment = int(float(row.get("t", row.get("treat", 0))))
            outcome = float(row.get("y", row.get("mort", 0)))
            x = {k: float(v) for k, v in row.items() if k not in ("t", "treat", "y", "mort")}
            yield x, treatment, outcome, None
        except (ValueError, KeyError):
            continue
