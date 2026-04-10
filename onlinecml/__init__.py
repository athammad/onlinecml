"""OnlineCML — Online Causal Machine Learning in Python.

Causal inference for the real world — one observation at a time.
"""

__version__ = "0.3.0"

from onlinecml import (
    base,
    datasets,
    diagnostics,
    evaluation,
    matching,
    metalearners,
    policy,
    propensity,
    reweighting,
)

__all__ = [
    "__version__",
    "base",
    "datasets",
    "diagnostics",
    "evaluation",
    "matching",
    "metalearners",
    "policy",
    "propensity",
    "reweighting",
]
