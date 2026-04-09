"""OnlineCML — Online Causal Machine Learning in Python.

Causal inference for the real world — one observation at a time.
"""

__version__ = "0.1.0"

from onlinecml import base, datasets, diagnostics, metalearners, policy, propensity, reweighting

__all__ = [
    "__version__",
    "base",
    "datasets",
    "diagnostics",
    "metalearners",
    "policy",
    "propensity",
    "reweighting",
]
