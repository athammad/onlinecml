"""Online matching methods for causal inference."""

from onlinecml.matching.caliper_matching import OnlineCaliperMatching
from onlinecml.matching.distance import (
    combined_distance,
    euclidean_distance,
    mahalanobis_distance,
    ps_distance,
)
from onlinecml.matching.kernel_matching import OnlineKernelMatching
from onlinecml.matching.online_matching import OnlineMatching

__all__ = [
    "OnlineMatching",
    "OnlineCaliperMatching",
    "OnlineKernelMatching",
    "euclidean_distance",
    "mahalanobis_distance",
    "ps_distance",
    "combined_distance",
]
