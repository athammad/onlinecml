"""Synthetic and real-world streaming datasets for causal inference."""

from onlinecml.datasets.continuous_treatment import ContinuousTreatmentStream
from onlinecml.datasets.drifting_causal import DriftingCausalStream
from onlinecml.datasets.heterogeneous_causal import HeterogeneousCausalStream
from onlinecml.datasets.linear_causal import LinearCausalStream
from onlinecml.datasets.real_world import load_ihdp, load_lalonde, load_news, load_twins
from onlinecml.datasets.unbalanced_causal import UnbalancedCausalStream

__all__ = [
    "LinearCausalStream",
    "HeterogeneousCausalStream",
    "DriftingCausalStream",
    "UnbalancedCausalStream",
    "ContinuousTreatmentStream",
    "load_lalonde",
    "load_ihdp",
    "load_news",
    "load_twins",
]
