"""Synthetic and real-world streaming datasets for causal inference."""

from onlinecml.datasets.heterogeneous_causal import HeterogeneousCausalStream
from onlinecml.datasets.linear_causal import LinearCausalStream
from onlinecml.datasets.real_world import load_ihdp, load_lalonde, load_news, load_twins

__all__ = [
    "LinearCausalStream",
    "HeterogeneousCausalStream",
    "load_lalonde",
    "load_ihdp",
    "load_news",
    "load_twins",
]
