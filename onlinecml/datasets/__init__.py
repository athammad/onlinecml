"""Synthetic streaming datasets for causal inference benchmarks."""

from onlinecml.datasets.heterogeneous_causal import HeterogeneousCausalStream
from onlinecml.datasets.linear_causal import LinearCausalStream

__all__ = [
    "LinearCausalStream",
    "HeterogeneousCausalStream",
]
