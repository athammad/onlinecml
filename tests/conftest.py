"""Shared fixtures for the OnlineCML test suite."""

import pytest
from river.linear_model import LinearRegression, LogisticRegression

from onlinecml.datasets import LinearCausalStream, HeterogeneousCausalStream


@pytest.fixture
def small_linear_stream():
    """A small linear causal stream for fast tests."""
    return LinearCausalStream(n=200, n_features=3, true_ate=2.0, seed=0)


@pytest.fixture
def small_het_stream():
    """A small heterogeneous CATE stream for fast tests."""
    return HeterogeneousCausalStream(n=200, n_features=3, true_ate=2.0, seed=0)


@pytest.fixture
def simple_regressor():
    """A fresh River linear regressor."""
    return LinearRegression()


@pytest.fixture
def simple_classifier():
    """A fresh River logistic regressor."""
    return LogisticRegression()
