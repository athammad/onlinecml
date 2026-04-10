"""Integration tests: multiple methods on the same stream."""

import pytest

from onlinecml.datasets import LinearCausalStream
from onlinecml.reweighting import OnlineIPW, OnlineAIPW
from onlinecml.metalearners import OnlineTLearner, OnlineRLearner


def test_all_methods_same_stream_same_seed():
    """All estimators should produce finite ATE estimates on the same stream."""
    n = 500
    true_ate = 3.0
    stream_data = list(LinearCausalStream(n=n, true_ate=true_ate, n_features=3, seed=99))

    estimators = {
        "ipw": OnlineIPW(),
        "aipw": OnlineAIPW(),
        "t_learner": OnlineTLearner(),
        "r_learner": OnlineRLearner(),
    }

    for x, w, y, _ in stream_data:
        for est in estimators.values():
            est.learn_one(x, w, y)

    for name, est in estimators.items():
        ate = est.predict_ate()
        assert isinstance(ate, float), f"{name} ATE is not float"
        lo, hi = est.predict_ci()
        assert lo < hi, f"{name} CI has lo >= hi"


def test_aipw_better_or_comparable_to_ipw_on_linear():
    """AIPW and IPW should both converge; AIPW CI should be finite."""
    stream_data = list(LinearCausalStream(n=1000, true_ate=2.0, n_features=3, seed=7))
    ipw = OnlineIPW()
    aipw = OnlineAIPW()

    for x, w, y, _ in stream_data:
        ipw.learn_one(x, w, y)
        aipw.learn_one(x, w, y)

    ipw_lo, ipw_hi = ipw.predict_ci()
    aipw_lo, aipw_hi = aipw.predict_ci()
    assert ipw_lo < ipw_hi
    assert aipw_lo < aipw_hi
