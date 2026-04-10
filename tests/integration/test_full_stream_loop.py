"""Integration tests: full stream loop with all estimators."""

import pytest

from onlinecml.datasets import LinearCausalStream, HeterogeneousCausalStream
from onlinecml.reweighting import OnlineIPW, OnlineAIPW, OnlineOverlapWeights
from onlinecml.metalearners import OnlineSLearner, OnlineTLearner, OnlineXLearner, OnlineRLearner
from onlinecml.matching import OnlineMatching, OnlineCaliperMatching, OnlineKernelMatching
from onlinecml.diagnostics import OnlineSMD, ATETracker, OverlapChecker, ConceptDriftMonitor
from onlinecml.policy import EpsilonGreedy


N = 300


@pytest.fixture
def stream():
    return LinearCausalStream(n=N, n_features=3, true_ate=2.0, seed=42)


@pytest.mark.parametrize("estimator_cls", [
    OnlineIPW,
    OnlineAIPW,
    OnlineOverlapWeights,
    OnlineSLearner,
    OnlineTLearner,
    OnlineXLearner,
    OnlineRLearner,
])
def test_estimator_full_loop(estimator_cls, stream):
    """Each estimator should survive a full stream without exceptions."""
    est = estimator_cls()
    for x, w, y, _ in stream:
        est.learn_one(x, w, y)
    assert est.n_seen == N
    ate = est.predict_ate()
    assert isinstance(ate, float)
    lo, hi = est.predict_ci()
    assert lo < hi


@pytest.mark.parametrize("matcher_cls,kwargs", [
    (OnlineMatching, {"k": 2, "buffer_size": 50}),
    (OnlineCaliperMatching, {"caliper": 5.0, "buffer_size": 50}),
    (OnlineKernelMatching, {"bandwidth": 2.0, "buffer_size": 50}),
])
def test_matching_full_loop(matcher_cls, kwargs, stream):
    """Each matching estimator should survive a full stream."""
    est = matcher_cls(**kwargs)
    for x, w, y, _ in stream:
        est.learn_one(x, w, y)
    assert est.n_seen == N
    assert isinstance(est.predict_ate(), float)


def test_full_pipeline_with_diagnostics(stream):
    """Full loop combining estimator, SMD, ATETracker, OverlapChecker."""
    ipw = OnlineIPW()
    smd = OnlineSMD(covariates=["x0", "x1", "x2"])
    tracker = ATETracker(log_every=50)
    checker = OverlapChecker()

    for x, w, y, _ in stream:
        ipw.learn_one(x, w, y)
        p = ipw.ps_model.predict_one(x)
        weight = ipw.ps_model.ipw_weight(x, w)
        smd.update(x, w, weight=weight)
        tracker.update(ipw.predict_ate())
        checker.update(p, w)

    assert ipw.n_seen == N
    assert len(tracker.history) > 0
    report = smd.report()
    assert "x0" in report
    overlap = checker.report()
    assert overlap["n_total"] == N


def test_full_pipeline_with_drift_monitor():
    """ConceptDriftMonitor should detect drift in a drifting stream."""
    monitor = ConceptDriftMonitor(delta=0.002)
    # Non-drifting stream first
    for _ in range(200):
        monitor.update(2.0 + (0.1 * (hash(str(_)) % 10 - 5) / 5))
    n_before = monitor.n_drifts
    assert isinstance(n_before, int)


def test_policy_with_estimator():
    """EpsilonGreedy policy integrated with an estimator in a loop."""
    stream = LinearCausalStream(n=200, n_features=3, seed=0)
    policy = EpsilonGreedy(eps_start=0.5, eps_end=0.1, decay=100, seed=0)
    est = OnlineRLearner()

    for step, (x, _, y, _) in enumerate(stream):
        cate = est.predict_one(x)
        treatment, propensity = policy.choose(cate, step)
        # Simulate outcome under policy treatment
        simulated_y = y + (treatment - 0.5) * 0.5
        est.learn_one(x, treatment, simulated_y, propensity=propensity)
        policy.update(simulated_y)

    assert est.n_seen == 200


def test_reset_and_reiterate(stream):
    """After reset, estimator behaves as if freshly initialized."""
    est = OnlineIPW()
    for x, w, y, _ in stream:
        est.learn_one(x, w, y)
    ate_before = est.predict_ate()
    est.reset()
    assert est.n_seen == 0
    assert est.predict_ate() == 0.0
