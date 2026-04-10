"""Throughput benchmark: observations processed per second per estimator.

Run with:
    python tests/benchmarks/bench_throughput.py

Reports mean obs/sec for learn_one + predict_one for each estimator class.
No assertions — this is a performance profiling tool, not a correctness test.
"""

import time
from typing import Any

from river.linear_model import LinearRegression, LogisticRegression

from onlinecml.datasets import LinearCausalStream
from onlinecml.matching import OnlineMatching
from onlinecml.metalearners import OnlineRLearner, OnlineSLearner, OnlineTLearner
from onlinecml.reweighting import OnlineAIPW, OnlineIPW, OnlineOverlapWeights

N_WARMUP = 100   # warm-up observations (not timed)
N_TIMED  = 1000  # timed observations


def _make_estimators() -> dict[str, Any]:
    return {
        "OnlineIPW":           OnlineIPW(),
        "OnlineAIPW":          OnlineAIPW(),
        "OnlineOverlapWeights": OnlineOverlapWeights(),
        "OnlineSLearner":      OnlineSLearner(model=LinearRegression()),
        "OnlineTLearner":      OnlineTLearner(
            treated_model=LinearRegression(),
            control_model=LinearRegression(),
        ),
        "OnlineRLearner":      OnlineRLearner(cate_model=LinearRegression()),
        "OnlineMatching":      OnlineMatching(k=3, buffer_size=100),
    }


def benchmark_throughput() -> None:
    """Measure and print obs/sec for each estimator."""
    stream_all = list(LinearCausalStream(n=N_WARMUP + N_TIMED, seed=0))
    warmup_data = stream_all[:N_WARMUP]
    timed_data  = stream_all[N_WARMUP:]

    print(f"\n{'Estimator':>22} | {'obs/sec':>10} | {'learn_one µs':>13} | {'predict_one µs':>15}")
    print("-" * 70)

    for name, model in _make_estimators().items():
        # Warm up
        for x, w, y, _ in warmup_data:
            model.learn_one(x, w, y)

        # Time learn_one
        t0 = time.perf_counter()
        for x, w, y, _ in timed_data:
            model.learn_one(x, w, y)
        t_learn = time.perf_counter() - t0

        # Time predict_one
        t0 = time.perf_counter()
        for x, _, _, _ in timed_data:
            model.predict_one(x)
        t_predict = time.perf_counter() - t0

        obs_per_sec  = N_TIMED / t_learn
        learn_us     = t_learn  / N_TIMED * 1e6
        predict_us   = t_predict / N_TIMED * 1e6
        print(f"{name:>22} | {obs_per_sec:>10,.0f} | {learn_us:>12.1f}µ | {predict_us:>14.1f}µ")


if __name__ == "__main__":
    benchmark_throughput()
