"""Memory benchmark: peak memory usage under long streams.

Run with:
    python tests/benchmarks/bench_memory.py

Uses tracemalloc to measure peak heap allocation for each estimator over
a 10 000-observation stream. No assertions — profiling tool only.
"""

import tracemalloc
from typing import Any

from river.linear_model import LinearRegression

from onlinecml.datasets import LinearCausalStream
from onlinecml.matching import OnlineMatching
from onlinecml.metalearners import OnlineRLearner, OnlineSLearner
from onlinecml.reweighting import OnlineAIPW, OnlineIPW

N = 10_000


def _make_estimators() -> dict[str, Any]:
    return {
        "OnlineIPW":      OnlineIPW(),
        "OnlineAIPW":     OnlineAIPW(),
        "OnlineSLearner": OnlineSLearner(model=LinearRegression()),
        "OnlineRLearner": OnlineRLearner(cate_model=LinearRegression()),
        "OnlineMatching": OnlineMatching(k=3, buffer_size=500),
    }


def benchmark_memory() -> None:
    """Measure peak memory for each estimator over N observations."""
    stream = list(LinearCausalStream(n=N, seed=0))

    print(f"\n{'Estimator':>18} | {'Peak MB':>9} | {'Current MB':>11}")
    print("-" * 46)

    for name, model in _make_estimators().items():
        tracemalloc.start()
        for x, w, y, _ in stream:
            model.learn_one(x, w, y)
        _, peak = tracemalloc.get_traced_memory()
        current, _ = tracemalloc.get_traced_memory()
        tracemalloc.stop()

        peak_mb    = peak    / 1024 / 1024
        current_mb = current / 1024 / 1024
        print(f"{name:>18} | {peak_mb:>9.3f} | {current_mb:>11.3f}")


if __name__ == "__main__":
    benchmark_memory()
