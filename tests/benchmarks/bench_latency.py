"""Latency benchmark: per-call learn_one and predict_one latency percentiles.

Run with:
    python tests/benchmarks/bench_latency.py

Reports p50, p95, and p99 latency in microseconds for each estimator.
No assertions — profiling tool only.
"""

import statistics
import time
from typing import Any

from river.linear_model import LinearRegression

from onlinecml.datasets import LinearCausalStream
from onlinecml.matching import OnlineMatching
from onlinecml.metalearners import OnlineRLearner, OnlineSLearner
from onlinecml.reweighting import OnlineAIPW, OnlineIPW

N_WARMUP = 200
N_TIMED  = 2000


def _make_estimators() -> dict[str, Any]:
    return {
        "OnlineIPW":      OnlineIPW(),
        "OnlineAIPW":     OnlineAIPW(),
        "OnlineSLearner": OnlineSLearner(model=LinearRegression()),
        "OnlineRLearner": OnlineRLearner(cate_model=LinearRegression()),
        "OnlineMatching": OnlineMatching(k=3, buffer_size=200),
    }


def _percentile(data: list[float], p: float) -> float:
    """Return the p-th percentile of data (0 < p <= 100)."""
    sorted_data = sorted(data)
    idx = int(len(sorted_data) * p / 100)
    return sorted_data[min(idx, len(sorted_data) - 1)]


def benchmark_latency() -> None:
    """Measure p50/p95/p99 latency for learn_one and predict_one."""
    stream_all = list(LinearCausalStream(n=N_WARMUP + N_TIMED, seed=0))
    warmup_data = stream_all[:N_WARMUP]
    timed_data  = stream_all[N_WARMUP:]

    header = (
        f"\n{'Estimator':>18} | "
        f"{'learn p50':>10} | {'learn p95':>10} | {'learn p99':>10} | "
        f"{'pred p50':>9} | {'pred p95':>9} | {'pred p99':>9}"
    )
    print(header)
    print("-" * len(header))

    for name, model in _make_estimators().items():
        # Warm up
        for x, w, y, _ in warmup_data:
            model.learn_one(x, w, y)

        # Collect per-call learn_one latencies
        learn_times: list[float] = []
        for x, w, y, _ in timed_data:
            t0 = time.perf_counter()
            model.learn_one(x, w, y)
            learn_times.append((time.perf_counter() - t0) * 1e6)

        # Collect per-call predict_one latencies
        pred_times: list[float] = []
        for x, _, _, _ in timed_data:
            t0 = time.perf_counter()
            model.predict_one(x)
            pred_times.append((time.perf_counter() - t0) * 1e6)

        lp50 = _percentile(learn_times, 50)
        lp95 = _percentile(learn_times, 95)
        lp99 = _percentile(learn_times, 99)
        pp50 = _percentile(pred_times, 50)
        pp95 = _percentile(pred_times, 95)
        pp99 = _percentile(pred_times, 99)

        print(
            f"{name:>18} | "
            f"{lp50:>8.1f}µs | {lp95:>8.1f}µs | {lp99:>8.1f}µs | "
            f"{pp50:>7.1f}µs | {pp95:>7.1f}µs | {pp99:>7.1f}µs"
        )


if __name__ == "__main__":
    benchmark_latency()
