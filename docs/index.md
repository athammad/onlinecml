# OnlineCML


> **Causal inference for the real world — one observation at a time.**


[![CI](https://github.com/athammad/onlinecml/actions/workflows/ci.yml/badge.svg)](https://github.com/athammad/onlinecml/actions/workflows/ci.yml)
\

OnlineCML is the first Python library for causal inference in a purely online,
one-observation-at-a-time setting. It is the streaming counterpart to EconML,
CausalML, and DoWhy — built on [River](https://riverml.xyz)'s infrastructure
and conventions.

## Why OnlineCML?

Most causal inference libraries assume you have a complete dataset before
you start. But the real world doesn't wait for your batch job: treatment
decisions are made continuously, data arrives as a stream, and effects
shift over time.

OnlineCML lets you:

- Estimate treatment effects **without storing data** (O(1) memory)
- Make treatment assignment decisions **in real time**
- Detect **concept drift** in causal relationships
- Stay compatible with **River pipelines** and any River model

## Quick Install

```bash
pip install onlinecml
```

## 5-Line Quickstart

```python
from onlinecml.datasets import LinearCausalStream
from onlinecml.reweighting import OnlineIPW

estimator = OnlineIPW()
for x, treatment, outcome, _ in LinearCausalStream(n=1000, true_ate=2.0, seed=42):
    estimator.learn_one(x, treatment, outcome)

print(f"ATE: {estimator.predict_ate():.3f}")
print(f"95% CI: {estimator.predict_ci()}")
```

## Methods (v1.0)

| Method | Class | CATE | Category |
|---|---|:---:|---|
| Inverse Probability Weighting | `OnlineIPW` | No | Reweighting |
| Augmented IPW (Doubly Robust) | `OnlineAIPW` | Yes | Reweighting |
| Overlap Weights | `OnlineOverlapWeights` | No | Reweighting |
| Covariate Balancing PS | `OnlineCBPS` | No | Reweighting |
| S-Learner | `OnlineSLearner` | Yes | Meta-Learner |
| T-Learner | `OnlineTLearner` | Yes | Meta-Learner |
| X-Learner | `OnlineXLearner` | Yes | Meta-Learner |
| R-Learner (Double ML) | `OnlineRLearner` | Yes | Meta-Learner |
| Causal Hoeffding Tree | `CausalHoeffdingTree` | Yes | Forest |
| Online Causal Forest | `OnlineCausalForest` | Yes | Forest |
| Nearest-Neighbour Matching | `OnlineMatching` | Yes | Matching |
| Caliper Matching | `OnlineCaliperMatching` | Yes | Matching |
| Kernel Matching | `OnlineKernelMatching` | Yes | Matching |
| Epsilon-Greedy | `EpsilonGreedy` | — | Policy |
| Thompson Sampling | `ThompsonSampling` | — | Policy |
| Gaussian Thompson Sampling | `GaussianThompsonSampling` | — | Policy |
| UCB | `UCB` | — | Policy |

## Navigation

- [Installation](getting_started/installation.md)
- [Quickstart](getting_started/quickstart.md)
- [Theory](theory/potential_outcomes.md)
- [API Reference](api_reference/base.md)
- [Contributing](contributing/contributing.md)
