# OnlineCML

> **Causal inference for the real world — one observation at a time.**

OnlineCML is the first Python library for causal inference in a purely online,
one-observation-at-a-time setting. It is the online counterpart to EconML,
CausalML, and DoWhy — built on River's infrastructure and conventions.

## Why OnlineCML?

Most causal inference libraries assume you have a complete dataset before
you start. But the real world doesn't wait for your batch job: treatment
decisions are made continuously, data arrives as a stream, and effects
shift over time.

OnlineCML lets you:

- Estimate treatment effects **without storing data**
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

## Methods (v0.1)

| Method | Class | Individual CATE |
|---|---|:---:|
| Inverse Probability Weighting | `OnlineIPW` | No |
| Augmented IPW (Doubly Robust) | `OnlineAIPW` | Yes |
| S-Learner | `OnlineSLearner` | Yes |
| T-Learner | `OnlineTLearner` | Yes |

## Navigation

- [Installation](getting_started/installation.md)
- [Quickstart](getting_started/quickstart.md)
- [API Reference](api_reference/base.md)
