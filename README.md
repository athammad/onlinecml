# OnlineCML

> Causal inference for the real world — one observation at a time.

[![Tests](https://github.com/yourname/onlinecml/actions/workflows/ci.yml/badge.svg)](https://github.com/yourname/onlinecml/actions)
[![Coverage](https://img.shields.io/badge/coverage-98%25-brightgreen)](https://github.com/yourname/onlinecml)
[![PyPI](https://img.shields.io/pypi/v/onlinecml)](https://pypi.org/project/onlinecml/)
[![Docs](https://img.shields.io/badge/docs-onlinecml.org-blue)](https://docs.onlinecml.org)

## Why OnlineCML?

Every major causal inference library (EconML, CausalML, DoWhy) requires a
complete dataset before you begin. But many real-world applications don't
have that luxury:

- A/B tests where you want decisions *now*, not after 30 days
- Clinical trials where treatment effects must be monitored continuously
- Marketing systems where customer data arrives as a stream
- Any setting where the treatment effect might shift over time

OnlineCML processes one observation at a time. No batches. No waiting.

## Installation

```bash
pip install onlinecml
```

## Quickstart

```python
from onlinecml.datasets import LinearCausalStream
from onlinecml.reweighting import OnlineIPW

estimator = OnlineIPW()
for x, treatment, outcome, _ in LinearCausalStream(n=1000, true_ate=2.0, seed=42):
    estimator.learn_one(x, treatment, outcome)

print(f"ATE:   {estimator.predict_ate():.3f}")  # → ~2.0
print(f"95%CI: {estimator.predict_ci()}")
```

## Methods (v0.1)

| Method | Class | ATE | Individual CATE | Doubly Robust |
|---|---|:---:|:---:|:---:|
| Inverse Probability Weighting | `OnlineIPW` | ✓ | — | — |
| Augmented IPW | `OnlineAIPW` | ✓ | ✓ | ✓ |
| S-Learner | `OnlineSLearner` | ✓ | ✓ | — |
| T-Learner | `OnlineTLearner` | ✓ | ✓ | — |

**Policies:** `EpsilonGreedy` (with exponential decay)

**Diagnostics:** `OnlineSMD`, `ATETracker` (with convergence plot)

**Datasets:** `LinearCausalStream`, `HeterogeneousCausalStream`

## How it differs from batch libraries

| | DoWhy | EconML | CausalML | **OnlineCML** |
|---|:---:|:---:|:---:|:---:|
| Online / streaming | ✗ | ✗ | ✗ | **✓** |
| One-obs-at-a-time | ✗ | ✗ | ✗ | **✓** |
| Concept drift | ✗ | ✗ | ✗ | **✓** |
| Exploration policy | ✗ | ✗ | ✗ | **✓** |
| River compatible | ✗ | ✗ | ✗ | **✓** |
| IPW / DR | ✓ | ✓ | ✓ | **✓** |
| Meta-learners | ✓ | ✓ | ✓ | **✓** |

## Documentation

Full documentation at [docs.onlinecml.org](https://docs.onlinecml.org).

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md). All PRs require unit tests and
must maintain >90% coverage.

## Citation

```bibtex
@software{onlinecml2025,
  title  = {OnlineCML: Online Causal Machine Learning in Python},
  year   = {2025},
  url    = {https://onlinecml.org}
}
```

## License

MIT
