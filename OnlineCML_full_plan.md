# OnlineCML — Full Project Plan
## The Reference Library for Online Causal Machine Learning

---

## 1. Project Vision

**OnlineCML** is the first Python library for causal inference in a
purely online, one-observation-at-a-time setting. It is the online
counterpart to EconML, CausalML, and DoWhy — built on River's
infrastructure and conventions, designed to be production-grade,
fully documented, and research-backed.

**Tagline:** *"Causal inference for the real world — one observation
at a time."*

---

## 2. Repository Structure

```
onlinecml/
│
├── .github/
│   ├── workflows/
│   │   ├── ci.yml                  # Tests on every PR
│   │   ├── cd.yml                  # Auto-publish to PyPI on tag
│   │   └── docs.yml                # Auto-deploy docs on merge to main
│   ├── ISSUE_TEMPLATE/
│   │   ├── bug_report.md
│   │   └── feature_request.md
│   └── PULL_REQUEST_TEMPLATE.md
│
├── onlinecml/                      # Main package
│   ├── __init__.py
│   ├── base/
│   ├── propensity/
│   ├── reweighting/
│   ├── matching/
│   ├── metalearners/
│   ├── forests/
│   ├── policy/
│   ├── diagnostics/
│   ├── datasets/
│   └── evaluation/
│
├── tests/                          # Full test suite
│   ├── unit/
│   ├── integration/
│   ├── regression/
│   └── benchmarks/
│
├── docs/                           # Documentation source (MkDocs)
│   ├── index.md
│   ├── getting_started/
│   ├── user_guide/
│   ├── api_reference/
│   ├── examples/
│   └── contributing/
│
├── examples/                       # Runnable notebooks and scripts
│   ├── notebooks/
│   └── scripts/
│
├── website/                        # Dedicated landing page
│   ├── index.html
│   ├── css/
│   ├── js/
│   └── assets/
│
├── benchmarks/                     # Performance benchmarks vs batch
│   ├── vs_econml.py
│   ├── vs_causalml.py
│   └── streaming_performance.py
│
├── paper/                          # Academic paper (LaTeX)
│   └── onlinecml_paper.tex
│
├── CHANGELOG.md
├── CONTRIBUTING.md
├── CODE_OF_CONDUCT.md
├── LICENSE                         # MIT
├── README.md
├── pyproject.toml
└── mkdocs.yml
```

---

## 3. Package Features (Full)

### 3.1 Base Infrastructure (`base/`)

**BaseOnlineEstimator**
Abstract class that every estimator inherits from:
```python
class BaseOnlineEstimator:
    def learn_one(self, x, treatment, outcome, propensity=None): ...
    def predict_one(self, x): ...           # CATE
    def predict_ate(self): ...              # running ATE
    def predict_ci(self, alpha=0.05): ...  # confidence interval
    def reset(self): ...                    # reset to untrained state
    def clone(self): ...                    # return fresh copy
    @property
    def n_seen(self): ...                   # observations processed
    @property
    def smd(self): ...                      # current SMD dict
```

**RunningStats**
Online mean, variance, covariance using Welford's algorithm:
```python
stats = RunningStats()
stats.update(x)
stats.mean, stats.variance, stats.std
```

**BasePolicy**
Abstract class for exploration policies:
```python
class BasePolicy:
    def choose(self, score, step): ...     # returns (treatment, propensity)
    def update(self, reward): ...
```

---

### 3.2 Propensity Score (`propensity/`)

**OnlinePropensityScore**
- Wraps any River classifier
- Tracks logged propensity per observation
- Methods: `learn_one`, `predict_one`, `ipw_weight`, `overlap_weight`

**OnlineCBPS** *(research)*
- Balancing constraint projected onto gradient updates
- Checks online SMD convergence after each step

---

### 3.3 Reweighting (`reweighting/`)

**OnlineIPW**
- Running ATE via importance-weighted average
- Trimming threshold for extreme weights
- Normalized IPW (stabilized weights) option

**OnlineAIPW**
- Doubly robust estimator — two River models
- Cross-fitting approximated via alternating updates
- DR correction per observation

**OnlineOverlapWeights**
- Bounded weights proportional to opposite-group probability
- More stable than IPW under near-positivity-violations

All share:
- `ate` property — running ATE
- `ate_ci(alpha)` — running confidence interval
- `weight_stats` — mean, max, min of recent weights

---

### 3.4 Matching (`matching/`)

**OnlineMatching**
- Separate treated/control SWINN buffers
- K nearest neighbors with IPW correction
- Running CATE per matched pair
- Distance options: euclidean, mahalanobis, PS-based, custom callable

**OnlineCaliperMatching**
- Max distance threshold — unmatched units tracked separately
- Reports % of units in common support

**OnlineKernelMatching**
- Weighted average of all control neighbors
- Gaussian kernel by default, custom kernel supported

**Distance utilities**
```python
from onlinecml.matching.distance import (
    euclidean_distance,
    mahalanobis_distance,
    ps_distance,
    combined_distance    # PS + Mahalanobis
)
```

---

### 3.5 Meta-Learners (`metalearners/`)

**OnlineSLearner**
- Single model, treatment as feature
- CATE = predict(x, W=1) - predict(x, W=0)
- Works with any River regressor

**OnlineTLearner**
- Two models: treated_model and control_model
- IPW correction per arm
- Handles unbalanced groups via sample weighting

**OnlineXLearner**
- Three-stage pipeline:
  1. T-Learner base models
  2. Imputed treatment effects per arm
  3. PS-weighted combination of two CATE models
- Best for unbalanced treatment groups

**OnlineRLearner**
- Robinson transformation online:
  - W_res = W - W_hat (online PS model)
  - Y_res = Y - Y_hat (online outcome model)
  - CATE trained on (W_res, Y_res) pairs
- Most theoretically grounded meta-learner
- Connects to DML

All meta-learners:
```python
model = OnlineRLearner(
    ps_model      = LogisticRegression(),
    outcome_model = LinearRegression(),
    cate_model    = LinearRegression(),
    policy        = EpsilonGreedy()
)
model.learn_one(x, w, y, p)
cate  = model.predict_one(x)
ate   = model.predict_ate()
ci    = model.predict_ci(alpha=0.05)
```

---

### 3.6 Online Causal Forests (`forests/`) *(novel)*

**CausalHoeffdingTree**
- Extends HoeffdingTreeRegressor with causal split criterion
- Split criterion: maximize treatment effect heterogeneity
- Robinson orthogonalization before splitting
- Honesty via separate streaming split/estimation buffers

**OnlineCausalForest**
- Ensemble of CausalHoeffdingTrees
- Online bagging via Poisson resampling (ARF style)
- ADWIN drift detection — trees replaced on drift
- Outputs CATE with approximate uncertainty estimate

```python
forest = OnlineCausalForest(
    n_trees          = 100,
    split_criterion  = "treatment_heterogeneity",
    honesty          = True,
    drift_detection  = True
)
forest.learn_one(x, w, y)
cate = forest.predict_one(x)
ci   = forest.predict_ci(x, alpha=0.05)
```

---

### 3.7 Exploration Policies (`policy/`)

**EpsilonGreedy**
```python
policy = EpsilonGreedy(
    eps_start = 0.5,
    eps_end   = 0.05,
    decay     = 2000
)
treatment, propensity = policy.choose(cate_score, step)
```

**ThompsonSampling**
- Beta-Bernoulli for binary outcomes
- Gaussian Thompson for continuous outcomes
- Posterior updated per observation

**UCB**
- Exploration bonus proportional to uncertainty
- Requires uncertainty estimate from CATE model
- Configurable confidence level

**FixedRandomization**
- Fixed probability treatment assignment
- Baseline for benchmarking (simulates RCT)

---

### 3.8 Diagnostics (`diagnostics/`)

**OnlineSMD**
```python
smd = OnlineSMD(covariates=["age", "bmi", "smoker"])
smd.update(x, treatment, weight=ipw_weight)
smd.report()           # dict: covariate -> (raw_smd, weighted_smd)
smd.is_balanced(thr=0.1)  # True if all |SMD| < threshold
```

**LiveLovePlot**
- Real-time matplotlib figure
- Updates every N steps
- Shows raw vs weighted SMD per covariate
- Saves to file on demand

**ATETracker**
- Running ATE with online confidence intervals
- Convergence plot: ATE estimate vs observations seen
- Early stopping signal when CI width < threshold

**OverlapChecker**
- Tracks PS distribution per arm
- Flags extreme PS values (< 0.05 or > 0.95)
- Reports % in common support
- Warning system for positivity violations

**ConceptDriftMonitor**
- Wraps River's ADWIN or Page-Hinkley detector
- Monitors ATE estimate for structural breaks
- Alerts when treatment effect distribution shifts

---

### 3.9 Datasets (`datasets/`)

**Synthetic streams (known ground truth):**
```python
from onlinecml.datasets import (
    LinearCausalStream,          # linear DGP, constant ATE
    HeterogeneousCausalStream,   # nonlinear CATE varies by X
    DriftingCausalStream,        # ATE shifts at known changepoint
    UnbalancedCausalStream,      # extreme PS, positivity stress test
    NetworkInterferenceStream,   # SUTVA violations
    ContinuousTreatmentStream,   # dose-response setting
)

stream = HeterogeneousCausalStream(
    n                   = 50000,
    n_features          = 10,
    true_ate            = -3.0,
    heterogeneity       = "nonlinear",
    confounding_strength = 0.7,
    seed                = 42
)
# Returns: (x_dict, treatment, outcome, true_cate)
```

**Real-world benchmark loaders:**
```python
from onlinecml.datasets import (
    load_lalonde,    # Jobs training program
    load_ihdp,       # Infant Health Development Program
    load_news,       # News dataset (high dimensional)
    load_twins,      # Twin births dataset
)
# All return River-compatible iterators
```

---

### 3.10 Evaluation (`evaluation/`)

**ProgressiveCausalScore**
- Causal-aware version of River's `progressive_val_score`
- Evaluates ATE error and PEHE at each step

```python
from onlinecml.evaluation import progressive_causal_score

results = progressive_causal_score(
    stream    = HeterogeneousCausalStream(),
    model     = OnlineRLearner(),
    metrics   = [ATEError(), PEHE(), UpliftAUC()],
    step      = 100           # evaluate every 100 obs
)
```

**Metrics:**
```python
from onlinecml.evaluation.metrics import (
    ATEError,          # |ATE_hat - ATE_true|
    PEHE,              # sqrt(mean((CATE_hat - CATE_true)^2))
    UpliftAUC,         # area under uplift curve
    QiniCoefficient,   # standard uplift metric
    CIWidth,           # mean confidence interval width
    CIcoverage,        # empirical CI coverage
)
```

---

## 4. Testing Strategy

### 4.1 Unit Tests (`tests/unit/`)

Every class has its own test file. Tests check:
- `learn_one` updates internal state correctly
- `predict_one` returns correct type and shape
- `reset()` returns to initial state
- Properties (`ate`, `smd`, `n_seen`) update correctly
- Edge cases: first observation, single treatment arm, extreme PS

```
tests/unit/
├── test_base_estimator.py
├── test_running_stats.py
├── test_online_ps.py
├── test_ipw.py
├── test_aipw.py
├── test_overlap_weights.py
├── test_online_matching.py
├── test_s_learner.py
├── test_t_learner.py
├── test_x_learner.py
├── test_r_learner.py
├── test_causal_hoeffding.py
├── test_online_causal_forest.py
├── test_epsilon_greedy.py
├── test_thompson.py
├── test_online_smd.py
├── test_ate_tracker.py
├── test_overlap_checker.py
├── test_datasets.py
└── test_metrics.py
```

### 4.2 Integration Tests (`tests/integration/`)

Test full pipelines end-to-end:
- River pipeline compatibility (`preprocessing | estimator`)
- Policy + estimator combinations
- Diagnostics + estimator combinations
- Multi-method comparison on same stream

```
tests/integration/
├── test_river_pipeline_compat.py   # works inside River pipelines
├── test_policy_estimator.py        # policy + any estimator
├── test_full_stream_loop.py        # simulate full production loop
└── test_multi_method_comparison.py # multiple methods same stream
```

### 4.3 Regression Tests (`tests/regression/`)

Verify that known-good results don't change across releases:
- ATE estimate within tolerance on fixed synthetic streams
- SMD convergence behavior
- Forest CATE estimates on fixed seeds

```
tests/regression/
├── test_ipw_ate_regression.py
├── test_rlearner_ate_regression.py
└── test_forest_cate_regression.py
```

### 4.4 Benchmark Tests (`tests/benchmarks/`)

Performance tests (not correctness):
- Observations per second for each estimator
- Memory usage under long streams
- Latency of `learn_one` and `predict_one`

```
tests/benchmarks/
├── bench_throughput.py        # obs/sec per estimator
├── bench_memory.py            # memory under 1M obs stream
└── bench_latency.py           # p50/p95/p99 latency
```

### 4.5 Testing Tools

- **pytest** — test runner
- **pytest-cov** — coverage reporting (target: > 90%)
- **pytest-benchmark** — performance benchmarks
- **hypothesis** — property-based testing for edge cases
- **CI target** — all tests pass on Python 3.10, 3.11, 3.12

---

## 5. Documentation

### 5.1 Tool: MkDocs + Material Theme

Professional, searchable, versioned documentation deployed
automatically to GitHub Pages on every merge to main.

### 5.2 Structure

```
docs/
├── index.md                        # Landing page with quick start
│
├── getting_started/
│   ├── installation.md             # pip install, dependencies
│   ├── quickstart.md               # 5-minute example
│   ├── core_concepts.md            # online learning, causal basics
│   └── vs_batch.md                 # when to use online vs batch
│
├── user_guide/
│   ├── stream_loop.md              # how to structure a stream loop
│   ├── propensity_score.md
│   ├── reweighting.md              # IPW, AIPW, overlap
│   ├── matching.md
│   ├── metalearners.md             # S/T/X/R learner guide
│   ├── causal_forests.md
│   ├── exploration_policies.md
│   ├── diagnostics.md              # SMD, love plot, ATE tracker
│   ├── concept_drift.md
│   └── evaluation.md
│
├── api_reference/                  # Auto-generated from docstrings
│   ├── propensity.md
│   ├── reweighting.md
│   ├── matching.md
│   ├── metalearners.md
│   ├── forests.md
│   ├── policy.md
│   ├── diagnostics.md
│   ├── datasets.md
│   └── evaluation.md
│
├── examples/                       # Rendered notebooks
│   ├── marketing_targeting.md      # email campaign uplift
│   ├── clinical_trial.md           # drug effect estimation
│   ├── pricing.md                  # dynamic pricing
│   ├── concept_drift.md            # handling drift
│   └── comparison_batch.md        # OnlineCML vs EconML comparison
│
├── theory/                         # Background reading
│   ├── potential_outcomes.md
│   ├── three_assumptions.md
│   ├── robinson_transformation.md
│   └── online_learning_basics.md
│
└── contributing/
    ├── contributing.md
    ├── development_setup.md
    ├── coding_standards.md
    └── adding_estimator.md         # guide for new contributors
```

### 5.3 Docstring Standard

Every public class and method has a NumPy-style docstring:

```python
class OnlineRLearner(BaseOnlineEstimator):
    """
    Online R-Learner for CATE estimation via Robinson transformation.

    Maintains two background River models (propensity score and outcome)
    and trains a CATE model on Robinson-residualized targets. Corrects
    for non-random treatment assignment via logged propensities.

    Parameters
    ----------
    ps_model : river.base.Classifier
        Model for estimating P(W=1|X). Any River classifier.
    outcome_model : river.base.Regressor
        Model for estimating E[Y|X]. Any River regressor.
    cate_model : river.base.Regressor
        Model for estimating CATE(X) from residualized targets.
    policy : BasePolicy, optional
        Exploration policy. Default: EpsilonGreedy().

    Attributes
    ----------
    n_seen : int
        Number of observations processed.
    ate : float
        Current running ATE estimate.

    Examples
    --------
    >>> from onlinecml.metalearners import OnlineRLearner
    >>> from onlinecml.datasets import LinearCausalStream
    >>> from river import linear_model
    >>>
    >>> model = OnlineRLearner(
    ...     ps_model      = linear_model.LogisticRegression(),
    ...     outcome_model = linear_model.LinearRegression(),
    ...     cate_model    = linear_model.LinearRegression(),
    ... )
    >>> for x, w, y, _ in LinearCausalStream(n=1000, seed=42):
    ...     model.learn_one(x, w, y)
    >>> print(f"ATE: {model.ate:.3f}")

    References
    ----------
    Robinson, P.M. (1988). Root-N-consistent semiparametric regression.
    Econometrica, 53, 931-954.

    Nie, X. and Wager, S. (2021). Quasi-oracle estimation of
    heterogeneous treatment effects. Biometrika, 108(2), 299-319.
    """
```

---

## 6. Examples

### 6.1 Jupyter Notebooks (`examples/notebooks/`)

```
examples/notebooks/
├── 01_quickstart.ipynb              # 5-minute intro
├── 02_ipw_vs_aipw.ipynb            # reweighting methods comparison
├── 03_metalearner_comparison.ipynb  # S/T/X/R side by side
├── 04_online_matching.ipynb         # KNN matching with SWINN
├── 05_causal_forest.ipynb           # OnlineCausalForest deep dive
├── 06_exploration_policies.ipynb    # epsilon-greedy vs Thompson
├── 07_diagnostics.ipynb             # SMD, love plot, ATE tracker
├── 08_concept_drift.ipynb           # handling shifting treatment effects
├── 09_marketing_case_study.ipynb    # email campaign uplift (realistic)
├── 10_clinical_trial.ipynb          # drug effect in streaming EHR data
└── 11_vs_econml_batch.ipynb         # online vs batch comparison
```

### 6.2 Scripts (`examples/scripts/`)

Lightweight runnable Python scripts (no notebook overhead):
```
examples/scripts/
├── minimal_example.py              # 20 lines, works out of the box
├── river_pipeline_example.py       # OnlineCML inside River pipeline
├── production_loop_example.py      # realistic production setup
└── benchmark_vs_batch.py           # ATE accuracy vs obs count plot
```

---

## 7. Website

### 7.1 Purpose

A dedicated landing page separate from the documentation. Its job is
to communicate the value proposition immediately to:
- Data scientists considering the library
- Researchers looking for a citation
- Engineers evaluating for production use

### 7.2 Pages

```
website/
├── index.html          # Landing page
├── docs.html           # Redirects to MkDocs site
├── benchmarks.html     # Performance comparison with batch methods
└── paper.html          # Link to arXiv paper + citation
```

### 7.3 Landing Page Sections

**Hero**
- Tagline: *"Causal inference for the real world — one observation at a time."*
- Animated code snippet showing the 5-line quickstart
- Buttons: "Get Started" → docs, "GitHub" → repo, "Paper" → arXiv

**The Problem**
- Visual: batch pipeline (wait → collect → train → deploy → freeze)
  vs online pipeline (learn → decide → learn → decide)
- Clear statement: "The world doesn't wait for your batch job."

**Features Grid**
- Online IPW, DR, Matching, Meta-Learners, Causal Forests
- Each with a 2-line code snippet and icon

**Benchmark Chart**
- ATE error vs observations seen: OnlineCML vs naive baseline
- Shows convergence — library works with limited data

**Ecosystem**
- "Built on River" — logo and link
- "Compatible with" — River pipelines, any River model
- "Compared to" — EconML, CausalML (with honest comparison)

**Quick Install**
```bash
pip install onlinecml
```

**Citation Block**
```bibtex
@software{onlinecml2025,
  title  = {OnlineCML: Online Causal Machine Learning in Python},
  author = {...},
  year   = {2025},
  url    = {https://onlinecml.org}
}
```

### 7.4 Hosting

- **GitHub Pages** — free, auto-deploys from `website/` folder
- **Custom domain** — `onlinecml.org` (or `.io`)
- **Docs subdomain** — `docs.onlinecml.org` → MkDocs site

---

## 8. CI/CD Pipeline

### 8.1 On Every Pull Request (`.github/workflows/ci.yml`)
```yaml
- Lint with ruff
- Type check with mypy
- Run unit tests (Python 3.10, 3.11, 3.12)
- Run integration tests
- Generate coverage report (fail if < 90%)
- Check docstrings completeness
```

### 8.2 On Merge to Main (`.github/workflows/docs.yml`)
```yaml
- Build MkDocs documentation
- Deploy to GitHub Pages (docs.onlinecml.org)
- Deploy website (onlinecml.org)
```

### 8.3 On Version Tag (`.github/workflows/cd.yml`)
```yaml
- Run full test suite
- Build package
- Publish to PyPI automatically
- Create GitHub Release with changelog
- Post to social channels (optional)
```

---

## 9. Package Configuration (`pyproject.toml`)

```toml
[project]
name            = "onlinecml"
version         = "0.1.0"
description     = "Online Causal Machine Learning in Python"
readme          = "README.md"
license         = {text = "MIT"}
requires-python = ">=3.10"

keywords = [
    "causal inference", "online learning", "streaming",
    "treatment effects", "uplift modeling", "machine learning"
]

classifiers = [
    "Development Status :: 4 - Beta",
    "Intended Audience :: Science/Research",
    "Topic :: Scientific/Engineering :: Artificial Intelligence",
    "License :: OSI Approved :: MIT License",
    "Programming Language :: Python :: 3.10",
    "Programming Language :: Python :: 3.11",
    "Programming Language :: Python :: 3.12",
]

dependencies = [
    "river>=0.23",
    "numpy>=1.24",
    "pandas>=2.0",
    "scipy>=1.10",
    "matplotlib>=3.7",
]

[project.optional-dependencies]
dev  = ["pytest", "pytest-cov", "pytest-benchmark", "hypothesis",
        "ruff", "mypy", "mkdocs-material"]
docs = ["mkdocs-material", "mkdocstrings[python]"]

[project.urls]
Homepage      = "https://onlinecml.org"
Documentation = "https://docs.onlinecml.org"
Repository    = "https://github.com/yourname/onlinecml"
Issues        = "https://github.com/yourname/onlinecml/issues"
Paper         = "https://arxiv.org/abs/XXXX.XXXXX"
```

---

## 10. README.md Structure

```markdown
# OnlineCML

> Causal inference for the real world — one observation at a time.

[![PyPI](badge)] [![Tests](badge)] [![Coverage](badge)] [![Docs](badge)]

## Why OnlineCML?

[2-paragraph motivation]

## Installation

pip install onlinecml

## Quickstart (5 lines)

[minimal working example]

## Methods

[table of all methods with links to docs]

## Benchmarks

[ATE error convergence chart]

## Documentation

[link to docs.onlinecml.org]

## Citation

[bibtex block]

## Contributing

[link to contributing guide]
```

---

## 11. Release Roadmap

### v0.1 — Foundation (Month 1-2)
**Goal:** installable, tested, documented core

- Base classes and River integration
- `OnlinePropensityScore`
- `OnlineIPW` and `OnlineAIPW`
- `OnlineSLearner` and `OnlineTLearner`
- `EpsilonGreedy` policy
- `OnlineSMD` and `ATETracker`
- Synthetic datasets (Linear, Heterogeneous)
- Unit tests for all above (>90% coverage)
- Full API docs
- README and quickstart
- Website landing page
- PyPI release

### v0.2 — Full Method Suite (Month 3-4)
**Goal:** complete the method library

- `OnlineXLearner` and `OnlineRLearner`
- `OnlineMatching` (SWINN + LazySearch)
- `OnlineCaliperMatching`
- `OnlineOverlapWeights`
- `ThompsonSampling` and `UCB` policies
- `LiveLovePlot` and `OverlapChecker`
- `ConceptDriftMonitor`
- Real-world datasets (LaLonde, IHDP, News)
- Integration tests
- 5 example notebooks
- PyPI release

### v0.3 — Evaluation + Benchmarks (Month 5)
**Goal:** make library trustworthy and comparable

- `ProgressiveCausalScore`
- Full metrics suite (PEHE, Qini, UpliftAUC)
- `DriftingCausalStream` and `UnbalancedCausalStream`
- Benchmarks vs EconML and CausalML (batch)
- Regression test suite
- Performance benchmarks (throughput, memory, latency)
- Benchmarks page on website
- Theory section in docs

### v1.0 — Novel Contribution + Paper (Month 6-9)
**Goal:** research contribution, academic credibility

- `CausalHoeffdingTree` (novel causal split criterion)
- `OnlineCausalForest` (novel)
- `OnlineCBPS` (research)
- `ContinuousTreatmentStream`
- All 11 example notebooks
- Full theory documentation
- arXiv paper submission
- Paper page on website
- Announcement blog post
- PyPI stable release

---

## 12. Academic Paper Plan

### Title
*"OnlineCML: A Python Library for Online Causal Machine Learning"*

### Sections
1. Introduction — the gap in the batch-only ecosystem
2. Background — potential outcomes, online learning, River
3. Library Design — architecture, API, River conventions
4. Methods — all estimators with theoretical motivation
5. Novel Contributions — CausalHoeffdingTree, OnlineCausalForest
6. Experiments — convergence, ATE accuracy, throughput
7. Conclusion

### Target Venues
- JMLR (Journal of Machine Learning Research) — software track
- NeurIPS or ICML workshop — causal inference track
- arXiv — preprint for immediate visibility

---

## 13. Community & Governance

### Contributing Guide
- How to add a new estimator (template provided)
- Coding standards (ruff, type hints, docstrings)
- Test requirements (unit + regression test for every PR)
- PR review process

### Issue Labels
- `bug` — something broken
- `feature` — new estimator or method
- `research` — theoretical question
- `docs` — documentation improvement
- `good first issue` — beginner friendly

### Code of Conduct
- Contributor Covenant standard

### Versioning
- Semantic versioning: MAJOR.MINOR.PATCH
- MAJOR — breaking API changes
- MINOR — new methods (backward compatible)
- PATCH — bug fixes and docs

---

## 14. Differentiation Summary

| | DoWhy | EconML | CausalML | **OnlineCML** |
|---|---|---|---|---|
| Online / streaming | ✗ | ✗ | ✗ | **✓** |
| One-obs-at-a-time | ✗ | ✗ | ✗ | **✓** |
| Concept drift | ✗ | ✗ | ✗ | **✓** |
| Exploration policy | ✗ | ✗ | ✗ | **✓** |
| River compatible | ✗ | ✗ | ✗ | **✓** |
| Online causal forest | ✗ | ✗ | ✗ | **✓** |
| IPW / DR / Overlap | ✓ | ✓ | ✓ | **✓** |
| Meta-learners | ✓ | ✓ | ✓ | **✓** |
| CATE estimation | ✓ | ✓ | ✓ | **✓** |
| >90% test coverage | varies | varies | varies | **✓** |
| Auto-generated docs | ✓ | ✓ | ✓ | **✓** |
| Dedicated website | ✗ | ✗ | ✗ | **✓** |
| Academic paper | ✓ | ✓ | ✓ | **✓** |
