# Quickstart

## The stream loop

Every OnlineCML estimator follows the same pattern:

```python
for x, treatment, outcome, true_cate in stream:
    cate = estimator.predict_one(x)   # predict BEFORE learning
    estimator.learn_one(x, treatment, outcome)

ate = estimator.predict_ate()
lo, hi = estimator.predict_ci(alpha=0.05)
```

`x` is a plain Python `dict`. `treatment` is `0` or `1`. `outcome` is a `float`.

## IPW

```python
from onlinecml.datasets import LinearCausalStream
from onlinecml.reweighting import OnlineIPW

estimator = OnlineIPW()
for x, w, y, _ in LinearCausalStream(n=2000, true_ate=3.0, seed=42):
    estimator.learn_one(x, w, y)

print(f"ATE:   {estimator.predict_ate():.3f}")
print(f"95%CI: {estimator.predict_ci()}")
```

## AIPW (Doubly Robust)

```python
from onlinecml.reweighting import OnlineAIPW

estimator = OnlineAIPW()
for x, w, y, _ in LinearCausalStream(n=2000, seed=42):
    estimator.learn_one(x, w, y)

# Individual CATE prediction
cate = estimator.predict_one({"x0": 0.5, "x1": -0.3, "x2": 0.1, "x3": 0.0, "x4": -0.2})
```

## R-Learner (Double ML)

```python
from onlinecml.metalearners import OnlineRLearner
from river.linear_model import LinearRegression

model = OnlineRLearner(cate_model=LinearRegression())
for x, w, y, _ in LinearCausalStream(n=2000, seed=42):
    model.learn_one(x, w, y)
```

## Causal Forest

```python
from onlinecml.forests import OnlineCausalForest

forest = OnlineCausalForest(n_trees=10, grace_period=100, seed=0)
for x, w, y, _ in LinearCausalStream(n=2000, seed=42):
    forest.learn_one(x, w, y)
```

## Exploration policy

```python
from onlinecml.policy import ThompsonSampling

policy = ThompsonSampling(seed=0)
for step, (x, w, y, _) in enumerate(LinearCausalStream(n=500, seed=0)):
    treatment, propensity = policy.choose(cate_score=0.0, step=step)
    policy.update(reward=float(y > 0))
```

## Diagnostics

```python
from onlinecml.diagnostics import OnlineSMD, ATETracker, OverlapChecker

smd     = OnlineSMD(covariates=["x0", "x1", "x2", "x3", "x4"])
tracker = ATETracker(log_every=50)
checker = OverlapChecker(ps_min=0.05, ps_max=0.95)

ipw = OnlineIPW()
for x, w, y, _ in LinearCausalStream(n=1000, seed=42):
    ps = ipw.ps_model.predict_one(x)
    checker.update(ps, treatment=w)
    smd.update(x, w, weight=ipw.ps_model.ipw_weight(x, w))
    ipw.learn_one(x, w, y)
    tracker.update(ipw.predict_ate())

print(smd.report())
print(f"Balanced:        {smd.is_balanced()}")
print(f"Overlap adequate: {checker.is_overlap_adequate()}")
tracker.plot()
```

## Progressive evaluation

```python
from onlinecml.evaluation import progressive_causal_score
from onlinecml.evaluation.metrics import ATEError, PEHE

results = progressive_causal_score(
    stream  = LinearCausalStream(n=1000, seed=0),
    model   = OnlineRLearner(),
    metrics = [ATEError(), PEHE()],
    step    = 100,
)
print(results["ATEError"])  # list of 10 values, one per 100 obs
```
