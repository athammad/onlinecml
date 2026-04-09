# Quickstart

## The stream loop

Every OnlineCML estimator follows the same pattern:

```python
for x, treatment, outcome, true_cate in stream:
    estimator.learn_one(x, treatment, outcome)
    cate = estimator.predict_one(x)
    ate  = estimator.predict_ate()
    ci   = estimator.predict_ci(alpha=0.05)
```

`x` is a plain Python dict. `treatment` is 0 or 1. `outcome` is a float.

## IPW example

```python
from onlinecml.datasets import LinearCausalStream
from onlinecml.reweighting import OnlineIPW

stream = LinearCausalStream(n=2000, true_ate=3.0, seed=42)
estimator = OnlineIPW()

for x, w, y, _ in stream:
    estimator.learn_one(x, w, y)

print(f"ATE:   {estimator.predict_ate():.3f}")
print(f"95%CI: {estimator.predict_ci()}")
```

## Doubly robust (AIPW) example

```python
from onlinecml.reweighting import OnlineAIPW

estimator = OnlineAIPW()
for x, w, y, _ in LinearCausalStream(n=2000, seed=42):
    estimator.learn_one(x, w, y)

# AIPW can also predict individual CATE
cate = estimator.predict_one({"x0": 0.5, "x1": -0.3, "x2": 0.1})
```

## Meta-learner example

```python
from onlinecml.metalearners import OnlineTLearner
from river.linear_model import LinearRegression

model = OnlineTLearner(
    treated_model=LinearRegression(),
    control_model=LinearRegression(),
)
for x, w, y, _ in LinearCausalStream(n=2000, seed=42):
    model.learn_one(x, w, y)
```

## Diagnostics example

```python
from onlinecml.diagnostics import OnlineSMD, ATETracker

smd = OnlineSMD(covariates=["x0", "x1", "x2"])
tracker = ATETracker(log_every=50)

from onlinecml.reweighting import OnlineIPW
estimator = OnlineIPW()

for x, w, y, _ in LinearCausalStream(n=1000, seed=42):
    estimator.learn_one(x, w, y)
    weight = estimator.ps_model.ipw_weight(x, w)
    smd.update(x, w, weight=weight)
    tracker.update(estimator.predict_ate())

print(smd.report())
print(f"Balanced: {smd.is_balanced()}")
tracker.plot()
```
