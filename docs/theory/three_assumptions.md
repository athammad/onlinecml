# The Three Causal Assumptions

All causal estimators — online or batch — require three assumptions to
identify treatment effects from observational data. This page explains
each assumption and shows how OnlineCML helps you check them.

## 1. Unconfoundedness

**Formal statement:**

$$Y_i(0), Y_i(1) \perp W_i \mid X_i$$

**In plain English:** after conditioning on the observed covariates $X_i$,
the treatment assignment contains no additional information about the
potential outcomes.

**What can go wrong:** unobserved confounders — variables that affect both
treatment and outcome but are not in $X_i$. Examples: unmeasured health
status, hidden user intent, genetic factors.

**How OnlineCML helps:** The `OnlineSMD` and `LiveLovePlot` tools check
whether the treated and control groups are balanced on *observed* covariates.
Balance on observables is a necessary (not sufficient) condition for
unconfoundedness.

```python
smd = OnlineSMD(covariates=["age", "income"])
for x, w, y, _ in stream:
    smd.update(x, treatment=w, weight=ipw_weight)
print(smd.is_balanced())  # |SMD| < 0.1 for all covariates
```

## 2. Overlap (Positivity)

**Formal statement:**

$$0 < e(x) < 1 \quad \forall\, x \in \text{support}(X)$$

where $e(x) = P(W=1 \mid X=x)$ is the propensity score.

**In plain English:** every type of unit has a non-zero probability of
being either treated or untreated.

**What can go wrong:** near-positivity violations occur when some subgroups
are almost never treated (or almost always treated). IPW weights become
extremely large, inflating variance.

**How OnlineCML helps:** `OverlapChecker` flags observations with propensity
scores outside `[ps_min, ps_max]`. `OnlineOverlapWeights` uses Li et al.
(2018)'s overlap weights $h(x) = e(x)(1-e(x))$ which are bounded and stable.

```python
checker = OverlapChecker(ps_min=0.05, ps_max=0.95)
for x, w, y, _ in stream:
    checker.update(ps_model.predict_one(x), treatment=w)
print(checker.is_overlap_adequate())
```

## 3. SUTVA

**Formal statement:** The potential outcome $Y_i(w)$ depends only on unit
$i$'s own treatment $W_i$, not on the treatments of other units.

**In plain English:** there are no spillover effects between units.

**What can go wrong:** network effects (social contagion), market-level
effects (price changes affect all buyers), household effects (one family
member's medication affects outcomes for others).

**How OnlineCML handles it:** OnlineCML assumes SUTVA by default. If your
setting has interference, results should be interpreted as **direct effects**
only. Future versions will include `NetworkInterferenceStream` for
SUTVA-violating settings.

## Checking Assumptions in Practice

```python
from onlinecml.diagnostics import OnlineSMD, OverlapChecker

smd     = OnlineSMD(covariates=[...])
checker = OverlapChecker()
ipw     = OnlineIPW()

for x, w, y, _ in stream:
    ps = ipw.ps_model.predict_one(x)
    checker.update(ps, treatment=w)
    weight = 1/ps if w == 1 else 1/(1-ps)
    smd.update(x, treatment=w, weight=weight)
    ipw.learn_one(x, w, y)

# Report
print("Overlap adequate:", checker.is_overlap_adequate())
print("Balance adequate:", smd.is_balanced())
```

## References

- Imbens, G.W. and Rubin, D.B. (2015). *Causal Inference for Statistics,
  Social, and Biomedical Sciences*. Cambridge University Press.
- Li, F., Morgan, K.L. and Zaslavsky, A.M. (2018). Balancing covariates via
  propensity score weighting. *Journal of the American Statistical
  Association*, 113(521), 390–400.
