# Potential Outcomes Framework

The **potential outcomes framework** (also called the Rubin Causal Model) is
the formal language behind all of OnlineCML's estimators.

## Setup

For each unit $i$, define:

- $Y_i(0)$ — the outcome unit $i$ would have *if not treated*
- $Y_i(1)$ — the outcome unit $i$ would have *if treated*
- $W_i \in \{0, 1\}$ — the observed treatment indicator
- $X_i$ — the observed pre-treatment covariates

The **individual treatment effect** is $\tau_i = Y_i(1) - Y_i(0)$.

Because we can only observe one potential outcome per unit
(the *fundamental problem of causal inference*), $\tau_i$ is never observed
directly. We instead estimate averages.

## Estimands

| Name | Formula | OnlineCML method |
|---|---|---|
| ATE | $E[\tau_i] = E[Y_i(1)] - E[Y_i(0)]$ | `predict_ate()` |
| CATE | $\tau(x) = E[\tau_i \mid X_i = x]$ | `predict_one(x)` |
| ATT | $E[\tau_i \mid W_i = 1]$ | *available via T-Learner* |

## Three Identifying Assumptions

All estimators in OnlineCML require three assumptions to hold:

### 1. Unconfoundedness (Ignorability)

$$Y_i(0), Y_i(1) \perp W_i \mid X_i$$

Treatment assignment is independent of potential outcomes given the observed
covariates. Violated when there are unobserved confounders.

### 2. Overlap (Positivity)

$$0 < P(W_i = 1 \mid X_i = x) < 1 \quad \forall\, x$$

Every unit has a positive probability of receiving either treatment. Use
`OverlapChecker` to monitor this continuously.

### 3. SUTVA (Stable Unit Treatment Value Assumption)

The potential outcome of unit $i$ does not depend on the treatment received
by any other unit. Violated in settings with network effects or spillovers.

## The Propensity Score

The propensity score $e(x) = P(W=1 \mid X=x)$ is a *balancing score*:
conditional on $e(X_i)$, treatment is independent of covariates
(Rosenbaum & Rubin, 1983). OnlineCML estimates $e(x)$ with any River classifier
via `OnlinePropensityScore`.

```python
from onlinecml.propensity import OnlinePropensityScore
from river.linear_model import LogisticRegression

ps = OnlinePropensityScore(classifier=LogisticRegression())
for x, w, y, _ in stream:
    p = ps.predict_one(x)
    ps.learn_one(x, w)
```

## References

- Rubin, D.B. (1974). Estimating causal effects of treatments in randomized
  and nonrandomized studies. *Journal of Educational Psychology*, 66(5), 688–701.
- Rosenbaum, P.R. and Rubin, D.B. (1983). The central role of the propensity
  score in observational studies for causal effects. *Biometrika*, 70(1), 41–55.
