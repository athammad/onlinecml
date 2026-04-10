# Robinson Transformation (R-Learner)

The **R-Learner** (Nie & Wager, 2021) is based on Robinson's (1988)
partial linear model decomposition. It is implemented in OnlineCML as
`OnlineRLearner`.

## The Partial Linear Model

Assume the outcome follows:

$$Y_i = m(X_i) + \tau(X_i) \cdot W_i + \varepsilon_i$$

where $m(x) = E[Y \mid X = x]$ is the baseline outcome surface and
$\tau(x)$ is the CATE.

## The Residual Transformation

Robinson (1988) showed that after partialling out the nuisance functions:

$$Y_i - m(X_i) = \tau(X_i) \cdot (W_i - e(X_i)) + \varepsilon_i$$

where $e(x) = E[W \mid X = x]$ is the propensity score. This is the
**Robinson decomposition**: the residualised outcome $\tilde{Y}_i = Y_i - m(X_i)$
regressed on the residualised treatment $\tilde{W}_i = W_i - e(X_i)$
identifies $\tau(X_i)$.

## The R-Learner Loss

Nie & Wager (2021) propose minimising:

$$\hat{\tau} = \arg\min_\tau \sum_i \left(\tilde{Y}_i - \tau(X_i)\tilde{W}_i\right)^2$$

which is equivalent to regressing the pseudo-outcome
$\tilde{Y}_i / \tilde{W}_i$ on $X_i$ with weight $\tilde{W}_i^2$.

## Online Approximation in OnlineCML

`OnlineRLearner` maintains three running River models:

1. `ps_model` — estimates $e(x) = P(W=1|X)$  
2. `outcome_model` — estimates $m(x) = E[Y|X]$  
3. `cate_model` — fits $\tau(x)$ from residualised targets

At each step:

```
W_res = W - ps_model.predict(X)        # treatment residual
Y_res = Y - outcome_model.predict(X)  # outcome residual
pseudo_outcome = Y_res / W_res         # only if |W_res| >= min_residual
weight = W_res^2
cate_model.learn_one(X, pseudo_outcome, w=weight)
```

The predict-then-learn protocol ensures the nuisance models are not
contaminated by the current observation when generating the pseudo-outcome.

## References

- Robinson, P.M. (1988). Root-N-consistent semiparametric regression.
  *Econometrica*, 56(4), 931–954.
- Nie, X. and Wager, S. (2021). Quasi-oracle estimation of heterogeneous
  treatment effects. *Biometrika*, 108(2), 299–319.
