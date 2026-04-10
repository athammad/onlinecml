# Online Learning Basics

OnlineCML is built on **online machine learning** — a paradigm where models
are updated one observation at a time rather than fitting on a complete
dataset. This page explains the core concepts.

## The Online Learning Loop

```python
for x, w, y, true_cate in stream:
    # 1. Predict BEFORE learning (no look-ahead bias)
    cate_hat = model.predict_one(x)

    # 2. Update with the new observation
    model.learn_one(x, w, y)
```

This **predict-then-learn** protocol is fundamental to online causal inference.
It ensures that the model's prediction for observation $i$ is based only on
observations $1, \ldots, i-1$, approximating the cross-fitting that batch
methods use to avoid look-ahead bias in pseudo-outcomes.

## Welford's Algorithm

All running statistics in OnlineCML use **Welford's (1962) online algorithm**
for numerically stable one-pass mean and variance estimation:

$$M_n = M_{n-1} + \frac{x_n - M_{n-1}}{n}$$
$$S_n = S_{n-1} + (x_n - M_{n-1})(x_n - M_n)$$

Sample variance: $\hat{\sigma}^2 = S_n / (n-1)$

This runs in **O(1) space** and **O(1) time** per observation, with superior
numerical stability to the naive two-pass algorithm. See `RunningStats`.

## Hoeffding Bound

The **Hoeffding bound** is used by `CausalHoeffdingTree` to decide when there
is enough evidence to split a node. For a random variable in range $[a, b]$,
after $n$ observations the bound is:

$$\varepsilon(n, \delta) = \sqrt{\frac{(b-a)^2 \ln(1/\delta)}{2n}}$$

With probability $1 - \delta$, the true mean of any function of the data
does not deviate from its sample mean by more than $\varepsilon$. A split is
triggered when the best split beats the second-best by more than $\varepsilon$.

## ADWIN (Adaptive Windowing)

`ConceptDriftMonitor` uses River's **ADWIN** (Bifet & Gavalda, 2007) algorithm.
ADWIN maintains a sliding window of variable length and detects drift when the
means of two sub-windows are statistically different, using the Hoeffding bound
as the test statistic.

ADWIN provides theoretical guarantees:
- **False positive rate** bounded by $\delta$ per test
- **Detection delay** proportional to $1 / \Delta\mu^2$ where $\Delta\mu$ is
  the magnitude of the drift

## Online Bagging

`OnlineCausalForest` uses **Oza's (2001) online bagging**: each observation is
presented to tree $k$ exactly $\text{Poisson}(\lambda)$ times, where $\lambda$
is the subsampling rate. When $\lambda = 1$, this simulates bootstrap resampling
in the online setting and gives approximately the same variance reduction as
batch random forests.

## References

- Welford, B.P. (1962). Note on a method for calculating corrected sums of
  squares and products. *Technometrics*, 4(3), 419–420.
- Hoeffding, W. (1963). Probability inequalities for sums of bounded random
  variables. *Journal of the American Statistical Association*, 58(301), 13–30.
- Bifet, A. and Gavalda, R. (2007). Learning from time-changing data with
  adaptive windowing. *Proceedings of SDM*, 443–448.
- Oza, N.C. (2001). Online bagging and boosting. *Proceedings of the
  American Statistical Association*, 229–234.
- Domingos, P. and Hulten, G. (2000). Mining high-speed data streams.
  *Proceedings of KDD*, 71–80.
