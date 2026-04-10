# Adding a New Estimator

This guide walks you through adding a new causal estimator to OnlineCML.
Follow these steps to ensure your estimator integrates cleanly with the
rest of the library.

## 1. Choose the right base class

| Base class | Use when |
|---|---|
| `BaseOnlineEstimator` | Estimating ATE or CATE from `(x, treatment, outcome)` triples |
| `BasePolicy` | Deciding which treatment to assign |
| `river.base.Base` | Helper classes (propensity models, diagnostics) |

## 2. Create the file

One class per file. Place it in the appropriate subpackage:

```
onlinecml/
├── reweighting/    # IPW-style weighting estimators
├── metalearners/   # S/T/X/R-Learner style estimators
├── matching/       # Matching-based estimators
├── forests/        # Tree and forest estimators
└── policy/         # Exploration policies
```

## 3. Implement the class

All constructor parameters must be stored as `self.param_name` (required for
`clone()` and `_get_params()` to work). Non-constructor state (running
statistics, counters) must be initialised in `__init__` but are **not**
constructor parameters.

```python
from onlinecml.base.base_estimator import BaseOnlineEstimator
from onlinecml.base.running_stats import RunningStats


class MyEstimator(BaseOnlineEstimator):
    """One-line summary.

    Longer description explaining the method.

    Parameters
    ----------
    alpha : float
        Description. Default 1.0.

    Examples
    --------
    >>> from onlinecml.datasets import LinearCausalStream
    >>> m = MyEstimator()
    >>> for x, w, y, _ in LinearCausalStream(n=100, seed=0):
    ...     m.learn_one(x, w, y)
    >>> isinstance(m.predict_ate(), float)
    True
    """

    def __init__(self, alpha: float = 1.0) -> None:
        # Constructor params → stored as self.param
        self.alpha = alpha
        # Non-constructor state
        self._n_seen: int = 0
        self._ate_stats = RunningStats()

    def learn_one(
        self,
        x: dict,
        treatment: int,
        outcome: float,
        propensity: float | None = None,
    ) -> None:
        """Process one observation (predict-then-learn protocol)."""
        # 1. Predict BEFORE updating any model
        # 2. Compute pseudo-outcome
        pseudo = ...
        self._ate_stats.update(pseudo)
        self._n_seen += 1
        # 3. Update any sub-models

    def predict_one(self, x: dict) -> float:
        """Return the CATE estimate for a single unit."""
        return self._ate_stats.mean  # or a model prediction
```

## 4. Register in `__init__.py`

Add the import and `__all__` entry to the subpackage's `__init__.py`:

```python
from onlinecml.reweighting.my_estimator import MyEstimator

__all__ = [..., "MyEstimator"]
```

## 5. Write tests

Create `tests/unit/test_my_estimator.py` with at minimum:

```python
class TestMyEstimator:
    def test_n_seen_starts_zero(self): ...
    def test_predict_ate_before_data_returns_zero(self): ...
    def test_learn_increments_n_seen(self): ...
    def test_predict_one_returns_float(self): ...
    def test_reset_clears_state(self): ...
    def test_clone_gives_fresh_model(self): ...
    def test_ci_finite_after_data(self): ...
```

## 6. Key design rules

- **Predict-then-learn**: always predict *before* updating any model
- **`reset()` works**: `BaseOnlineEstimator.reset()` uses `clone()` internals — don't override it unless necessary
- **`clone()` gives a fresh model**: `n_seen == 0` after cloning
- **No pandas, no batches**: all processing must work one observation at a time
- **O(1) memory**: avoid growing unbounded lists (use `deque(maxlen=...)` for buffers)
- **Thread-safe (read-only)**: `predict_one` should not modify state

## 7. Checklist before PR

- [ ] Docstring on class and all public methods (NumPy style)
- [ ] Type hints on all parameters and return values
- [ ] `learn_one` / `predict_one` / `reset` / `clone` all work
- [ ] Unit tests cover the standard test set above
- [ ] `ruff check` passes
- [ ] `mypy` passes
- [ ] `pytest tests/ --cov` shows ≥ 90% total coverage
