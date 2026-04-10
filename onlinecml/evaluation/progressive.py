"""Progressive causal evaluation: predict-before-learn scoring over a stream."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from onlinecml.base.base_estimator import BaseOnlineEstimator


def progressive_causal_score(
    stream,  # noqa: ANN001
    model: "BaseOnlineEstimator",
    metrics: list,
    step: int = 100,
) -> dict[str, Any]:
    """Evaluate a causal model progressively over a streaming dataset.

    Implements the predict-before-learn protocol: for each observation, the
    model is scored **before** it sees the label, then updated. This gives an
    unbiased estimate of online generalisation performance.

    At every ``step`` observations, each metric's current ``score`` is recorded.

    Parameters
    ----------
    stream : iterable of (x, treatment, outcome, true_cate)
        Any OnlineCML dataset or iterable yielding 4-tuples.
    model : BaseOnlineEstimator
        An unfitted (or partially fitted) causal estimator. Must implement
        ``learn_one(x, treatment, outcome)`` and ``predict_one(x)``.
    metrics : list
        List of metric objects (e.g., ``[ATEError(), PEHE()]``). Each must
        implement ``update(x, w, y, true_cate, cate_hat, model)`` and a
        ``score`` property.
    step : int
        Record metric scores every ``step`` observations. Default 100.

    Returns
    -------
    results : dict
        Dictionary with key ``"steps"`` (list of checkpoint indices) and one
        key per metric class name (list of scores at each checkpoint).

    Examples
    --------
    >>> from onlinecml.datasets import LinearCausalStream
    >>> from onlinecml.reweighting import OnlineIPW
    >>> from onlinecml.evaluation import progressive_causal_score
    >>> from onlinecml.evaluation.metrics import ATEError, PEHE
    >>>
    >>> results = progressive_causal_score(
    ...     stream  = LinearCausalStream(n=500, seed=0),
    ...     model   = OnlineIPW(),
    ...     metrics = [ATEError(), PEHE()],
    ...     step    = 100,
    ... )
    >>> len(results["steps"])
    5
    """
    history: dict[str, list[float]] = {m.__class__.__name__: [] for m in metrics}
    steps_list: list[int] = []

    for i, (x, w, y, true_cate) in enumerate(stream):
        # Predict-before-learn
        cate_hat = model.predict_one(x)

        # Update each metric with the current prediction
        for m in metrics:
            m.update(x, w, y, true_cate, cate_hat, model)

        # Train the model
        model.learn_one(x, w, y)

        # Record checkpoint
        if (i + 1) % step == 0:
            steps_list.append(i + 1)
            for m in metrics:
                history[m.__class__.__name__].append(m.score)

    return {"steps": steps_list, **history}
