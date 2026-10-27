"""Decision metrics for a mule-risk score.

You write this file. Keep the signatures exactly as they are: the grader calls them by name.

Rules for every function:
- y_true and scores are lists or 1-D arrays. Raise ValueError if the lengths differ, the
  input is empty, a label is not 0 or 1, or a score is NaN.
- A case is flagged when score >= threshold.
- Precision is 0.0 when nothing is flagged; recall is 0.0 when there are no positives.
"""
from __future__ import annotations

import math

def _validate_inputs(y_true, scores):
    y_true = list(y_true)
    scores = list(scores)

    if not y_true or not scores or len(y_true) != len(scores):
        raise ValueError("Inputs must have the same non-zero length")

    if any(label not in (0, 1) for label in y_true):
        raise ValueError("Labels must be 0 or 1")

    if any(math.isnan(float(score)) for score in scores):
        raise ValueError("Scores must not contain NaN")

    return y_true, scores


def confusion_at(y_true, scores, threshold: float) -> dict:
    y_true, scores = _validate_inputs(y_true, scores)

    tp = fp = tn = fn = 0

    for actual, score in zip(y_true, scores):
        if score >= threshold:
            if actual == 1:
                tp += 1
            else:
                fp += 1
        else:
            if actual == 0:
                tn += 1
            else:
                fn += 1

    return {'tp': tp, 'fp': fp, 'tn': tn, 'fn': fn}


    



def precision_recall_at(
    y_true, scores, threshold: float
) -> tuple[float, float]:
    y_true, scores = _validate_inputs(y_true, scores)

    confusion = confusion_at(y_true, scores, threshold)

    tp = confusion['tp']
    fp = confusion['fp']
    fn = confusion['fn']

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0

    return precision, recall


def precision_at_k(y_true, scores, k: int) -> float:
    """Share of positives among the k highest scores."""
    y_true, scores = _validate_inputs(y_true, scores)

    if k <= 0:
        raise ValueError("k must be greater than 0")

    k = min(k, len(y_true))

    sorted_ys = sorted(
        zip(y_true, scores),
        key=lambda x: x[1],
        reverse=True
    )

    positives = sum(label == 1 for label, _ in sorted_ys[:k])

    return positives / k



def expected_cost(y_true, scores, threshold: float, *, cost_fp: float, cost_fn: float) -> float:
    """fp * cost_fp + fn * cost_fn at `threshold`."""
    confusion = confusion_at(y_true, scores, threshold)
    return confusion['fp'] * cost_fp + confusion['fn'] * cost_fn




def best_threshold(y_true, scores, *, cost_fp: float, cost_fn: float) -> float:
    """Return the highest threshold among those with the lowest expected cost."""
    y_true, scores = _validate_inputs(y_true, scores)

    possible_thresholds = sorted(
        set(scores) | {float('inf')},
        reverse=True
    )

    best_cost = float('inf')
    best_threshold_value = possible_thresholds[0]

    for threshold in possible_thresholds:
        cost = expected_cost(
            y_true,
            scores,
            threshold,
            cost_fp=cost_fp,
            cost_fn=cost_fn
        )

        if cost < best_cost:
            best_cost = cost
            best_threshold_value = threshold

    return best_threshold_value



def threshold_for_precision(y_true, scores, min_precision: float) -> float | None:
    """Lowest distinct score whose precision is >= min_precision, or None if none reaches it."""
    y_true, scores = _validate_inputs(y_true, scores)

    possible_thresholds = sorted(set(scores))

    for threshold in possible_thresholds:
        precision = precision_recall_at(y_true, scores, threshold)[0]
        if precision >= min_precision:
            return threshold

    return None
