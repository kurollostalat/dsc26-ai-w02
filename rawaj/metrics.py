"""Decision metrics for a mule-risk score.

You write this file. Keep the signatures exactly as they are: the grader calls them by name.

Rules for every function:
- y_true and scores are lists or 1-D arrays. Raise ValueError if the lengths differ, the
  input is empty, a label is not 0 or 1, or a score is NaN.
- A case is flagged when score >= threshold.
- Precision is 0.0 when nothing is flagged; recall is 0.0 when there are no positives.
"""
from __future__ import annotations


def confusion_at(y_true, scores, threshold: float) -> dict:
    
    tp= fp= tn = fn= 0
    for actual, score in zip(y_true , scores):
        if score >= threshold:
            if actual ==1:
                tp += 1
            else: 
                 fp+= 1
        else:
            if actual ==0:
                tn += 1
            else :
                fn +=1
                    
    return  {'tp':tp , 'fp': fp  , 'tn': tn , 'fn': fn} 
    


def precision_recall_at(y_true, scores, threshold: float) -> tuple[float, float]:
    
    precision = recall = 0.0
    confusion = confusion_at(y_true, scores, threshold)
    precision =(confusion['tp']/(confusion['tp']+confusion['fp'])) if (confusion['tp']+confusion['fp'])>0 else 0.0
    recall = (confusion['tp']/(confusion['tp']+confusion['fn'])) if (confusion['tp']+confusion['fn'])>0 else 0.0
    
    return (precision, recall) 



def precision_at_k(y_true, scores, k: int) -> float:
    """Share of positives among the k highest scores."""
    if k <= 0: raise ValueError("k must be greater than 0")
    if k > len(y_true): 
        k = len(y_true)
    sorted_ys= sorted(zip(y_true, scores), key=lambda x: x[1], reverse=True)
    precision = 0.0
    n = len(y_true)
    for i in range(min(k, n)):
        if sorted_ys [i][0] == 1:
            precision += 1
    precision /= min(k, n)   
    
    return precision


def expected_cost(y_true, scores, threshold: float, *, cost_fp: float, cost_fn: float) -> float:
    """fp * cost_fp + fn * cost_fn at `threshold`."""
    confusion = confusion_at(y_true, scores, threshold)
    return confusion['fp'] * cost_fp + confusion['fn'] * cost_fn


def best_threshold(y_true, scores, *, cost_fp: float, cost_fn: float) -> float:
    """Threshold with the lowest expected cost."""
    possible_thresholds = sorted(set(scores) | {float('inf')}, reverse=True)
    best_cost = float('inf')
    for threshold in possible_thresholds:
        cost = expected_cost(y_true, scores, threshold, cost_fp=cost_fp, cost_fn=cost_fn)
        if cost < best_cost:
            best_cost = cost
            best_threshold_value = threshold
    return best_threshold_value
    """Candidates are every distinct score plus math.inf ("flag nothing"). On a tie in cost,
    return the highest threshold (fewer flags for the same cost).
    """
    


def threshold_for_precision(y_true, scores, min_precision: float) -> float | None:
    """Lowest distinct score whose precision is >= min_precision, or None if none reaches it."""
    possible_thresholds = sorted(set(scores))
    for thresholds in  possible_thresholds:
        precision= precision_recall_at(y_true, scores, thresholds)[0]
        if precision >= min_precision:
            return thresholds
    return None   
