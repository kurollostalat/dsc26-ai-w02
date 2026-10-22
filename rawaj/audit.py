"""Data audit for Rawaj Pay transfer extracts.

You write this file. Keep the signature exactly as it is: the grader calls it by name.
Allowed dependencies: numpy, pandas, scikit-learn (and the standard library).
"""
from __future__ import annotations


import pandas as pd

def calculate_auc(y_true, scores):
    pairs = sorted(zip(scores, y_true))
    n_positive = sum(y == 1 for y in y_true)
    n_negative = sum(y == 0 for y in y_true)

    if n_positive == 0 or n_negative == 0:
        return None

    rank_sum = 0
    i = 0

    while i < len(pairs):
        j = i
        while j < len(pairs) and pairs[j][0] == pairs[i][0]:
            j += 1

        average_rank = ((i + 1) + j) / 2

        for k in range(i, j):
            if pairs[k][1] == 1:
                rank_sum += average_rank

        i = j

    return (
        rank_sum - n_positive * (n_positive + 1) / 2
    ) / (n_positive * n_negative)

def audit(df: pd.DataFrame, *, target: str, id_col: str, time_col: str) -> dict:
  n_rows = len(df)
  
  missing_mask = df.apply(
    lambda col: col.apply(
        lambda value: pd.isnull(value)
        or (isinstance(value, str) and value.strip() == "")
    )
)
  missing = missing_mask.mean().to_dict()
  
  ids = df.loc[~missing_mask[id_col], id_col]
  duplicate_ids = sorted(
      ids[ids.duplicated(keep=False)].unique().tolist()
  )
  
  duplicate_rows= df.duplicated().sum()
  
  constant_columns= sorted([col for col in df.columns if col != target and df[col].nunique() <= 1])
  
  unlabelled_rows=df[target].apply(
    lambda value: pd.isnull(value) or (isinstance(value, str) and value.strip() == "")
  ).sum()
  
  labelled_mask = df[target].apply(
    lambda value: not (
        pd.isnull(value) or
        (isinstance(value, str) and value.strip() == "")
    )
)

  labelled_rows = df[labelled_mask]
  
  
  positive_rate = (
      (labelled_rows[target] == 1).sum() / len(labelled_rows)
      if len(labelled_rows) > 0
      else 0.0
  )
  
  parsed_times = pd.to_datetime(df[time_col], errors='coerce')

  bad_timestamps = (
    parsed_times.isnull() & ~missing_mask[time_col]
).sum()
  
  valid_times = parsed_times.dropna()

  time_range = [
      valid_times.min().isoformat(),
      valid_times.max().isoformat()
  ] if not valid_times.empty else [None, None]
  
  leakage_suspects = []
  candidate_columns = [
      col for col in df.columns
      if col not in [target, id_col, time_col]
  ]
  
  can_check_leakage = (
    len(labelled_rows) >= 20
    and labelled_rows[target].nunique() == 2
)
  
  if can_check_leakage:
    y = labelled_rows[target]

    for col in candidate_columns:
        if df[col].nunique(dropna=False) <= 1:
            continue

        # Check whether missingness predicts the target
        missing_indicator = missing_mask.loc[labelled_rows.index, col].astype(int)

        if missing_indicator.nunique() > 1:
            missing_auc = calculate_auc(y.tolist(), missing_indicator.tolist())

            if missing_auc >= 0.98 or missing_auc <= 0.02:
                leakage_suspects.append(col)
                continue

        # Check numeric and boolean columns
        if (
            pd.api.types.is_numeric_dtype(df[col])
            or pd.api.types.is_bool_dtype(df[col])
        ):
            values = labelled_rows[col].dropna()

            if len(values) >= 20:
                y_valid = labelled_rows.loc[values.index, target]

                if y_valid.nunique() == 2:
                    auc = calculate_auc(y_valid.tolist(), values.tolist())

                    if auc >= 0.98 or auc <= 0.02:
                        leakage_suspects.append(col)

  leakage_suspects = sorted(set(leakage_suspects))
  return {
    "n_rows": n_rows,
    "missing": missing,
    "duplicate_ids": duplicate_ids,
    "duplicate_rows": duplicate_rows,
    "constant_columns": constant_columns,
    "unlabelled_rows": unlabelled_rows,
    "positive_rate": positive_rate,
    "bad_timestamps": bad_timestamps,
    "time_range": time_range,
    "leakage_suspects": leakage_suspects
  }
  """Audit a transfer extract before anyone trains a model on it.

    Raises KeyError if `target`, `id_col` or `time_col` is not a column of `df`.

    Returns a dict with exactly these keys:

    - "n_rows" (int): number of rows.
    - "missing" ({column: float}): fraction missing, for every column. Missing means
      NaN/None, or a string that is empty or only whitespace.
    - "duplicate_ids" (list): sorted non-missing `id_col` values that appear more than once.
    - "duplicate_rows" (int): rows that exactly repeat an earlier row.
    - "constant_columns" (list): sorted columns other than `target` with at most one
      distinct non-missing value.
    - "unlabelled_rows" (int): rows where `target` is missing.
    - "positive_rate" (float): share of labelled rows with target == 1; 0.0 if none is labelled.
    - "bad_timestamps" (int): non-missing values in `time_col` that do not parse as dates.
    - "time_range" (list): [min, max] of the parsed timestamps as ISO strings, or [None, None].
    - "leakage_suspects" (list): sorted columns (not target, id_col or time_col) that look
      like leaks. See README.md for the exact rule.
    """
    
