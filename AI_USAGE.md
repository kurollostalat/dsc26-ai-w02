# AI Usage

## Tools I used, and for which functions

I used ChatGPT to understand the project requirements, discuss implementation approaches, and review and debug parts of `rawaj/audit.py` and `rawaj/metrics.py`.

I used AI assistance to work through data-quality checks, including missing-value handling, duplicate detection, timestamp validation, and potential data-leakage detection using ROC AUC. I also used it to discuss threshold-selection logic and understand the evaluation requirements. I ran the provided pytest tests to check my implementations.

## One place where the AI was wrong, incomplete, or not what the spec said

One issue was missing-value handling. Using `notnull()` alone does not identify empty strings or whitespace-only strings as missing, even though the project specification requires treating them as missing. I identified this mismatch by comparing the implementation with the specification and adjusted the missing-value logic accordingly.

I also reviewed the duplicate-ID logic to ensure that missing IDs were excluded rather than mistakenly selected.

## The function I wrote without AI, and why

I independently implemented `confusion_at()`, `precision_recall_at()`, and `expected_cost()` in `metrics.py`, along with the `time_range` calculation in `audit.py`, without using AI to write these parts.

I chose to implement these parts myself to understand the underlying logic. This helped me learn how thresholds affect classification results, how precision and recall are calculated, how false positives and false negatives contribute to expected cost, and how valid timestamps determine a dataset's time range.
