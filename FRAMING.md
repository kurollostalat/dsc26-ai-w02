# Framing: Rawaj Pay Mule Detection

## Decision

The model should help Rawaj Pay's fraud and risk team prioritize suspicious transactions and accounts for investigation. High-risk cases should be reviewed by analysts rather than automatically treated as confirmed fraud. False positives waste investigation time and may inconvenience legitimate customers, while false negatives allow fraudulent activity and money-mule accounts to go undetected. The team's daily investigation capacity and the actual costs of these errors must be established with the client before selecting a final threshold.

## Metric

Precision and recall are both important for this decision. Precision measures how many flagged cases are genuinely suspicious, helping the team use its investigation capacity efficiently, while recall measures how many actual positive cases are detected. Accuracy alone can be misleading when fraudulent cases are rare, because a model can appear accurate while missing most fraud. The threshold should be selected based on the trade-off between precision, recall, false-positive costs, false-negative costs, and the number of cases analysts can review.

## Baseline

Before deploying the model, Rawaj Pay should compare its performance against the existing fraud-review process or current rules, if available. If historical decisions and outcomes are available, they can help establish the current detection rate, false-positive burden, and investigation workload. The model must demonstrate measurable improvement over this baseline using the client's priorities, while maintaining a manageable review volume and avoiding unnecessary disruption to legitimate customers. Any comparison should use consistent evaluation data and clearly documented assumptions.

## Classical ML or LLM

<!-- Classical model, LLM, a rule, or a mix? Argue from the decision table in the session. -->
