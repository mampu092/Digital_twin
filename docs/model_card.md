# Model card

Intended use: local research and competition demonstration of recorded-status
classification, synthetic anomaly detection and hypothetical input scenarios.
Prohibited interpretation: imminent-event prediction, clinical diagnosis, causal
treatment benefit, medication selection or autonomous patient triage.

Primary target: heart_disease; secondary target: hypertension. Binary recorded
labels lack timing. Features exclude both targets, patient_id and bmi_gap.
Median/mode imputation, numeric indicators, standardization and one-hot encoding
are fit in training pipelines. Class weighting is used for logistic, random forest,
HistGradientBoosting and LightGBM. MLP uses its native unweighted loss with early
stopping; AP selection and thresholding address its imbalance tradeoff.

| target | features | model | roc_auc | pr_auc | sensitivity | specificity | precision | f1 | brier |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| heart_disease | clinical | random_forest | 0.8893 | 0.2584 | 0.8892 | 0.7499 | 0.1442 | 0.2482 | 0.0372 |
| heart_disease | combined | hist_gradient_boosting | 0.8876 | 0.2617 | 0.8652 | 0.7617 | 0.1469 | 0.2511 | 0.0372 |
| hypertension | clinical | hist_gradient_boosting | 0.7809 | 0.2341 | 0.7657 | 0.6586 | 0.1839 | 0.2966 | 0.0757 |
| hypertension | combined | lightgbm | 0.7806 | 0.2331 | 0.7591 | 0.6668 | 0.1863 | 0.2992 | 0.0757 |

See reports/model_comparison.md for intervals, threshold tradeoffs and matched
ablation. Stored artifacts: four calibrated baseline bundles, fusion.pt,
fusion_transform.joblib. Only load trusted joblib files; they can execute code.

Limitations: unknown label timing, observational confounding, low-cardinality
wearables, unverified patient linkage, missingness, unknown scales, no external
cohort, limited gender representation, single split and modest search budget.
Training references and min/max checks are not a full multivariate OOD detector.
Data-rights verification remains required before publishing weights.
