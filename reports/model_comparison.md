# Model comparison

Five successfully trained families: logistic regression, random forest,
HistGradientBoosting, multilayer perceptron, and LightGBM. Each was trained for
both targets with clinical-only and combined features (20 searches, two candidates
and two grouped CV folds each). Only the four selected calibrated pipelines are
retained as deployable baseline artifacts. The best fusion checkpoint is separate.

## Held-out test results

| target | features | model | roc_auc | pr_auc | sensitivity | specificity | precision | f1 | brier |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| heart_disease | clinical | random_forest | 0.8893 | 0.2584 | 0.8892 | 0.7499 | 0.1442 | 0.2482 | 0.0372 |
| heart_disease | combined | hist_gradient_boosting | 0.8876 | 0.2617 | 0.8652 | 0.7617 | 0.1469 | 0.2511 | 0.0372 |
| hypertension | clinical | hist_gradient_boosting | 0.7809 | 0.2341 | 0.7657 | 0.6586 | 0.1839 | 0.2966 | 0.0757 |
| hypertension | combined | lightgbm | 0.7806 | 0.2331 | 0.7591 | 0.6668 | 0.1863 | 0.2992 | 0.0757 |

PR-AUC is implemented as average precision, not trapezoidal interpolation.
Every reported score is for recorded disease status, not a future event.
Test prevalence: heart disease 4.53%;
hypertension 9.13%.
Thresholds optimize tuning balanced accuracy, which creates substantial false-positive
burden and low precision at this prevalence. They are not clinically approved cutoffs.

## Uncertainty

- heart_disease, clinical: ROC-AUC 0.8893 (95% CI 0.8793-0.8984); average precision 0.2584 (95% CI 0.2241-0.2894).
- heart_disease, combined: ROC-AUC 0.8876 (95% CI 0.8756-0.8977); average precision 0.2617 (95% CI 0.2281-0.3000).
- hypertension, clinical: ROC-AUC 0.7809 (95% CI 0.7708-0.7922); average precision 0.2341 (95% CI 0.2166-0.2510).
- hypertension, combined: ROC-AUC 0.7806 (95% CI 0.7696-0.7915); average precision 0.2331 (95% CI 0.2136-0.2491).

Intervals use 200 percentile bootstrap resamples of patient/profile clusters,
conditional on the fitted model and fixed threshold. They do not capture training
variation, source uncertainty, donor reuse or transportability. Individual
prediction confidence intervals are not available.

## Ablation

Separately optimized pipelines give the results in `outputs/metrics/ablation.json`.
Their choice of algorithm differs and can confound a feature-only comparison.
The secondary matched-family analysis holds algorithm and hyperparameters fixed:

- heart_disease, fixed hist_gradient_boosting: combined-minus-clinical ROC-AUC -0.0017, 95% CI -0.0035 to -0.0002; AP +0.0008, 95% CI -0.0072 to +0.0093.
- hypertension, fixed lightgbm: combined-minus-clinical ROC-AUC -0.0008, 95% CI -0.0017 to -0.0001; AP -0.0018, 95% CI -0.0047 to +0.0005.

No reliable clinical benefit follows from these small internal differences. The
source-linkage and semantic concerns remain even when an interval excludes zero.
See source_signature_holdout.csv for a separate 43-signature sensitivity analysis.
That secondary analysis reuses primary hyperparameters; it is not an external or
nested donor-validation result. No labels were fabricated for these tests.

## Tuning results

| target | features | model | validation_pr_auc | training_cv_pr_auc |
| --- | --- | --- | --- | --- |
| heart_disease | clinical | logistic | 0.2942 | 0.2711 |
| heart_disease | clinical | random_forest | 0.3056 | 0.2654 |
| heart_disease | clinical | hist_gradient_boosting | 0.3037 | 0.2674 |
| heart_disease | clinical | mlp | 0.2949 | 0.2692 |
| heart_disease | clinical | lightgbm | 0.2940 | 0.2668 |
| heart_disease | combined | logistic | 0.2947 | 0.2697 |
| heart_disease | combined | random_forest | 0.2841 | 0.2587 |
| heart_disease | combined | hist_gradient_boosting | 0.2977 | 0.2624 |
| heart_disease | combined | mlp | 0.2620 | 0.2250 |
| heart_disease | combined | lightgbm | 0.2949 | 0.2640 |
| hypertension | clinical | logistic | 0.2287 | 0.2216 |
| hypertension | clinical | random_forest | 0.2329 | 0.2238 |
| hypertension | clinical | hist_gradient_boosting | 0.2361 | 0.2228 |
| hypertension | clinical | mlp | 0.2247 | 0.2119 |
| hypertension | clinical | lightgbm | 0.2357 | 0.2254 |
| hypertension | combined | logistic | 0.2287 | 0.2206 |
| hypertension | combined | random_forest | 0.2230 | 0.2213 |
| hypertension | combined | hist_gradient_boosting | 0.2311 | 0.2224 |
| hypertension | combined | mlp | 0.2235 | 0.1922 |
| hypertension | combined | lightgbm | 0.2347 | 0.2256 |

## Calibration and explainability

Independent sigmoid calibration follows model selection. Calibration plots and
Brier scores are saved for every selected feature-set/target pair.
Decision curves use illustrative probability thresholds and recorded labels; no
clinical net-benefit claim is warranted. Permutation importance uses 1,500 held-out
rows and three repeats without retuning. Tree SHAP uses 150 rows where supported.
SHAP explains raw tree output, not calibrated probability. Patient explanations
use one-at-a-time training-reference replacement and are non-additive/noncausal.
