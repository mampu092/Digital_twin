# Validation findings and clinical status

## Track A - Cross-sectional classification

| target | features | model | roc_auc | pr_auc | sensitivity | specificity | precision | f1 | brier |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| heart_disease | clinical | random_forest | 0.8893 | 0.2584 | 0.8892 | 0.7499 | 0.1442 | 0.2482 | 0.0372 |
| heart_disease | combined | hist_gradient_boosting | 0.8876 | 0.2617 | 0.8652 | 0.7617 | 0.1469 | 0.2511 | 0.0372 |
| hypertension | clinical | hist_gradient_boosting | 0.7809 | 0.2341 | 0.7657 | 0.6586 | 0.1839 | 0.2966 | 0.0757 |
| hypertension | combined | lightgbm | 0.7806 | 0.2331 | 0.7591 | 0.6668 | 0.1863 | 0.2992 | 0.0757 |

Actual partitions: {'calibration': 8275, 'test': 16546, 'train': 49668, 'tune': 8278}. Test patients remain untouched by preprocessing,
model selection, calibration and operating-threshold choice. Exact clinical
predictor profiles and repeated patient IDs are grouped. Unknown source donor
overlap cannot be eliminated without source metadata.

Additional evidence: bootstrap_ci95.json; subgroup_metrics.csv (age/gender, at least
100 rows and 20 positives); missingness_sensitivity.csv; train_test_distribution.csv;
source_signature_holdout.csv; matched_family_ablation.json.
Single-split evaluation and small hyperparameter search limit precision and stability.

## Track B - Synthetic temporal evaluation

180 independent synthetic patients, 4137
scored observations after enrollment, coverage 99.93%.
ROC-AUC 0.9568; sensitivity
0.8525; specificity 0.9623;
F1 0.9133.
The fixed index threshold is 2.5. Bootstrap resampling uses whole synthetic patients.
The monotonic index-to-[0,1] mapping is not a calibrated risk probability.
The synthetic positive prevalence is designed and unrelated to clinical prevalence.

The fusion network was trained on 4096 real static
rows paired with independent synthetic sequences; tune 768,
test 1024. Its disease-head ROC-AUC is
0.8676 on this subset; synthetic sequence-anomaly
ROC-AUC is 0.999920.
Near-perfect sequence discrimination reflects strong engineered shifts across long
windows and an easier sequence-level label. It is not comparable to clinical event
prediction or to per-observation anomaly detection. The disease head is uncalibrated.

## Track C - System behavior

Test runs (counts are parsed from actual JUnit outputs): {"core_tests": {"tests": "23", "failures": "0", "errors": "0", "skipped": "0", "time": "15.673"}, "dashboard_tests": {"tests": "2", "failures": "0", "errors": "0", "skipped": "0", "time": "10.430"}, "final_tests": {"tests": "45", "failures": "0", "errors": "0", "skipped": "0", "time": "23.858"}}.
The suite covers training-only imputation, grouping, artifact inference, neural
gradients, missing sensors, invalid chronology/identity/ranges, idempotent state
updates, SQLite round trips, intervention isolation, PDF content and all seven UI pages.
PDF rendering and live-browser visual inspection complement automated assertions.

## Not established

No external, prospective, interventional, multi-site or real wearable temporal
validation has been performed. No medical-device certification, clinical utility,
causal effect, real-world alert burden or individual uncertainty guarantee is claimed.
Python 3.12 on Windows CPU was executed. Python 3.11, Linux/macOS, Docker and CUDA
are documented paths but have not been run in this session.
