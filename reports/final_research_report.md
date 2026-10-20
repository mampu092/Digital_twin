# Cardiovascular Digital Twin - executed research report

Built for the requested Happiest Health Digital Twin Challenge demonstration.
Team and institutional attribution remain user-editable placeholders.

## Objective and evidence

Integrate static clinical context, wearable-associated variables and changing
physiological observations while keeping associations, synthetic deviations and
clinical outcome evidence distinct. All 82,767 rows of the source CSV
were audited; actual width is 32. No genuine longitudinal source
data or future-event labels were available.

## Major data-quality findings

No missing patient IDs, repeated IDs or exact duplicate full rows. There are
23 duplicate clinical profiles under the audit
definition, and 21,071 complete cases. SpO2 is 60.14% missing;
stress/sleep scores 47.99%; NREM HR/RMSSD/respiration 39.27%.
Only 43 NREM HR/RMSSD signatures are observed. No physiological or score unit
was certified from names alone. Source provenance and license were not supplied.
The raw CSV is omitted from the distribution.

## Methods and actual classification results

| target | features | model | roc_auc | pr_auc | sensitivity | specificity | precision | f1 | brier |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| heart_disease | clinical | random_forest | 0.8893 | 0.2584 | 0.8892 | 0.7499 | 0.1442 | 0.2482 | 0.0372 |
| heart_disease | combined | hist_gradient_boosting | 0.8876 | 0.2617 | 0.8652 | 0.7617 | 0.1469 | 0.2511 | 0.0372 |
| hypertension | clinical | hist_gradient_boosting | 0.7809 | 0.2341 | 0.7657 | 0.6586 | 0.1839 | 0.2966 | 0.0757 |
| hypertension | combined | lightgbm | 0.7806 | 0.2331 | 0.7591 | 0.6668 | 0.1863 | 0.2992 | 0.0757 |

Both targets are recorded-status classification. Train-only preprocessing,
patient/profile splits, grouped randomized CV, independent calibration and untouched
test evaluation implement the real-data track. Balanced-accuracy thresholding gives
high sensitivity at low precision; a clinical alert policy has not been validated.

## Wearable incremental value

- heart_disease, fixed hist_gradient_boosting: combined-minus-clinical ROC-AUC -0.0017, 95% CI -0.0035 to -0.0002; AP +0.0008, 95% CI -0.0072 to +0.0093.
- hypertension, fixed lightgbm: combined-minus-clinical ROC-AUC -0.0008, 95% CI -0.0017 to -0.0001; AP -0.0018, 95% CI -0.0047 to +0.0005.

The small internal changes do not establish prospective or clinical value. A
source-signature holdout, missingness-only model and complete-case comparison
investigate sensitivity; none repairs absent provenance.

## Synthetic temporal and fusion experiments

The robust patient-relative detector uses first-seven-night medians and MAD scales,
with minimum noise floors, then scores the two largest absolute standardized
deviations. Synthetic scenarios create correlated multivariate shifts, autoregressive
noise, random missingness and varied baselines.
Independent synthetic test performance: AUC 0.9568,
F1 0.9133, specificity 0.9623.
These quantify recovery of injected perturbations, not cardiac events.

The EHR embedding + GRU/attention + gated fusion model executes on CPU, supports
CUDA when available, and has two heads. Ten training epochs were run with an
independent tuning subset; the checkpoint was selected by tuning loss. High
synthetic sequence AUC (0.999920) is
expected for strong simulator shifts and is not clinical evidence.

## Working system

The twin validates observation identity, provenance, ranges and strict chronology;
preserves a static profile and personal reference; updates a smoothed state;
tracks missingness and heuristic process uncertainty; persists to SQLite;
executes noncausal scenarios; explains predictions; and exports a patient PDF.
Streamlit provides eight pages and Plotly charts, including the version 0.2 loneliness and BP research extension. Screenshots are captured from
the real running application using an invented patient.

## Interpretation and next research steps

Biological hypotheses are in docs/clinical_hypotheses.md with primary NHLBI sources.
Neither attention, SHAP nor reference substitutions identify causal pathways.
First verify source cohorts, consent/license, source donor IDs, device units,
score direction and merge code. Then collect prospective linked sensor/EHR data
with event adjudication and timestamps; pre-register endpoints and horizons;
use patient/donor/time-separated development; test independent sites and devices;
assess calibration, sensitivity, specificity, alert burden and subgroup utility;
validate uncertainty and interventions with appropriate study designs; and obtain
clinical, privacy, security and regulatory review before any patient-care use.

The current system is an executable research prototype. Genuine event forecasting,
causal treatment modeling, secure clinical deployment and external validation
remain unimplemented because the necessary data and evidence are unavailable.

Source and methods receipts: reports/audit_summary.json,
reports/feature_dictionary.csv, outputs/metrics/*.json and *.csv, source scripts,
and three inspectable companion notebooks.
