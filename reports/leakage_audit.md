# Leakage investigation

Patient ID and bmi_gap are excluded. Both heart_disease and hypertension are excluded
from predictors, except each target is used as its own outcome. Other historical
conditions remain phenotype correlates, not known pre-outcome predictors.
All learned preprocessing, tuning and feature references use the training partition.
Exact matching predictor profiles and repeated patient IDs are assigned together.
Labels with unknown timing prohibit prospective claims regardless of model accuracy.

Full-data descriptive correlations are diagnostic only; they are not used to select
features or hyperparameters. See correlations.csv and wearable_label_associations.csv.
Distinct observed wearable values: {'nightly_temperature': 64, 'nremhr': 43, 'rmssd': 43, 'spo2': 28, 'full_sleep_breathing_rate': 43, 'stress_score': 37, 'sleep_points_percentage': 37, 'TENSE/ANXIOUS': 44, 'TIRED': 51, 'ENTERTAINMENT': 34}.
Shared missingness is examined by disease, gender and diabetes. Low cardinality and
high reuse cannot establish authentic patient-device linkage. No source records are
available to confirm or refute target-conditioned generation.

Required next evidence: source cohort and Fitbit donor IDs, collection timestamps,
merge code and feature-generation code; then split by every source patient/donor.
Current patient grouping cannot prevent unknown donor overlap. Treat wearable
ablation results as internal associations only. Independent sensor validation is required.
