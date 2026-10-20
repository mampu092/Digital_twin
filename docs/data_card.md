# Data card

Source: user-supplied `FINAL_dataset_with_fitbit.csv`, inspected locally.
Audit fingerprint: `reports/audit_summary.json`.
Size: 82,767 records, 32 columns; 82,767 distinct patient IDs.
No timestamps, no known repeated sensor observations, no incident cardiac outcome.
The two outcome columns are recorded heart disease and hypertension.

No source paper, cohort provenance, codebook, consent, merger script, or licensing
statement was supplied. The user requested continuing without those materials.
The system does not assert legal permission, anonymity, valid source linkage, or
that these wearable values are real device observations.

Patient ID and unknown-derived bmi_gap are excluded. Both target fields are
excluded from predictors for either task. All other selected covariates remain
subject to unknown temporal ordering and source meaning.

Missing rows are retained using training-only median/mode imputation, numeric
missingness indicators, scaling and categorical encoding inside sklearn pipelines.
No target-conditioned imputation or pre-split oversampling is used.
Complete-case and missingness-only analyses are secondary sensitivity checks.

Wearable fields show very low cardinality relative to row count. NREM HR/RMSSD
signatures are used for an additional grouped sensitivity test, but inferred
signatures cannot substitute for actual donor identifiers. Unknown donor reuse
can invalidate ordinary patient-level wearable generalization claims.

The distributable omits raw data, row-level predictions and private split manifests.
Aggregate metrics and trained models are included for reproducible local inference.
Their redistribution terms still depend on unknown source terms. No membership-
inference or model inversion privacy guarantee has been established.
