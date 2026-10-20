# Dataset limitations

Processed 82,767 rows and 32 columns (the supplied brief says 33).
There are 82,767 unique patient IDs, no verified timestamps,
and no incident-event label. Each row is a cross-sectional record.
Existing heart disease and hypertension status cannot establish prospective event risk.

No data dictionary, source cohort citation, linkage method, consent documentation,
device export, or redistribution license accompanied the CSV. Units and score direction
are unverified. In particular nightly_temperature is NOT established core temperature,
and a higher stress_score is NOT established to mean more stress.
sleep_points_percentage has a fractional observed range, not a confirmed percentage scale.
TENSE/ANXIOUS, TIRED and ENTERTAINMENT are numeric aggregates with unknown semantics.
bmi_gap is excluded as an unexplained derived/matching field.

Wearable fields contain repeated values and shared missingness patterns. These may
reflect aggregation, donor reuse, merging, imputation or synthetic assignment. Numerical
inspection cannot verify which process was used or rule out label-conditioned generation.
No causal or clinical interpretation of their incremental performance is warranted.
21,071 rows have no missing values; complete-case comparisons are selection-biased.
No external validation or temporal transportability has been established.
Numeric IDs do not establish anonymity. Raw data, row-level predictions, and patient
split manifests are private local outputs and are excluded from the public-ready ZIP.
