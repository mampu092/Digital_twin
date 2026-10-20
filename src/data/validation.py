"""Reproducible, aggregate-only audit. Associations do not establish provenance."""
import hashlib
import html
import numpy as np
import pandas as pd
from src.common import ROOT, dump_json
from src.data.loader import WEARABLE, TARGETS


def audit_dataset(df, path):
    out = ROOT / "reports"
    missing = pd.DataFrame({"column": df.columns, "missing_count": df.isna().sum().values,
                            "missing_fraction": df.isna().mean().values})
    missing.to_csv(out / "missingness.csv", index=False)
    balance = pd.concat([df[c].value_counts().rename_axis("value").reset_index(name="n").assign(target=c)
                         for c in TARGETS + ["stroke", "diabetes"]], ignore_index=True)
    balance["fraction"] = balance.n / len(df)
    balance.to_csv(out / "class_balance.csv", index=False)
    dictionary = []
    for c in df:
        numeric = pd.api.types.is_numeric_dtype(df[c])
        dictionary.append(dict(column=c, dtype=str(df[c].dtype), unique=df[c].nunique(),
            missing_fraction=df[c].isna().mean(), minimum=df[c].min() if numeric else "",
            maximum=df[c].max() if numeric else "",
            role="identifier" if c == "patient_id" else "target" if c in TARGETS else
                 "excluded_unknown_derived" if c == "bmi_gap" else "candidate_predictor",
            provenance="not supplied", units_and_scale="unverified; do not infer device units" if c in WEARABLE else
            "coding unverified; name suggests " + c, interpretation="no measurement time or source metadata"))
    pd.DataFrame(dictionary).to_csv(out / "feature_dictionary.csv", index=False)
    df.describe(include="all").T.to_csv(out / "descriptive_statistics.csv")
    cat = {c: {str(k): int(v) for k,v in df[c].value_counts(dropna=False).items()}
           for c in df if df[c].nunique() < 20}
    dump_json(cat, out / "categorical_distributions.json")
    subgroups = []
    for group in TARGETS + ["gender", "diabetes"]:
        for value, part in df.groupby(group):
            for col in WEARABLE:
                subgroups.append(dict(group=group, value=value, column=col, n=len(part),
                                      missing_fraction=part[col].isna().mean()))
    pd.DataFrame(subgroups).to_csv(out / "missingness_by_subgroup.csv", index=False)
    corr = df.select_dtypes("number").drop(columns="patient_id").corr()
    corr.to_csv(out / "correlations.csv")
    associations = []
    for target in TARGETS:
        for c in WEARABLE:
            associations.append(dict(target=target, feature=c,
                correlation=corr.loc[c, target],
                distinct_observed=df[c].nunique(),
                missing_target0=df.loc[df[target] == 0, c].isna().mean(),
                missing_target1=df.loc[df[target] == 1, c].isna().mean(),
                mean_target0=df.loc[df[target] == 0, c].mean(),
                mean_target1=df.loc[df[target] == 1, c].mean()))
    assoc = pd.DataFrame(associations)
    assoc.to_csv(out / "wearable_label_associations.csv", index=False)
    ranges = {"age": (0,120), "bmi": (5,100), "sleep_hours": (0,24),
              "bad_mental_days": (0,30), "bad_physical_days": (0,30)}
    failures = {c:int((~df[c].between(*bounds) & df[c].notna()).sum()) for c,bounds in ranges.items()}
    profile_cols = [c for c in df if c != "patient_id"]
    clinical_cols = [c for c in profile_cols if c not in WEARABLE]
    summary = dict(rows=len(df), columns=len(df.columns), unique_patients=df.patient_id.nunique(),
        repeated_ids=int(df.patient_id.duplicated().sum()), duplicate_rows=int(df.duplicated().sum()),
        duplicate_profiles=int(df[profile_cols].duplicated().sum()),
        duplicate_clinical_profiles=int(df[clinical_cols].duplicated().sum()),
        timestamps=[c for c in df if "timestamp" in c.lower() or c.lower().endswith("_date")],
        complete_cases=int(df.dropna().shape[0]), range_failures=failures,
        source_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
        raw_redistribution="unknown: excluded from distributable ZIP",
        wearable_unique_values={c:int(df[c].nunique()) for c in WEARABLE})
    dump_json(summary, out / "audit_summary.json")
    limits = f"""# Dataset limitations

Processed {len(df):,} rows and {len(df.columns)} columns (the supplied brief says 33).
There are {df.patient_id.nunique():,} unique patient IDs, no verified timestamps,
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
{summary['complete_cases']:,} rows have no missing values; complete-case comparisons are selection-biased.
No external validation or temporal transportability has been established.
Numeric IDs do not establish anonymity. Raw data, row-level predictions, and patient
split manifests are private local outputs and are excluded from the public-ready ZIP.
"""
    (out / "dataset_limitations.md").write_text(limits, encoding="utf-8")
    leakage = f"""# Leakage investigation

Patient ID and bmi_gap are excluded. Both heart_disease and hypertension are excluded
from predictors, except each target is used as its own outcome. Other historical
conditions remain phenotype correlates, not known pre-outcome predictors.
All learned preprocessing, tuning and feature references use the training partition.
Exact matching predictor profiles and repeated patient IDs are assigned together.
Labels with unknown timing prohibit prospective claims regardless of model accuracy.

Full-data descriptive correlations are diagnostic only; they are not used to select
features or hyperparameters. See correlations.csv and wearable_label_associations.csv.
Distinct observed wearable values: {summary['wearable_unique_values']}.
Shared missingness is examined by disease, gender and diabetes. Low cardinality and
high reuse cannot establish authentic patient-device linkage. No source records are
available to confirm or refute target-conditioned generation.

Required next evidence: source cohort and Fitbit donor IDs, collection timestamps,
merge code and feature-generation code; then split by every source patient/donor.
Current patient grouping cannot prevent unknown donor overlap. Treat wearable
ablation results as internal associations only. Independent sensor validation is required.
"""
    (out / "leakage_audit.md").write_text(leakage, encoding="utf-8")
    body = "<h1>Cardiovascular Digital Twin: data audit</h1>"
    body += f"<p>{len(df):,} records | {len(df.columns)} columns | Cross-sectional | Source provenance unverified</p>"
    body += "<h2>Completeness</h2>" + missing.to_html(index=False)
    body += "<h2>Labels</h2>" + balance.to_html(index=False)
    body += "<h2>Feature dictionary</h2>" + pd.DataFrame(dictionary).to_html(index=False)
    body += "<h2>Scientific limitations</h2><pre>" + html.escape(limits) + "</pre>"
    (out / "data_audit.html").write_text("<!doctype html><meta charset='utf-8'><style>"
        "body{font:15px system-ui;margin:40px;color:#17354b}table{border-collapse:collapse}"
        "td,th{padding:6px;border:1px solid #cddce5}pre{white-space:pre-wrap}</style>" + body, encoding="utf-8")
    return summary
