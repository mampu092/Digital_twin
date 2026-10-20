import os
import hashlib
from pathlib import Path
import pandas as pd
from src.common import ROOT

TARGETS = ["heart_disease", "hypertension"]
WEARABLE = ["nightly_temperature", "nremhr", "rmssd", "spo2",
            "full_sleep_breathing_rate", "stress_score", "sleep_points_percentage",
            "TENSE/ANXIOUS", "TIRED", "ENTERTAINMENT"]
SENSORS = ["nremhr", "rmssd", "spo2", "full_sleep_breathing_rate", "sleep_hours", "stress_score"]
CATEGORICAL = ["gender", "smoking_history", "physical_activity", "alcohol", "gen_health"]
# Matching/derived feature of unknown origin; deliberately excluded.
EXCLUDED = ["patient_id", "bmi_gap"]


def load_dataset(path=None):
    path = Path(path or os.environ.get("CARDIO_DATA_PATH", ROOT / "data/raw/FINAL_dataset_with_fitbit.csv"))
    if not path.is_file():
        raise FileNotFoundError(f"Authorized CSV required at {path}. Supply --data PATH or CARDIO_DATA_PATH.")
    df = pd.read_csv(path)
    df.attrs["source_sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()
    required = {"patient_id", "heart_disease", "hypertension", "gender", "age", *SENSORS}
    if missing := required - set(df):
        raise ValueError(f"Missing required columns: {sorted(missing)}")
    if df.patient_id.isna().any():
        raise ValueError("patient_id must not be missing.")
    for target in TARGETS:
        if not df[target].isin([0, 1]).all():
            raise ValueError(f"{target} must contain nonmissing binary labels.")
    return df


def feature_columns(df, target, mode="combined"):
    excluded = set(EXCLUDED + [target])
    # Both labels removed from each other's models to avoid circular diagnosis proxies.
    excluded.update(TARGETS)
    if mode == "clinical":
        excluded.update(WEARABLE)
    return [c for c in df if c not in excluded]
