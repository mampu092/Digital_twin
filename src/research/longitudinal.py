"""Time-safe measurement contracts and descriptive physiology.

No conversion from a diagnosis flag to blood pressure, or entertainment to
loneliness, is permitted. Thresholds below are data-quality checks, not diagnoses.
"""
import numpy as np
import pandas as pd


def aware_time(value):
    stamp = pd.Timestamp(value)
    if pd.isna(stamp) or stamp.tzinfo is None:
        raise ValueError("A timezone-aware time is required")
    return stamp.tz_convert("UTC")


def history_as_of(measurements, patient_id, cutoff):
    """Use both collection and availability time; never backward-fill future data."""
    cutoff = aware_time(cutoff)
    required = {"patient_id", "timestamp", "available_at", "variable", "value", "unit", "quality"}
    if not required.issubset(measurements):
        raise ValueError(f"Missing measurement fields: {sorted(required-set(measurements))}")
    frame = measurements.copy()
    frame["timestamp"] = frame.timestamp.map(aware_time)
    frame["available_at"] = frame.available_at.map(aware_time)
    if (frame.available_at < frame.timestamp).any():
        raise ValueError("Availability cannot precede collection")
    if frame.duplicated(["patient_id", "timestamp", "variable"]).any():
        raise ValueError("Duplicate measurement keys require upstream resolution")
    return frame.loc[(frame.patient_id.astype(str) == str(patient_id)) &
                     (frame.timestamp <= cutoff) & (frame.available_at <= cutoff)].sort_values("timestamp")


def bp_summary(frame):
    """Paired validated cuff/ABPM readings; no inferred BP or hypertension diagnosis.

    Input is one paired reading per row. Means weight readings equally; the caller
    must supply a standardized sampling protocol. Sleep state is observed, not
    inferred from wall-clock time. Minimum counts are software guards only.
    """
    required = {"timestamp", "sbp", "dbp", "asleep", "quality", "device_validated"}
    if not required.issubset(frame):
        raise ValueError(f"Missing BP fields: {sorted(required-set(frame))}")
    if frame.empty:
        return {"status": "unavailable", "n": 0}
    if "patient_id" in frame and frame.patient_id.nunique(dropna=False)!=1:
        raise ValueError("Summarize one participant at a time")
    times = frame.timestamp.map(aware_time)
    if times.duplicated().any():
        raise ValueError("Duplicate BP timestamps")
    good = frame.loc[frame.quality.eq("valid") & frame.device_validated.eq(True)].copy()
    if not good.asleep.isin([True, False]).all():
        raise ValueError("Observed asleep state must be boolean")
    if not np.isfinite(good[["sbp", "dbp"]].to_numpy(dtype=float)).all():
        raise ValueError("Missing or nonfinite BP values")
    if not ((good.sbp.between(50, 260)) & (good.dbp.between(30, 160)) & (good.sbp > good.dbp)).all():
        raise ValueError("Implausible paired BP; review source rather than clip")
    if len(good) < 2:
        return {"status": "insufficient_valid_readings", "n": len(good)}
    wake, sleep = good.loc[~good.asleep.astype(bool)], good.loc[good.asleep.astype(bool)]
    result = {"status": "descriptive_only", "n": len(good), "excluded_n": len(frame)-len(good),
              "mean_sbp_mmhg": float(good.sbp.mean()), "mean_dbp_mmhg": float(good.dbp.mean()),
              "mean_pulse_pressure_mmhg": float((good.sbp-good.dbp).mean()),
              "approx_mean_arterial_pressure_mmhg": float((good.dbp+(good.sbp-good.dbp)/3).mean()),
              "awake_n": len(wake), "asleep_n": len(sleep), "sbp_dip_percent": None,
              "interpretation": "Descriptive readings; MAP is a resting approximation; no diagnostic classification"}
    if min(len(wake), len(sleep)) >= 2:
        result["sbp_dip_percent"] = float(100*(wake.sbp.mean()-sleep.sbp.mean())/wake.sbp.mean())
    return result


def capability_matrix(columns):
    """Presence alone does not certify measurement semantics, alignment or outcomes."""
    columns = set(columns)
    specs = [
        ("Recorded disease", "Cross-sectional", ["heart_disease", "hypertension"], "internal classification only"),
        ("Loneliness pathway", "Repeated weeks/months", ["loneliness_score", "loneliness_instrument", "timestamp"], "validated scale and repeated aligned observations"),
        ("Acute stress", "Minutes/hours", ["timestamp", "stress_label", "stress_label_available_at", "ecg_or_ppg", "respiration", "activity_state"], "independent stress labels; signal quality and context"),
        ("BP trajectory", "Weeks/months", ["timestamp", "sbp", "dbp", "device_validated"], "repeated validated BP with future target times"),
        ("Nocturnal BP", "Sleep/wake cycles", ["timestamp", "sbp", "dbp", "asleep", "device_validated"], "ABPM and recorded sleep state"),
        ("Incident CVD", "Years", ["index_time", "event_time", "censor_time", "event_adjudicated"], "incident endpoints, follow-up and censoring"),
        ("Intervention effect", "12-week trial", ["action", "assignment_probability", "eligible", "outcome", "outcome_time"], "randomization or defensible exchangeability"),
        ("HPA / immune state", "Repeated assays", ["cortisol", "cortisol_collection_time", "hs_crp", "il6"], "timed assays; latent states require identification"),
    ]
    return pd.DataFrame([dict(component=name, horizon=horizon,
                             available=", ".join(c for c in needed if c in columns) or "none",
                             missing=", ".join(c for c in needed if c not in columns) or "none",
                             source_status="available_with_limitations" if set(needed)<=columns else "not_estimable_from_source",
                             requirement=requirement)
                         for name, horizon, needed, requirement in specs])


def validate_trial_log(frame):
    """One row per decision with outcome before the next decision.

    This deliberately rejects overlapping proximal windows, unknown propensities,
    missing outcomes and endpoint probabilities. It is not a general longitudinal
    causal estimator with treatment/confounding feedback.
    """
    fields = {"patient_id", "decision_time", "outcome_time", "action", "assignment_probability",
              "eligible", "outcome"}
    if not fields.issubset(frame):
        raise ValueError(f"Missing trial fields: {sorted(fields-set(frame))}")
    data = frame.copy()
    if data.empty or data.patient_id.isna().any():
        raise ValueError("Nonempty identified trial log required")
    for col in ["decision_time", "outcome_time"]:
        data[col] = data[col].map(aware_time)
    if data.duplicated(["patient_id", "decision_time"]).any():
        raise ValueError("Duplicate decision")
    if not data.eligible.isin([0, 1, False, True]).all():
        raise ValueError("Eligibility must be observed binary")
    if not data.action.isin([0, 1]).all():
        raise ValueError("Binary actions required")
    if (data.eligible.eq(0) & data.action.ne(0)).any():
        raise ValueError("Ineligible decisions cannot assign an intervention")
    if (data.outcome_time <= data.decision_time).any():
        raise ValueError("Outcomes must follow decisions")
    data = data.sort_values(["patient_id", "decision_time"])
    next_time = data.groupby("patient_id").decision_time.shift(-1)
    if ((next_time.notna()) & (data.outcome_time >= next_time)).any():
        raise ValueError("Overlapping decision windows require a longitudinal estimator")
    data = data.loc[data.eligible.eq(1)].copy()
    if data.empty or not np.isfinite(data[["assignment_probability", "outcome"]].to_numpy(float)).all():
        raise ValueError("Observed eligible outcomes and known probabilities required")
    if not data.assignment_probability.between(0, 1, inclusive="neither").all():
        raise ValueError("Strict positivity requires 0 < p(action=1) < 1")
    return data
