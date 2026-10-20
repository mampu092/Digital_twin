import numpy as np
import pandas as pd
import pytest
from src.data.simulation import simulate,SCENARIOS,BOUNDS
from src.models.anomaly_detector import PatientAnomalyDetector


def test_seed_provenance_bounds_and_missingness():
    a=simulate(seed=42);b=simulate(seed=42)
    pd.testing.assert_frame_equal(a,b)
    assert a.timestamp.is_monotonic_increasing and a.timestamp.is_unique
    assert set(a.provenance)=={"synthetic"}
    assert a[list(BOUNDS)].isna().sum().sum()>0
    for c,(lo,hi) in BOUNDS.items():assert a[c].dropna().between(lo,hi).all()


def test_past_not_changed_by_future_scenario():
    a=simulate(scenario="stable")
    b=simulate(scenario="sustained_stress")
    pd.testing.assert_frame_equal(a.iloc[:10][list(BOUNDS)],b.iloc[:10][list(BOUNDS)])
    assert b.synthetic_deviation.sum()==20


def test_patient_relative_baseline():
    frame=simulate(scenario="stable",missing=0)
    records=frame.to_dict("records")
    detector=PatientAnomalyDetector();detector.fit(records[:7])
    altered=dict(records[8]);altered["nremhr"]+=40;altered["rmssd"]-=30
    assert detector.score(altered)["score"]>detector.score(records[8])["score"]


@pytest.mark.parametrize("kwargs",[{"days":4},{"scenario":"unknown"},{"missing":1}])
def test_invalid_simulation_request(kwargs):
    with pytest.raises(ValueError):simulate(**kwargs)
