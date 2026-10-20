import numpy as np
import pytest
from src.twin.digital_twin import CardiovascularDigitalTwin
from src.data.simulation import simulate
from src.twin.intervention_engine import DISCLAIMER


def records(profile):
    return simulate(profile["patient_id"],"sustained_stress").replace({np.nan:None}).to_dict("records")


def test_ingestion_persistence_idempotence_and_static_score(profile,models,tmp_path):
    path=tmp_path/"state.sqlite"
    twin=CardiovascularDigitalTwin(profile,models,path)
    original=twin.estimate_baseline_risk()["heart_disease"]["score"]
    for row in records(profile):
        twin.ingest_wearable_data(row)
    assert len(twin.state.anomalies)==30
    assert twin.state.latent_state>0
    twin.update_patient_state()
    assert len(twin.state.anomalies)==30
    restored=CardiovascularDigitalTwin.restore(path,profile["patient_id"],models)
    assert restored.state.latent_state==twin.state.latent_state
    assert restored.state.baseline==twin.state.baseline
    assert twin.estimate_baseline_risk()["heart_disease"]["score"]==original


@pytest.mark.parametrize("mutation",[{"nremhr":-2},{"spo2":float("inf")},{"patient_id":"other"},
                                   {"timestamp":"2026-01-01"},{"provenance":"observed"}])
def test_invalid_observation_atomic(profile,models,mutation):
    twin=CardiovascularDigitalTwin(profile,models)
    row=records(profile)[0];row.update(mutation)
    with pytest.raises(ValueError):twin.ingest_wearable_data(row)
    assert not twin.state.observations


def test_out_of_order_and_duplicate_rejected(profile,models):
    twin=CardiovascularDigitalTwin(profile,models)
    rows=records(profile)
    twin.ingest_wearable_data(rows[1])
    for row in rows[:2]:
        with pytest.raises(ValueError):twin.ingest_wearable_data(row)


def test_all_missing_is_not_a_normal_reading(profile,models):
    twin=CardiovascularDigitalTwin(profile,models)
    from src.data.loader import SENSORS
    for row in records(profile)[:10]:
        row.update({s:None for s in SENSORS});twin.ingest_wearable_data(row)
    assert twin.state.anomalies[-1]["score"] is None
    assert twin.state.anomalies[-1]["status"]!="within_reference"


def test_intervention_does_not_mutate_profile(profile,models):
    twin=CardiovascularDigitalTwin(profile,models)
    result=twin.simulate_intervention(dict(intervention_type="sleep_regularity",duration_days=14,
                                         assumed_feature_changes={"sleep_hours":7.5}))
    assert twin.state.profile["sleep_hours"]==profile["sleep_hours"]
    assert result["disclaimer"]==DISCLAIMER
    assert len(twin.state.interventions)==1
    with pytest.raises(ValueError):
        twin.simulate_intervention(dict(intervention_type="care_adherence",duration_days=14,
                                       assumed_feature_changes={"drug_dose":20}))


def test_pdf_export_real_document(profile,models):
    twin=CardiovascularDigitalTwin(profile,models)
    content=twin.export_patient_report()
    assert content.startswith(b"%PDF") and len(content)>1500
    import fitz
    doc=fitz.open(stream=content,filetype="pdf")
    text=" ".join(p.get_text() for p in doc)
    assert "Synthetic patient" in text and "not a certified medical device" in text
