import copy
import numpy as np
import pandas as pd
import pytest
import torch
from src.research.longitudinal import history_as_of, bp_summary, validate_trial_log, capability_matrix
from src.research.causal import estimate_ate, policy_value, safe_utility_choice
from src.research.synthetic_cohort import cohort, randomized_trial
from src.research.multimodal import LongitudinalMultimodal, discrete_survival_loss, multitask_loss, cumulative_event_probability
from src.research.state_space import ResearchStateSpace


def test_asof_blocks_unavailable_results_and_future_collection():
    rows=pd.DataFrame([
        dict(patient_id="p",timestamp="2026-01-01T00:00Z",available_at="2026-01-03T00:00Z",variable="crp",value=2,unit="mg/L",quality="valid"),
        dict(patient_id="p",timestamp="2026-01-02T00:00Z",available_at="2026-01-02T00:00Z",variable="sbp",value=125,unit="mmHg",quality="valid"),
        dict(patient_id="p",timestamp="2026-01-04T00:00Z",available_at="2026-01-04T00:00Z",variable="sbp",value=130,unit="mmHg",quality="valid")])
    assert list(history_as_of(rows,"p","2026-01-02T12:00Z").variable)==["sbp"]
    with pytest.raises(ValueError):history_as_of(rows,"p","2026-01-02")


def test_bp_equations_and_quality_exclusion():
    rows=pd.DataFrame(dict(timestamp=pd.date_range("2026-01-01",periods=5,tz="UTC"),
                           sbp=[120,120,108,108,220],dbp=[80,80,70,70,100],
                           asleep=[False,False,True,True,False],quality=["valid"]*4+["invalid"],
                           device_validated=[True]*5))
    result=bp_summary(rows)
    assert result["mean_sbp_mmhg"]==114
    assert result["sbp_dip_percent"]==pytest.approx(10)
    assert result["mean_pulse_pressure_mmhg"]==39
    assert result["approx_mean_arterial_pressure_mmhg"]==88
    rows.loc[0,"sbp"]=60
    with pytest.raises(ValueError):bp_summary(rows)


def test_bp_does_not_mix_participants():
    frame=pd.DataFrame(dict(patient_id=["a","b"],timestamp=["2026-01-01T00:00Z","2026-01-02T00:00Z"],
                            sbp=[120,120],dbp=[80,80],asleep=[False,True],
                            quality=["valid","valid"],device_validated=[True,True]))
    with pytest.raises(ValueError):bp_summary(frame)


def test_saved_synthetic_checkpoint_inference():
    from src.research.inference import synthetic_prediction
    result=synthetic_prediction(1000)
    assert result["provenance"]=="ENTIRELY SYNTHETIC"
    assert 50<result["next_week_mean_sbp_mmhg"]<260
    p=result["cumulative_event_probability_by_illustrative_year"]
    assert np.all(np.diff(p)>=0) and 0<=p[0]<=p[-1]<=1


def balanced_trial():
    frame=randomized_trial(n=100)
    frame["eligible"]=1
    frame["assignment_probability"]=.5
    frame["action"]=np.tile([0,1],50)
    frame["outcome"]=10+frame.action*2
    return frame


def test_ipw_and_ate_match_independent_manual_equations():
    frame=balanced_trial()
    assert estimate_ate(frame,repeats=100)["estimate"]==2
    # Always treat: half the rows contribute 12/.5, half contribute zero.
    result=policy_value(frame,np.ones(100),repeats=100)
    assert result["ipw_value"]==12
    assert result["effective_sample_size"]==50


@pytest.mark.parametrize("field,value",[("assignment_probability",0),("assignment_probability",1),
                                       ("assignment_probability",np.nan),("outcome",np.nan),
                                       ("action",2),("eligible",2)])
def test_trial_invalid_inputs_fail_closed(field,value):
    frame=balanced_trial()
    frame.loc[0,field]=value
    with pytest.raises(ValueError):validate_trial_log(frame)


def test_overlapping_outcomes_rejected():
    frame=randomized_trial(n=20,decisions=2)
    frame.loc[0,"outcome_time"]=frame.loc[1,"outcome_time"]
    with pytest.raises(ValueError):validate_trial_log(frame)


def test_eligibility_precedes_assignment():
    simulated=randomized_trial(n=100,decisions=3)
    assert simulated.loc[simulated.eligible.eq(0),"action"].eq(0).all()
    invalid=balanced_trial()
    invalid.loc[1,"eligible"]=0
    with pytest.raises(ValueError):validate_trial_log(invalid)


def test_unmeasured_state_axes_and_missing_uncertainty():
    state=ResearchStateSpace()
    measured=state.update([1,1,np.nan,np.nan,1,1])
    assert measured["measurement_supported"]==[True,False,False,True,True]
    assert np.linalg.eigvalsh(state.covariance).min()>0
    before=state.covariance.copy()
    missing=state.update([np.nan]*6,elapsed_days=10)
    assert missing["sd"][0]>np.sqrt(before[0,0])
    with pytest.raises(ValueError):state.update([0]*6,provenance="observed")


def test_multimodal_missing_values_cannot_change_prediction():
    torch.manual_seed(8)
    values,masks,time,labels,availability=cohort(n=8,days=7)
    model=LongitudinalMultimodal().eval()
    original=model(values,masks,time)
    changed={k:torch.where(masks[k].bool(),v,torch.full_like(v,9999)) for k,v in values.items()}
    masked=model(changed,masks,time)
    for head in original:torch.testing.assert_close(original[head],masked[head])
    loss,_=multitask_loss(original,labels,availability)
    loss.backward()
    assert torch.isfinite(loss)
    assert all(p.grad is None or torch.isfinite(p.grad).all() for p in model.parameters())


def test_survival_likelihood_censoring_manual_and_monotonicity():
    logits=torch.zeros(2,3,requires_grad=True)
    loss=discrete_survival_loss(logits,torch.tensor([2.,3.]),torch.tensor([1.,0.]))
    # event in bin2: .5*.5; censored after bin3: .5^3; average NLL=2.5 log2
    assert loss.item()==pytest.approx(2.5*np.log(2))
    loss.backward()
    assert logits.grad[0,2]==0  # No risk contribution after the event.
    probabilities=cumulative_event_probability(logits).detach().numpy()
    np.testing.assert_allclose(probabilities[0],[.5,.75,.875])
    with pytest.raises(ValueError):discrete_survival_loss(logits,torch.tensor([0.,3.]),torch.tensor([1.,0.]))


def test_missing_labels_do_not_train_heads():
    values,masks,time,labels,available=cohort(n=8,days=7)
    out=LongitudinalMultimodal()(values,masks,time)
    for k in available:available[k][:]=False
    with pytest.raises(ValueError):multitask_loss(out,labels,available)
    available["bp"][:]=True
    labels["acute"][:]=float("nan")
    loss,components=multitask_loss(out,labels,available)
    assert set(components)=={"bp"} and torch.isfinite(loss)


def test_capability_does_not_map_entertainment_to_loneliness():
    result=capability_matrix(["ENTERTAINMENT","TENSE/ANXIOUS","hypertension","heart_disease"])
    assert result.loc[result.component.eq("Loneliness pathway"),"source_status"].item()=="not_estimable_from_source"
    assert safe_utility_choice([{"name":"unsupported","safe":True}])["selected"] is None


def test_enrollment_reading_never_scores_against_itself(profile,models):
    from src.twin.digital_twin import CardiovascularDigitalTwin
    from src.data.simulation import simulate
    twin=CardiovascularDigitalTwin(profile,models)
    rows=simulate(profile["patient_id"],missing=0).to_dict("records")
    for row in rows[:7]:
        assert twin.ingest_wearable_data(row)["score"] is None
    assert twin.ingest_wearable_data(rows[7])["score"] is not None
