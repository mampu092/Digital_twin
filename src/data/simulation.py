"""Synthetic nightly observations. No observed patient trajectory is reconstructed."""
import numpy as np
import pandas as pd
from src.data.loader import SENSORS

SCENARIOS=["stable","sustained_stress","sleep_deprivation","reduced_recovery",
           "respiratory_disturbance","combined_vulnerability"]
BOUNDS={"nremhr":(35,125),"rmssd":(5,180),"spo2":(80,100),
        "full_sleep_breathing_rate":(8,28),"sleep_hours":(2,12),"stress_score":(0,100)}
UNITS={"nremhr":"bpm (simulation assumption)","rmssd":"ms (simulation assumption)",
       "spo2":"% (simulation assumption)","full_sleep_breathing_rate":"breaths/min (simulation assumption)",
       "sleep_hours":"hours","stress_score":"0-100 simulated stress burden; higher = more"}
SHIFTS={
    "stable":[0,0,0,0,0,0], "sustained_stress":[7,-12,-.2,1,-.7,16],
    "sleep_deprivation":[5,-10,-.2,.5,-2,10], "reduced_recovery":[9,-19,-.3,1,-.5,12],
    "respiratory_disturbance":[6,-8,-4,3,-.8,5],
    "combined_vulnerability":[10,-16,-2,2,-1.5,15]}
NOISE=np.array([2.3,5,.35,.5,.45,4.0])


def demo_profile():
    """Entirely invented adult, not copied from the supplied data."""
    return dict(patient_id="SYNTH-DEMO-001",gender="Female",age=58,hypertension=0,heart_disease=0,
        smoking_history="never",bmi=29.0,HbA1c_level=5.8,blood_glucose_level=125,diabetes=0,
        physical_activity=0,alcohol=0,stroke=0,kidney_disease=0,asthma=0,diff_walking=0,
        bad_physical_days=2,bad_mental_days=8,gen_health=2,sleep_hours=6.5,skin_cancer=0,
        nightly_temperature=None,nremhr=64.,rmssd=40.,spo2=96.,
        full_sleep_breathing_rate=15.,stress_score=50.,sleep_points_percentage=None,
        **{"TENSE/ANXIOUS":None,"TIRED":None,"ENTERTAINMENT":None})


def simulate(patient_id="SYNTH-DEMO-001",scenario="stable",days=30,seed=42,missing=.08,start="2026-01-01"):
    if scenario not in SCENARIOS:
        raise ValueError(f"Unknown scenario: {scenario}")
    if not 7<=days<=30 or not 0<=missing<1:
        raise ValueError("days must be 7..30 and missing must be 0..1 (exclusive).")
    rng=np.random.default_rng(seed)
    base=np.array([62,45,96.5,15,7.2,45])+rng.normal(size=6)*[7,12,.7,1,.5,10]
    base=np.array([np.clip(v,*BOUNDS[c]) for c,v in zip(SENSORS,base)])
    residual=np.zeros(6)
    onset=10
    rows=[]
    severity=rng.uniform(.65,1.4)
    for day in range(days):
        residual=.65*residual+rng.normal(size=6)*NOISE*np.sqrt(1-.65**2)
        gradual=min(max((day-onset+1)/6,0),1)
        intensity=gradual if scenario!="respiratory_disturbance" else float(day>=onset)
        # Correlated multivariate shifts, random severity, benign noise; no clinical event labels.
        value=base+residual+np.array(SHIFTS[scenario])*intensity*severity
        row={"patient_id":str(patient_id),"timestamp":(pd.Timestamp(start,tz="UTC")+pd.Timedelta(days=day,hours=6)).isoformat(),
             "provenance":"synthetic","scenario":scenario,"synthetic_deviation":int(scenario!="stable" and day>=onset),
             "perturbation_intensity":float(intensity*severity)}
        for c,v in zip(SENSORS,value):
            row[c]=float(np.clip(v,*BOUNDS[c])) if rng.random()>=missing else None
        rows.append(row)
    return pd.DataFrame(rows)
