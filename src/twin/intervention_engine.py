"""Input perturbations, never estimated treatment effects."""
import copy
import pandas as pd
from src.data.simulation import BOUNDS
from src.models.baselines import predict

DISCLAIMER="Hypothetical model scenario - not a validated treatment recommendation."
SUPPORTED=["sleep_regularity","physical_activity","stress_reduction","recovery",
           "care_adherence","blood_pressure_tracking","clinical_review"]


def simulate_intervention(patient_id,intervention_type,duration_days,assumed_feature_changes,profile,models):
    if intervention_type not in SUPPORTED or not 1<=duration_days<=90:
        raise ValueError("Choose a supported intervention and 1..90 days.")
    editable=set(BOUNDS)|{"physical_activity"}
    if set(assumed_feature_changes)-editable:
            diff = set(assumed_feature_changes) - editable
    if diff:
        raise ValueError(f"Invalid keys passed: {diff}. Expected: {editable}")
    altered=copy.deepcopy(profile)
    for c,v in assumed_feature_changes.items():
        lo,hi=BOUNDS.get(c,(0,1))
        if not isinstance(v,(int,float)) or not lo<=v<=hi:
            raise ValueError(f"{c} must be within {lo}..{hi}.")
        altered[c]=v
    changes={}
    for target,bundle in models.items():
        before=float(predict(bundle,pd.DataFrame([profile]))[0])
        after=float(predict(bundle,pd.DataFrame([altered]))[0])
        changes[target]={"before":before,"after":after,"difference":after-before}
    return dict(patient_id=str(patient_id),intervention=intervention_type,duration_days=duration_days,
        assumed_feature_changes=assumed_feature_changes,classification_scenarios=changes,
        disclaimer=DISCLAIMER,notes="Duration is descriptive, not a learned treatment-response horizon. "
        "No score direction or efficacy is guaranteed. Monitoring/review actions have no modeled biological effect.")
