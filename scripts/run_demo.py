import _bootstrap
import argparse
import numpy as np
import pandas as pd
from dataclasses import asdict
from src.common import ROOT,dump_json
from src.data.simulation import simulate,demo_profile,SCENARIOS
from src.models.baselines import load_bundle
from src.twin.digital_twin import CardiovascularDigitalTwin
from src.reporting.pdf_report import architecture_pdf
from src.models.fusion_training import infer_fusion

if __name__=="__main__":
    parser=argparse.ArgumentParser(description="Run an entirely synthetic patient through the saved models and twin.")
    parser.add_argument("--scenario",choices=SCENARIOS,default="sustained_stress")
    parser.add_argument("--seed",type=int,default=42)
    args=parser.parse_args()
    profile=demo_profile()
    models={t:load_bundle(t) for t in ["heart_disease","hypertension"]}
    twin=CardiovascularDigitalTwin(profile,models,ROOT/"outputs/demo_twin.sqlite")
    trajectory=simulate(profile["patient_id"],args.scenario,seed=args.seed)
    for record in trajectory.replace({np.nan:None}).to_dict("records"):
        twin.ingest_wearable_data(record)
    twin.simulate_intervention(dict(intervention_type="sleep_regularity",duration_days=14,
        assumed_feature_changes={"sleep_hours":7.5,"physical_activity":1,"rmssd":50.0}))
    twin.export_patient_report(ROOT/"outputs/patient_reports/synthetic_patient_report.pdf")
    architecture_pdf()
    dump_json(asdict(twin.state),ROOT/"outputs/demo_state.json")
    pd.DataFrame(twin.state.anomalies).drop(columns="z").to_csv(ROOT/"outputs/metrics/demo_state_updates.csv",index=False)
    fusion=infer_fusion(profile,trajectory) if (ROOT/"models/saved/fusion.pt").exists() else {"status":"fusion_not_trained"}
    dump_json({"status":"completed","synthetic_observations":len(twin.state.observations),
        "flagged_observations":sum(x["flag"] for x in twin.state.anomalies),
        "classification":twin.estimate_baseline_risk(),"current_synthetic_scenario":twin.estimate_current_scenario(),"fusion":fusion,
        "report":"outputs/patient_reports/synthetic_patient_report.pdf"},ROOT/"outputs/metrics/demo_status.json")
    print("Synthetic twin demo completed, SQLite persisted, PDF exported.")
