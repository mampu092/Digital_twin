"""A locked-family, locked-hyperparameter feature ablation complementary to model selection."""
import _bootstrap
import argparse
import joblib
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.pipeline import Pipeline
from threadpoolctl import threadpool_limits
from src.common import ROOT,dump_json
from src.data.loader import load_dataset,feature_columns
from src.data.preprocessing import make_preprocessor
from src.models.baselines import load_bundle,predict,verify_training_dataset
from src.evaluation.calibration import fit_calibrator,calibrate,select_threshold
from src.evaluation.metrics import metrics,paired_ablation


def run(df):
    verify_training_dataset(df)
    manifest=pd.read_csv(ROOT/"data/processed/private_split_manifest.csv")
    tr,va,ca,te=(np.flatnonzero(manifest.split.eq(s)) for s in ["train","tune","calibration","test"])
    result={}
    for target in ["heart_disease","hypertension"]:
        combined=load_bundle(target)
        cols=feature_columns(df,target,"clinical")
        pipe=Pipeline([("preprocess",make_preprocessor(cols)),
                       ("model",clone(combined["pipeline"].named_steps["model"]))])
        y=df[target].to_numpy()
        pipe.fit(df.iloc[tr][cols],y[tr])
        cal=fit_calibrator(y[ca],pipe.predict_proba(df.iloc[ca][cols])[:,1])
        threshold=select_threshold(y[va],calibrate(cal,pipe.predict_proba(df.iloc[va][cols])[:,1]))
        clinical=calibrate(cal,pipe.predict_proba(df.iloc[te][cols])[:,1])
        full=predict(combined,df.iloc[te])
        result[target]={"fixed_family":combined["model_name"],
            "clinical_metrics":metrics(y[te],clinical,threshold),
            "combined_metrics":metrics(y[te],full,combined["threshold"]),
            "paired_difference":paired_ablation(y[te],clinical,full,manifest.group.to_numpy()[te]),
            "scope":"Same algorithm and hyperparameters selected on combined training/tune data. "
                    "Preprocessing and calibration fitted independently. Secondary locked analysis, not model selection."}
    dump_json(result,ROOT/"outputs/metrics/matched_family_ablation.json")
    print(result)


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data",default=None)
    args=parser.parse_args()
    with threadpool_limits(limits=2):run(load_dataset(args.data))
