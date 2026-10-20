"""Grouped train/tune/calibration/test partitions and comparable model families."""
import logging
import json
import numpy as np
import pandas as pd
import joblib
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import StratifiedGroupKFold, RandomizedSearchCV
from sklearn.pipeline import Pipeline
from sklearn.metrics import average_precision_score
from src.common import ROOT, dump_json
from src.data.loader import feature_columns, CATEGORICAL
from src.data.preprocessing import make_preprocessor
from src.evaluation.calibration import fit_calibrator, calibrate, select_threshold


def profile_groups(df):
    """Union repeated IDs and identical predictors, even if outcomes conflict."""
    parent=np.arange(len(df))
    def find(x):
        while parent[x]!=x:
            parent[x]=parent[parent[x]]
            x=parent[x]
        return x
    def union(a,b):
        parent[find(b)]=find(a)
    cols=feature_columns(df,"heart_disease","clinical")
    for keys in [df.patient_id.astype(str).to_numpy(),
                 pd.util.hash_pandas_object(df[cols],index=False).to_numpy()]:
        seen={}
        for i,k in enumerate(keys):
            if k in seen: union(seen[k],i)
            else: seen[k]=i
    return np.array([find(i) for i in range(len(df))])


def split_dataset(df):
    groups=profile_groups(df)
    labels=df.heart_disease.astype(str)+"_"+df.hypertension.astype(str)
    folds=np.empty(len(df),dtype=int)
    for f,(_,ix) in enumerate(StratifiedGroupKFold(10,shuffle=True,random_state=42).split(df,labels,groups)):
        folds[ix]=f
    split=np.where(folds<6,"train",np.where(folds==6,"tune",np.where(folds==7,"calibration","test")))
    assert pd.DataFrame({"g":groups,"s":split}).groupby("g").s.nunique().max()==1
    return split,groups


def candidates():
    result={
        "logistic":(LogisticRegression(max_iter=700,class_weight="balanced",random_state=42),
                    {"model__C":[.1,1.0]}),
        "random_forest":(RandomForestClassifier(n_estimators=100,max_depth=12,class_weight="balanced",
                           n_jobs=2,random_state=42),{"model__min_samples_leaf":[15,40]}),
        "hist_gradient_boosting":(HistGradientBoostingClassifier(max_iter=120,max_leaf_nodes=15,
                           class_weight="balanced",random_state=42),{"model__learning_rate":[.05,.1]}),
        "mlp":(MLPClassifier(hidden_layer_sizes=(64,32),max_iter=100,early_stopping=True,
                           n_iter_no_change=8,random_state=42),{"model__alpha":[.01,.1]})}
    try:
        from lightgbm import LGBMClassifier
        result["lightgbm"]=(LGBMClassifier(n_estimators=120,num_leaves=15,class_weight="balanced",
                            n_jobs=2,verbosity=-1,random_state=42),{"model__learning_rate":[.03,.08]})
    except ImportError:
        logging.warning("LightGBM unavailable; four required core model families remain.")
    return result


def train_baselines(df, search_iter=2):
    dump_json({"source_sha256":df.attrs.get("source_sha256"),"rows":len(df),"columns":list(df.columns),
               "seed":42,"scope":"Source fingerprint; no patient identifiers stored here"},
               ROOT/"models/saved/training_provenance.json")
    split,groups=split_dataset(df)
    pd.DataFrame({"row":np.arange(len(df)),"patient_id":df.patient_id,"split":split,"group":groups}).to_csv(
        ROOT/"data/processed/private_split_manifest.csv",index=False)
    tr,va,ca=(np.flatnonzero(split==s) for s in ["train","tune","calibration"])
    table=[]
    for target in ["heart_disease","hypertension"]:
        y=df[target].to_numpy()
        for mode in ["clinical","combined"]:
            cols=feature_columns(df,target,mode)
            X=df[cols]
            finalists=[]
            for name,(model,space) in candidates().items():
                logging.info("Training %s / %s / %s",target,mode,name)
                pipeline=Pipeline([("preprocess",make_preprocessor(cols)),("model",model)])
                search=RandomizedSearchCV(pipeline,space,n_iter=search_iter,
                    cv=StratifiedGroupKFold(2,shuffle=True,random_state=42),
                    scoring="average_precision",random_state=42,n_jobs=1,error_score="raise")
                search.fit(X.iloc[tr],y[tr],groups=groups[tr])
                pipe=search.best_estimator_
                raw=pipe.predict_proba(X.iloc[va])[:,1]
                ap=float(average_precision_score(y[va],raw))
                row=dict(target=target,features=mode,model=name,validation_pr_auc=ap,
                    training_cv_pr_auc=float(search.best_score_),best_params=str(search.best_params_),
                    train_n=len(tr),tune_n=len(va),calibration_n=len(ca))
                table.append(row)
                finalists.append((ap,name,pipe))
            _,name,pipe=max(finalists,key=lambda z:z[0])
            cal=fit_calibrator(y[ca],pipe.predict_proba(X.iloc[ca])[:,1])
            threshold=select_threshold(y[va],calibrate(cal,pipe.predict_proba(X.iloc[va])[:,1]))
            reference={}
            bounds={}
            for c in cols:
                reference[c]=X.iloc[tr][c].mode().iloc[0] if c in CATEGORICAL else float(X.iloc[tr][c].median())
                if pd.api.types.is_numeric_dtype(X[c]):
                    bounds[c]=[float(X.iloc[tr][c].min()),float(X.iloc[tr][c].max())]
            bundle=dict(pipeline=pipe,calibrator=cal,threshold=threshold,features=cols,
                target=target,mode=mode,model_name=name,reference=reference,bounds=bounds,
                training_n=len(tr),label="Recorded disease classification; not future event risk")
            joblib.dump(bundle,ROOT/f"models/saved/{target}_{mode}.joblib",compress=3)
            logging.info("Selected %s, validation AP %.4f",name,max(finalists,key=lambda z:z[0])[0])
    pd.DataFrame(table).to_csv(ROOT/"outputs/metrics/model_selection.csv",index=False)
    dump_json({s:int((split==s).sum()) for s in np.unique(split)},ROOT/"outputs/metrics/split_counts.json")
    return table


def predict(bundle, frame):
    return calibrate(bundle["calibrator"],bundle["pipeline"].predict_proba(frame[bundle["features"]])[:,1])


def load_bundle(target="heart_disease",mode="combined"):
    path=ROOT/f"models/saved/{target}_{mode}.joblib"
    if not path.exists():
        raise FileNotFoundError("Train the baseline models first: python scripts/train_baselines.py --data PATH")
    return joblib.load(path)


def verify_training_dataset(df):
    path=ROOT/"models/saved/training_provenance.json"
    meta=json.loads(path.read_text(encoding="utf-8"))
    if meta["source_sha256"]!=df.attrs.get("source_sha256") or meta["rows"]!=len(df) or meta["columns"]!=list(df.columns):
        raise ValueError("Input CSV differs from the source used for training. Re-run baseline training before evaluation.")
