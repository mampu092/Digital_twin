"""Real-source domain ablations and adjusted cross-sectional associations.

This is a post hoc extension on an already inspected test partition, not a fresh
confirmatory holdout. Neither model fitting nor covariate adjustment proves causation.
"""
import _bootstrap
import argparse
import numpy as np
import pandas as pd
from scipy.special import expit
from scipy.stats import norm
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from src.common import ROOT, dump_json
from src.data.loader import load_dataset
from src.data.preprocessing import make_preprocessor
from src.models.baselines import split_dataset, verify_training_dataset
from src.evaluation.calibration import fit_calibrator, calibrate, select_threshold
from src.evaluation.metrics import metrics, paired_ablation
from src.research.longitudinal import capability_matrix

CLINICAL = ["age","gender","bmi","HbA1c_level","blood_glucose_level","diabetes","smoking_history",
            "physical_activity","alcohol","stroke","kidney_disease","asthma","diff_walking",
            "bad_physical_days","gen_health","sleep_hours","skin_cancer"]
PSYCH = ["bad_mental_days","TENSE/ANXIOUS","TIRED"]
WEARABLE = ["nightly_temperature","nremhr","rmssd","spo2","full_sleep_breathing_rate",
            "stress_score","sleep_points_percentage"]
DOMAINS = {"clinical_context":CLINICAL, "clinical_psych_proxies":CLINICAL+PSYCH,
           "clinical_wearable":CLINICAL+WEARABLE, "clinical_psych_wearable":CLINICAL+PSYCH+WEARABLE}


def adjusted_associations(df, groups):
    records = []
    for exposure in ["bad_mental_days","TENSE/ANXIOUS"]:
        observed = df[exposure].notna()
        data = df.loc[observed].copy()
        group = groups[observed]
        # Complete exposure cases. Covariates below are nonmissing in this CSV.
        controls = pd.get_dummies(data[["age","gender","bmi","diabetes","smoking_history"]],
                                  columns=["gender","smoking_history"],drop_first=True,dtype=float)
        controls = (controls-controls.mean())/controls.std().replace(0,1)
        x = ((data[exposure]-data[exposure].mean())/data[exposure].std()).to_numpy()
        for target in ["hypertension","heart_disease"]:
            design = pd.DataFrame({"exposure_per_observed_sd":x}, index=data.index)
            if target=="heart_disease":
                design["recorded_hypertension"] = data.hypertension.astype(float)
                design["exposure_x_hypertension"] = x*data.hypertension.to_numpy()
            design = pd.concat([design,controls],axis=1)
            X, y = design.to_numpy(float), data[target].to_numpy(float)
            model = LogisticRegression(C=1e6, max_iter=1000, tol=1e-9).fit(X,y)
            Z = np.column_stack([np.ones(len(X)),X])
            coefficient = np.r_[model.intercept_,model.coef_.ravel()]
            p = expit(Z@coefficient)
            information = Z.T@(Z*(p*(1-p))[:,None])
            inverse = np.linalg.pinv(information)
            cluster_scores = pd.DataFrame(Z*(y-p)[:,None]).groupby(group).sum().to_numpy()
            covariance = inverse@(cluster_scores.T@cluster_scores)@inverse
            se = np.sqrt(np.maximum(0,np.diag(covariance)))
            for name in ["exposure_per_observed_sd","recorded_hypertension","exposure_x_hypertension"]:
                if name not in design:
                    continue
                j = list(design).index(name)+1
                beta, error = coefficient[j], se[j]
                records.append(dict(target=target, exposure=exposure, term=name,
                                    n=len(data), excluded_exposure_missing=int((~observed).sum()),
                                    odds_ratio=float(np.exp(beta)),ci95_low=float(np.exp(beta-1.96*error)),
                                    ci95_high=float(np.exp(beta+1.96*error)),
                                    p_unadjusted=float(2*norm.sf(abs(beta/error))),
                                    scope="cross-sectional conditional association; unknown raw-score direction"))
    result = pd.DataFrame(records)
    # Benjamini-Hochberg across every coefficient reported in this exploratory table.
    order = np.argsort(result.p_unadjusted.to_numpy())
    ranked = result.p_unadjusted.to_numpy()[order]*len(result)/np.arange(1,len(result)+1)
    adjusted = np.minimum.accumulate(ranked[::-1])[::-1].clip(0,1)
    result["q_bh"] = 0.
    result.loc[result.index[order],"q_bh"] = adjusted
    return result


def run(path, repeats=200):
    df = load_dataset(path)
    verify_training_dataset(df)
    split, groups = split_dataset(df)
    tr, va, ca, te = [np.flatnonzero(split==s) for s in ["train","tune","calibration","test"]]
    rows, contrasts, feature_lists = [], {}, {}
    for target in ["heart_disease","hypertension"]:
        y = df[target].to_numpy()
        predictions = {}
        for domain, features in DOMAINS.items():
            print(f"Domain comparison: {target} / {domain}",flush=True)
            model = Pipeline([("preprocess",make_preprocessor(features)),
                              ("model",HistGradientBoostingClassifier(max_iter=120,max_leaf_nodes=15,
                                  learning_rate=.05,early_stopping=False,l2_regularization=1,random_state=742))])
            model.fit(df.iloc[tr][features],y[tr])
            cal = fit_calibrator(y[ca],model.predict_proba(df.iloc[ca][features])[:,1])
            threshold = select_threshold(y[va],calibrate(cal,model.predict_proba(df.iloc[va][features])[:,1]))
            p = calibrate(cal,model.predict_proba(df.iloc[te][features])[:,1])
            predictions[domain] = p
            rows.append(dict(target=target,domain=domain,**metrics(y[te],p,threshold)))
            feature_lists[domain] = features
        contrasts[target] = {
            "psych_vs_clinical":paired_ablation(y[te],predictions["clinical_context"],predictions["clinical_psych_proxies"],groups[te],repeats),
            "wearable_vs_clinical":paired_ablation(y[te],predictions["clinical_context"],predictions["clinical_wearable"],groups[te],repeats),
            "psych_added_to_wearables":paired_ablation(y[te],predictions["clinical_wearable"],predictions["clinical_psych_wearable"],groups[te],repeats),
        }
    pd.DataFrame(rows).to_csv(ROOT/"outputs/metrics/psychosocial_domain_metrics.csv",index=False)
    dump_json({"contrasts":contrasts,"features":feature_lists,"source_sha256":df.attrs["source_sha256"],
               "partitions":{s:int((split==s).sum()) for s in np.unique(split)},
               "model":"Fixed HGB 120 iterations, 15 leaves, lr .05, L2=1; all domains identical settings",
               "scope":"Exploratory post hoc reuse of existing test partition; source linkage unknown; no external validation",
               "excluded":"ENTERTAINMENT is not loneliness; both disease labels and bmi_gap excluded from predictors",
               "clinical_context":"Includes sleep and activity; estimates conditional predictive addition, not total psychosocial effect"},
              ROOT/"outputs/metrics/psychosocial_ablation.json")
    association = adjusted_associations(df,groups)
    association.to_csv(ROOT/"outputs/metrics/psychosocial_associations.csv",index=False)
    table = []
    for exposure in ["bad_mental_days","TENSE/ANXIOUS"]:
        # Quartiles are descriptive rank bins, never diagnostic anxiety categories.
        bins = pd.qcut(df[exposure],4,duplicates="drop").astype("string").fillna("Missing")
        grouped = df.assign(exposure_bin=bins).groupby(["hypertension","exposure_bin"],observed=True)
        for (htn,band), subset in grouped:
            table.append(dict(exposure=exposure,raw_value_bin=band,hypertension=htn,n=len(subset),
                              recorded_heart_disease_n=int(subset.heart_disease.sum()),
                              recorded_heart_disease_fraction=float(subset.heart_disease.mean())))
    pd.DataFrame(table).to_csv(ROOT/"outputs/metrics/hypertension_psych_strata.csv",index=False)
    capability_matrix(df.columns).to_csv(ROOT/"reports/research_capabilities.csv",index=False)
    print(association.to_string(index=False))


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data",required=True)
    parser.add_argument("--bootstrap",type=int,default=200)
    args=parser.parse_args()
    run(args.data,args.bootstrap)
