import logging
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.base import clone
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_curve,precision_recall_curve,average_precision_score,roc_auc_score
from sklearn.calibration import calibration_curve
from sklearn.inspection import permutation_importance
from scipy.stats import ks_2samp
from src.common import ROOT,dump_json
from src.data.loader import WEARABLE,feature_columns
from src.models.baselines import load_bundle,predict,profile_groups,verify_training_dataset
from src.evaluation.metrics import metrics,bootstrap,paired_ablation
from src.evaluation.calibration import fit_calibrator,calibrate,select_threshold
from src.data.simulation import simulate,SCENARIOS
from src.models.anomaly_detector import PatientAnomalyDetector


def evaluate_models(df,repeats=200):
    verify_training_dataset(df)
    manifest=pd.read_csv(ROOT/"data/processed/private_split_manifest.csv")
    if len(manifest)!=len(df):
        raise ValueError("Split manifest and dataset row counts differ. Re-run training.")
    tr,va,ca,te=(np.flatnonzero(manifest.split.eq(s)) for s in ["train","tune","calibration","test"])
    all_metrics=[]; subgroups=[]; sensitivity=[]; ablations={}; cis={}; curves=[]
    for target in ["heart_disease","hypertension"]:
        y=df[target].to_numpy()
        preds={}
        for mode in ["clinical","combined"]:
            bundle=load_bundle(target,mode)
            p=predict(bundle,df.iloc[te]); preds[mode]=p
            row=metrics(y[te],p,bundle["threshold"])
            row.update(target=target,features=mode,model=bundle["model_name"])
            all_metrics.append(row)
            cis[f"{target}_{mode}"]=bootstrap(y[te],p,manifest.group.to_numpy()[te],bundle["threshold"],repeats)
            for category,values in [("gender",df.iloc[te].gender),
                ("age_band",pd.cut(df.iloc[te].age,[0,40,60,130],labels=["18-40","41-60","61+"]))]:
                for value in values.dropna().unique():
                    mask=np.asarray(values==value)
                    if mask.sum()>=100 and y[te][mask].sum()>=20:
                        r=metrics(y[te][mask],p[mask],bundle["threshold"])
                        r.update(target=target,features=mode,group=category,value=str(value))
                        subgroups.append(r)
            pd.DataFrame({"row":te,"label":y[te],"score":p}).to_csv(
                ROOT/f"outputs/predictions/private_{target}_{mode}.csv",index=False)
            fpr,tpr,_=roc_curve(y[te],p)
            precision,recall,_=precision_recall_curve(y[te],p)
            obs,expected=calibration_curve(y[te],p,n_bins=10,strategy="quantile")
            fig,axes=plt.subplots(1,3,figsize=(13,3.6))
            axes[0].plot(fpr,tpr,color="#0c8b88");axes[0].plot([0,1],[0,1],"--",color="gray")
            axes[0].set(xlabel="False positive rate",ylabel="Sensitivity",title="ROC")
            axes[1].plot(recall,precision,color="#0c8b88")
            axes[1].axhline(y[te].mean(),ls="--",color="gray")
            axes[1].set(xlabel="Recall",ylabel="Precision",title="Precision-recall")
            axes[2].plot(expected,obs,"o-",color="#0c8b88");axes[2].plot([0,1],[0,1],"--",color="gray")
            axes[2].set(xlabel="Mean classified probability",ylabel="Observed prevalence",title="Calibration",xlim=(0,1),ylim=(0,1))
            fig.suptitle(f"Recorded {target} | {mode} | held-out patients")
            fig.tight_layout();fig.savefig(ROOT/f"outputs/figures/{target}_{mode}_validation.png",dpi=140);plt.close(fig)
            for x,z in zip(expected,obs):
                curves.append({"target":target,"features":mode,"predicted":x,"observed":z})
            # Descriptive decision curves: thresholds are illustrative, not clinician utility estimates.
            dca=[]
            for t in np.linspace(.01,.5,50):
                pred=p>=t
                tp=np.mean(pred & (y[te]==1));fp=np.mean(pred & (y[te]==0))
                prevalence=y[te].mean()
                dca.append(dict(threshold=t,model=tp-fp*t/(1-t),
                                classify_all=prevalence-(1-prevalence)*t/(1-t),classify_none=0))
            pd.DataFrame(dca).to_csv(ROOT/f"outputs/metrics/{target}_{mode}_decision_curve.csv",index=False)
        ablations[target]=paired_ablation(y[te],preds["clinical"],preds["combined"],manifest.group.to_numpy()[te],repeats)
        bundle=load_bundle(target,"combined")
        cols=bundle["features"]
        complete=df[cols].notna().all(axis=1).to_numpy()
        a,b,c,d=(ix[complete[ix]] for ix in [tr,va,ca,te])
        if min(y[a].sum(),y[b].sum(),y[c].sum(),y[d].sum())>=20:
            cc=clone(bundle["pipeline"]).fit(df.iloc[a][cols],y[a])
            cc_cal=fit_calibrator(y[c],cc.predict_proba(df.iloc[c][cols])[:,1])
            cc_thresh=select_threshold(y[b],calibrate(cc_cal,cc.predict_proba(df.iloc[b][cols])[:,1]))
            pp=calibrate(cc_cal,cc.predict_proba(df.iloc[d][cols])[:,1])
            for method,ppp,threshold in [("complete_case_trained",pp,cc_thresh),
                ("imputed_on_same_complete_test",predict(bundle,df.iloc[d]),bundle["threshold"])]:
                sensitivity.append(dict(target=target,analysis=method,**metrics(y[d],ppp,threshold)))
        miss=df[WEARABLE].isna().astype(float)
        model=LogisticRegression(max_iter=300).fit(miss.iloc[tr],y[tr])
        sensitivity.append(dict(target=target,analysis="missingness_only",
                                **metrics(y[te],model.predict_proba(miss.iloc[te])[:,1])))
        # Post-hoc evaluation of the locked model: no model-selection feedback.
        sample=te[np.random.default_rng(42).choice(len(te),min(1500,len(te)),replace=False)]
        imp=permutation_importance(bundle["pipeline"],df.iloc[sample][cols],y[sample],
                                   scoring="average_precision",n_repeats=3,random_state=42,n_jobs=1)
        pd.DataFrame({"feature":cols,"ap_decrease":imp.importances_mean,"std":imp.importances_std}).sort_values(
            "ap_decrease",ascending=False).to_csv(ROOT/f"outputs/metrics/{target}_permutation_importance.csv",index=False)
        try:
            import shap
            pipe=bundle["pipeline"]
            transformed=pipe.named_steps["preprocess"].transform(df.iloc[sample[:150]][cols])
            explainer=shap.TreeExplainer(pipe.named_steps["model"])
            vals=explainer.shap_values(transformed)
            vals=vals[1] if isinstance(vals,list) else vals
            if vals.ndim==3:vals=vals[:,:,1]
            pd.DataFrame({"transformed_feature":pipe.named_steps["preprocess"].get_feature_names_out(),
                "mean_absolute_shap":np.abs(vals).mean(axis=0)}).to_csv(ROOT/f"outputs/metrics/{target}_shap.csv",index=False)
            dump_json({"status":"completed","scope":"raw tree output; not calibrated probability or causality",
                       "n":len(transformed)},ROOT/f"outputs/metrics/{target}_shap_status.json")
        except Exception as error:
            dump_json({"status":"not_supported_for_selected_model","reason":str(error),
                       "fallback":"permutation importance + one-feature substitutions"},ROOT/f"outputs/metrics/{target}_shap_status.json")
    pd.DataFrame(all_metrics).to_csv(ROOT/"outputs/metrics/test_metrics.csv",index=False)
    pd.DataFrame(subgroups).to_csv(ROOT/"outputs/metrics/subgroup_metrics.csv",index=False)
    pd.DataFrame(sensitivity).to_csv(ROOT/"outputs/metrics/missingness_sensitivity.csv",index=False)
    pd.DataFrame(curves).to_csv(ROOT/"outputs/metrics/calibration_points.csv",index=False)
    dump_json(cis,ROOT/"outputs/metrics/bootstrap_ci95.json")
    dump_json(ablations,ROOT/"outputs/metrics/ablation.json")
    shift=[]
    for col in df.select_dtypes("number").columns:
        if col=="patient_id":continue
        x,z=df.iloc[tr][col].dropna(),df.iloc[te][col].dropna()
        shift.append(dict(feature=col,ks_statistic=float(ks_2samp(x,z).statistic),
            train_mean=float(x.mean()),test_mean=float(z.mean()),
            train_missing=float(df.iloc[tr][col].isna().mean()),test_missing=float(df.iloc[te][col].isna().mean())))
    pd.DataFrame(shift).to_csv(ROOT/"outputs/metrics/train_test_distribution.csv",index=False)
    donor_sensitivity(df)
    return all_metrics


def donor_sensitivity(df):
    """Candidate source signatures, not known Fitbit people; secondary stress test."""
    usable=df.dropna(subset=["nremhr","rmssd"]).copy()
    donor=pd.util.hash_pandas_object(usable[["nremhr","rmssd"]],index=False).to_numpy()
    labels=usable.heart_disease.astype(str)+"_"+usable.hypertension.astype(str)
    train,test=next(StratifiedGroupKFold(5,shuffle=True,random_state=42).split(usable,labels,donor))
    # Remove identical clinical predictor profiles across this secondary boundary.
    keys=pd.util.hash_pandas_object(usable[feature_columns(usable,"heart_disease","clinical")],index=False).to_numpy()
    train=train[~np.isin(keys[train],keys[test])]
    rows=[]
    for target in ["heart_disease","hypertension"]:
        for mode in ["clinical","combined"]:
            bundle=load_bundle(target,mode);cols=bundle["features"]
            model=clone(bundle["pipeline"]).fit(usable.iloc[train][cols],usable.iloc[train][target])
            p=model.predict_proba(usable.iloc[test][cols])[:,1]; y=usable.iloc[test][target].to_numpy()
            rows.append(dict(target=target,features=mode,n_train=len(train),n_test=len(test),
                observed_source_signatures=len(np.unique(donor)),test_source_signatures=len(np.unique(donor[test])),
                roc_auc=float(roc_auc_score(y,p)),pr_auc=float(average_precision_score(y,p)),
                interpretation="secondary source-signature holdout; source IDs unknown; hyperparameters locked from primary analysis"))
    pd.DataFrame(rows).to_csv(ROOT/"outputs/metrics/source_signature_holdout.csv",index=False)


def synthetic_evaluation(patients=180,repeats=200):
    """Independent seeds from demo and fusion cohorts, fixed detector threshold."""
    rows=[]
    for i in range(patients):
        scenario=SCENARIOS[i%len(SCENARIOS)]
        frame=simulate(f"SYNTH-TEST-{i}",scenario,seed=900000+i)
        detector=PatientAnomalyDetector()
        records=frame.replace({np.nan:None}).to_dict("records")
        detector.fit(records[:7])
        for row in records[7:]:
            result=detector.score(row)
            if result["score"] is not None:
                rows.append(dict(patient=i,scenario=scenario,label=row["synthetic_deviation"],
                                 score=result["score"],flag=result["flag"]))
    data=pd.DataFrame(rows)
    # Deviation index maps monotonically to [0,1] only for metric API;
    # it is not a calibrated probability.
    p=1-np.exp(-data.score.to_numpy()/3)
    threshold=1-np.exp(-2.5/3)
    result={"scope":"SYNTHETIC ONLY; injected multivariate perturbations, no cardiac-event labels",
        "patients":patients,"evaluated_observations":len(data),"evaluation_coverage":len(data)/(patients*23),
        "metrics":metrics(data.label,p,threshold),
        "ci95":bootstrap(data.label,p,data.patient,threshold,repeats),
        "brier_interpretation":"descriptive index only; not probability calibration",
        "threshold_rule":"fixed deviation index >= 2.5, not optimized on these test patients"}
    by_scenario=[]
    for name,part in data.groupby("scenario"):
        by_scenario.append(dict(scenario=name,**metrics(part.label,1-np.exp(-part.score/3),threshold)))
    pd.DataFrame(by_scenario).to_csv(ROOT/"outputs/metrics/synthetic_by_scenario.csv",index=False)
    dump_json(result,ROOT/"outputs/metrics/synthetic_evaluation.json")
    return result
