"""Research pathway view backed by saved calculations, never raw patient records."""
import json
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from src.common import ROOT


def render_research_lab():
    st.info("Research hypothesis: loneliness → anxiety/stress → physiological change → sustained BP change → cardiovascular outcomes. This causal chain has not been established by the supplied dataset.")
    tabs = st.tabs(["Evidence and data", "Psychological addition", "Longitudinal model", "Physiological state", "Intervention research"])
    def read(name):
        return json.loads((ROOT/"outputs/metrics"/name).read_text(encoding="utf-8"))
    with tabs[0]:
        st.subheader("Three horizons, distinct outcomes")
        st.table(pd.DataFrame([
            {"Horizon":"Minutes to hours","Target":"Independently labeled stress/autonomic episode","Current evidence":"Architecture only; synthetic next-step exercise is nightly"},
            {"Horizon":"Weeks to months","Target":"Repeated SBP/DBP, nocturnal patterns, resting HR","Current evidence":"Synthetic BP and state-space calculation"},
            {"Horizon":"Years","Target":"Incident hypertension / adjudicated CVD with follow-up","Current evidence":"Synthetic right-censored survival head"}]))
        path=ROOT/"reports/research_capabilities.csv"
        if path.exists():
            st.dataframe(pd.read_csv(path),hide_index=True,width="stretch")
        st.warning("Loneliness, measured BP, validated anxiety scores, intervention assignments and incident-event follow-up are absent. ENTERTAINMENT is not a loneliness scale; TENSE/ANXIOUS has unverified scale meaning.")
        st.caption("Research protocol, biomarker priorities, equations and screenshot traceability: docs/research_extension.md and reports/enhancement_audit.md.")
    with tabs[1]:
        path=ROOT/"outputs/metrics/psychosocial_domain_metrics.csv"
        if not path.exists():
            st.info("Run scripts/analyze_psychosocial.py with the authorized source CSV.")
        else:
            st.subheader("What do the available psychological proxies add?")
            result=pd.read_csv(path)
            target=st.selectbox("Domain comparison target",["heart_disease","hypertension"],key="research_target")
            result=result.loc[result.target.eq(target)]
            st.dataframe(result[["domain","roc_auc","pr_auc","brier","sensitivity","specificity"]],hide_index=True,width="stretch")
            st.caption("Identical HGB settings across four feature sets; preprocessing trained on training data, calibration separate. This is an exploratory reuse of the previously inspected test set.")
            st.json(read("psychosocial_ablation.json")["contrasts"][target],expanded=False)
            st.subheader("Adjusted cross-sectional associations")
            association=pd.read_csv(ROOT/"outputs/metrics/psychosocial_associations.csv")
            st.dataframe(association.loc[association.target.eq(target)],hide_index=True,width="stretch")
            st.caption("Odds ratios per observed SD of the raw variable; adjustment for age, gender, BMI, diabetes and smoking. Heart-disease analysis additionally includes hypertension and its interaction with the proxy. These are conditional associations, not causal mediation.")
            st.dataframe(pd.read_csv(ROOT/"outputs/metrics/hypertension_psych_strata.csv"),hide_index=True,width="stretch")
    with tabs[2]:
        path=ROOT/"outputs/metrics/longitudinal_synthetic_evaluation.json"
        if not path.exists():
            st.info("Run scripts/run_research_demo.py.")
        else:
            result=read(path.name)
            st.warning("Entirely synthetic participants, loneliness exposure, assays, BP and outcomes. These scores are software demonstrations.")
            st.markdown("Four separate encoders combine physiology, psychological measures, assays and personal context. Missingness masks and elapsed time feed a history-only GRU, with acute, BP, psychological and survival heads.")
            cols=st.columns(3)
            cols[0].metric("Synthetic SBP MAE",f"{result['bp']['sbp_mae_mmhg']:.2f} mmHg")
            cols[1].metric("Synthetic acute AUROC",f"{result['acute']['roc_auc']:.3f}")
            cols[2].metric("Held-out invented participants",result["participants"]["test"])
            st.json(result,expanded=False)
            participant=int(st.number_input("Invented held-out participant",1000,1199,1000))
            if st.button("Run saved synthetic model"):
                from src.research.inference import synthetic_prediction
                st.json(synthetic_prediction(participant))
            st.caption("BP loss is Huber; acute loss is binary cross-entropy plus Brier; event likelihood respects right censoring. Missing labels do not contribute to loss. Brier loss alone does not establish calibration.")
            probs=result["survival"]["mean_cumulative_probability"]
            st.plotly_chart(px.line(x=list(range(1,len(probs)+1)),y=probs,markers=True,
                labels={"x":"Illustrative annual interval","y":"Synthetic mean cumulative event probability"}),use_container_width=True)
    with tabs[3]:
        path=ROOT/"data/synthetic/research_state_trajectory.csv"
        if path.exists():
            frame=pd.read_csv(path)
            axis=st.selectbox("Synthetic latent coordinate",list(frame.axis.unique()))
            data=frame.loc[frame.axis.eq(axis)]
            figure=go.Figure()
            figure.add_scatter(x=data.day,y=data["mean"]+1.96*data.sd,line=dict(width=0),showlegend=False)
            figure.add_scatter(x=data.day,y=data["mean"]-1.96*data.sd,line=dict(width=0),fill="tonexty",name="Assumed-model band")
            figure.add_scatter(x=data.day,y=data["mean"],name=axis,line=dict(color="#058b8c",width=3))
            figure.update_layout(xaxis_title="Synthetic day",yaxis_title="Unitless latent coordinate")
            st.plotly_chart(figure,use_container_width=True)
            if not data.measurement_supported.any():
                st.warning("No measurements support this coordinate. The trajectory is driven only by invented model assumptions.")
            st.caption("Kalman updates propagate measurement and process uncertainty, including missing readings. Coordinates are not identified biological mechanisms; bands depend on assumed coefficients.")
            st.json(read("synthetic_bp_summary.json"),expanded=False)
            st.caption("BP summaries calculate pulse pressure, approximate resting MAP and sleep/wake SBP dip. No diagnosis is inferred. Sleep state and validated-device flags are invented in this example.")
    with tabs[4]:
        path=ROOT/"outputs/metrics/synthetic_causal_evaluation.json"
        if path.exists():
            data=read(path.name)
            est=data["parallel_trial_centered"]
            cols=st.columns(2)
            cols[0].metric("Injected synthetic effect",f"{data['known_parallel_effect']:.1f} mmHg")
            cols[1].metric("Estimated synthetic effect",f"{est['estimate']:.2f} mmHg")
            st.write(f"95% participant-bootstrap interval: {est['ci95'][0]:.2f} to {est['ci95'][1]:.2f} mmHg.")
            st.caption("The centered estimator subtracts a prespecified 130 mmHg constant before weighting. Raw HT results are retained to show its higher finite-sample variance.")
            st.json(data,expanded=False)
            mrt=data["mrt_proximal_centered"]
            truth=data["known_mrt_eligible_mean_effect"]
            if not mrt["ci95"][0]<=truth<=mrt["ci95"][1]:
                st.caption("In this synthetic run the proximal-trial interval misses the injected mean effect. One run does not validate interval coverage; this finite-sample result is retained.")
            st.warning("This validates estimator behavior under invented randomization assumptions. It does not demonstrate that loneliness interventions lower BP or justify treatment selection.")
            st.markdown("A real study needs validated loneliness/anxiety instruments, repeated cuff BP, treatment and medication logs, eligibility and assignment probabilities, prespecified outcomes, and independent clinical validation.")
