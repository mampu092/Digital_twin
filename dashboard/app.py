from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import json
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from src.common import ROOT, setup
from src.data.simulation import demo_profile, simulate, SCENARIOS
from src.data.loader import SENSORS
from src.models.baselines import load_bundle, predict
from src.twin.digital_twin import CardiovascularDigitalTwin
from src.twin.intervention_engine import DISCLAIMER
from src.evaluation.explainability import data_quality
from dashboard.components.charts import sensor_chart

setup()
st.set_page_config(page_title="Cardio Twin | Clinical Research", page_icon="🩸", layout="wide")
st.markdown("""<style>
.stApp{background:#f5f8fa;color:#17354b}
.block-container{padding-top:4rem;padding-bottom:1rem}
[data-testid="stSidebar"]{background:#14354b}
[data-testid="stSidebar"] *{color:#e9f3f7}
[data-testid="stSidebar"] [data-baseweb="select"] *{color:#17354b}
[data-testid="stMetric"]{background:white;border:1px solid #dde7ed;border-radius:12px;padding:18px}
[data-testid="stMetricLabel"]{color:#526c7e}
.eyebrow{letter-spacing:.18em;font-size:12px;font-weight:700;color:#058b8c}
.headline{font-size:32px;line-height:1.15;font-weight:750;margin:8px 0}
.subline{font-size:15px;color:#61788a;margin-bottom:24px}
.card{background:white;border:1px solid #dde7ed;border-radius:12px;padding:20px;margin:12px 0}
.badge{display:inline-block;border-radius:20px;padding:5px 12px;font-size:12px;background:#e4f3f1;color:#087f83;font-weight:600}
</style>""", unsafe_allow_html=True)

@st.cache_resource
def get_models():
    return {target: load_bundle(target) for target in ["heart_disease", "hypertension"]}

@st.cache_data
def read_csv(relative):
    return pd.read_csv(ROOT / relative)

def load_json(relative):
    return json.loads((ROOT / relative).read_text(encoding="utf-8"))

def render():
    with st.sidebar:
        st.markdown("## CARDIO TWIN")
        st.caption("Clinical research workspace")
        
        # Added "Monte Carlo Simulation" to the navigation list
        page = st.radio("Navigation", ["Patient Overview", "Physiological Monitoring", "Cardiovascular Risk Analysis",
                                       "Digital Twin Simulation", "Intervention Explorer", "Research and Model Validation",
                                       "Loneliness and BP Research", "Monte Carlo Simulation", "Patient Report"])
        st.divider()
        scenario = st.selectbox("Synthetic scenario", SCENARIOS, format_func=lambda x: x.replace("_", " ").title(), index=1)
        days = st.slider("Replay observations", 1, 30, 30)
        st.caption("All patient and sensor displays are synthetic. Research use only.")

    st.markdown('<div class="eyebrow">CARDIOVASCULAR DIGITAL TWIN / RESEARCH PROTOTYPE</div>', unsafe_allow_html=True)
    st.markdown(f'<div class="headline">{page}</div>', unsafe_allow_html=True)
    st.markdown('<div class="subline">Static clinical phenotyping meets a changing, patient-relative physiological state.</div>', unsafe_allow_html=True)
    st.caption("Cross-sectional disease classification • Synthetic temporal demonstration • No validated event forecast")

    try:
        models = get_models()
    except FileNotFoundError as error:
        st.error(str(error)); st.stop()

    profile = demo_profile()
    frame = simulate(scenario=scenario).iloc[:days]
    twin = CardiovascularDigitalTwin(profile, models)

    for row in frame.replace({np.nan: None}).to_dict("records"):
        twin.ingest_wearable_data(row)

    scores = twin.estimate_baseline_risk()
    current = twin.state.anomalies[-1]

    if page == "Patient Overview":
        st.markdown('<span class="badge">SYNTHETIC PATIENT · SYNTH-DEMO-001</span>', unsafe_allow_html=True)
        st.subheader("Patient state and physiological change")
        cols = st.columns(4)
        cols[0].metric("Heart disease", f"{scores['heart_disease']['score']:.1%}", help="Recorded-status classification score")
        cols[1].metric("Hypertension", f"{scores['hypertension']['score']:.1%}", help="Recorded-status classification score")
        cols[2].metric("Deviation index", f"{twin.state.latent_state[-1]:.2f}", help="Smoothed patient-relative physiological deviation")
        cols[3].metric("Sensors available", f"{(frame[SENSORS].iloc[-1].notna()).mean():.0%}")

        left, right = st.columns([2, 1])
        with left:
            st.plotly_chart(sensor_chart(frame, "nremhr", twin.state.baseline, twin.state.anomalies), use_container_width=True)
            st.caption(f"Last update: {frame.timestamp.iloc[-1]} · {len(frame)} simulated nightly observations")
        with right:
            st.markdown('<div class="card"><b>Clinical context</b><br>58 years · Female<br>BMI 29.0 · Never smoked<br>Invented profile for demonstration</div>', unsafe_allow_html=True)
            st.info(current["status"].replace("_", " ").capitalize())
            st.caption("Deviation reflects change from the first seven nights. It is not a cardiac-event probability.")
        quality = scores["heart_disease"]["quality"]
        st.warning("Imputed baseline fields: " + ", ".join(quality["imputed_fields"]))
        st.caption(quality["uncertainty"])

    elif page == "Physiological Monitoring":
        st.info("Simulation units are explicitly assumed. Original wearable units and stress-score direction are unverified.")
        for row in range(3):
            columns = st.columns(2)
            for col, sensor in zip(columns, SENSORS[row*2:row*2+2]):
                with col: st.plotly_chart(sensor_chart(frame, sensor, twin.state.baseline, twin.state.anomalies), use_container_width=True)
        st.caption("Shaded band: enrollment median ± 2 robust scale units. Diamond: multisensor deviation; not a diagnostic threshold.")

    elif page == "Cardiovascular Risk Analysis":
        target = st.selectbox("Recorded classification target", list(models))
        bundle = models[target]
        clinical = load_bundle(target, "clinical")
        combined = scores[target]["score"]
        clinical_score = float(predict(clinical, pd.DataFrame([profile]))[0])
        cols = st.columns(3)
        cols[0].metric("Clinical-only score", f"{clinical_score:.1%}")
        cols[1].metric("Clinical + wearable score", f"{combined:.1%}")
        cols[2].metric("Calibrated decision threshold", f"{bundle['threshold']:.3f}")

        explanations = pd.DataFrame(twin.explain_prediction()[target]).head(12)
        fig = px.bar(explanations.sort_values("score_difference"), x="score_difference", y="feature", orientation="h",
                     color="score_difference", color_continuous_scale=["#8e64b0", "#058b8c"])
        fig.update_layout(height=460, xaxis_title="Change when replacing one feature with training reference", yaxis_title="")
        st.plotly_chart(fig, use_container_width=True)
        st.caption("One-feature reference substitutions are non-additive and noncausal. Global tree SHAP summaries are in the validation outputs where supported.")

        if (ROOT/"outputs/metrics/calibration_points.csv").exists():
            curve = read_csv("outputs/metrics/calibration_points.csv").query("target == @target")
            st.plotly_chart(px.line(curve, x="predicted", y="observed", color="features", markers=True, title="Held-out calibration"), use_container_width=True)
        st.json(data_quality(bundle, profile))
        st.warning("Disease labels have no event times. These outputs cannot predict an imminent heart attack or demonstrate treatment benefit.")

    elif page == "Digital Twin Simulation":
        st.info("Use the replay slider to ingest the first 1–30 observations. The enrollment reference remains fixed.")
        states = pd.DataFrame(twin.state.anomalies)
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=states.timestamp, y=states.latent_state, name="Smoothed deviation", line=dict(color="#058b8c", width=3)))
        fig.add_trace(go.Scatter(x=states.timestamp, y=states.latent_state+1.96*states.process_sd, name="Heuristic state band", line=dict(width=0), showlegend=False))
        fig.add_trace(go.Scatter(x=states.timestamp, y=states.latent_state-1.96*states.process_sd, fill="tonexty",
                                 fillcolor="rgba(5,139,140,.12)", line=dict(width=0), showlegend=False))
        st.plotly_chart(fig, use_container_width=True)
        st.caption("Band propagates an assumed process variance. It is not a calibrated clinical confidence interval.")

        current_scenario = twin.estimate_current_scenario()
        st.write("Classification scores under the latest synthetic inputs")
        for target, result in current_scenario.items():
            st.metric(target.replace("_", " ").title(), f"{result['score']:.1%}",
                      f"{result['score']-scores[target]['score']:+.3f} vs static input score", delta_color="off")
        st.caption("These are input-perturbation scores, not a change in established diagnosis or validated event risk.")

        chosen = st.selectbox("Compare trajectory", SENSORS, index=4)
        shift = st.slider("Assumed additive sensor change", -5.0, 5.0, 1.0)
        altered = frame.copy()
        from src.data.simulation import BOUNDS
        altered[chosen] = (altered[chosen]+shift).clip(*BOUNDS[chosen])
        st.plotly_chart(sensor_chart(frame, chosen, comparison=altered), use_container_width=True)
        if st.button("Evaluate experimental fusion network"):
            from src.models.fusion_training import infer_fusion
            st.json(infer_fusion(profile, frame))
        st.caption("Fusion combines real-label training with synthetic sequences. Attention is not evidence of biological causality.")

    elif page == "Intervention Explorer":
        st.warning(DISCLAIMER)
        left, right = st.columns([1, 2])
        with left:
            sleep = st.slider("Assumed sleep duration (hours)", 2.0, 12.0, 7.5)
            activity = st.selectbox("Assumed physical activity code", [0, 1], index=1)
            stress = st.slider("Assumed stress score (raw source scale)", 0.0, 100.0, 50.0)
            recovery = st.slider("Assumed RMSSD (simulation units)", 5.0, 180.0, 50.0)
            duration = st.slider("Illustrative duration (days)", 1, 90, 14)
            st.caption("Original activity coding and stress-score direction are unverified. Duration has no learned response effect.")
            result = twin.simulate_intervention(dict(intervention_type="sleep_regularity", duration_days=duration,
                                                    assumed_feature_changes={"sleep_hours": sleep, "physical_activity": activity, "stress": stress, "rmssd": recovery}))
            st.session_state["scenario_result"] = result
        with right:
            comp = pd.DataFrame([{"target": t, "state": s, "score": r["s"]} for t, r in result["classification_scenarios"].items() for s in ["before", "after"]])
            st.plotly_chart(px.bar(comp, x="target", y="score", color="state", barmode="group",
                                   color_discrete_sequence=["#6f8899", "#058b8c"]), use_container_width=True)
            st.write(result["notes"])
            for t, r in result["classification_scenarios"].items():
                st.write(f"{t}: {r['difference']:+.3f} classification-score change.")
        st.info("Care adherence, blood-pressure tracking and clinical review are supported as monitoring scenarios, with no fabricated physiological effect.")

    elif page == "Research and Model Validation":
        if not (ROOT/"outputs/metrics/test_metrics.csv").exists():
            st.info("Run scripts/evaluate_models.py to generate validation outputs."); return
        audit = load_json("reports/audit_summary.json")
        cols = st.columns(4)
        cols[0].metric("Source records", f"{audit['rows']:,}")
        cols[1].metric("Source columns", f"{audit['columns']:,}")
        cols[2].metric("Complete cases", f"{audit['complete_cases']:,}")
        cols[3].metric("Verified longitudinal data", "None")
        st.subheader("Track A · Real-data recorded-status classification")
        result = read_csv("outputs/metrics/test_metrics.csv")
        st.dataframe(result, width="stretch", hide_index=True)
        st.caption("Model selection uses tune data; calibration uses separate calibration data. Test estimates include 95% cluster-bootstrap intervals.")
        st.json(load_json("outputs/metrics/bootstrap_c195.json"), expanded=False)
        target = st.selectbox("Validation target", ["heart_disease", "hypertension"])
        st.image(str(ROOT/f"outputs/figures/{target}_combined_validation.png"))
        row = result[(result.target==target)&(result.features=="combined")].iloc[0]
        st.plotly_chart(px.imshow([[row.tn,row.fp],[row.fn,row.tp]], text_auto=True,
                                 labels=dict(x="Predicted status",y="Recorded status"),title="Held-out confusion matrix",
                                 color_continuous_scale="Teal"), use_container_width=True)
        st.subheader("Wearable ablation and provenance sensitivity")
        st.json(load_json("outputs/metrics/ablation.json"))
        if (ROOT/"outputs/metrics/matched_family_ablation.json").exists():
            with st.expander("Matched algorithm and hyperparameter ablation"):
                st.json(load_json("outputs/metrics/matched_family_ablation.json"))
        st.dataframe(read_csv("outputs/metrics/source_signature_holdout.csv"), hide_index=True)
        st.caption("Candidate signatures are repeated NREM-HR/RMSSD pairs, not verified source patient IDs.")
        with st.expander("Model selection, subgroups, missingness and distribution"):
            for path in ["model_selection", "subgroup_metrics", "missingness_sensitivity", "train_test_distribution"]:
                st.write(path.replace("_"," ").title()); st.dataframe(read_csv(f"outputs/metrics/{path}.csv"), hide_index=True)
        st.dataframe(read_csv("reports/missingness.csv"), hide_index=True)
        st.dataframe(read_csv("reports/class_balance.csv"), hide_index=True)
        st.subheader("Track B · Synthetic temporal evaluation")
        st.json(load_json("outputs/metrics/synthetic_evaluation.json"))
        if (ROOT/"outputs/metrics/fusion_evaluation.json").exists():
            st.subheader("Experimental mixed-source fusion"); st.json(load_json("outputs/metrics/fusion_evaluation.json"))
        st.warning("No external clinical validation. Source provenance, source-donor overlap, sensor semantics and temporal label ordering remain unresolved.")

    elif page == "Loneliness and BP Research":
        from dashboard.components.research_lab import render_research_lab
        render_research_lab()

    elif page == "Monte Carlo Simulation":
        st.info("Run the Monte Carlo simulation to explore outcome distributions and uncertainty bounds.")
        st.markdown("""
            This section utilizes the new `Monte_Carlo` module to run stochastic simulations based on the current patient state.
        """)
        
        if st.button("Run Simulation"):
            with st.spinner("Running iterations..."):
                try:
                    # --- NOTE: CHANGE THIS TO MATCH YOUR FOLDER/FILE NAMES ---
                    # Change 'Monte_Carlo_Core.simulation' to your actual folder and file path
                    # Change 'run_simulation' to your actual function name
                    from Monte_Carlo_Core.simulation import run_simulation 
                    
                    # Run your function (pass the data it needs, like `frame` or `profile`)
                    mc_results = run_simulation(frame)
                    
                    st.success("Simulation completed successfully!")
                    
                    # --- NOTE: UPDATE THIS TO PLOT YOUR ACTUAL RESULTS ---
                    st.subheader("Results")
                    # If your function returns a figure, plot it like this:
                    # st.plotly_chart(mc_results, use_container_width=True)
                    
                    # If it returns raw data, plot it:
                    # st.line_chart(mc_results)
                    
                    st.json({"status": "success", "message": "Insert your plotting code here."})
                    
                except Exception as e:
                    st.error(f"Error running simulation: {e}")
                    st.info("Did you update the import statement in the 'Monte Carlo Simulation' section of app.py to match your folder structure?")

    elif page == "Patient Report":
        if "scenario_result" in st.session_state:
            twin.state.interventions.append(st.session_state["scenario_result"])
        st.write("Download a report of the current synthetic trajectory, baseline outputs, feature substitutions, data quality and any explored scenario.")
        content=twin.export_patient_report()
        st.download_button("Download synthetic patient PDF", data=content, file_name="synthetic_patient_report.pdf", mime="application/pdf")
        st.caption("The report contains the invented demonstration patient only.")
        st.json({"patient": profile["patient_id"], "observations": len(frame), "current_status": current["status"]})

render()