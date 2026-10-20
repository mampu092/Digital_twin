"""Local clinician-facing research UI. Run with streamlit run dashboard.py."""
import json
import os
from pathlib import Path
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

st.set_page_config(page_title='Virtual Patient Lab',layout='wide')
ROOT = Path(__file__).parent
OUT = Path(os.environ.get('TWIN_OUTPUT_DIR',str(ROOT/'outputs')))
path = OUT/'dashboard_data.json'
st.title('Virtual Patient Lab')
st.caption('Fitbit summary data and simulated EHR / wearable fusion. Research prototype.')
if not path.exists():
    st.info('Run monte_carlo_digital_twin.py first. See README.md.')
    st.stop()
data = json.loads(path.read_text(encoding='utf-8'))
data['patients'].sort(key=lambda p:p['profile']['weekly_risk'])
S,A = data['summary'],data['audit']
st.warning('All trajectories, future events and intervention effects are simulated. Source licensing and anonymization are unverified. This is not a clinical decision tool.')
tabs = st.tabs(['Virtual patient','Model evaluation','Data and privacy'])

with tabs[0]:
    labels = [p['profile']['virtual_patient_id'] for p in data['patients']]
    chosen = st.selectbox('Held-out virtual patient',labels,index=len(labels)-1)
    ex = data['patients'][labels.index(chosen)]
    profile,series,decision = ex['profile'],ex['series'],ex['decision']
    c1,c2,c3,c4 = st.columns(4)
    c1.metric('Age band',profile['age_band'].replace('15-19','18-19').replace('60-64','60'))
    c2.metric('Diabetes flag',profile['diabetes'])
    c3.metric('Hypertension flag',profile['hypertension'])
    c4.metric('Modeled weekly event risk',f"{profile['weekly_risk']:.1%}")
    st.caption('Weekly baseline risk and next-two-hour spike risk have different targets. Test patients were selected by baseline-risk quantiles, without looking at future events.')
    hour = st.slider('Replay time (hours since simulation start)',24,164,72,2)
    threshold = st.slider('Alert threshold for this UI',.01,.99,float(round(decision['threshold'],2)),.01)
    retrospective = st.checkbox('Retrospective view: reveal future trajectory and labels',False)
    hours = np.array(series['hours'])
    dh = np.array(decision['hours'])
    now = int(np.flatnonzero(dh<=hour)[-1])
    risk = decision['p'][now]
    st.metric('Predicted spike risk over the next 2 hours',f'{risk:.1%}')
    st.info('Review alert: inspect recent glucose, HR and signal quality.' if risk>=threshold else 'Risk is below the selected alert threshold.')
    st.caption(f"Latest decision at hour {dh[now]:.1f}. Run-0 validation F2 threshold: {decision['threshold']:.3f}. Changing the UI threshold changes displayed alerts only.")
    mask = (hours>=max(0,hour-24))&(hours<= (hour+2 if retrospective else hour))
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=hours[mask],y=np.array(series['G'])[mask],name='Simulated glucose',line={'color':'#285a91'}))
    fig.add_hline(y=180,line_dash='dash',annotation_text='Modeled event threshold: 180 mg/dL')
    fig.add_vline(x=hour,line_dash='dot')
    fig.update_layout(title='Glucose history',xaxis_title='Simulation hour',yaxis_title='mg/dL',height=330)
    st.plotly_chart(fig,use_container_width=True)
    f = go.Figure()
    visible = (dh>=hour-24)&(dh<=hour)
    f.add_trace(go.Scatter(x=dh[visible],y=np.array(decision['p'])[visible],mode='lines+markers',name='Predicted risk'))
    f.add_hline(y=threshold,line_dash='dash',annotation_text='Selected threshold')
    f.update_layout(title='Two-hour spike predictions',xaxis_title='Simulation hour',yaxis_title='Probability',yaxis_range=[0,1],height=270)
    st.plotly_chart(f,use_container_width=True)
    left,right = st.columns(2)
    for area,key,title,unit in [(left,'HR','Heart rate','bpm'),(right,'SBP','Systolic blood pressure','mmHg')]:
        with area:
            fig = go.Figure(go.Scatter(x=hours[mask],y=np.array(series[key])[mask],name=title))
            fig.update_layout(title='Simulated '+title.lower(),xaxis_title='Simulation hour',yaxis_title=unit,height=260)
            st.plotly_chart(fig,use_container_width=True)
    st.text_area('Clinician review note (session only)',placeholder='Record evidence to review, signal quality concerns, and follow-up needed.')
    note = {'virtual_patient':chosen,'simulation_hour':hour,'probability':risk,'ui_threshold':threshold,
            'status':'simulated research replay; no patient care action recorded'}
    st.download_button('Download replay snapshot',json.dumps(note,indent=2),file_name='virtual_patient_snapshot.json')

with tabs[1]:
    st.caption(f"{S['mc_runs']} runs, {S['patients_per_run']} bootstrap trajectories per run, 7 simulated days, 15-minute steps. Only held-out identities contribute to metrics.")
    cols = ['static','wearable_no_cgm','fusion','full']
    names = ['Clinical only','Clinical + wearables','Add simulated CGM','Add psychological context']
    rows = []
    for col,name in zip(cols,names):
        z = S['metrics'][col+'_auc']
        rows.append({'Model':name,'AUROC mean':z['mean'],'MC low':z['low'],'MC high':z['high'],
                     'Average precision':S['metrics'][col+'_ap']['mean']})
    st.dataframe(pd.DataFrame(rows),hide_index=True,width='stretch')
    st.caption(S['interval_definition'])
    st.image(str(OUT/'fig2_model_performance.png'))
    st.image(str(OUT/'fig4_value_decomposition.png'))
    selected = S['metrics']
    x,y,z = st.columns(3)
    x.metric('Validation-selected precision',f"{selected['selected_precision']['mean']:.1%}")
    y.metric('Validation-selected recall',f"{selected['selected_recall']['mean']:.1%}")
    z.metric('Alerts per 100 windows',f"{selected['selected_alerts_per_100_windows']['mean']:.1f}")
    st.caption('Thresholds maximize F2 on validation data only. Positive event windows are rare; AUROC alone does not describe alert usefulness.')
    st.image(str(OUT/'fig3_causal_effect.png'))
    st.caption('MRT effects are imposed inside the simulator. Treatment-centered regression estimates the effect from randomized observed outcomes with trajectory-cluster robust uncertainty. It does not establish a clinical treatment benefit.')

with tabs[2]:
    st.image(str(OUT/'fig1_data_fusion.png'))
    st.dataframe(pd.DataFrame([{'Field':k,'Observed coverage':v} for k,v in A['observed_coverage'].items()]),hide_index=True)
    st.write('The source contains one row per patient and has no measurement timestamps. A patient-ID join is assumed by the supplied table. Its original EHR-to-Fitbit linkage cannot be verified.')
    if 'wearable_profile_reuse' in A:
        reuse=A['wearable_profile_reuse']
        st.write(f"Complete resting-HR/HRV/breathing summaries contain {reuse['distinct_triplets']} distinct triplets across {reuse['complete_hr_hrv_breathing_rows']:,} rows. Repeated summaries do not establish independent Fitbit wearers.")
    st.write('Missing values are imputed with training-only medians and accompanied by missingness indicators. Simulated data are not measured continuous glucose, blood pressure, or real-time Fitbit observations.')
    st.json(A['privacy'])
    st.write('Original patient IDs are excluded from the dashboard. Age is coarsened. These controls reduce exposure but do not establish anonymization, HIPAA compliance, DPDP compliance, or a redistribution license.')
    st.markdown('[HHS de-identification guidance](https://www.hhs.gov/hipaa/for-professionals/special-topics/de-identification/index.html) · [MeitY DPDP Rules](https://www.meity.gov.in/documents/act-and-policies/digital-personal-data-protection-rules-2025-gDOxUjMtQWa?pageTitle=Digit)')
    st.json(data['provenance'])
