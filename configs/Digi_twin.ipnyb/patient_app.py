import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

st.set_page_config(page_title="Individual Patient Dashboard", layout="wide")
st.title(" Individual Patient Digital Twin")
st.markdown("Select a patient ID to view their specific clinical and Fitbit profile.")

@st.cache_data
def load_data():
    df = pd.read_csv('FINAL_dataset_with_fitbit.csv')

    if 'patient_id' not in df.columns:
        df.insert(0, 'patient_id', df.index)
        
    return df

try:
    df = load_data()
except FileNotFoundError:
    st.error("⚠️ `FINAL_dataset_with_fitbit.csv` not found. Please ensure the file is in the same folder.")
    st.stop()

st.sidebar.header("Patient Search")

min_id = int(df['patient_id'].min())
max_id = int(df['patient_id'].max())

selected_id = st.sidebar.number_input(
    "Enter Patient ID:", 
    min_value=min_id, 
    max_value=max_id, 
    value=min_id, 
    step=1
)

patient_data = df[df['patient_id'] == selected_id].iloc[0]

cohort_avg = df.mean(numeric_only=True)

st.header(f"Patient Profile: #{selected_id}")

col1, col2, col3 = st.columns(3)

with col1:

    age = patient_data.get('age', 'N/A')
    st.metric("Age", f"{age}")
    
with col2:
    gender = patient_data.get('gender', 'N/A')
    if gender == 0 or gender == '0': gender = "Female/Male (0)"
    elif gender == 1 or gender == '1': gender = "Male/Female (1)"
    st.metric("Gender", str(gender))
    
with col3:
    bmi = patient_data.get('bmi', 'N/A')
    st.metric("BMI", f"{bmi:.1f}" if isinstance(bmi, (int, float)) else bmi)

st.subheader("🩺 Clinical Conditions")
clinic_col1, clinic_col2, clinic_col3 = st.columns(3)

def display_condition(col, label, value):
    if pd.isna(value):
        col.info(f"**{label}**: Unknown")
    elif value == 1 or str(value).lower() == 'yes' or str(value).lower() == 'true':
        col.error(f"**{label}**: Positive")
    else:
        col.success(f"**{label}**: Negative")

with clinic_col1:
    display_condition(clinic_col1, "Diabetes", patient_data.get('diabetes', None))
with clinic_col2:
    display_condition(clinic_col2, "Heart Disease", patient_data.get('heart_disease', None))
with clinic_col3:
    display_condition(clinic_col3, "Hypertension", patient_data.get('hypertension', None))

st.markdown("---")


st.subheader("🏃 Assigned Fitbit Vitals")
st.markdown("*Note: Fitbit data is synthetically assigned based on a random match, not real-time device data.*")

fitbit_cols = ['stress_score', 'spo2', 'rmssd', 'nightly_temperature', 'nremhr', 'sleep_points_percentage']
available_fitbit = [col for col in fitbit_cols if col in df.columns]

if available_fitbit:
    fitbit_metrics_cols = st.columns(len(available_fitbit))
    
    for idx, col_name in enumerate(available_fitbit):
        val = patient_data.get(col_name, None)
        if pd.notna(val):
            fitbit_metrics_cols[idx].metric(
                label=col_name.replace('_', ' ').title(), 
                value=f"{val:.2f}"
            )
        else:
            fitbit_metrics_cols[idx].metric(label=col_name, value="N/A")
else:
    st.info("No Fitbit columns found in the dataset.")

st.markdown("---")

st.subheader("How this patient compares to the cohort")
metrics_to_compare = ['age', 'bmi', 'spo2', 'stress_score']
valid_metrics = [m for m in metrics_to_compare if m in df.columns]

if valid_metrics:
    comp_cols = st.columns(len(valid_metrics))
    
    for idx, metric in enumerate(valid_metrics):
        with comp_cols[idx]:
            p_val = patient_data.get(metric, 0)
            c_val = cohort_avg.get(metric, 0)
            
            fig = go.Figure(data=[
                go.Bar(name='Patient', x=[''], y=[p_val], marker_color='#ff7f0e'),
                go.Bar(name='Cohort Avg', x=[''], y=[c_val], marker_color='#1f77b4')
            ])
            
            fig.update_layout(
                title=f"{metric.replace('_', ' ').title()}",
                barmode='group',
                height=300,
                margin=dict(l=20, r=20, t=40, b=20),
                showlegend=False
            )
            st.plotly_chart(fig, use_container_width=True)